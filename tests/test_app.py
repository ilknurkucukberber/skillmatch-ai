import sys
import ast
import string
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
import i18n

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


class MockUpload:
    def __init__(self, name, payload):
        self.name = name
        self._payload = payload

    def getvalue(self):
        return self._payload


def test_skill_detection_supports_turkish_and_english_aliases():
    text = "Python, SQL, makine öğrenmesi, Docker ve NLP deneyimi; ayrıca PyTorch ve AWS kullanıyorum."
    skills = app.find_skills(text)

    assert {"Python", "SQL", "Machine Learning", "Docker", "NLP", "PyTorch", "AWS"} <= skills


def test_translation_dicts_are_aligned():
    assert set(i18n.TRANSLATIONS["tr"]) == set(i18n.TRANSLATIONS["en"])


def test_resume_improvement_recommendations_are_evidence_based_and_non_duplicate():
    resume_text = """
    İletişim: ayse@example.com
    Hedef: Çalışkan, öğrenmeye açık bir öğrenciyim.
    Projeler:
    - Metin sınıflandırma modeli geliştirdim. Python, scikit-learn ve pandas kullanıldı.
    - SQL sorguları ile veri analizi yaptım.
    Beceriler: Python, SQL, Pandas
    """
    job_text = """
    Software Engineering Intern
    Required: Python, SQL, Docker, REST APIs, Git. Must have strong project experience and clear technical summaries.
    """
    summary = app.build_recommendations(
        resume_text=resume_text,
        job_text=job_text,
        matched={"Python", "SQL"},
        missing={"Docker", "REST API", "Git"},
        total_job_skills=5,
    )

    assert summary["recommendations"]
    localized = [
        app.localize_recommendation(item, "tr")
        for item in summary["recommendations"]
    ]
    titles = [item["title"] for item in localized]
    assert len(titles) == len(set(titles))
    assert any("özet" in item["title"].lower() for item in localized)
    assert any("proje" in item["title"].lower() for item in localized)
    assert 4 <= len(summary["recommendations"]) <= 6


def test_resume_section_detection_supports_turkish_and_english_headings():
    turkish = "Özet\nPython ve SQL ile veri analizi yapıyorum.\nProjeler\n - Metin analizi\nBeceriler\nPython, SQL\nEğitim\nBilgisayar Mühendisliği"
    english = "Summary\nMachine learning intern with Python projects.\nProjects\n- Recommendation system\nSkills\nPython, SQL\nEducation\nComputer Engineering"

    tr_sections = app.detect_resume_sections(turkish)
    en_sections = app.detect_resume_sections(english)

    assert tr_sections["summary"]
    assert tr_sections["skills"]
    assert en_sections["summary"]
    assert en_sections["projects"]


def test_missing_summary_and_weak_summary_are_flagged_without_fabrication():
    resume = """
    Beceriler: Python, SQL
    Deneyim: 2024 yazında staj yaptım.
    - Veri temizleme ve analiz çalışmaları yaptım.
    """
    recs = app.build_resume_improvement_recommendations(
        resume_text=resume,
        job_text="We need a software engineering intern with a strong, job-focused summary and clear project ownership.",
        missing={"Docker"},
    )

    assert recs
    localized = [app.localize_recommendation(item, "en") for item in recs]
    assert any(item["action"] in {"Add", "Improve"} for item in localized)
    assert all("not found" not in str(item["resume_evidence"]).lower() for item in recs if item["resume_evidence"])


def test_skill_emphasis_recommendation_uses_resume_evidence_only():
    resume = """
    Summary: Computer engineering student.
    Projects: Built a text classification project using Python, pandas, and scikit-learn.
    Skills: Python
    """
    recs = app.build_resume_improvement_recommendations(
        resume_text=resume,
        job_text="Need Python, SQL, Git, API development, machine learning, and project communication.",
        missing={"SQL", "Git", "Machine Learning"},
    )

    localized = [app.localize_recommendation(item, "en") for item in recs]
    assert any("project" in item["title"].lower() or "skills" in item["title"].lower() for item in localized)
    assert any("Python" in item["resume_evidence"] or "python" in item["resume_evidence"] for item in recs)


def test_resume_extraction_rejects_empty_and_corrupt_files():
    with pytest.raises(ValueError, match="empty|invalid|corrupt|only PDF"):
        app.extract_resume(MockUpload("empty.txt", b""))

    with pytest.raises(ValueError, match="invalid|corrupt"):
        app.extract_resume(MockUpload("broken.pdf", b"not a valid pdf"))


def test_model_load_uses_cache_resource(monkeypatch):
    class DummyModel:
        def __init__(self):
            self.called = 0

        def eval(self):
            return None

        def parameters(self):
            return []

        def encode(self, chunks, **kwargs):
            self.called += 1
            assert kwargs["batch_size"] == 16
            return np.ones((len(chunks), 384), dtype=np.float32)

    dummy = DummyModel()
    app.load_model.clear()
    monkeypatch.setattr(app, "load_model", lambda: dummy)

    first = app.load_model()
    second = app.load_model()

    assert first is second


def test_embedding_caching_works_for_same_input(monkeypatch):
    class DummyModel:
        def __init__(self):
            self.calls = 0

        def encode(self, chunks, **kwargs):
            self.calls += 1
            assert kwargs["normalize_embeddings"] is True
            return np.ones((len(chunks), 384), dtype=np.float32)

    dummy = DummyModel()
    monkeypatch.setattr(app, "load_model", lambda: dummy)
    app.build_embedding.clear()
    first = app.build_embedding("Python SQL NLP")
    second = app.build_embedding("Python SQL NLP")
    assert first.shape == second.shape
    assert dummy.calls == 1


def test_semantic_similarity_score_preserves_cosine_behavior():
    assert app.semantic_similarity_score(
        np.array([1.0, 0.0]), np.array([1.0, 0.0])
    ) == pytest.approx(1.0)
    assert app.semantic_similarity_score(
        np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ) == pytest.approx(0.0)
    assert app.semantic_similarity_score(
        np.array([1.0, 0.0]), np.array([-1.0, 0.0])
    ) == pytest.approx(-1.0)


def sample_analysis():
    resume = (
        "Özet\nPython ve SQL ile veri analizi yapıyorum.\n"
        "Projeler\nMetin sınıflandırma projesinde NLP kullandım.\n"
        "Beceriler\nPython, SQL, NLP"
    )
    job = "Required: Python, SQL, Docker, and NLP for software engineering."
    required = app.classify_job_skills(job)["required"]
    matched = sorted(app.find_skills(resume) & set(required))
    missing = sorted(set(required) - app.find_skills(resume))
    recommendation_data = app.build_recommendations(
        resume, job, matched, missing, len(required)
    )
    return {
        "semantic_similarity": 0.731,
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_coverage": len(matched) / len(required) * 100,
        "job_skill_count": len(required),
        "required_skills": sorted(required),
        "preferred_skills": [],
        "job_requirements": app.classify_job_skills(job),
        "resume_skills": sorted(app.find_skills(resume)),
        **recommendation_data,
        "model_load_time_seconds": 0.2,
        "inference_time_seconds": 0.1,
    }


def test_app_test_switches_languages_without_replacing_analysis(monkeypatch):
    analysis = sample_analysis()
    load_calls = []
    monkeypatch.setattr(app, "load_model", lambda: load_calls.append(True))

    at = AppTest.from_file(str(ROOT / "app.py"))
    at.session_state["analysis"] = analysis
    at.session_state["lang"] = "tr"
    at.run(timeout=10)
    assert not at.exception
    assert any("Analiz sonuçları" in item.value for item in at.subheader)
    assert any("CV'nizi Geliştirecek Öneriler" in item.value for item in at.markdown)
    turkish_text = "\n".join(item.value for item in at.markdown)
    tr_description = app.localize_recommendation(analysis["recommendations"][0], "tr")["description"]
    assert tr_description in turkish_text
    assert i18n.TRANSLATIONS["en"]["advisor_summary_rewrite"] not in turkish_text
    resume_job_text = "Required: Python, SQL, Docker, and NLP for software engineering."
    at.text_area(key="job_description").set_value(resume_job_text).run(timeout=10)

    at.selectbox(key="lang").set_value("en").run(timeout=10)
    assert not at.exception
    assert any("Analysis results" in item.value for item in at.subheader)
    assert any("Recommendations to Improve Your Resume" in item.value for item in at.markdown)
    english_text = "\n".join(item.value for item in at.markdown)
    en_description = app.localize_recommendation(analysis["recommendations"][0], "en")["description"]
    assert en_description in english_text
    assert i18n.TRANSLATIONS["tr"]["advisor_summary_rewrite"] not in english_text
    assert at.session_state["analysis"]["semantic_similarity"] == 0.731
    assert at.session_state["analysis"]["skill_coverage"] == analysis["skill_coverage"]
    assert at.text_area(key="job_description").value == resume_job_text

    at.selectbox(key="lang").set_value("tr").run(timeout=10)
    assert any("Analiz sonuçları" in item.value for item in at.subheader)
    assert at.session_state["analysis"]["matched_skills"] == analysis["matched_skills"]
    assert load_calls == []


def test_all_catalog_keys_are_unique_and_placeholders_are_aligned():
    source = (ROOT / "i18n.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "TRANSLATIONS" for target in node.targets)
    )
    translations_node = assignment.value
    key_sets = {}
    for language_key, language_dict in zip(translations_node.keys, translations_node.values):
        language = ast.literal_eval(language_key)
        keys = [ast.literal_eval(key) for key in language_dict.keys]
        assert len(keys) == len(set(keys)), f"Duplicate keys for {language}"
        key_sets[language] = set(keys)
        for template in language_dict.values:
            string.Formatter().parse(ast.literal_eval(template))

    assert key_sets["tr"] == key_sets["en"]
    for key in key_sets["tr"]:
        tr_fields = {
            field for _, field, _, _ in string.Formatter().parse(i18n.TRANSLATIONS["tr"][key]) if field
        }
        en_fields = {
            field for _, field, _, _ in string.Formatter().parse(i18n.TRANSLATIONS["en"][key]) if field
        }
        assert tr_fields == en_fields, key


@pytest.mark.parametrize("language, expected", [("tr", "Eşleşti"), ("en", "Matched")])
def test_language_specific_chart_categories(language, expected):
    chart = app.build_chart_data(["Python"], ["Docker"], language)
    assert expected in chart["Category"].tolist()
    assert chart["Count"].tolist() == [1, 1]
    figure = app.build_skill_chart(["Python"], ["Docker"], language)
    assert figure.layout.title.text == i18n.TRANSLATIONS[language]["chart_title"]
    assert figure.layout.xaxis.title.text == i18n.TRANSLATIONS[language]["chart_category_axis"]
    assert figure.layout.yaxis.title.text == i18n.TRANSLATIONS[language]["chart_count_axis"]
    assert i18n.TRANSLATIONS[language]["chart_count_axis"] in figure.data[0].hovertemplate


@pytest.mark.parametrize(
    ("resume", "job"),
    [
        ("Özet\nPython, SQL ve Docker deneyimim var.", "Gerekli: Python, SQL ve Docker deneyimi."),
        ("Summary\nPython, SQL and Docker experience.", "Gerekli: Python, SQL ve Docker deneyimi."),
        ("Özet\nPython, SQL ve Docker deneyimim var.", "Required: Python, SQL, Docker experience."),
        ("Summary\nPython, SQL and Docker experience.", "Required: Python, SQL, Docker experience."),
    ],
)
def test_bilingual_resume_and_job_skill_extraction(resume, job):
    required = set(app.classify_job_skills(job)["required"])
    assert {"Python", "SQL", "Docker"} <= required
    assert {"Python", "SQL", "Docker"} <= app.find_skills(resume)


def test_advisor_templates_and_report_follow_selected_language():
    analysis = sample_analysis()
    tr_item = app.localize_recommendation(analysis["recommendations"][0], "tr")
    en_item = app.localize_recommendation(analysis["recommendations"][0], "en")
    assert tr_item["description"] != en_item["description"]
    assert tr_item["priority_label"] in {"Yüksek", "Orta", "Düşük"}

    tr_report = app.build_report(analysis, "tr")
    en_report = app.build_report(analysis, "en")
    assert tr_report["language"] == "tr"
    assert en_report["language"] == "en"
    assert tr_report["resume_improvement_recommendations"][0]["description"] != en_report["resume_improvement_recommendations"][0]["description"]
    assert tr_report["recommendations"] == tr_report["resume_improvement_recommendations"]
    assert "Beceri kapsamı paydasına" in tr_report["denominator_definition"]
    assert "skill-coverage denominator" in en_report["denominator_definition"]


def test_translation_failures_and_upload_errors_are_visible_and_localized():
    with pytest.raises(KeyError, match="Missing"):
        app.t("not_a_translation_key", "tr")
    with pytest.raises(ValueError, match="Unsupported language"):
        app.t("app_title", "fr")

    with pytest.raises(ValueError, match="Dosya yüklenmedi"):
        app.extract_resume(None, "tr")
    with pytest.raises(ValueError, match="No file was uploaded"):
        app.extract_resume(None, "en")


def test_app_test_empty_upload_warning_uses_selected_language():
    at = AppTest.from_file(str(ROOT / "app.py"))
    at.session_state["lang"] = "en"
    at.run(timeout=10)
    at.button(key="analyze").click().run()
    assert any(
        warning.value == i18n.TRANSLATIONS["en"]["empty_resume_warning"]
        for warning in at.warning
    )


def test_every_advisor_recommendation_type_localizes_in_both_languages():
    for recommendation_type in app.RECOMMENDATION_PRESENTATION:
        recommendation = {
            "type": recommendation_type,
            "priority": "high",
            "facts": {"skill": "NLP"} if recommendation_type in {"skill_emphasis", "skill_gap"} else {},
            "resume_evidence": "Python and NLP",
            "job_evidence": "Required: NLP",
            "icon": "✨",
        }
        turkish = app.localize_recommendation(recommendation, "tr")
        english = app.localize_recommendation(recommendation, "en")
        assert turkish["description"]
        assert english["description"]
        assert turkish["suggested_revision"]
        assert english["suggested_revision"]
        assert turkish["description"] != english["description"]


def test_all_skill_guidance_has_both_languages():
    for skill, guidance in app.SKILL_GUIDANCE.items():
        assert set(guidance) == {"tr", "en"}, skill
        assert guidance["tr"]["why"]
        assert guidance["en"]["why"]
        assert guidance["tr"]["actions"]
        assert guidance["en"]["actions"]


def test_source_evidence_is_preserved_verbatim():
    resume = "Projects\n  Built a model with Python and NLP.\n"
    job = "Required: Python and NLP."
    recommendations = app.build_resume_improvement_recommendations(
        resume, job, missing={"Python"}
    )
    project = next(item for item in recommendations if item["type"] == "project_clarity")
    skill = next(item for item in recommendations if item["type"] == "skill_emphasis")
    assert "Projects\n  Built a model with Python and NLP." == project["resume_evidence"]
    assert skill["job_evidence"] == job


def test_txt_and_pdf_resume_uploads_still_work():
    assert "Python" in app.extract_resume(
        MockUpload("resume.txt", b"Python and SQL resume content.")
    )
    pdf = app.fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Python and SQL resume content.")
    payload = pdf.tobytes()
    pdf.close()
    assert "Python" in app.extract_resume(MockUpload("resume.pdf", payload))
