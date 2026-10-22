import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timezone

import fitz
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from i18n import LANGUAGE_OPTIONS, t

MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024
MAX_PDF_PAGES = 15
MAX_TEXT_CHARS = 12000
MAX_EMBEDDING_CHARS = 1800


class ResumeExtractionError(ValueError):
    pass


SKILLS = {
    "Python": ["python", "py"],
    "SQL": ["sql", "postgresql", "mysql", "database queries"],
    "Machine Learning": ["machine learning", "ml", "makine ogrenmesi", "makine öğrenmesi"],
    "Deep Learning": ["deep learning", "derin öğrenme", "cnn", "rnn"],
    "NLP": ["nlp", "natural language processing", "doğal dil işleme", "text mining"],
    "PyTorch": ["pytorch"],
    "TensorFlow": ["tensorflow"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "Docker": ["docker"],
    "Git": ["git"],
    "GitHub": ["github"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "azure devops"],
    "REST API": ["rest api", "restful api", "api development"],
    "FastAPI": ["fastapi"],
    "Streamlit": ["streamlit"],
    "Flask": ["flask"],
    "Django": ["django"],
    "React": ["react", "reactjs", "react.js"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript", "ts"],
    "HTML": ["html"],
    "CSS": ["css", "sass", "tailwind"],
    "C++": ["c++", "cpp"],
    "Java": ["java"],
    "Linux": ["linux", "ubuntu", "bash"],
    "Firebase": ["firebase"],
    "Kubernetes": ["kubernetes", "k8s"],
    "CI/CD": ["ci/cd", "continuous integration", "continuous delivery", "devops"],
    "Data Analysis": ["data analysis", "veri analizi", "analytics"],
    "Computer Vision": ["computer vision", "bilgisayarla görme", "opencv"],
    "LLM": ["llm", "large language model", "büyük dil modeli", "buyuk dil modeli"],
}

SKILL_GUIDANCE = {
    "Python": {
        "en": {"why": "Python is commonly required for scripting, automation, data processing, and backend or ML tasks.", "actions": ["Build a small CLI data project with pytest and pandas.", "Convert a notebook workflow into a reusable Python package or script."]},
        "tr": {"why": "Python, otomasyon, veri işleme ve arka uç/ML görevleri için sık kullanılan bir beceridir.", "actions": ["Pandas ve pytest ile küçük bir CLI veri projesi geliştirin.", "Bir not defteri iş akışını tekrar kullanılabilir bir Python paketi veya betiğine dönüştürün."]},
    },
    "SQL": {
        "en": {"why": "The job description signals a need for querying, joins, and data quality checks.", "actions": ["Create an SQLite analytics notebook with joins, CTEs, and KPI queries.", "Write a repository of SQL questions for filtering, aggregation, and reporting."]},
        "tr": {"why": "İş ilanı sorgulama, join ve veri kalitesi kontrolü ihtiyacını gösteriyor.", "actions": ["Join, CTE ve KPI sorguları içeren bir SQLite analizi hazırlayın.", "Filtreleme, toplama ve raporlama için SQL soruları içeren bir örnek seti hazırlayın."]},
    },
    "Machine Learning": {
        "en": {"why": "This is a core skill for model training, evaluation, and feature interpretation tasks.", "actions": ["Train a small scikit-learn model and explain feature importance.", "Publish a short notebook with preprocessing, metrics, and model trade-offs."]},
        "tr": {"why": "Bu beceri, model eğitimi, değerlendirme ve öznitelik yorumlama için temel bir gerekliliktir.", "actions": ["Küçük bir scikit-learn modeli eğitin ve öznitelik önemini açıklayın.", "Ön işleme, metrikler ve model trade-off'larını gösteren kısa bir not defteri yayınlayın."]},
    },
    "Deep Learning": {
        "en": {"why": "The target role appears to value neural-network modeling and representation learning.", "actions": ["Train a CNN on a public dataset and document loss curves.", "Compare a simple neural-net model against a baseline model in a small notebook."]},
        "tr": {"why": "Hedef rol, sinir ağı modelleme ve temsil öğrenimi konularını önemsemektedir.", "actions": ["Halka açık bir veri seti üzerinde CNN eğitin ve loss eğrilerini belgeleyin.", "Küçük bir not defterinde basit bir sinir ağı modeli ile bir baseline'i karşılaştırın."]},
    },
    "NLP": {
        "en": {"why": "The role emphasizes text pipelines, semantic understanding, and language-model-based tasks.", "actions": ["Build a sentiment-analysis demo with preprocessing and evaluation notes.", "Create a small semantic-search or text-classification prototype."]},
        "tr": {"why": "Rol, metin boru hatları, anlamsal anlama ve dil modeli tabanlı görevleri vurguluyor.", "actions": ["Ön işleme ve değerlendirme notlarıyla duygu analizi demosu hazırlayın.", "Küçük bir anlamsal arama veya metin sınıflandırma prototipi oluşturun."]},
    },
    "Docker": {
        "en": {"why": "Containerized execution and reproducible setup are often essential in production-oriented roles.", "actions": ["Containerize a small Flask or FastAPI service with Dockerfile and compose settings.", "Document local startup, environment variables, and production run steps."]},
        "tr": {"why": "Konteynerli çalışma ve tekrar üretilebilir kurulum, üretim odaklı roller için genellikle kritiktir.", "actions": ["Küçük bir Flask veya FastAPI servisini Dockerfile ve compose ayarlarıyla konteynerleştirin.", "Yerel başlatma, ortam değişkenleri ve üretim çalıştırma adımlarını belgeleyin."]},
    },
    "Git": {
        "en": {"why": "Version control discipline is a baseline expectation for collaborative engineering work.", "actions": ["Use a feature-branch workflow for a small project and document commit patterns.", "Create a branch, PR-like summary, and resolve a small merge conflict in a demo repo."]},
        "tr": {"why": "Versiyon kontrolü, işbirlikçi mühendislik için temel bir beklentidir.", "actions": ["Küçük bir proje için feature branch akışı kullanın ve commit yapılarınızı belgeleyin.", "Demo bir depoda branch, PR benzeri özet ve küçük bir merge conflict çözümü gösterin."]},
    },
    "GitHub": {
        "en": {"why": "The role likely expects collaboration workflows, code review, and project visibility through GitHub.", "actions": ["Publish a clean repository with README, issue templates, and project structure.", "Create a small pull request workflow demo with a changelog and test notes."]},
        "tr": {"why": "Rol, GitHub üzerinden işbirliği, kod inceleme ve proje görünürlüğü beklentisi taşıyabilir.", "actions": ["README, issue şablonları ve proje yapısı içeren temiz bir depo yayınlayın.", "Değişiklik günlüğü ve test notlarıyla küçük bir pull request akışı demosu oluşturun."]},
    },
    "Pandas": {
        "en": {"why": "Analytical roles often require robust data cleaning and transformation workflows.", "actions": ["Build a mini ETL script with pandas and show the transformations clearly.", "Create a notebook that cleans messy CSV files and explains the data quality decisions."]},
        "tr": {"why": "Analitik roller, veri temizleme ve dönüştürme akışlarını güçlü şekilde gerektirir.", "actions": ["Pandas ile küçük bir ETL betiği oluşturup dönüşümleri net şekilde gösterin.", "Dağınık CSV dosyalarını temizleyen ve veri kalitesi kararlarını açıklayan bir not defteri hazırlayın."]},
    },
    "NumPy": {
        "en": {"why": "NumPy is frequently needed for vectorized numerical work and preprocessing pipelines.", "actions": ["Write a short array-transformation script and explain the performance benefit.", "Solve a small numeric problem with broadcasting, reshaping, and indexing examples."]},
        "tr": {"why": "NumPy, vektörize sayısal işlemler ve ön işleme boru hatları için sık kullanılır.", "actions": ["Kısa bir dizi dönüştürme betiği yazıp performans avantajını açıklayın.", "Broadcasting, reshape ve indexing örnekleriyle küçük bir sayısal problem çözün."]},
    },
    "Streamlit": {
        "en": {"why": "Interactive dashboards and proof-of-concept apps are often expected in data roles.", "actions": ["Build a simple Streamlit dashboard with filters and a small chart.", "Turn a notebook analysis into a polished interactive app with sidebar controls."]},
        "tr": {"why": "Etkileşimli dashboard'lar ve kavram kanıtı uygulamalar, veri rollerinde sık istenir.", "actions": ["Filtreler ve küçük bir grafik içeren simple bir Streamlit dashboard hazırlayın.", "Bir not defteri analizini kenar çubuğu kontrolleriyle güzel bir etkileşimli uygulamaya dönüştürün."]},
    },
    "FastAPI": {
        "en": {"why": "API delivery and backend integrations are common technical requirements in product roles.", "actions": ["Create a minimal CRUD API with FastAPI and document its endpoints.", "Show request validation, error handling, and a small test suite."]},
        "tr": {"why": "API teslimi ve arka uç entegrasyonları, ürün odaklı roller için sık görülen teknik gereksinimlerdir.", "actions": ["FastAPI ile minimal bir CRUD API oluşturup endpoint'lerini belgeleyin.", "İstek doğrulama, hata yönetimi ve küçük bir test serisi gösterin."]},
    },
    "Flask": {
        "en": {"why": "Flask experience often matters for lightweight backend services and internal tools.", "actions": ["Build a small Flask app with routes, templates, and environment configuration.", "Show how to structure the app for debugging and simple deployment."]},
        "tr": {"why": "Flask deneyimi, hafif arka uç servisleri ve dahili araçlar için sıklıkla önemlidir.", "actions": ["Rotalar, şablonlar ve ortam yapılandırması içeren küçük bir Flask uygulaması hazırlayın.", "Uygulamayı debugging ve basit dağıtım için nasıl yapılandırdığınızı gösterin."]},
    },
    "PyTorch": {
        "en": {"why": "The job likely values practical deep-learning implementation and experimentation skills.", "actions": ["Train a small PyTorch classifier on a simple dataset and explain the training loop.", "Document the model architecture, loss curve, and evaluation metrics clearly."]},
        "tr": {"why": "İş ilanı olasılıkla pratik derin öğrenme uygulamaları ve deneyim gerektiriyor.", "actions": ["Küçük bir PyTorch sınıflandırıcısını basit bir veri seti üzerinde eğitin ve eğitim döngüsünü açıklayın.", "Model mimarisini, loss eğrisini ve değerlendirme metriklerini net şekilde belgeleyin."]},
    },
    "TensorFlow": {
        "en": {"why": "The role may require practical deep-learning model development in a framework ecosystem.", "actions": ["Create a TensorFlow training example and explain the data pipeline and model layers.", "Document how you compare a simple baseline model to a neural-net model."]},
        "tr": {"why": "Rol, framework ekosisteminde pratik derin öğrenme modeli geliştirme deneyimi gerektirebilir.", "actions": ["Bir TensorFlow eğitim örneği oluşturup veri pipeline'ını ve model katmanlarını açıklayın.", "Küçük bir baseline modeli ile sinir ağı modelini nasıl karşılaştırdığınızı belgeleyin."]},
    },
    "Scikit-learn": {
        "en": {"why": "This is a practical requirement for classic ML pipelines, evaluation, and feature engineering.", "actions": ["Create a baseline model comparison notebook with train/validation splits.", "Show feature engineering, model choice, and metric reporting clearly."]},
        "tr": {"why": "Bu beceri, klasik ML boru hatları, değerlendirme ve öznitelik mühendisliği için pratik bir gerekliliktir.", "actions": ["Eğitim/doğrulama bölünmeleriyle temel bir model karşılaştırma not defteri oluşturun.", "Öznitelik mühendisliği, model seçimi ve metrik raporlamasını net şekilde gösterin."]},
    },
    "AWS": {
        "en": {"why": "Cloud deployment and operational workflows can be important in data and product teams.", "actions": ["Deploy a lightweight app or dataset pipeline to AWS and document each step.", "Show how environment variables, storage, and service configuration are managed."]},
        "tr": {"why": "Bulut dağıtımı ve operasyonel akışlar, veri ve ürün ekiplerinde önemli olabilir.", "actions": ["Hafif bir uygulamayı veya veri pipeline'ını AWS'ye dağıtın ve her adımı belgeleyin.", "Ortam değişkenleri, depolama ve servis yapılandırmasının nasıl yönetildiğini gösterin."]},
    },
    "REST API": {
        "en": {"why": "API design and integration are frequently part of product and data workflows.", "actions": ["Build a minimal REST service with clear endpoint documentation and sample payloads.", "Show authentication, validation, and error-handling patterns in a demo app."]},
        "tr": {"why": "API tasarımı ve entegrasyonu, ürün ve veri iş akışlarının sık parçasıdır.", "actions": ["Açık endpoint dokumentasyonu ve örnek payload'lar içeren minimal bir REST servisi oluşturun.", "Demo uygulamada kimlik doğrulama, doğrulama ve hata yönetimi desenlerini gösterin."]},
    },
    "Data Analysis": {
        "en": {"why": "Business-facing roles often need clear exploratory analysis and presentation of findings.", "actions": ["Build a dashboard with KPIs and explain the main business insights.", "Turn a raw dataset into a concise, story-driven analysis notebook."]},
        "tr": {"why": "İş odaklı roller, bulguları net şekilde keşif ve sunum şeklinde ortaya koymayı gerektirir.", "actions": ["KPI'lar içeren bir dashboard hazırlayıp ana iş içgörülerini açıklayın.", "Ham veri setini kısa, hikâye odaklı bir analiz not defterine dönüştürün."]},
    },
    "LLM": {
        "en": {"why": "The role may involve prompting, evaluation, and practical language-model workflows.", "actions": ["Build a notebook comparing prompts and model outputs on a small task set.", "Document guardrails, evaluator metrics, and bias or risk considerations."]},
        "tr": {"why": "Rol, istem oluşturma, değerlendirme ve pratik dil modeli iş akışlarını içerebilir.", "actions": ["Küçük bir görev seti üzerinde istemler ve model çıktıları karşılaştıran bir not defteri oluşturun.", "Koruyucu kuraları, değerlendirme metriklerini ve bias/risk yönlerini belgeleyin."]},
    },
}

def normalize_text(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value).lower())
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_negated(text_fragment):
    fragment = normalize_text(text_fragment)
    patterns = ["no experience with", "not familiar with", "not required", "lack of", "does not require", "without"]
    return any(pattern in fragment for pattern in patterns)


def extract_context_excerpt(text, skill_name, max_chars=140, language="en"):
    if not text or not skill_name:
        return t("no_direct_job_evidence", language)
    skill_aliases = SKILLS.get(skill_name, [])
    for alias in skill_aliases:
        match = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - 80)
            end = min(len(text), match.end() + 80)
            return resume_evidence_excerpt(text[start:end], max_chars)
    return t("no_direct_job_evidence", language)


def classify_job_skills(job_text):
    normalized = normalize_text(job_text)
    required = []
    preferred = []
    alternative = []
    alt_pairs = [
        ("PyTorch", "TensorFlow"),
        ("Flask", "FastAPI"),
        ("JavaScript", "TypeScript"),
    ]

    for skill in sorted(SKILLS):
        alias_norms = [normalize_text(alias) for alias in SKILLS[skill] if normalize_text(alias)]
        if not alias_norms:
            continue
        if not any(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized) for alias in alias_norms):
            continue

        status = "required"
        skill_key = skill.lower().replace("-", " ")
        if re.search(rf"\b(preferred|nice to have|nice-to-have|bonus|plus|good to have|desirable|advantage)\b.*{re.escape(skill_key)}|{re.escape(skill_key)}.*\b(preferred|nice to have|nice-to-have|bonus|plus|good to have|desirable|advantage)\b", normalized):
            status = "preferred"
        elif any(re.search(rf"\b(required|must have|mandatory|strong experience with|hands on|proficiency in|experience with|knowledge of|comfortable with|responsible for|using)\b.*{re.escape(alias)}|{re.escape(alias)}.*\b(required|must have|mandatory|strong experience with|hands on|proficiency in|experience with|knowledge of|comfortable with|responsible for|using)\b", normalized) for alias in alias_norms):
            status = "required"
        if is_negated(job_text):
            status = "preferred"

        if status == "required":
            required.append(skill)
        else:
            preferred.append(skill)

    for left, right in alt_pairs:
        if left in required and right in required:
            required = [item for item in required if item not in {left, right}]
            alternative.extend([left, right])

    return {
        "required": sorted(set(required)),
        "preferred": sorted(set(preferred)),
        "alternative": sorted(set(alternative)),
    }


def detect_resume_sections(resume_text):
    text = re.sub(r"\r\n?", "\n", str(resume_text or ""))
    lines = text.split("\n")
    sections = {"summary": [], "skills": [], "projects": [], "experience": [], "education": [], "other": []}
    current = "other"

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        lowered = normalize_text(line)
        if re.search(r"\b(summary|overview|profile|objective|about|ozet|profil|hedef|hakkinda|hakkında|about me|adres|identity)\b", lowered):
            current = "summary"
            sections["summary"].append(raw_line)
            continue
        if re.search(r"\b(skills|technical skills|competencies|tooling|yetenekler|beceriler|teknik beceriler|teknikler)\b", lowered):
            current = "skills"
            sections["skills"].append(raw_line)
            continue
        if re.search(r"\b(projects|project experience|portfolio|projeler|calismalar|çalışmalar|uygulamalar)\b", lowered):
            current = "projects"
            sections["projects"].append(raw_line)
            continue
        if re.search(r"\b(experience|work experience|internship|staj|deneyim|is deneyimi|iş deneyimi)\b", lowered):
            current = "experience"
            sections["experience"].append(raw_line)
            continue
        if re.search(r"\b(education|school|universit|universite|egitim|eğitim|okul|lise|university)\b", lowered):
            current = "education"
            sections["education"].append(raw_line)
            continue
        if re.search(r"\b(interests|hobbies|additional|extra|ilgi alanlari|ilgi alanları|hobiler|ek bilgi|kisisel bilgi|kişisel bilgi)\b", lowered):
            current = "other"
            sections["other"].append(raw_line)
            continue
        sections[current].append(raw_line)

    return {key: "\n".join(value).strip() for key, value in sections.items()}


def is_generic_summary(summary_text):
    if not summary_text:
        return False
    lowered = normalize_text(summary_text)
    generic_markers = [
        "caliskan", "çalışkan", "öğrenmeye açık", "ogrenmeye acik", "team player", "self motivated",
        "motivated", "hardworking", "results driven", "problem solver", "detail oriented",
        "işe uygun", "soru", "öğrenmeye istekli", "learn quickly"
    ]
    if any(marker in lowered for marker in generic_markers):
        return True
    job_keywords = ["python", "sql", "ml", "machine learning", "nlp", "docker", "api", "project", "data", "model", "analysis", "intern", "student"]
    return not any(keyword in lowered for keyword in job_keywords)


def resume_evidence_excerpt(text, max_chars=180):
    excerpt = str(text or "").strip()
    if len(excerpt) <= max_chars:
        return excerpt
    return excerpt[:max_chars].rstrip() + "…"


def make_resume_recommendation(recommendation_type, priority, resume_evidence, job_evidence, facts=None, icon="✨"):
    return {
        "type": recommendation_type,
        "priority": priority,
        "resume_evidence": resume_evidence,
        "job_evidence": job_evidence,
        "facts": facts or {},
        "icon": icon,
    }


def build_resume_improvement_recommendations(resume_text, job_text, missing=None):
    """Build language-neutral recommendation identifiers and source evidence."""
    resume_text = resume_text or ""
    job_text = job_text or ""
    missing = sorted(set(missing or []))

    sections = detect_resume_sections(resume_text)
    summary = sections.get("summary", "").strip()
    skills_section = sections.get("skills", "").strip()
    projects = sections.get("projects", "").strip()
    experience = sections.get("experience", "").strip()
    other = sections.get("other", "").strip()
    job_excerpt = resume_evidence_excerpt(job_text[:500], 180)
    recs = []

    if summary and (len(summary.split()) > 220 or is_generic_summary(summary)):
        recs.append(make_resume_recommendation("summary_rewrite", "high", resume_evidence_excerpt(summary), job_excerpt, icon="✍️"))
    elif not summary:
        recs.append(make_resume_recommendation("summary_add", "high", "", job_excerpt, icon="📝"))

    relevant_skill = next((skill for skill in missing if skill.lower() in normalize_text(projects + " " + experience)), None)
    if relevant_skill:
        evidence = projects or experience
        recs.append(make_resume_recommendation(
            "skill_emphasis", "high", resume_evidence_excerpt(evidence, 200), job_excerpt,
            {"skill": relevant_skill}, "💡",
        ))
    elif missing:
        recs.append(make_resume_recommendation(
            "skill_gap", "medium", "", job_excerpt, {"skill": missing[0]}, "📌",
        ))

    if projects:
        recs.append(make_resume_recommendation("project_clarity", "high", resume_evidence_excerpt(projects), job_excerpt, icon="🚀"))
    elif experience:
        recs.append(make_resume_recommendation("experience_clarity", "medium", resume_evidence_excerpt(experience), job_excerpt, icon="🧩"))

    if other and len(other.split()) > 20:
        recs.append(make_resume_recommendation("trim_content", "medium", resume_evidence_excerpt(other), job_excerpt, icon="🧹"))

    if not summary and (projects or experience):
        recs.append(make_resume_recommendation(
            "reorder_sections", "medium", resume_evidence_excerpt(projects or experience), job_excerpt, icon="📚",
        ))
    elif skills_section and not summary:
        recs.append(make_resume_recommendation("reorder_skills", "medium", resume_evidence_excerpt(skills_section), job_excerpt, icon="🔀"))

    if len(resume_text.split()) > 220 and not all(sections.get(key, "").strip() for key in ("summary", "skills", "projects", "experience", "education")):
        recs.append(make_resume_recommendation("ats_format", "medium", resume_evidence_excerpt(resume_text), job_excerpt, icon="📖"))

    if not skills_section:
        recs.append(make_resume_recommendation("skills_section_add", "medium", "", job_excerpt, icon="✅"))
    if experience and not projects:
        recs.append(make_resume_recommendation("experience_bullets", "medium", resume_evidence_excerpt(experience), job_excerpt, icon="🛠️"))
    unique = []
    seen = set()
    for recommendation in recs:
        if recommendation["type"] not in seen:
            unique.append(recommendation)
            seen.add(recommendation["type"])

    concise_evidence = resume_evidence_excerpt(resume_text)
    for recommendation_type, icon in (
        ("focus_top", "✨"),
        ("quantify_impact", "📊"),
        ("consistent_format", "🗂️"),
        ("concise_bullets", "📖"),
    ):
        if len(unique) >= 4:
            break
        if recommendation_type not in seen:
            unique.append(
                make_resume_recommendation(
                    recommendation_type,
                    "low",
                    concise_evidence,
                    job_excerpt,
                    icon=icon,
                )
            )
            seen.add(recommendation_type)
    return unique[:6]


def build_chart_data(matched, missing, language="en"):
    return pd.DataFrame({
        "Category": [t("matched_bar", language), t("missing_bar", language)],
        "Count": [len(matched), len(missing)],
    })


def build_skill_chart(matched, missing, language):
    fig = px.bar(
        build_chart_data(matched, missing, language),
        x="Category",
        y="Count",
        color="Category",
        title=t("chart_title", language),
        labels={
            "Category": t("chart_category_axis", language),
            "Count": t("chart_count_axis", language),
        },
        color_discrete_map={
            t("matched_bar", language): "#5ec7ff",
            t("missing_bar", language): "#ff7f7f",
        },
    )
    fig.update_layout(
        showlegend=False,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_yaxes(title_text=t("chart_count_axis", language), rangemode="tozero")
    fig.update_xaxes(title_text=t("chart_category_axis", language))
    fig.update_traces(
        hovertemplate=(
            "%{x}<br>"
            + t("chart_count_axis", language)
            + ": %{y}<extra></extra>"
        )
    )
    return fig


def build_recommendation_item(skill_name, job_text, resume_text, language):
    guidance = SKILL_GUIDANCE.get(skill_name)
    localized = guidance[language] if guidance else {}
    reason = localized.get("why", t("default_reason", language))
    actions = localized.get("actions", [t("default_learning", language), t("default_portfolio", language)])
    learning = actions[0] if actions else t("default_learning", language)
    portfolio = actions[1] if len(actions) > 1 else t("default_portfolio", language)
    evidence_excerpt = extract_context_excerpt(job_text, skill_name, language=language)
    skill_detected = skill_name in find_skills(resume_text)
    priority = "high"
    if skill_name in {"Git", "GitHub", "Docker", "Python", "SQL"}:
        priority = "high"
    elif skill_name in {"Pandas", "NumPy", "Streamlit", "FastAPI", "Flask", "NLP"}:
        priority = "medium"
    else:
        priority = "low"

    return {
        "skill": skill_name,
        "priority": t(f"priority_{priority}", language),
        "reason": reason,
        "learning": learning,
        "portfolio": portfolio,
        "resume_evidence": t("detected_in_resume", language) if skill_detected else t("not_in_resume", language),
        "job_evidence": evidence_excerpt,
    }


def build_recommendations(resume_text, job_text, matched, missing, total_job_skills):
    extra_count = min(3, len(missing))
    current_coverage = (len(matched) / total_job_skills * 100) if total_job_skills else 0.0
    projected_coverage = (
        (len(matched) + extra_count) / total_job_skills * 100
        if total_job_skills
        else 0.0
    )
    return {
        "current_skill_coverage_percent": round(current_coverage, 2),
        "projected_skill_coverage_percent": round(projected_coverage, 2),
        "extra_skill_count": extra_count,
        "recommendations": build_resume_improvement_recommendations(
            resume_text, job_text, missing
        ),
    }


RECOMMENDATION_PRESENTATION = {
    "summary_rewrite": ("rec_summary_title", "action_improve"),
    "summary_add": ("rec_add_summary_title", "action_add"),
    "skill_emphasis": ("rec_skill_title", "action_expand"),
    "skill_gap": ("rec_skill_title", "action_add"),
    "project_clarity": ("rec_project_title", "action_expand"),
    "experience_clarity": ("rec_project_title", "action_expand"),
    "trim_content": ("rec_irrelevant_title", "action_shorten"),
    "reorder_sections": ("rec_structure_title", "action_reorder"),
    "reorder_skills": ("rec_structure_title", "action_reorder"),
    "ats_format": ("rec_ats_title", "action_improve"),
    "skills_section_add": ("rec_skill_title", "action_add"),
    "experience_bullets": ("rec_project_title", "action_expand"),
    "focus_top": ("rec_ats_title", "action_improve"),
    "quantify_impact": ("rec_project_title", "action_improve"),
    "consistent_format": ("rec_ats_title", "action_improve"),
    "concise_bullets": ("rec_ats_title", "action_improve"),
}


def localize_recommendation(recommendation, language):
    recommendation_type = recommendation["type"]
    title_key, action_key = RECOMMENDATION_PRESENTATION[recommendation_type]
    values = recommendation["facts"]
    return {
        **recommendation,
        "title": t(title_key, language),
        "action": t(action_key, language),
        "priority_label": t(f"priority_{recommendation['priority']}", language),
        "description": t(f"advisor_{recommendation_type}", language, **values),
        "suggested_revision": t(
            f"advisor_{recommendation_type}_example", language, **values
        ),
    }


def build_report(response, language):
    localized_recommendations = [
        localize_recommendation(item, language)
        for item in response["recommendations"]
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "language": language,
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "semantic_similarity_percent": round(response["semantic_similarity"] * 100, 2),
        "skill_coverage_percent": (
            round(response["skill_coverage"], 2)
            if response["skill_coverage"] is not None
            else None
        ),
        "denominator_definition": t("report_denominator_definition", language),
        "job_required_skills": response["required_skills"],
        "job_preferred_skills": response["preferred_skills"],
        "job_requirements": response["job_requirements"],
        "recommendation_types": [
            item["type"] for item in response["recommendations"]
        ],
        "matched_required_skills": response["matched_skills"],
        "skills_not_detected": response["missing_skills"],
        "resume_skills_detected": response["resume_skills"],
        "resume_improvement_recommendations": localized_recommendations,
        "recommendations": localized_recommendations,
        "overall_assessment": (
            f"{t('resume_overall_assessment', language)}: "
            + t(
                "overall_assessment_message",
                language,
                matched_count=len(response["matched_skills"]),
                missing_count=len(response["missing_skills"]),
            )
        ),
        "closing_summary": t("resume_advisor_conclusion", language),
        "conditional_message": (
            t(
                "coverage_projection",
                language,
                current_coverage=f"{response['current_skill_coverage_percent']:.1f}",
                projected_coverage=f"{response['projected_skill_coverage_percent']:.1f}",
            )
            if response["extra_skill_count"]
            else ""
        ),
        "disclaimer": t("report_disclaimer", language),
        "analysis_metadata": {
            "version": "1.2",
            "text_truncation_limit_chars": MAX_TEXT_CHARS,
            "model_device": "cpu",
            "cached_model": True,
        },
    }


def render_language_selector():
    language = st.session_state.get("lang", "tr")
    st.selectbox(
        t("language_label", language),
        options=tuple(LANGUAGE_OPTIONS),
        format_func=LANGUAGE_OPTIONS.__getitem__,
        key="lang",
    )


def render_resume_improvement_advisor(response, language):
    recs = response["recommendations"]
    if not recs:
        st.info(t("nothing_to_recommend", language))
        return

    st.markdown("---")
    st.markdown(f"## ✨ {t('resume_advisor_title', language)}")
    st.caption(t('advisor_notice', language))
    st.write(
        f"{t('resume_overall_assessment', language)}: "
        + t(
            "overall_assessment_message",
            language,
            matched_count=len(response["matched_skills"]),
            missing_count=len(response["missing_skills"]),
        )
    )

    for index, item in enumerate(recs, start=1):
        localized = localize_recommendation(item, language)
        with st.container():
            st.markdown(f"### {index}. {item.get('icon', '✨')} {localized['title']}")
            st.caption(
                f"{t('recommendation_action', language)}: {localized['action']} • "
                f"{t('priority_label', language)}: {localized['priority_label']}"
            )
            st.write(localized["description"])
            st.caption(f"{t('suggested_revision_label', language)}: {localized['suggested_revision']}")
            if item.get("resume_evidence"):
                st.markdown(
                    f"**{t('evidence_resume_label', language)}**\n\n"
                    f"> {item['resume_evidence']}"
                )
            if item.get("job_evidence"):
                st.markdown(
                    f"**{t('evidence_job_label', language)}**\n\n"
                    f"> {item['job_evidence']}"
                )

    st.success(t("resume_advisor_conclusion", language))


def render_results(response, language):
    similarity = response["semantic_similarity"]
    matched = response["matched_skills"]
    missing = response["missing_skills"]
    coverage = response["skill_coverage"]

    st.subheader(t("analysis_results", language))
    c1, c2, c3 = st.columns(3)
    c1.metric(t("semantic_similarity", language), f"{similarity * 100:.1f}%")
    c2.metric(t("matched_skills", language), str(len(matched)))
    c3.metric(
        t("skill_coverage", language),
        f"{coverage:.0f}%" if coverage is not None else t("not_available", language),
    )
    st.info(t("similarity_explainer", language))

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(f"**{t('common_skills', language)}**")
        if matched:
            st.write(", ".join(matched))
        else:
            st.write(t("no_matched_skills", language))
    with right_col:
        st.markdown(f"**{t('missing_skills', language)}**")
        if missing:
            st.write(", ".join(missing))
        else:
            st.write(t("no_missing_skills", language))

    fig = build_skill_chart(matched, missing, language)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    render_resume_improvement_advisor(response, language)

    if response["extra_skill_count"]:
        st.info(
            t(
                "coverage_projection",
                language,
                current_coverage=f"{response['current_skill_coverage_percent']:.1f}",
                projected_coverage=f"{response['projected_skill_coverage_percent']:.1f}",
            )
        )

    st.caption(t("semantic_vs_coverage", language))
    st.caption(t("style_note", language))
    st.caption(t("future_llm_hint", language))


def score_skill_evidence(text):
    if not text or not text.strip():
        return {}
    normalized = normalize_text(text)
    result = {}
    for skill, aliases in SKILLS.items():
        score = 0
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if not alias_norm:
                continue
            score += len(re.findall(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", normalized))
        if score > 0:
            result[skill] = score
    return result


def find_skills(text):
    if not text or not text.strip():
        return set()
    normalized = normalize_text(text)
    found = set()
    for skill, aliases in SKILLS.items():
        for alias in aliases:
            alias_norm = normalize_text(alias)
            if not alias_norm:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", normalized):
                if not is_negated(f"{skill} {alias_norm}"):
                    found.add(skill)
                    break
    return found


def extract_resume(file, language="en"):
    if file is None:
        raise ResumeExtractionError(t("file_missing", language))
    data = file.getvalue()
    if len(data) == 0:
        raise ResumeExtractionError(t("file_empty", language))
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise ResumeExtractionError(t("file_too_large", language, max_size=MAX_FILE_SIZE_BYTES // (1024 * 1024)))

    lower_name = (file.name or "").lower()
    if lower_name.endswith(".txt"):
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ResumeExtractionError(t("txt_decode_error", language)) from exc
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            raise ResumeExtractionError(t("file_empty", language))
        return cleaned

    if lower_name.endswith(".pdf"):
        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                if len(doc) > MAX_PDF_PAGES:
                    raise ResumeExtractionError(t("pdf_too_many_pages", language, max_pages=MAX_PDF_PAGES))
                pages = [page.get_text("text") for page in doc]
                joined = "\n".join(page for page in pages if page)
                cleaned = re.sub(r"\s+", " ", joined).strip()
                if not cleaned:
                    raise ResumeExtractionError(t("pdf_no_text", language))
                return cleaned
        except fitz.FileDataError as exc:
            raise ResumeExtractionError(t("pdf_invalid", language)) from exc

    raise ResumeExtractionError(t("file_type_unsupported", language))


@st.cache_resource(show_spinner=False)
def load_model():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@st.cache_data(show_spinner=False)
def build_embedding(text):
    if not text or not text.strip():
        return np.zeros(384, dtype=np.float32)
    model = load_model()
    chunks = chunk_text_for_embedding(text)
    with torch.inference_mode():
        vectors = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, batch_size=16)
    if vectors.ndim == 1:
        return vectors.astype(np.float32)
    return vectors.mean(axis=0).astype(np.float32)


def chunk_text_for_embedding(text, max_chars=MAX_EMBEDDING_CHARS):
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return chunks[:12]


def semantic_similarity_score(resume_embedding, job_embedding):
    similarity = float(
        cosine_similarity([resume_embedding], [job_embedding])[0][0]
    )
    return max(-1.0, min(1.0, similarity))


def main():
    st.set_page_config(page_title="SkillMatch AI", page_icon="🎯", layout="wide")
    if "lang" not in st.session_state:
        st.session_state.lang = "tr"

    language = st.session_state.get("lang", "tr")
    left_col, right_col = st.columns([6, 2], gap="small")
    with right_col:
        st.write("")
        render_language_selector()
    language = st.session_state.lang
    with left_col:
        st.title(t("app_title", language))
        st.caption(t("tagline", language))

    st.write(t("intro", language))

    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        resume_file = st.file_uploader(
            t("resume_label", language),
            type=["pdf", "txt"],
            help=t("resume_help", language),
            key="resume_upload",
        )
    with right_col:
        job_text = st.text_area(
            t("job_label", language),
            height=240,
            placeholder=t("job_placeholder", language),
            key="job_description",
        )

    if st.button(
        t("analyze_button", language),
        type="primary",
        use_container_width=True,
        key="analyze",
    ):
        st.session_state.pop("analysis", None)
        if not resume_file or not job_text.strip():
            st.warning(t("empty_resume_warning", language))
        else:
            try:
                resume_text = extract_resume(resume_file, language)
                if len(resume_text) < 40:
                    st.error(t("resume_extract_error", language))
                elif len(job_text.strip()) < 40:
                    st.error(t("job_min_length_error", language))
                else:
                    with st.spinner(t("model_loading", language)):
                        model_start = time.perf_counter()
                        load_model()
                        model_load_time = time.perf_counter() - model_start

                    resume_text_for_similarity = re.sub(
                        r"\s+", " ", resume_text[:MAX_TEXT_CHARS]
                    ).strip()
                    job_text_for_similarity = re.sub(
                        r"\s+", " ", job_text[:MAX_TEXT_CHARS]
                    ).strip()

                    inference_start = time.perf_counter()
                    resume_embedding = build_embedding(resume_text_for_similarity)
                    job_embedding = build_embedding(job_text_for_similarity)
                    inference_time = time.perf_counter() - inference_start

                    similarity = semantic_similarity_score(
                        resume_embedding, job_embedding
                    )

                    job_profile = classify_job_skills(job_text)
                    required_skills = set(job_profile["required"])
                    preferred_skills = set(job_profile["preferred"])
                    resume_skills = find_skills(resume_text)
                    matched = sorted(resume_skills & required_skills)
                    missing = sorted(required_skills - resume_skills)
                    coverage = (
                        len(matched) / len(required_skills) * 100
                        if required_skills
                        else None
                    )
                    recommendation_data = build_recommendations(
                        resume_text,
                        job_text,
                        matched,
                        missing,
                        len(required_skills),
                    )
                    st.session_state.analysis = {
                        "semantic_similarity": similarity,
                        "matched_skills": matched,
                        "missing_skills": missing,
                        "skill_coverage": coverage,
                        "job_skill_count": len(required_skills),
                        "required_skills": sorted(required_skills),
                        "preferred_skills": sorted(preferred_skills),
                        "job_requirements": job_profile,
                        "resume_sections": detect_resume_sections(resume_text),
                        "resume_skills": sorted(resume_skills),
                        "recommendations": recommendation_data["recommendations"],
                        "recommendation_types": [
                            item["type"]
                            for item in recommendation_data["recommendations"]
                        ],
                        "detected_issues": [
                            {
                                "type": item["type"],
                                "priority": item["priority"],
                            }
                            for item in recommendation_data["recommendations"]
                        ],
                        "current_skill_coverage_percent": recommendation_data[
                            "current_skill_coverage_percent"
                        ],
                        "projected_skill_coverage_percent": recommendation_data[
                            "projected_skill_coverage_percent"
                        ],
                        "extra_skill_count": recommendation_data["extra_skill_count"],
                        "model_load_time_seconds": round(model_load_time, 2),
                        "inference_time_seconds": round(inference_time, 2),
                    }
            except ResumeExtractionError as exc:
                st.error(str(exc))
            except Exception:
                logging.exception("SkillMatch analysis failed")
                st.error(t("analysis_error_generic", language))

    response = st.session_state.get("analysis")
    if response:
        st.caption(
            f"{t('model_ready', language)} "
            f"{response['model_load_time_seconds']:.2f}s"
        )
        st.caption(
            f"{t('inference_time', language)}: "
            f"{response['inference_time_seconds']:.2f}s"
        )
        render_results(response, language)
        report = build_report(response, language)
        st.download_button(
            t("download_report", language),
            json.dumps(report, ensure_ascii=False, indent=2),
            "skillmatch_report.json",
            "application/json",
            key="download_report",
        )

    st.divider()
    st.caption(t("privacy_notice", language))


if __name__ == "__main__":
    main()
