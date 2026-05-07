import re
from datetime import date
from urllib.parse import unquote

import google.generativeai as genai
import pandas as pd
import requests
import streamlit as st
from tavily import TavilyClient


SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
REQUEST_HEADERS = {"User-Agent": "AcademicSearch/1.0"}
REQUEST_TIMEOUT = 15

PAPER_FIELDS = (
    "paperId,title,year,venue,abstract,tldr,citationCount,authors.name,"
    "externalIds,url,publicationDate"
)
LINEAGE_FIELDS = (
    "paperId,title,year,venue,citationCount,authors.name,abstract,tldr,"
    "references.paperId,references.title,references.year,references.citationCount,"
    "citations.paperId,citations.title,citations.year,citations.citationCount"
)

GEMINI_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
]

GEMINI_FALLBACK_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
]

ACADEMIC_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "europepmc.org",
    "semanticscholar.org",
    "crossref.org",
    "biorxiv.org",
    "medrxiv.org",
    "arxiv.org",
    "nature.com",
    "springer.com",
    "sciencedirect.com",
    "wiley.com",
    "acs.org",
    "rsc.org",
    "iopscience.iop.org",
    "aip.scitation.org",
    "frontiersin.org",
    "tandfonline.com",
    "mdpi.com",
]

RESEARCH_TEMPLATES = {
    "通用學術搜尋": {
        "expansion": [
            "recent research",
            "systematic review",
            "meta-analysis",
            "DOI",
            "citation",
        ],
        "domains": [],
        "report_focus": """
請使用通用學術報告格式：核心結論、重要研究、方法與證據等級、限制、未來方向、實務或研究建議。
""",
    },
    "運動科學 / 生理學": {
        "expansion": [
            "exercise physiology",
            "training adaptation",
            "concurrent training",
            "interference effect",
            "endurance training",
            "resistance training",
            "strength",
            "hypertrophy",
            "power",
            "VO2max",
            "trained participants",
        ],
        "domains": [
            "pubmed.ncbi.nlm.nih.gov",
            "pmc.ncbi.nlm.nih.gov",
            "europepmc.org",
            "frontiersin.org",
            "tandfonline.com",
            "sciencedirect.com",
        ],
        "report_focus": """
請特別分開討論：肌力、肌肥大、爆發力、VO2max；有氧模式如跑步/腳踏車/HIIT/Zone 2；訓練安排如同日、間隔數小時、不同日；並標出對訓練實務最有用的結論。
""",
    },
    "醫學 / 臨床研究": {
        "expansion": [
            "clinical trial",
            "randomized controlled trial",
            "cohort study",
            "systematic review",
            "meta-analysis",
            "guideline",
            "PMID",
            "PubMed",
        ],
        "domains": [
            "pubmed.ncbi.nlm.nih.gov",
            "pmc.ncbi.nlm.nih.gov",
            "europepmc.org",
            "cochranelibrary.com",
            "nejm.org",
            "thelancet.com",
            "jamanetwork.com",
        ],
        "report_focus": """
請特別分開討論：研究設計、受試者/病人族群、主要與次要 outcome、效益與風險、臨床適用性、指南或共識是否一致。
""",
    },
    "材料科學 / 顯微分析": {
        "expansion": [
            "transmission electron microscopy",
            "TEM",
            "scanning transmission electron microscopy",
            "STEM",
            "electron energy loss spectroscopy",
            "EELS",
            "STEM-EELS",
            "monochromated EELS",
            "core-loss EELS",
            "low-loss EELS",
            "spectrum imaging",
            "chemical mapping",
            "oxidation state",
            "plasmon",
            "phonon EELS",
            "4D-STEM",
            "in situ TEM",
            "cryo TEM",
            "machine learning EELS",
        ],
        "domains": [
            "nature.com",
            "sciencedirect.com",
            "springer.com",
            "wiley.com",
            "acs.org",
            "rsc.org",
            "iopscience.iop.org",
            "aip.scitation.org",
            "arxiv.org",
        ],
        "report_focus": """
請特別分開討論：儀器硬體進展、能量/空間解析度、spectrum imaging 與定量方法、denoising/機器學習分析、in situ/cryo/4D-STEM 整合、代表性材料應用，以及未來 3-5 年趨勢。
""",
    },
    "AI / Computer Science": {
        "expansion": [
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "benchmark",
            "dataset",
            "arXiv",
            "conference",
            "NeurIPS",
            "ICML",
            "ICLR",
            "ACL",
            "CVPR",
        ],
        "domains": [
            "arxiv.org",
            "openreview.net",
            "proceedings.neurips.cc",
            "proceedings.mlr.press",
            "aclanthology.org",
            "thecvf.com",
            "semanticscholar.org",
        ],
        "report_focus": """
請特別分開討論：核心方法、資料集與 benchmark、與前作比較、實驗設計、可重現性、開源資源、實際限制與後續研究方向。
""",
    },
    "自訂模板": {
        "expansion": [],
        "domains": [],
        "report_focus": """
請依使用者問題本身判斷最合適的學術報告結構，仍需保持 citation 與證據品質。
""",
    },
}

PREPRINT_MARKERS = ["arxiv", "biorxiv", "medrxiv", "preprint"]
SOURCE_TYPE_SCORE = {
    "正式論文": 3,
    "預印本": 2,
    "學術網頁": 1,
    "一般網頁": 0,
}
EVIDENCE_PATTERNS = [
    ("系統性回顧/統合分析", 5, ["systematic review", "meta-analysis", "meta analysis"]),
    ("隨機對照試驗", 4, ["randomized", "randomised", "randomized controlled trial", "rct"]),
    ("世代/縱向研究", 3, ["cohort", "longitudinal", "prospective"]),
    ("病例對照/橫斷研究", 2, ["case-control", "cross-sectional", "observational"]),
    ("敘述性回顧/評論", 1, ["review", "narrative review", "commentary", "editorial"]),
]


st.set_page_config(
    page_title="Gemini x 學術搜尋",
    page_icon="🔎",
    layout="wide",
)


def init_session_state():
    defaults = {
        "papers": [],
        "web_sources": [],
        "report": "",
        "lineage": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_secret_or_empty(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def normalize_query(query):
    return re.sub(r"\s+", " ", query.strip())


def get_template(template_name):
    return RESEARCH_TEMPLATES.get(template_name, RESEARCH_TEMPLATES["通用學術搜尋"])


def template_expansion(template_name):
    return " ".join(get_template(template_name).get("expansion", []))


def template_domains(template_name):
    domains = get_template(template_name).get("domains", [])
    merged = domains + [domain for domain in ACADEMIC_DOMAINS if domain not in domains]
    return merged


def build_academic_query(query, start_year, template_name="通用學術搜尋", custom_terms=""):
    terms = [
        query,
        query_expansion(query, template_name, custom_terms),
        "recent",
        "systematic review",
        "meta-analysis",
        "randomized controlled trial",
        "cohort study",
        "DOI",
        "PMID",
        "PubMed",
        str(start_year),
        str(date.today().year),
    ]
    return " ".join(terms)


def query_expansion(query, template_name="通用學術搜尋", custom_terms=""):
    expansions = {
        "有氧": "aerobic exercise endurance training",
        "肌力": "resistance training strength training",
        "重訓": "resistance training strength training",
        "肌肥大": "hypertrophy muscle growth",
        "不相容": "interference effect concurrent training",
        "併行訓練": "concurrent training interference effect",
        "穿透式電子顯微鏡": "transmission electron microscopy TEM STEM",
        "電子顯微鏡": "electron microscopy TEM STEM",
        "EELS": "electron energy loss spectroscopy EELS STEM-EELS",
        "能量損失": "electron energy loss spectroscopy EELS",
        "材料": "materials science characterization",
        "顯微": "microscopy characterization",
    }
    extra_terms = [value for key, value in expansions.items() if key in query]
    return " ".join(extra_terms + [template_expansion(template_name), custom_terms])


def paper_authors(paper):
    authors = paper.get("authors") or []
    names = [author.get("name") for author in authors if author.get("name")]
    if not names:
        return "Unknown"
    if len(names) <= 3:
        return ", ".join(names)
    return f"{names[0]} et al."


def paper_summary(paper, limit=280):
    tldr = paper.get("tldr") or {}
    text = tldr.get("text") or paper.get("abstract") or ""
    return text[:limit] + ("..." if len(text) > limit else "")


def paper_identifier(paper):
    external_ids = paper.get("externalIds") or {}
    doi = external_ids.get("DOI")
    pmid = external_ids.get("PubMed")
    if doi:
        return f"DOI: {doi}"
    if pmid:
        return f"PMID: {pmid}"
    return paper.get("paperId", "")


def normalize_identifier(value):
    if not value:
        return ""
    return re.sub(r"\s+", "", str(value).strip().lower())


def normalize_title(title):
    title = re.sub(r"<[^>]+>", "", title or "")
    title = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", " ", title.lower())
    return normalize_query(title)


def paper_dedupe_keys(paper):
    external_ids = paper.get("externalIds") or {}
    keys = []
    doi = normalize_identifier(external_ids.get("DOI"))
    pmid = normalize_identifier(external_ids.get("PubMed"))
    paper_id = normalize_identifier(paper.get("paperId"))
    title = normalize_title(paper.get("title", ""))

    if doi:
        keys.append(f"doi:{doi}")
    if pmid:
        keys.append(f"pmid:{pmid}")
    if paper_id:
        keys.append(f"id:{paper_id}")
    if title:
        keys.append(f"title:{title}")
    return keys


def classify_source_type(paper):
    source_blob = " ".join(
        [
            str(paper.get("source", "")),
            str(paper.get("venue", "")),
            str(paper.get("url", "")),
            str(paper_identifier(paper)),
        ]
    ).lower()
    if any(marker in source_blob for marker in PREPRINT_MARKERS):
        return "預印本"
    external_ids = paper.get("externalIds") or {}
    if external_ids.get("DOI") or external_ids.get("PubMed"):
        return "正式論文"
    if paper.get("source") in {"PubMed", "Europe PMC", "Semantic Scholar"}:
        return "正式論文"
    return "學術網頁"


def classify_evidence_level(paper):
    text = " ".join(
        [
            paper.get("title", ""),
            paper.get("abstract", ""),
            (paper.get("tldr") or {}).get("text", ""),
            " ".join(paper.get("publication_types") or []),
        ]
    ).lower()
    for label, score, patterns in EVIDENCE_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return label, score
    return "原始研究/未判定", 0


def annotate_paper(paper):
    annotated = paper.copy()
    source_type = classify_source_type(annotated)
    evidence_level, evidence_score = classify_evidence_level(annotated)
    annotated["source_type"] = source_type
    annotated["evidence_level"] = evidence_level
    annotated["evidence_score"] = evidence_score
    return annotated


def paper_sort_key(paper):
    return (
        SOURCE_TYPE_SCORE.get(paper.get("source_type", ""), 0),
        paper.get("evidence_score") or 0,
        paper.get("year") or 0,
        paper.get("citationCount") or 0,
    )


def merge_paper_records(existing, incoming):
    merged = existing.copy()
    incoming = incoming.copy()

    for key in ["title", "year", "venue", "url", "abstract", "tldr"]:
        if not merged.get(key) and incoming.get(key):
            merged[key] = incoming[key]

    existing_ids = merged.get("externalIds") or {}
    incoming_ids = incoming.get("externalIds") or {}
    merged["externalIds"] = {**incoming_ids, **existing_ids}

    if not merged.get("authors") and incoming.get("authors"):
        merged["authors"] = incoming["authors"]

    merged["citationCount"] = max(merged.get("citationCount") or 0, incoming.get("citationCount") or 0)

    sources = {
        source
        for source in [merged.get("source"), incoming.get("source")]
        if source
    }
    if sources:
        merged["source"] = " + ".join(sorted(sources))

    return annotate_paper(merged)


def dedupe_papers(papers):
    key_to_index = {}
    deduped = []
    for paper in papers:
        annotated = annotate_paper(paper)
        keys = paper_dedupe_keys(annotated)
        matched_index = next((key_to_index[key] for key in keys if key in key_to_index), None)

        if matched_index is not None:
            deduped[matched_index] = merge_paper_records(deduped[matched_index], annotated)
            key_to_index.update({key: matched_index for key in paper_dedupe_keys(deduped[matched_index])})
            continue

        if not keys:
            continue

        key_to_index.update({key: len(deduped) for key in keys})
        deduped.append(annotated)
    return deduped


def classify_web_source(source):
    url = (source.get("url") or "").lower()
    if any(marker in url for marker in ["biorxiv.org", "medrxiv.org", "arxiv.org"]):
        return "預印本"
    if any(domain in url for domain in ["pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "europepmc.org"]):
        return "正式論文"
    if any(domain in url for domain in ACADEMIC_DOMAINS):
        return "學術網頁"
    return "一般網頁"


def annotate_web_sources(sources):
    annotated = []
    for source in sources:
        item = source.copy()
        item["source_type"] = classify_web_source(item)
        annotated.append(item)
    return sorted(
        annotated,
        key=lambda item: SOURCE_TYPE_SCORE.get(item.get("source_type", ""), 0),
        reverse=True,
    )


def abstract_from_inverted_index(inverted_index):
    if not inverted_index:
        return ""
    positioned_words = []
    for word, positions in inverted_index.items():
        for position in positions:
            positioned_words.append((position, word))
    return " ".join(word for _, word in sorted(positioned_words))


def crossref_year(item):
    for key in ["published-print", "published-online", "published"]:
        date_parts = (item.get(key) or {}).get("date-parts") or []
        if date_parts and date_parts[0]:
            return date_parts[0][0]
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def search_semantic_scholar(query, limit, start_year, template_name="通用學術搜尋", custom_terms=""):
    params = {
        "query": f"{query} {query_expansion(query, template_name, custom_terms)}",
        "limit": min(limit * 3, 100),
        "fields": PAPER_FIELDS,
        "year": f"{start_year}-",
    }
    try:
        response = requests.get(
            SEMANTIC_SCHOLAR_SEARCH_URL,
            params=params,
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 400:
            fallback_params = {key: value for key, value in params.items() if key != "year"}
            response = requests.get(
                SEMANTIC_SCHOLAR_SEARCH_URL,
                params=fallback_params,
                headers=REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
        response.raise_for_status()
        papers = response.json().get("data", [])
    except requests.RequestException:
        return []

    for paper in papers:
        paper["source"] = "Semantic Scholar"
    recent = sorted(papers, key=paper_sort_key, reverse=True)[:limit]
    cited = sorted(papers, key=lambda paper: paper.get("citationCount") or 0, reverse=True)[:limit]
    return dedupe_papers(recent + cited)[: limit * 2]


@st.cache_data(ttl=3600, show_spinner=False)
def search_pubmed(query, limit, start_year, template_name="通用學術搜尋", custom_terms=""):
    try:
        search_response = requests.get(
            PUBMED_SEARCH_URL,
            params={
                "db": "pubmed",
                "term": f"{query} {query_expansion(query, template_name, custom_terms)}",
                "retmode": "json",
                "retmax": limit,
                "sort": "pub date",
                "datetype": "pdat",
                "mindate": start_year,
                "maxdate": date.today().year,
            },
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        search_response.raise_for_status()
        ids = search_response.json().get("esearchresult", {}).get("idlist", [])
    except requests.RequestException:
        return []

    if not ids:
        return []

    try:
        summary_response = requests.get(
            PUBMED_SUMMARY_URL,
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "json",
            },
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        summary_response.raise_for_status()
        result = summary_response.json().get("result", {})
    except requests.RequestException:
        return []

    papers = []
    for pubmed_id in result.get("uids", []):
        item = result.get(pubmed_id, {})
        pubdate = item.get("pubdate", "")
        year_match = re.search(r"\d{4}", pubdate)
        article_ids = item.get("articleids", [])
        doi = next((entry.get("value") for entry in article_ids if entry.get("idtype") == "doi"), None)
        authors = [{"name": author.get("name")} for author in item.get("authors", [])]
        papers.append(
            {
                "paperId": f"PMID:{pubmed_id}",
                "title": item.get("title", "Untitled"),
                "year": int(year_match.group(0)) if year_match else None,
                "venue": item.get("source", "PubMed"),
                "abstract": "",
                "tldr": None,
                "citationCount": 0,
                "authors": authors,
                "externalIds": {"DOI": doi, "PubMed": pubmed_id},
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/",
                "source": "PubMed",
                "publication_types": item.get("pubtype", []),
            }
        )
    return papers


@st.cache_data(ttl=3600, show_spinner=False)
def search_europe_pmc(query, limit, start_year, template_name="通用學術搜尋", custom_terms=""):
    current_year = date.today().year
    try:
        response = requests.get(
            EUROPE_PMC_SEARCH_URL,
            params={
                "query": (
                    f"({query} {query_expansion(query, template_name, custom_terms)}) "
                    f"AND FIRST_PDATE:[{start_year}-01-01 TO {current_year}-12-31]"
                ),
                "format": "json",
                "pageSize": limit,
                "sort": "FIRST_PDATE_D desc",
            },
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 400:
            response = requests.get(
                EUROPE_PMC_SEARCH_URL,
                params={
                    "query": f"{query} {query_expansion(query, template_name, custom_terms)}",
                    "format": "json",
                    "pageSize": limit,
                    "sort": "FIRST_PDATE_D desc",
                },
                headers=REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
        response.raise_for_status()
        results = response.json().get("resultList", {}).get("result", [])
    except requests.RequestException:
        return []

    papers = []
    for item in results:
        year = item.get("pubYear")
        pmid = item.get("pmid")
        doi = item.get("doi")
        author_string = item.get("authorString", "")
        authors = [{"name": name.strip()} for name in author_string.split(",") if name.strip()]
        full_text_urls = (item.get("fullTextUrlList") or {}).get("fullTextUrl") or []
        full_text_url = full_text_urls[0].get("url") if full_text_urls else ""
        papers.append(
            {
                "paperId": item.get("id") or f"EPMC:{item.get('source', '')}:{item.get('title', '')}",
                "title": item.get("title", "Untitled"),
                "year": int(year) if year and str(year).isdigit() else None,
                "venue": item.get("journalTitle", "Europe PMC"),
                "abstract": item.get("abstractText", ""),
                "tldr": None,
                "citationCount": int(item.get("citedByCount", 0) or 0),
                "authors": authors,
                "externalIds": {"DOI": doi, "PubMed": pmid},
                "url": full_text_url
                or f"https://europepmc.org/article/{item.get('source', 'MED')}/{item.get('id', '')}",
                "source": "Europe PMC",
                "publication_types": [item.get("pubType")] if item.get("pubType") else [],
            }
        )
    return papers


@st.cache_data(ttl=3600, show_spinner=False)
def search_openalex(query, limit, start_year, template_name="通用學術搜尋", custom_terms=""):
    try:
        response = requests.get(
            OPENALEX_WORKS_URL,
            params={
                "search": f"{query} {query_expansion(query, template_name, custom_terms)}",
                "per-page": limit,
                "filter": f"from_publication_date:{start_year}-01-01",
                "sort": "publication_date:desc",
            },
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except requests.RequestException:
        return []

    papers = []
    for item in results:
        source = ((item.get("primary_location") or {}).get("source") or {}).get("display_name")
        ids = item.get("ids") or {}
        doi = (ids.get("doi") or "").replace("https://doi.org/", "") or None
        pmid_match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", ids.get("pmid", "") or "")
        authors = [
            {"name": (authorship.get("author") or {}).get("display_name")}
            for authorship in item.get("authorships", [])
            if (authorship.get("author") or {}).get("display_name")
        ]
        papers.append(
            {
                "paperId": item.get("id"),
                "title": item.get("display_name", "Untitled"),
                "year": item.get("publication_year"),
                "venue": source or "OpenAlex",
                "abstract": abstract_from_inverted_index(item.get("abstract_inverted_index")),
                "tldr": None,
                "citationCount": item.get("cited_by_count", 0),
                "authors": authors,
                "externalIds": {"DOI": doi, "PubMed": pmid_match.group(1) if pmid_match else None},
                "url": item.get("doi") or item.get("id"),
                "source": "OpenAlex",
                "publication_types": [item.get("type")] if item.get("type") else [],
            }
        )
    return papers


@st.cache_data(ttl=3600, show_spinner=False)
def search_crossref(query, limit, start_year, template_name="通用學術搜尋", custom_terms=""):
    try:
        response = requests.get(
            CROSSREF_WORKS_URL,
            params={
                "query.bibliographic": f"{query} {query_expansion(query, template_name, custom_terms)}",
                "rows": limit,
                "filter": f"from-pub-date:{start_year}-01-01",
                "sort": "published",
                "order": "desc",
            },
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json().get("message", {}).get("items", [])
    except requests.RequestException:
        return []

    papers = []
    for item in results:
        authors = [
            {"name": " ".join(part for part in [author.get("given"), author.get("family")] if part)}
            for author in item.get("author", [])
        ]
        titles = item.get("title") or []
        venues = item.get("container-title") or []
        papers.append(
            {
                "paperId": f"DOI:{item.get('DOI')}" if item.get("DOI") else item.get("URL"),
                "title": titles[0] if titles else "Untitled",
                "year": crossref_year(item),
                "venue": venues[0] if venues else "Crossref",
                "abstract": re.sub(r"<[^>]+>", "", item.get("abstract", "")),
                "tldr": None,
                "citationCount": item.get("is-referenced-by-count", 0),
                "authors": authors,
                "externalIds": {"DOI": item.get("DOI"), "PubMed": None},
                "url": item.get("URL"),
                "source": "Crossref",
                "publication_types": [item.get("type")] if item.get("type") else [],
            }
        )
    return papers


@st.cache_data(ttl=1800, show_spinner=False)
def search_tavily(query, api_key, max_results, start_year, template_name="通用學術搜尋", custom_terms=""):
    if not api_key:
        return []

    tavily = TavilyClient(api_key=api_key)
    search_query = build_academic_query(query, start_year, template_name, custom_terms)
    domains = template_domains(template_name)
    try:
        response = tavily.search(
            query=search_query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            include_raw_content=True,
            include_domains=domains,
        )
    except TypeError:
        domain_query = " OR ".join([f"site:{domain}" for domain in domains[:8]])
        try:
            response = tavily.search(
                query=f"{search_query} {domain_query}",
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
                include_raw_content=True,
            )
        except Exception:
            return []
    except Exception:
        return []
    return response.get("results", [])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_paper_lineage(user_input):
    clean_input = unquote(user_input).strip().replace('"', "")
    if not clean_input:
        return None

    doi_match = re.search(r"(10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+)", clean_input)
    arxiv_match = re.search(r"(\d{4}\.\d{4,5})", clean_input)

    lookup_id = None
    if doi_match:
        lookup_id = f"DOI:{doi_match.group(1)}"
    elif arxiv_match:
        lookup_id = f"arXiv:{arxiv_match.group(1)}"
    elif len(clean_input) >= 30 and re.fullmatch(r"[a-fA-F0-9]+", clean_input):
        lookup_id = clean_input

    if not lookup_id:
        search_result = search_semantic_scholar(clean_input, 1, 1900)
        if not search_result:
            return None
        lookup_id = search_result[0]["paperId"]

    try:
        response = requests.get(
            f"{SEMANTIC_SCHOLAR_PAPER_URL}/{lookup_id}",
            params={"fields": LINEAGE_FIELDS},
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        paper = response.json()
    except requests.RequestException:
        return None

    references = sorted(
        [item for item in paper.get("references", []) if item.get("paperId")],
        key=lambda item: item.get("citationCount") or 0,
        reverse=True,
    )[:8]
    citations = sorted(
        [item for item in paper.get("citations", []) if item.get("paperId")],
        key=lambda item: item.get("year") or 0,
        reverse=True,
    )[:8]

    return {"paper": paper, "references": references, "citations": citations}


def format_paper_context(papers):
    lines = []
    for index, paper in enumerate(papers, start=1):
        lines.append(
            "\n".join(
                [
                    f"[論文{index}] {paper.get('title', 'Untitled')}",
                    f"年份: {paper.get('year', 'N/A')}",
                    f"來源資料庫: {paper.get('source', 'Semantic Scholar')}",
                    f"來源類型: {paper.get('source_type', '未判定')}",
                    f"證據等級: {paper.get('evidence_level', '未判定')}",
                    f"期刊/會議: {paper.get('venue', 'N/A')}",
                    f"作者: {paper_authors(paper)}",
                    f"引用數: {paper.get('citationCount', 0)}",
                    f"識別碼: {paper_identifier(paper)}",
                    f"摘要: {paper_summary(paper, 900)}",
                ]
            )
        )
    return "\n\n".join(lines)


def format_web_context(sources):
    lines = []
    for index, source in enumerate(sources, start=1):
        content = source.get("raw_content") or source.get("content") or ""
        lines.append(
            "\n".join(
                [
                    f"[網頁{index}] {source.get('title', 'Untitled')}",
                    f"來源類型: {source.get('source_type', '未判定')}",
                    f"網址: {source.get('url', '')}",
                    f"內容: {content[:1200]}",
                ]
            )
        )
    return "\n\n".join(lines)


def generate_with_gemini(prompt, api_key, preferred_model):
    genai.configure(api_key=api_key)
    candidate_models = [preferred_model] + [
        model for model in GEMINI_FALLBACK_MODELS if model != preferred_model
    ]
    last_error = None

    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            text = getattr(response, "text", "")
            if not text:
                text = "⚠️ Gemini 已回應，但沒有產生可顯示文字。"
            if model_name != preferred_model:
                text = f"⚠️ 原選模型 `{preferred_model}` 不可用，已自動改用 `{model_name}`。\n\n{text}"
            return text
        except Exception as error:
            last_error = error
            continue

    return (
        "❌ Gemini 生成失敗：目前選擇的模型與備援模型都無法使用。\n\n"
        f"最後錯誤：{type(last_error).__name__}: {last_error}"
    )


def generate_keyword_suggestions(query, template_name, api_key, model_name):
    template = get_template(template_name)
    prompt = f"""
你是一位跨領域學術搜尋策略專家。請根據研究問題與領域模板，產生適合資料庫搜尋的英文關鍵字。

【研究問題】
{query}

【研究領域模板】
{template_name}

【此模板既有關鍵字】
{" ".join(template.get("expansion", []))}

【輸出要求】
1. 只輸出英文搜尋關鍵字與常見同義詞，不要寫解釋。
2. 使用空白分隔，必要時保留常用縮寫，例如 TEM、EELS、RCT、VO2max。
3. 包含方法、技術、研究設計、結果指標、應用場景等搜尋詞。
4. 控制在 25-45 個關鍵詞以內。
5. 不要輸出 citation、Markdown 標題或項目符號。
"""
    return generate_with_gemini(prompt, api_key, model_name).strip()


def build_citation_section(papers, web_sources):
    lines = ["\n\n## Citation Sources"]
    for index, paper in enumerate(papers, start=1):
        identifier = paper_identifier(paper) or "No DOI/PMID"
        source = paper.get("source", "Semantic Scholar")
        year = paper.get("year", "N/A")
        venue = paper.get("venue", "N/A")
        url = paper.get("url", "")
        lines.append(
            f"- [論文{index}] {paper.get('title', 'Untitled')} ({year}). "
            f"{venue}. {identifier}. Source: {source}. {url}"
        )

    for index, source in enumerate(web_sources, start=1):
        lines.append(
            f"- [網頁{index}] {source.get('title', 'Untitled')}. "
            f"{source.get('url', '')}"
        )

    return "\n".join(lines)


def generate_report(query, papers, web_sources, api_key, model_name, template_name):
    template = get_template(template_name)

    prompt = f"""
你是一位嚴謹的學術研究助理。請用台灣繁體中文回答，並以證據品質為優先。

【研究問題】
{query}

【研究領域模板】
{template_name}

【領域特化輸出要求】
{template.get("report_focus", "")}

【整合學術資料庫論文資料】
{format_paper_context(papers)}

【Tavily 最新網頁資料】
{format_web_context(web_sources)}

【輸出格式】
1. 先用 5-8 句話回答核心結論。
2. 整理最重要研究，包含年份、研究設計、族群/樣本、主要發現、限制。
3. 優先引用「正式論文」與較高「證據等級」的資料；預印本與一般網頁只能作為最新線索。
4. 清楚區分系統性回顧/統合分析、RCT、觀察研究、評論文章，並指出來源資料庫。
5. 每一段只要包含研究發現、年份、數據、比較或建議，就必須附 citation，格式只能使用 [論文1]、[論文2]、[網頁1] 這種標記。
6. 如果資料不足或來源不像正式論文，請直接指出，不要假裝確定。
7. 最後給實務建議，分成「較有把握」與「仍需保留」，每一點也必須附 citation。
8. 不要輸出沒有 citation 的學術結論；如果沒有足夠來源，請明確說「目前來源不足」。
"""
    report = generate_with_gemini(prompt, api_key, model_name)
    return report + build_citation_section(papers, web_sources)


def generate_lineage_report(lineage, question, api_key, model_name):
    paper = lineage["paper"]
    references = lineage["references"]
    citations = lineage["citations"]
    context = "\n\n".join(
        [
            "【主論文】",
            format_paper_context([paper]),
            "【高引用參考文獻】",
            format_paper_context(references),
            "【近期引用文獻】",
            format_paper_context(citations),
        ]
    )

    prompt = f"""
你是一位學術脈絡分析師。請用台灣繁體中文，根據以下主論文、參考文獻與引用文獻回答。

【使用者關注】
{question or "請分析這篇論文在領域中的位置、上游基礎與後續發展。"}

{context}

請輸出：
1. 主論文的核心貢獻。
2. 它承接了哪些上游研究。
3. 後續研究如何延伸或修正它。
4. 對閱讀這個領域的建議路徑。
"""
    return generate_with_gemini(prompt, api_key, model_name)


def render_paper_table(papers):
    if not papers:
        st.info("尚無論文資料。")
        return

    rows = [
        {
            "來源": paper.get("source", "Semantic Scholar"),
            "類型": paper.get("source_type", "未判定"),
            "證據等級": paper.get("evidence_level", "未判定"),
            "年份": paper.get("year"),
            "標題": paper.get("title"),
            "期刊/會議": paper.get("venue"),
            "引用數": paper.get("citationCount", 0),
            "作者": paper_authors(paper),
            "識別碼": paper_identifier(paper),
        }
        for paper in papers
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_result_summary(papers, web_sources):
    formal_count = sum(1 for paper in papers if paper.get("source_type") == "正式論文")
    preprint_count = sum(1 for paper in papers if paper.get("source_type") == "預印本")
    high_evidence_count = sum(1 for paper in papers if (paper.get("evidence_score") or 0) >= 4)
    latest_year = max([paper.get("year") or 0 for paper in papers], default=0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("去重後論文", len(papers))
    col2.metric("正式論文", formal_count)
    col3.metric("高證據等級", high_evidence_count)
    col4.metric("最新年份", latest_year or "N/A")

    if preprint_count or web_sources:
        st.caption(f"另含 {preprint_count} 筆預印本標記，以及 {len(web_sources)} 筆最新網頁線索。")


def render_sources(papers, web_sources):
    with st.expander("查看論文來源", expanded=False):
        for index, paper in enumerate(papers, start=1):
            url = paper.get("url")
            title = paper.get("title", "Untitled")
            st.markdown(f"**[論文{index}] {title}**")
            st.caption(
                f"{paper.get('source', 'Semantic Scholar')} | {paper.get('year', 'N/A')} | "
                f"{paper.get('source_type', '未判定')} | {paper.get('evidence_level', '未判定')} | "
                f"{paper.get('venue', 'N/A')} | "
                f"Cited: {paper.get('citationCount', 0)} | {paper_identifier(paper)}"
            )
            if url:
                st.markdown(url)
            if summary := paper_summary(paper):
                st.write(summary)
            st.divider()

    if web_sources:
        with st.expander("查看最新網頁來源", expanded=False):
            for index, source in enumerate(web_sources, start=1):
                st.markdown(f"**[網頁{index}] [{source.get('title', 'Untitled')}]({source.get('url', '')})**")
                st.caption(f"{source.get('source_type', '未判定')} | {(source.get('content') or '')[:300]}")
                st.divider()


def sidebar():
    with st.sidebar:
        st.header("設定")

        default_gemini_key = get_secret_or_empty("GOOGLE_API_KEY")
        default_tavily_key = get_secret_or_empty("TAVILY_API_KEY")

        gemini_key = st.text_input(
            "Gemini API Key",
            value=default_gemini_key,
            type="password",
        )
        tavily_key = st.text_input(
            "Tavily API Key（可選，用於最新網頁補強）",
            value=default_tavily_key,
            type="password",
        )

        model_name = st.selectbox(
            "Gemini 模型",
            GEMINI_MODELS,
            index=0,
            help="僅使用 Gemini 3 preview 系列；若 API 尚未開放這些模型，報告會直接提示失敗。",
        )
        template_name = st.selectbox(
            "研究領域模板",
            list(RESEARCH_TEMPLATES.keys()),
            index=0,
            help="依領域自動補英文關鍵字、Tavily 學術網域與報告結構。",
        )
        topic_query = st.session_state.get("topic_query", "").strip()
        if st.button("用 Gemini 建議關鍵字", use_container_width=True):
            if not gemini_key:
                st.warning("請先輸入 Gemini API Key。")
            elif not topic_query:
                st.warning("請先在主畫面輸入研究問題。")
            else:
                with st.spinner("Gemini 正在產生搜尋關鍵字..."):
                    st.session_state.custom_terms = generate_keyword_suggestions(
                        topic_query,
                        template_name,
                        gemini_key,
                        model_name,
                    )
                st.rerun()
        custom_terms = st.text_area(
            "Gemini 建議 / 自訂補充關鍵字（可編輯）",
            key="custom_terms",
            placeholder="例如：STEM-EELS phonon EELS battery materials",
            height=80,
            help="可由 Gemini 依研究問題產生，也可手動修改；會附加到所有資料庫查詢。",
        )
        start_year = st.number_input(
            "搜尋年份（起始）",
            min_value=1900,
            max_value=date.today().year,
            value=max(date.today().year - 3, 1900),
            step=1,
            help="只搜尋此年份之後的文獻，適合追最新研究。",
        )
        paper_limit = st.slider("Semantic Scholar 論文數", 5, 30, 12)
        biomedical_limit = st.slider("PubMed / Europe PMC 論文數", 5, 30, 12)
        index_limit = st.slider("OpenAlex / Crossref 論文數", 5, 30, 12)
        web_limit = st.slider("Tavily 最新來源數", 0, 15, 5)

        st.caption("預設僅使用 Gemini 3 preview 與近年正式論文；可用研究領域模板切換 query expansion、來源優先權與報告格式。")

    return (
        gemini_key,
        tavily_key,
        model_name,
        template_name,
        custom_terms,
        paper_limit,
        biomedical_limit,
        index_limit,
        web_limit,
        start_year,
    )


def topic_search_tab(
    gemini_key,
    tavily_key,
    model_name,
    template_name,
    custom_terms,
    paper_limit,
    biomedical_limit,
    index_limit,
    web_limit,
    start_year,
):
    query = st.text_input(
        "研究問題",
        placeholder="例如：有氧運動與肌力訓練不相容，或 TEM-EELS 技術近年發展",
        key="topic_query",
    )
    col_search, col_report = st.columns([1, 1])

    with col_search:
        search_clicked = st.button("搜尋文獻", type="primary", use_container_width=True)
    with col_report:
        report_clicked = st.button("產生學術報告", use_container_width=True)

    if search_clicked:
        clean_query = normalize_query(query)
        if not clean_query:
            st.warning("請先輸入研究問題。")
            return

        with st.status("正在搜尋多個學術資料庫與最新網頁來源...", expanded=True) as status:
            st.write(f"套用研究領域模板：{template_name}")
            st.write("搜尋 Semantic Scholar：近年優先，並補高引用結果。")
            semantic_papers = search_semantic_scholar(clean_query, paper_limit, start_year, template_name, custom_terms)
            st.write(f"Semantic Scholar：{len(semantic_papers)} 筆")
            st.write("搜尋 PubMed 與 Europe PMC。")
            pubmed_papers = search_pubmed(clean_query, biomedical_limit, start_year, template_name, custom_terms)
            europe_pmc_papers = search_europe_pmc(clean_query, biomedical_limit, start_year, template_name, custom_terms)
            st.write(f"PubMed：{len(pubmed_papers)} 筆；Europe PMC：{len(europe_pmc_papers)} 筆")
            st.write("搜尋 OpenAlex 與 Crossref。")
            openalex_papers = search_openalex(clean_query, index_limit, start_year, template_name, custom_terms)
            crossref_papers = search_crossref(clean_query, index_limit, start_year, template_name, custom_terms)
            st.write(f"OpenAlex：{len(openalex_papers)} 筆；Crossref：{len(crossref_papers)} 筆")
            st.write("搜尋 Tavily 學術站點。")
            web_sources = (
                search_tavily(clean_query, tavily_key, web_limit, start_year, template_name, custom_terms)
                if web_limit
                else []
            )
            st.write(f"Tavily：{len(web_sources)} 筆")
            papers = dedupe_papers(
                semantic_papers
                + pubmed_papers
                + europe_pmc_papers
                + openalex_papers
                + crossref_papers
            )
            papers = sorted(papers, key=paper_sort_key, reverse=True)
            st.session_state.papers = papers
            st.session_state.web_sources = annotate_web_sources(web_sources)
            st.session_state.report = ""
            if not papers and not web_sources:
                status.update(label="搜尋完成，但各資料源都沒有回傳結果", state="complete", expanded=True)
                st.warning("沒有找到可用結果。可能是資料庫暫時限流、查詢太窄，或搜尋年份太新。可放寬年份或改用英文關鍵字再試。")
            else:
                status.update(label=f"搜尋完成：找到 {len(papers)} 筆去重後論文", state="complete", expanded=False)

    if st.session_state.papers or st.session_state.web_sources:
        st.subheader("搜尋結果")
        render_result_summary(st.session_state.papers, st.session_state.web_sources)
        render_paper_table(st.session_state.papers)
        render_sources(st.session_state.papers, st.session_state.web_sources)

    if report_clicked:
        if not query.strip():
            st.warning("請先輸入研究問題。")
        elif not gemini_key:
            st.error("請先輸入 Gemini API Key。")
        elif not st.session_state.papers and not st.session_state.web_sources:
            st.warning("請先搜尋文獻，再產生報告。")
        else:
            with st.spinner(f"正在使用 {model_name} 產生報告..."):
                st.session_state.report = generate_report(
                    query,
                    st.session_state.papers,
                    st.session_state.web_sources,
                    gemini_key,
                    model_name,
                    template_name,
                )

    if st.session_state.report:
        st.subheader("學術報告")
        st.markdown(st.session_state.report)
        st.download_button(
            "下載報告 Markdown",
            st.session_state.report,
            file_name="academic_report.md",
            mime="text/markdown",
        )


def lineage_tab(gemini_key, model_name):
    user_input = st.text_input(
        "輸入 DOI、arXiv ID、Semantic Scholar Paper ID 或論文標題",
        placeholder="例如：10.xxxx/xxxxx",
    )
    question = st.text_input(
        "想分析的角度（可選）",
        placeholder="例如：這篇論文如何影響後續 concurrent training 研究？",
    )

    col_fetch, col_ai = st.columns([1, 1])
    with col_fetch:
        fetch_clicked = st.button("建立引用脈絡", type="primary", use_container_width=True)
    with col_ai:
        analyze_clicked = st.button("分析脈絡", use_container_width=True)

    if fetch_clicked:
        if not user_input.strip():
            st.warning("請先輸入論文資訊。")
            return
        with st.spinner("正在讀取引用與參考文獻..."):
            st.session_state.lineage = fetch_paper_lineage(user_input)
        if not st.session_state.lineage:
            st.error("找不到這篇論文，請改用 DOI、Paper ID 或更完整標題。")

    if st.session_state.lineage:
        lineage = st.session_state.lineage
        st.subheader("主論文")
        render_paper_table([lineage["paper"]])

        left, right = st.columns(2)
        with left:
            st.subheader("高引用參考文獻")
            render_paper_table(lineage["references"])
        with right:
            st.subheader("近期引用文獻")
            render_paper_table(lineage["citations"])

    if analyze_clicked:
        if not gemini_key:
            st.error("請先輸入 Gemini API Key。")
        elif not st.session_state.lineage:
            st.warning("請先建立引用脈絡。")
        else:
            with st.spinner(f"正在使用 {model_name} 分析脈絡..."):
                result = generate_lineage_report(
                    st.session_state.lineage,
                    question,
                    gemini_key,
                    model_name,
                )
            st.subheader("脈絡分析")
            st.markdown(result)


def main():
    init_session_state()
    (
        gemini_key,
        tavily_key,
        model_name,
        template_name,
        custom_terms,
        paper_limit,
        biomedical_limit,
        index_limit,
        web_limit,
        start_year,
    ) = sidebar()

    st.title("🔎 Gemini x 學術搜尋")
    st.caption("整合 Semantic Scholar 論文資料、Tavily 最新網頁線索與 Gemini 3 preview 報告生成。")

    tab_topic, tab_lineage = st.tabs(["主題搜尋", "單篇論文脈絡"])
    with tab_topic:
        topic_search_tab(
            gemini_key,
            tavily_key,
            model_name,
            template_name,
            custom_terms,
            paper_limit,
            biomedical_limit,
            index_limit,
            web_limit,
            start_year,
        )
    with tab_lineage:
        lineage_tab(gemini_key, model_name)

    st.divider()
    st.caption("Designed for academic research. 請以原始論文與正式資料庫作為最終判讀依據。")


if __name__ == "__main__":
    main()
