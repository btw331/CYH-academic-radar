"""
學術雷達 V12.9 (Future Proof) - 優化版
功能完整的學術論文分析與搜尋系統
"""
import streamlit as st
import google.generativeai as genai
import requests
import pandas as pd
import re
import json
from urllib.parse import unquote
import time
from typing import Optional, Dict, List, Any, Tuple
from functools import wraps
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 0. 基礎設定與 CSS
# ==========================================
st.set_page_config(page_title="學術雷達 V12.9 (Future Proof)", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&family=Roboto+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans TC', sans-serif; color: #333; }
    
    .report-container {
        background-color: #fdfbf7; 
        border: 2px solid #5d4037; 
        border-radius: 12px;
        padding: 30px; 
        margin-bottom: 25px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        line-height: 1.7;
    }
    
    .pi-box {
        background-color: #e3f2fd; border: 2px solid #1565c0; border-radius: 12px;
        padding: 25px; margin-top: 20px; margin-bottom: 25px;
    }
    
    .source-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.85em; font-weight: 700; margin-bottom: 15px;
        background-color: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7;
    }
    
    .bib-container { background-color: #fff8e1; padding: 20px; border-radius: 10px; border: 1px solid #ffe082; margin-top: 20px; }
    
    .auth-tag-first { color: #d32f2f; font-weight: bold; }
    .auth-tag-last { color: #1976d2; font-weight: bold; }
    
    .search-card {
        background-color: #ffffff; padding: 20px; border-radius: 10px;
        border: 1px solid #e0e0e0; margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); transition: transform 0.2s;
    }
    .search-card:hover { transform: translateY(-3px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
    .sc-title { font-size: 1.1em; font-weight: bold; color: #1a237e; margin-bottom: 8px; }

    .chat-box { background-color: #f1f8e9; padding: 15px; border-radius: 10px; border: 1px solid #c5e1a5; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 常數定義
# ==========================================
HEADERS = {"User-Agent": "AcademicRadar/12.9"}
API_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_DELAY = 1

LIGHT_FIELDS = "paperId,title,year,citationCount,venue,authors.name,references.paperId,references.citationCount,references.year,citations.paperId,citations.citationCount,citations.year"
RICH_FIELDS = "paperId,title,year,citationCount,venue,authors.name,authors.authorId,abstract,tldr"
BROAD_FIELDS = "paperId,title,year,citationCount,venue,authors.name,abstract,tldr"
AUTHOR_FIELDS = "authorId,name,citationCount,hIndex,paperCount,papers.title,papers.year,papers.citationCount,papers.venue"

# ==========================================
# 2. 工具函數
# ==========================================
def retry_on_failure(max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY):
    """API 調用重試裝飾器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, requests.Timeout, requests.ConnectionError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"API 調用失敗 (嘗試 {attempt + 1}/{max_retries}): {str(e)}")
                        time.sleep(delay * (attempt + 1))  # 指數退避
                    else:
                        logger.error(f"API 調用最終失敗: {str(e)}")
                except Exception as e:
                    logger.error(f"未預期的錯誤: {str(e)}")
                    raise
            raise last_exception
        return wrapper
    return decorator

def validate_input(query: str, min_length: int = 1) -> bool:
    """驗證輸入"""
    if not query or not isinstance(query, str):
        return False
    return len(query.strip()) >= min_length

# ==========================================
# 3. 核心搜尋引擎
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
@retry_on_failure()
def search_broad_papers(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    廣度搜尋論文
    
    Args:
        query: 搜尋關鍵字
        limit: 返回結果數量限制
        
    Returns:
        論文列表
    """
    if not validate_input(query):
        return []
    
    try:
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": limit, "fields": BROAD_FIELDS},
            headers=HEADERS,
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        return response.json().get('data', [])
    except requests.HTTPError as e:
        logger.error(f"HTTP 錯誤: {e.response.status_code} - {e.response.text}")
        return []
    except requests.RequestException as e:
        logger.error(f"請求錯誤: {str(e)}")
        return []

@st.cache_data(ttl=3600, show_spinner=False)
@retry_on_failure()
def fetch_network_skeleton(user_input: str) -> Optional[Dict[str, Any]]:
    """
    獲取論文引用網絡骨架
    
    Args:
        user_input: DOI、arXiv ID 或論文標題
        
    Returns:
        包含 hero、all_ancestors、all_descendants 的字典，失敗返回 None
    """
    if not validate_input(user_input):
        return None
    
    clean_input = unquote(user_input).strip().replace('"', '')
    lookup_id = None
    
    # 識別 DOI 或 arXiv ID
    doi_match = re.search(r'(10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+)', clean_input)
    arxiv_match = re.search(r'(\d{4}\.\d{4,5})', clean_input)
    
    if doi_match:
        lookup_id = f"DOI:{doi_match.group(1)}"
    elif arxiv_match:
        lookup_id = f"arXiv:{arxiv_match.group(1)}"
    
    def fetch_paper(paper_id: str) -> Optional[Dict[str, Any]]:
        """獲取單篇論文資料"""
        try:
            response = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
                params={"fields": LIGHT_FIELDS},
                headers=HEADERS,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            logger.error(f"獲取論文失敗 {paper_id}: {e.response.status_code}")
            return None
        except requests.RequestException as e:
            logger.error(f"請求錯誤: {str(e)}")
            return None
    
    # 嘗試直接查找
    hero = fetch_paper(lookup_id) if lookup_id else None
    
    # 如果直接查找失敗，使用搜尋
    if not hero:
        try:
            response = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": clean_input, "limit": 1, "fields": "paperId"},
                headers=HEADERS,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            data = response.json().get('data')
            if data and len(data) > 0:
                hero = fetch_paper(data[0]['paperId'])
        except requests.RequestException as e:
            logger.error(f"搜尋論文失敗: {str(e)}")
    
    if not hero or not hero.get('paperId'):
        return None
    
    # 排序引用和參考文獻
    refs = sorted(
        [r for r in (hero.get('references') or []) if r.get('paperId')],
        key=lambda x: (x.get('citationCount') or 0),
        reverse=True
    )
    cites = sorted(
        [c for c in (hero.get('citations') or []) if c.get('paperId')],
        key=lambda x: (x.get('year') or 0),
        reverse=True
    )
    
    return {'hero': hero, 'all_ancestors': refs, 'all_descendants': cites}

@retry_on_failure()
def enrich_segment(paper_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    豐富論文資料（添加摘要、作者 ID 等）
    
    Args:
        paper_objects: 論文對象列表
        
    Returns:
        豐富後的論文列表
    """
    if not paper_objects:
        return []
    
    ids = [p['paperId'] for p in paper_objects if p.get('paperId')]
    if not ids:
        return paper_objects
    
    enriched_map = {}
    try:
        response = requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            params={"fields": RICH_FIELDS},
            json={"ids": ids},
            headers=HEADERS,
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        for p in response.json():
            if p:
                enriched_map[p['paperId']] = p
    except requests.RequestException as e:
        logger.error(f"批量獲取論文失敗: {str(e)}")
    
    enriched_list = []
    for p in paper_objects:
        pid = p['paperId']
        if pid in enriched_map:
            full_data = enriched_map[pid].copy()
            if 'code' in p:
                full_data['code'] = p['code']
            enriched_list.append(full_data)
        else:
            enriched_list.append(p)
    
    return enriched_list

@retry_on_failure()
def fetch_author_profile_no_cache(author_id: str) -> Optional[Dict[str, Any]]:
    """
    獲取作者資料（不使用快取）
    
    Args:
        author_id: 作者 ID
        
    Returns:
        作者資料字典，失敗返回 None
    """
    if not validate_input(author_id):
        return None
    
    try:
        response = requests.get(
            f"https://api.semanticscholar.org/graph/v1/author/{author_id}",
            params={"fields": AUTHOR_FIELDS},
            headers=HEADERS,
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        logger.error(f"獲取作者資料失敗 {author_id}: {e.response.status_code}")
        return None
    except requests.RequestException as e:
        logger.error(f"請求錯誤: {str(e)}")
        return None

# ==========================================
# 4. AI 分析功能
# ==========================================
def generate_deep_analysis_classic(
    hero: Dict[str, Any],
    ancestors: List[Dict[str, Any]],
    descendants: List[Dict[str, Any]],
    api_key: str,
    model_name: str
) -> str:
    """
    生成深度分析報告
    
    Args:
        hero: 主角論文
        ancestors: 祖先文獻列表
        descendants: 後代文獻列表
        api_key: Gemini API 金鑰
        model_name: 模型名稱
        
    Returns:
        分析報告（Markdown 格式）
    """
    if not api_key:
        return "❌ 錯誤：未提供 API 金鑰"
    
    genai.configure(api_key=api_key)
    
    def format_paper(p: Dict[str, Any], code: str) -> str:
        """格式化論文資訊"""
        title = p.get('title', 'Unknown Title')
        year = p.get('year', 'N/A')
        cite = p.get('citationCount', 0)
        
        auth_list = p.get('authors', [])
        if not auth_list:
            auth_str = "Unknown"
        elif len(auth_list) <= 4:
            auth_str = ", ".join([a.get('name', '?') for a in auth_list])
        else:
            first = auth_list[0].get('name', '?')
            last_3 = [a.get('name', '?') for a in auth_list[-3:]]
            auth_str = f"First:{first} ... Last3:{', '.join(last_3)}"
        
        return f"[{code}] {title} ({year}) | {auth_str} | Cited:{cite}"
    
    context = f"主角論文: {format_paper(hero, 'Hero')}\n\n"
    context += "【祖先文獻】:\n" + "\n".join([format_paper(a, a.get('code', 'A')) for a in ancestors]) + "\n\n"
    context += "【後代文獻】:\n" + "\n".join([format_paper(d, d.get('code', 'D')) for d in descendants])
    
    system_prompt = """
    你是一位精通「學術系譜學」的 AI 專家。
    請基於提供的論文列表，進行深度的數據推論、概念流變分析，並預測未來的可能性。
    
    【重要指令】：
    1. **語言**：所有輸出必須使用 **繁體中文 (Traditional Chinese, Taiwan)**。
    2. **表格呈現**：概念流變請務必使用 **Markdown 表格** 呈現。
    
    【輸出報告格式】：
    ### 📜 學術雷達深度報告
    
    #### 1. 🌊 概念流變表 (Concept Flow Table)
    | 階段 | 核心關鍵詞 | 演變描述 |
    | :--- | :--- | :--- |
    | **A系列 (起源)** | ... | ... |
    | **Hero (轉折)** | ... | ... |
    | **D系列 (應用)** | ... | ... |
    
    #### 2. 🧩 領域分類與聚類
    * **群組 A (理論基石)**：[A1], [A3]...
    * **群組 B (方法突破)**：[Hero], [D1]...
    
    #### 3. 👑 領域領袖與師承
    * **核心實驗室 (PI)**：(觀察作者群的最後幾位，推論核心實驗室)
    * **第一作者 (執行者)**：(觀察第一作者的貢獻)
    
    #### 4. 🔗 技術演進詳解
    **4.1 ⏪ 向前溯源**
    * **[A?]** (PI: ...): **[貢獻]** ... 
    
    **4.2 ⏩ 向後展望**
    * **[D?]** (PI: ...): **[貢獻]** ... 
    
    #### 5. 🔮 未來可能性圓錐 (The Cone of Possibilities)
    *(針對 Hero 論文，預測未來)*
    * **🎯 核心 (Probable)**：...
    * **🚀 擴展 (Plausible)**：...
    * **🌌 邊界 (Possible)**：...
    """
    
    try:
        model = genai.GenerativeModel(model_name)
        result = model.generate_content(system_prompt + context)
        return result.text
    except Exception as e:
        logger.error(f"AI 分析失敗: {str(e)}")
        return f"❌ 分析失敗: {str(e)}"

def generate_author_analysis(
    author_name: str,
    selected_papers: List[Dict[str, Any]],
    api_key: str,
    model_name: str
) -> str:
    """
    生成作者分析報告
    
    Args:
        author_name: 作者姓名
        selected_papers: 選中的論文列表
        api_key: Gemini API 金鑰
        model_name: 模型名稱
        
    Returns:
        分析報告（Markdown 格式）
    """
    if not api_key:
        return "❌ 錯誤：未提供 API 金鑰"
    
    if not selected_papers:
        return "❌ 錯誤：未選擇論文"
    
    genai.configure(api_key=api_key)
    
    papers_str = "\n".join([
        f"- {p.get('title', 'Unknown')} ({p.get('year', 'N/A')}) | Cited: {p.get('citationCount', 0)}"
        for p in selected_papers
    ])
    
    system_prompt = f"""
    你是一位「學術星探」。請分析這位 PI (或研究員)。
    【注意】：**已排除同名同姓的干擾資料**，以下提供的論文確定皆為同一人所著。
    
    【檔案】姓名: {author_name}
    【經確認的代表作】:
    {papers_str}
    
    【任務】：請用**條列式**分析：
    1. **學術江湖地位** (是資深大佬、實驗室主持人，還是新銳研究員？)
    2. **核心研究版圖** (根據上述論文，精準定位其專長)
    3. **研究風格與專長**
    """
    
    try:
        model = genai.GenerativeModel(model_name)
        result = model.generate_content(system_prompt)
        return result.text
    except Exception as e:
        logger.error(f"作者分析失敗: {str(e)}")
        return f"❌ 分析失敗: {str(e)}"

def ask_historian(
    question: str,
    context_data: List[Dict[str, Any]],
    api_key: str,
    model_name: str
) -> str:
    """
    歷史學家問答功能
    
    Args:
        question: 用戶問題
        context_data: 上下文資料
        api_key: Gemini API 金鑰
        model_name: 模型名稱
        
    Returns:
        AI 回答
    """
    if not api_key:
        return "❌ 錯誤：未提供 API 金鑰"
    
    if not validate_input(question):
        return "❌ 錯誤：問題不能為空"
    
    # 格式化上下文資料，避免 datetime 等複雜對象造成問題
    def format_context_item(item: Dict[str, Any]) -> str:
        """格式化單個上下文項目"""
        parts = []
        if 'code' in item:
            parts.append(f"代號: {item['code']}")
        if 'title' in item:
            parts.append(f"標題: {item['title']}")
        if 'year' in item:
            # 處理年份，可能是數字或字符串
            year = item['year']
            if year and year != 'N/A':
                parts.append(f"年份: {year}")
        return " | ".join(parts)
    
    # 格式化所有上下文資料
    formatted_context = "\n".join([
        f"- {format_context_item(item)}"
        for item in context_data
        if isinstance(item, dict)
    ])
    
    # 限制上下文長度
    if len(formatted_context) > 2000:
        formatted_context = formatted_context[:2000] + "..."
    
    genai.configure(api_key=api_key)
    prompt = f"""你是一位學術顧問。請用繁體中文回答。

【背景資料 - 相關論文列表】：
{formatted_context}

【問題】：
{question}

請基於上述論文資料，提供專業的分析和回答。"""
    
    try:
        model = genai.GenerativeModel(model_name)
        result = model.generate_content(prompt)
        return result.text
    except Exception as e:
        logger.error(f"問答失敗: {str(e)}")
        return f"❌ 回答失敗: {str(e)}"

def generate_multilingual_abstract(
    text_content: str,
    api_key: str,
    model_name: str
) -> str:
    """
    生成多語言摘要
    
    Args:
        text_content: 原始文本內容
        api_key: Gemini API 金鑰
        model_name: 模型名稱
        
    Returns:
        多語言摘要
    """
    if not api_key:
        return "❌ 錯誤：未提供 API 金鑰"
    
    genai.configure(api_key=api_key)
    prompt = f"""請將報告總結為 **100 字摘要**。輸出：繁體中文、English、日本語。\n內容：\n{text_content[:2000]}"""
    
    try:
        model = genai.GenerativeModel(model_name)
        result = model.generate_content(prompt)
        return result.text
    except Exception as e:
        logger.error(f"摘要生成失敗: {str(e)}")
        return f"❌ 摘要生成失敗: {str(e)}"

# ==========================================
# 5. 存檔功能
# ==========================================
def export_state_to_json() -> str:
    """
    匯出狀態到 JSON
    
    Returns:
        JSON 字符串
    """
    def json_serializer(obj):
        """自定義 JSON 序列化器，處理 datetime 等特殊對象"""
        from datetime import datetime, date
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return str(obj)
    
    data = {
        k: st.session_state[k]
        for k in ['skeleton', 'full_lineage', 'offsets', 'deep_dive_result', 'pi_analysis_result']
        if k in st.session_state
    }
    return json.dumps(data, default=json_serializer, ensure_ascii=False, indent=2)

# ==========================================
# 6. UI 邏輯
# ==========================================
# 初始化 session state
if 'skeleton' not in st.session_state:
    st.session_state.skeleton = None
if 'full_lineage' not in st.session_state:
    st.session_state.full_lineage = {'hero': {}, 'ancestors': [], 'descendants': []}
if 'offsets' not in st.session_state:
    st.session_state.offsets = {'a': 0, 'd': 0}
if 'deep_dive_result' not in st.session_state:
    st.session_state.deep_dive_result = None
if 'pi_analysis_result' not in st.session_state:
    st.session_state.pi_analysis_result = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pre_fill_doi' not in st.session_state:
    st.session_state.pre_fill_doi = ""
if 'read_only_mode' not in st.session_state:
    st.session_state.read_only_mode = False
if 'pi_raw_data' not in st.session_state:
    st.session_state.pi_raw_data = None

# 側邊欄
with st.sidebar:
    st.title("🔬 參數設定")
    
    # Secrets 自動讀取邏輯
    if "GOOGLE_API_KEY" in st.secrets:
        st.success("✅ 已自動載入 Gemini Key (系統託管)")
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = st.text_input("Gemini API Key", type="password", help="請輸入 Google Gemini API 金鑰")
    
    model_name = st.selectbox(
        "模型",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"],
        index=0,
        help="選擇 Gemini 模型版本"
    )
    
    st.divider()
    st.markdown("### 📥 知識庫存檔")
    
    if st.session_state.deep_dive_result:
        st.download_button(
            "下載進度 (JSON)",
            export_state_to_json(),
            "radar_fix.json",
            "application/json",
            help="完整資料備份"
        )
        st.download_button(
            "下載報告 (.md)",
            st.session_state.deep_dive_result,
            "academic_report.md",
            "text/markdown",
            help="下載 Markdown 格式報告"
        )
    
    with st.expander("📂 讀取舊檔案 (JSON/MD)", expanded=True):
        uploaded_file = st.file_uploader("拖曳檔案到此", type=["json", "md"], help="支援 JSON 進度檔或 Markdown 報告檔")
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".json"):
                    data = json.load(uploaded_file)
                    for k, v in data.items():
                        st.session_state[k] = v
                    st.session_state.read_only_mode = False
                    st.toast("✅ JSON 進度還原成功！")
                    time.sleep(1)
                    st.rerun()
                elif uploaded_file.name.endswith(".md"):
                    content = uploaded_file.read().decode("utf-8")
                    st.session_state.deep_dive_result = content
                    st.session_state.read_only_mode = True
                    st.toast("📖 進入純閱讀模式")
                    time.sleep(1)
                    st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"JSON 解析失敗: {e}")
            except Exception as e:
                st.error(f"讀取失敗: {e}")

st.title("🧬 學術雷達 V12.9 (Future Proof)")
st.caption("核心：**同名同姓篩選** + **Streamlit 參數修正** + **Secrets 管理**。")

# === 核心處理邏輯 ===
def process_mining(doi_target: str, action: str = 'init') -> None:
    """
    處理深度挖掘
    
    Args:
        doi_target: DOI、arXiv ID 或論文標題
        action: 操作類型 ('init', 'older', 'newer', 'expand_both')
    """
    with st.status("正在啟動 V10.2 經典引擎...", expanded=True) as status:
        if action == 'init':
            st.write("📡 掃描引用網絡骨架...")
            skeleton = fetch_network_skeleton(doi_target)
            if not skeleton:
                status.update(label="❌ 找不到資料", state="error")
                st.error("找不到資料。請檢查輸入是否正確。")
                return
            st.session_state.skeleton = skeleton
            st.session_state.offsets = {'a': 0, 'd': 0}
            hero_enriched = enrich_segment([skeleton['hero']])
            if hero_enriched:
                st.session_state.full_lineage = {'hero': hero_enriched[0], 'ancestors': [], 'descendants': []}
            else:
                st.error("無法豐富主角論文資料")
                return
            st.session_state.chat_history = []
            st.session_state.pi_analysis_result = None
            st.session_state.pi_raw_data = None
            st.session_state.read_only_mode = False
        
        st.write("🔍 擴充詳細資料 (PI、摘要)...")
        sk = st.session_state.skeleton
        if not sk:
            st.error("骨架資料不存在")
            return
        
        off = st.session_state.offsets
        new_a_objs, new_d_objs = [], []
        
        if action == 'init':
            new_a_objs = sk['all_ancestors'][0:5]
            new_d_objs = sk['all_descendants'][0:5]
            st.session_state.offsets = {'a': 5, 'd': 5}
        elif action == 'older':
            new_a_objs = sk['all_ancestors'][off['a']:off['a']+5]
            st.session_state.offsets['a'] += 5
        elif action == 'newer':
            new_d_objs = sk['all_descendants'][off['d']:off['d']+5]
            st.session_state.offsets['d'] += 5
        elif action == 'expand_both':
            new_a_objs = sk['all_ancestors'][off['a']:off['a']+5]
            new_d_objs = sk['all_descendants'][off['d']:off['d']+5]
            st.session_state.offsets['a'] += 5
            st.session_state.offsets['d'] += 5
        
        enriched_a = enrich_segment(new_a_objs)
        enriched_d = enrich_segment(new_d_objs)
        
        exist_a = len(st.session_state.full_lineage['ancestors'])
        for i, p in enumerate(enriched_a):
            p['code'] = f"A{exist_a + i + 1}"
        
        exist_d = len(st.session_state.full_lineage['descendants'])
        for i, p in enumerate(enriched_d):
            p['code'] = f"D{exist_d + i + 1}"
        
        st.session_state.full_lineage['ancestors'].extend(enriched_a)
        st.session_state.full_lineage['descendants'].extend(enriched_d)
        
        st.write("🧠 AI 正在進行深度推論...")
        if not api_key:
            st.error("❌ 請先設定 API 金鑰")
            return
        
        analysis = generate_deep_analysis_classic(
            st.session_state.full_lineage['hero'],
            st.session_state.full_lineage['ancestors'],
            st.session_state.full_lineage['descendants'],
            api_key,
            model_name
        )
        st.session_state.deep_dive_result = analysis
        status.update(label="✅ 分析完成", state="complete", expanded=False)
        
        time.sleep(0.5)
        st.rerun()

# === 頁籤介面 ===
if st.session_state.read_only_mode:
    st.warning("⚠️ 純閱讀模式 (Read-Only)。")
    st.markdown('<span class="source-badge">📄 Archived Report</span>', unsafe_allow_html=True)
    with st.container():
        if st.session_state.deep_dive_result:
            st.markdown(f'<div class="report-container">{st.session_state.deep_dive_result}</div>', unsafe_allow_html=True)
        else:
            st.info("沒有可顯示的報告")

else:
    tab_insight, tab_broad = st.tabs(["🕵️‍♀️ 深度挖掘 (Deep Dive)", "🔭 廣度搜尋 (Broad Search)"])

    with tab_insight:
        c1, c2 = st.columns([3, 1])
        with c1:
            doi_input = st.text_input(
                "輸入 DOI 或 網址",
                value=st.session_state.pre_fill_doi,
                key="deep_input",
                help="可輸入 DOI、arXiv ID 或論文標題"
            )
        with c2:
            st.write("")
            st.write("")
            btn_analyze = st.button("🔍 執行深掘", use_container_width=True)

        if st.session_state.skeleton:
            st.divider()
            st.caption("🔄 擴展搜尋範圍")
            cb1, cb2, cb3 = st.columns([1, 1, 1])
            btn_older = cb1.button("⬅️ 找更早祖先", use_container_width=True)
            btn_both = cb2.button("↔️ 雙向同時擴展", use_container_width=True)
            btn_newer = cb3.button("找更新後代 ➡️", use_container_width=True)
            
            if btn_older:
                process_mining(doi_input, 'older')
            if btn_newer:
                process_mining(doi_input, 'newer')
            if btn_both:
                process_mining(doi_input, 'expand_both')

        if btn_analyze and doi_input:
            if not api_key:
                st.error("❌ 請先在側邊欄設定 API 金鑰")
            else:
                process_mining(doi_input, 'init')

        if st.session_state.get('deep_dive_result'):
            st.markdown('<span class="source-badge">✅ V10.2 Logic Report</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="report-container">{st.session_state.deep_dive_result}</div>', unsafe_allow_html=True)
            
            st.divider()
            st.markdown("### 🕵️‍♂️ PI 深度偵探 (Identity Verification)")
            st.caption("同名同姓是資料庫常見錯誤。請在下方 **「驗明正身」**，剔除不屬於該作者的論文。")
            
            all_papers = (
                st.session_state.full_lineage['ancestors'] +
                [st.session_state.full_lineage['hero']] +
                st.session_state.full_lineage['descendants']
            )
            pi_options = {}
            
            for p in all_papers:
                auths = p.get('authors', [])
                if not auths:
                    continue
                safe_title = (p.get('title', 'Unknown')[:20] + "...")
                
                def add_opt(a_obj: Dict[str, Any], role: str) -> None:
                    """添加作者選項"""
                    if a_obj.get('authorId'):
                        lbl = f"[{role}] {a_obj.get('name')} (from {safe_title})"
                        if lbl not in pi_options:
                            pi_options[lbl] = a_obj['authorId']

                add_opt(auths[0], "第一作者")
                if len(auths) > 1:
                    add_opt(auths[-1], "最後作者")
                if len(auths) >= 3:
                    add_opt(auths[-2], "倒數第二")
                if len(auths) >= 4:
                    add_opt(auths[-3], "倒數第三")
            
            if pi_options:
                col_pi_sel, col_pi_btn = st.columns([3, 1])
                with col_pi_sel:
                    selected_pi_label = st.selectbox(
                        "1️⃣ 選擇要分析的作者",
                        options=list(pi_options.keys()),
                        help="選擇要分析的作者"
                    )
                
                if st.button("2️⃣ 載入論文列表 (驗明正身)", use_container_width=True) and selected_pi_label:
                    target_author_id = pi_options[selected_pi_label]
                    with st.spinner("正在調閱學術檔案..."):
                        raw_data = fetch_author_profile_no_cache(target_author_id)
                        st.session_state.pi_raw_data = raw_data
                        st.session_state.pi_analysis_result = None

                if st.session_state.pi_raw_data:
                    author_name = st.session_state.pi_raw_data.get('name', 'Unknown')
                    raw_papers = st.session_state.pi_raw_data.get('papers', [])
                    
                    st.markdown(f"**{author_name}** 的高引用論文列表 (共 {len(raw_papers)} 篇)：")
                    st.info("💡 請勾選 **「真正屬於這位作者」** 的論文。若看到領域不符的（如同名同姓），請取消勾選。")
                    
                    if raw_papers:
                        df_papers = pd.DataFrame(raw_papers)
                        df_papers['Select'] = True
                        cols = ['Select', 'title', 'year', 'venue', 'citationCount']
                        valid_cols = [c for c in cols if c in df_papers.columns or c == 'Select']
                        df_papers = df_papers[valid_cols]
                        
                        edited_df = st.data_editor(
                            df_papers,
                            column_config={
                                "Select": st.column_config.CheckboxColumn("納入分析", help="勾選以納入 AI 分析", default=True),
                                "title": "論文標題",
                                "year": "年份",
                                "venue": "期刊/會議",
                                "citationCount": "引用數"
                            },
                            disabled=["title", "year", "venue", "citationCount"],
                            hide_index=True,
                            width='stretch'
                        )
                        
                        selected_rows = edited_df[edited_df['Select'] == True]
                        count_sel = len(selected_rows)
                        
                        if st.button(f"3️⃣ 確認 ({count_sel} 篇) 並執行 AI 分析", type="primary", use_container_width=True) and api_key:
                            if count_sel == 0:
                                st.error("請至少選擇一篇論文！")
                            else:
                                selected_paper_list = selected_rows.to_dict('records')
                                with st.spinner(f"AI 正在閱讀這 {count_sel} 篇論文並分析風格..."):
                                    pi_report = generate_author_analysis(author_name, selected_paper_list, api_key, model_name)
                                    st.session_state.pi_analysis_result = pi_report
                    else:
                        st.warning("此作者沒有找到相關論文資料。")
            else:
                st.info("暫無可分析的作者資料")

            if st.session_state.pi_analysis_result:
                st.markdown('<div class="pi-box">', unsafe_allow_html=True)
                st.markdown(st.session_state.pi_analysis_result)
                st.markdown('</div>', unsafe_allow_html=True)

            st.divider()
            st.subheader("⏳ 技術與作者演進表")
            table_data = []
            
            def get_auth_display(p: Dict[str, Any]) -> str:
                """獲取作者顯示字串"""
                auths = p.get('authors', [])
                if not auths:
                    return "Unknown"
                if len(auths) == 1:
                    return auths[0].get('name', 'Unknown')
                if len(auths) == 2:
                    return f"{auths[0].get('name', 'Unknown')} & {auths[1].get('name', 'Unknown')}"
                first = auths[0].get('name', 'Unknown')
                last = auths[-1].get('name', 'Unknown')
                last_2 = auths[-2].get('name', 'Unknown')
                return f"{first} ... {last_2}, {last}"

            for p in all_papers:
                role = (
                    "🟨 主角" if p == st.session_state.full_lineage['hero']
                    else ("🟦 基石" if p in st.session_state.full_lineage['ancestors'] else "🟩 後續")
                )
                tldr_text = (p.get('tldr') or {}).get('text')
                abs_text = p.get('abstract')
                smry = (tldr_text or abs_text or "")[:100]
                table_data.append({
                    "角色": role,
                    "代號": p.get('code', ''),
                    "年份": p.get('year'),
                    "關鍵作者群": get_auth_display(p),
                    "標題": p.get('title', 'Unknown'),
                    "摘要重點": smry
                })
            
            if table_data:
                st.dataframe(pd.DataFrame(table_data), width='stretch', hide_index=True)
            else:
                st.info("暫無資料可顯示")

            st.subheader("💬 追問歷史學家")
            user_q = st.text_input("有疑問嗎？", key="chat_input", help="輸入問題，AI 會基於當前論文資料回答")
            if st.button("送出") and user_q and api_key:
                with st.spinner("AI 思考中..."):
                    ctx = [
                        {
                            "code": p.get('code', 'Hero'),
                            "title": p.get('title', 'Unknown'),
                            "year": p.get('year', 'N/A')
                        }
                        for p in all_papers
                    ]
                    ans = ask_historian(user_q, ctx, api_key, model_name)
                    st.session_state.chat_history.append({"q": user_q, "a": ans})
            
            for chat in reversed(st.session_state.chat_history):
                st.markdown(f"<div class='chat-box'><b>Q: {chat['q']}</b><br>A: {chat['a']}</div>", unsafe_allow_html=True)

            st.markdown("#### 📚 完整文獻詳情")
            st.markdown('<div class="bib-container">', unsafe_allow_html=True)
            for p in all_papers:
                auth_html = ""
                auths = p.get('authors', [])
                if auths:
                    auth_html += f"<span class='auth-tag-first'>{auths[0].get('name', 'Unknown')} (1st)</span>"
                    if len(auths) > 1:
                        if len(auths) > 3:
                            auth_html += ", ... "
                            auth_html += f", {auths[-2].get('name', 'Unknown')} (2nd Last)"
                        auth_html += f", <span class='auth-tag-last'>{auths[-1].get('name', 'Unknown')} (Last)</span>"
                else:
                    auth_html = "Unknown"
                
                st.markdown(
                    f"**[{p.get('code', 'Hero')}]** {p.get('title', 'Unknown')} ({p.get('year', 'N/A')})<br>"
                    f"🏛️ {p.get('venue', 'N/A')} | 🔗 Cited: {p.get('citationCount', 0)}<br>"
                    f"👤 {auth_html}",
                    unsafe_allow_html=True
                )
                st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("🌍 生成中/英/日 總結卡"):
                if not api_key:
                    st.error("❌ 請先設定 API 金鑰")
                else:
                    with st.spinner("翻譯中..."):
                        summary = generate_multilingual_abstract(st.session_state.deep_dive_result, api_key, model_name)
                        st.info("多語言摘要卡")
                        st.markdown(summary)

    with tab_broad:
        st.markdown("### 🔭 技術關鍵字搜尋")
        broad_query = st.text_input("輸入關鍵字", key="broad_input", help="輸入關鍵字搜尋相關論文")
        limit = st.slider("搜尋數量", 5, 20, 10, help="選擇要返回的論文數量")
        
        if st.button("🚀 搜尋", key="btn_broad"):
            if not validate_input(broad_query):
                st.warning("⚠️ 請輸入有效的搜尋關鍵字")
            else:
                with st.spinner("搜尋 Semantic Scholar 資料庫..."):
                    results = search_broad_papers(broad_query, limit)
                    if results:
                        json_str = json.dumps(results, indent=2, ensure_ascii=False)
                        st.download_button(
                            "📥 下載搜尋結果列表 (JSON)",
                            json_str,
                            "broad_search_results.json",
                            "application/json",
                            help="下載搜尋結果為 JSON 格式"
                        )
                        
                        st.success(f"找到 {len(results)} 篇相關論文")
                        for p in results:
                            with st.container():
                                t_text = (p.get('tldr') or {}).get('text')
                                a_text = p.get('abstract')
                                s_text = (t_text or a_text or "")[:200]
                                st.markdown(f"""
                                <div class="search-card">
                                    <div class="sc-title">{p.get('title', 'Unknown')}</div>
                                    <div style="font-size:0.9em; color:#616161; margin:5px 0;">
                                        📅 {p.get('year', 'N/A')} | 🏛️ {p.get('venue','N/A')} | 🔗 Cited: {p.get('citationCount', 0)}
                                    </div>
                                    <div style="font-size:0.95em; color:#424242;">{s_text}...</div>
                                </div>
                                """, unsafe_allow_html=True)
                                if st.button(f"📥 深度分析 (ID: {p['paperId']})", key=f"btn_{p['paperId']}"):
                                    st.session_state.pre_fill_doi = p['paperId']
                                    st.info(f"已選定論文 ID: {p['paperId']}，請切換至「深度洞察」頁籤並點擊執行。")
                                    st.rerun()
                    else:
                        st.warning("找不到相關論文。請嘗試其他關鍵字。")
