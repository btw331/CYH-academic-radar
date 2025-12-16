# ==========================================
# 0. 優先執行：警告屏蔽與套件設定
# ==========================================
import warnings
import os
import json
warnings.filterwarnings("ignore")
os.environ["on_bad_lines"] = "skip"

import streamlit as st
import re
import pandas as pd
import time
import requests
import concurrent.futures
import random
from urllib.parse import urlparse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from tavily import TavilyClient

# ==========================================
# 1. 基礎設定與 CSS樣式
# ==========================================
st.set_page_config(page_title="全域觀點解析 V34.3", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stButton button[kind="secondary"] { border: 2px solid #673ab7; color: #673ab7; font-weight: bold; }
    
    .report-paper {
        background-color: #fdfbf7; 
        color: #2c3e50; 
        padding: 40px; 
        border-radius: 4px; 
        margin-bottom: 15px; 
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        font-family: "Microsoft JhengHei", "Georgia", serif;
        line-height: 1.8;
        font-size: 1.05rem;
    }
    
    /* [V34.3] 引用樣式優化：灰底小字，降低干擾 */
    .citation {
        font-size: 0.75em;          /* 字體縮小 */
        color: #777777;             /* 灰色文字 */
        background-color: #f4f4f4;  /* 淡灰背景 */
        padding: 2px 6px;           /* 內距 */
        border-radius: 4px;         /* 圓角 */
        margin: 0 4px;              /* 左右留白 */
        font-family: sans-serif; 
        border: 1px solid #e0e0e0;  /* 極淡邊框 */
        font-weight: 400;           /* 不加粗 */
        vertical-align: 1px;        /* 微調垂直對齊 */
        display: inline-block;      /* 確保整塊顯示 */
    }

    /* 關鍵時序卷軸表格 (HTML Style) */
    .scrollable-table-container {
        height: 600px; 
        overflow-y: auto; 
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background-color: white;
        margin-bottom: 20px;
    }
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-family: "Microsoft JhengHei", sans-serif;
        font-size: 0.95em;
    }
    .custom-table th {
        position: sticky;
        top: 0;
        background-color: #f1f3f4;
        color: #333;
        font-weight: bold;
        padding: 12px 15px;
        text-align: left;
        border-bottom: 2px solid #ddd;
        z-index: 2;
    }
    .custom-table td {
        padding: 10px 15px;
        border-bottom: 1px solid #f0f0f0;
        vertical-align: middle;
        color: #333;
    }
    .custom-table tr:hover {
        background-color: #f8f9fa;
    }
    .custom-table a {
        color: #1a73e8;
        text-decoration: none;
        font-weight: 500;
        font-size: 1.05em;
    }
    .custom-table a:hover {
        text-decoration: underline;
        color: #1557b0;
    }
    
    .methodology-text {
        font-size: 0.9em;
        line-height: 1.6;
        color: #444;
    }
    .methodology-header {
        font-weight: bold;
        color: #1a237e;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 資料庫與共用常數 (Strict Domain Lists)
# ==========================================
TAIWAN_WHITELIST = [
    "udn.com", "ltn.com.tw", "chinatimes.com", "cna.com.tw", 
    "storm.mg", "setn.com", "ettoday.net", "tvbs.com.tw", 
    "mirrormedia.mg", "thenewslens.com", "upmedia.mg", 
    "rwnews.tw", "news.pts.org.tw", "ctee.com.tw", "businessweekly.com.tw",
    "news.yahoo.com.tw", "ftvnews.com.tw", "newtalk.tw", "nownews.com", "mygopen.com"
]

INDIE_WHITELIST = [
    "twreporter.org", "theinitium.com", "thenewslens.com", 
    "mindiworldnews.com", "vocus.cc", "matters.town", 
    "plainlaw.me", "whogovernstw.org", "rightplus.org", 
    "biosmonthly.com", "storystudio.tw", "womany.net", "dq.yam.com"
]

INTL_WHITELIST = [
    "bbc.com", "cnn.com", "reuters.com", "apnews.com", "bloomberg.com", 
    "wsj.com", "nytimes.com", "dw.com", "voanews.com", "nikkei.com", "nhk.or.jp", "rfi.fr"
]

# 分類對照表 (用於前端顯示 Emoji)
DB_MAP = {
    "CHINA": ["xinhuanet", "people.com.cn", "huanqiu", "cctv", "chinadaily", "taiwan.cn", "gwytb", "guancha"],
    "GREEN": ["ltn", "ftv", "setn", "rti.org", "newtalk", "mirrormedia", "dpp.org", "libertytimes"],
    "BLUE": ["udn", "chinatimes", "tvbs", "cti", "nownews", "ctee", "kmt.org", "uniteddaily"],
    "OFFICIAL": ["cna.com", "pts.org", "mnd.gov", "mac.gov", "tfc-taiwan", "gov.tw"],
    "INDIE": ["twreporter", "theinitium", "thenewslens", "upmedia", "storm.mg", "mindiworld", "vocus", "matters", "plainlaw"],
    "INTL": ["bbc", "cnn", "reuters", "apnews", "bloomberg", "wsj", "nytimes", "dw.com", "voanews", "rfi.fr"],
    "FARM": ["kknews", "read01", "ppfocus", "buzzhand", "bomb01", "qiqi", "inf.news", "toutiao"]
}

# 雜訊黑名單
NOISE_BLACKLIST = [
    "zhihu.com", "baidu.com", "pinterest.com", "instagram.com", 
    "facebook.com", "tiktok.com", "youtube.com", "dcard.tw", "ptt.cc"
]

def get_domain_name(url):
    try: return urlparse(url).netloc.replace("www.", "")
    except: return ""

def classify_source(url):
    if not url or url == "#": return "OTHER"
    try:
        domain = urlparse(url).netloc.lower()
        clean_domain = domain.replace("www.", "")
    except: return "OTHER"

    for cat, keywords in DB_MAP.items():
        for kw in keywords:
            if kw in domain:
                return cat
    return "OTHER"

def get_category_meta(cat):
    meta = {
        "CHINA": ("🇨🇳 中國官媒", "#d32f2f"),
        "FARM": ("⛔ 內容農場", "#ef6c00"),
        "BLUE": ("🔵 泛藍觀點", "#1565c0"),
        "GREEN": ("🟢 泛綠觀點", "#2e7d32"),
        "OFFICIAL": ("⚪ 官方/中立", "#546e7a"),
        "INDIE": ("🕵️ 獨立/深度", "#fbc02d"),
        "INTL": ("🌏 國際媒體", "#f57c00"),
        "VIDEO": ("🟣 影音社群", "#7b1fa2"),
        "OTHER": ("📄 其他來源", "#9e9e9e")
    }
    return meta.get(cat, ("📄 其他來源", "#9e9e9e"))

# [V34.3 Fix] 萬能引用格式化函式
def format_citation_style(text):
    if not text: return ""
    
    def replacement(match):
        # 提取括號內的所有數字
        nums = re.findall(r'\d+', match.group(0))
        if not nums: return match.group(0)
        # 去重並排序
        unique_nums = sorted(list(set(nums)), key=int)
        # 返回 HTML 格式
        return f'<span class="citation">Source {", ".join(unique_nums)}</span>'

    # 1. 捕捉連續單一引用: [Source 1][Source 2]
    text = re.sub(r'(\[Source \d+\](?:[,;]?\s*\[Source \d+\])*)', replacement, text)
    
    # 2. 捕捉合併引用 (含全形/半形括號): (Source 1, 2), （Source 1, 2）, [Source 1, 2]
    # 正則解釋: [\[\(（] 匹配任意左括號, \s*Source\s+ 匹配 Source 字樣, [\d,，、\s]+ 匹配數字與分隔符
    text = re.sub(r'([\[\(（]\s*Source\s+[\d,，、\s]+[\]\)）])', replacement, text)
    
    return text

# 網址日期提取器
def extract_date_from_url(url):
    if not url: return None
    patterns = [
        r'/(\d{4})[-/](\d{2})[-/](\d{2})/',
        r'/(\d{4})(\d{2})(\d{2})/',
        r'-(\d{4})(\d{2})(\d{2})'
    ]
    for p in patterns:
        match = re.search(p, url)
        if match:
            y, m, d = match.groups()
            return f"{y}-{m}-{d}"
    return None

def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

# ==========================================
# 3. 核心功能模組
# ==========================================

def search_cofacts(query):
    url = "https://cofacts-api.g0v.tw/graphql"
    graphql_query = """
    query ListArticles($text: String!) {
      ListArticles(filter: {q: $text}, orderBy: [{_score: DESC}], first: 3) {
        edges { node { text articleReplies(status: NORMAL) { reply { text type } } } }
      }
    }
    """
    try:
        response = requests.post(url, json={'query': graphql_query, 'variables': {'text': query}}, timeout=3)
        if response.status_code == 200:
            data = response.json()
            articles = data.get('data', {}).get('ListArticles', {}).get('edges', [])
            result_text = ""
            if articles:
                result_text += "【Cofacts 查核資料庫】\n"
                for i, art in enumerate(articles):
                    node = art.get('node', {})
                    rumor = node.get('text', '')[:50]
                    replies = node.get('articleReplies', [])
                    if replies:
                        r_type = replies[0].get('reply', {}).get('type')
                        result_text += f"- 謠言: {rumor}... (判定: {r_type})\n"
            return result_text
    except: return ""
    return ""

# 三軌平行搜尋
def execute_tri_track_search(query, api_key_tavily, search_params, is_strict_mode):
    if search_params['max_results'] <= 20 and not is_strict_mode:
        tavily = TavilyClient(api_key=api_key_tavily)
        return tavily.search(query=query, **search_params).get('results', [])

    queries = [
        f"{query} 新聞 事件 時間軸", 
        f"{query} 評論 觀點 爭議 分析", 
        f"{query} 懶人包 重點 影響"
    ]
    
    sub_params = search_params.copy()
    sub_params['max_results'] = 20 
    
    all_results = []
    seen_urls = set()
    
    def fetch(q):
        try:
            t = TavilyClient(api_key=api_key_tavily)
            return t.search(query=q, **sub_params).get('results', [])
        except: return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fetch, q) for q in queries]
        for future in concurrent.futures.as_completed(futures):
            res_list = future.result()
            for item in res_list:
                url = item.get('url')
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(item)
    
    return all_results

def get_search_context(query, api_key_tavily, days_back, selected_regions, max_results, context_report=None):
    try:
        search_params = {
            "search_depth": "advanced",
            "topic": "general",
            "days": days_back,
            "max_results": max_results,
            "exclude_domains": NOISE_BLACKLIST
        }

        target_domains = []
        is_strict_mode = False
        
        if not isinstance(selected_regions, list): selected_regions = [selected_regions]

        for r in selected_regions:
            if "台灣" in r:
                target_domains.extend(TAIWAN_WHITELIST)
                is_strict_mode = True
            if "獨立" in r:
                target_domains.extend(INDIE_WHITELIST)
                is_strict_mode = True
            if "亞洲" in r or "歐洲" in r or "美洲" in r:
                target_domains.extend(INTL_WHITELIST)
                is_strict_mode = True
        
        if is_strict_mode and target_domains:
            target_domains = list(set(target_domains))
            search_params["include_domains"] = target_domains

        results = execute_tri_track_search(query, api_key_tavily, search_params, is_strict_mode)
        results = results[:max_results]
        
        context_text = ""
        for i, res in enumerate(results):
            title = res.get('title', 'No Title')
            url = res.get('url', '#')
            
            pub_date = res.get('published_date')
            if not pub_date:
                url_date = extract_date_from_url(url)
                if url_date:
                    pub_date = url_date
                else:
                    pub_date = "Missing"
            else:
                pub_date = pub_date[:10]
            
            res['final_date'] = pub_date
            
            content = res.get('content', '')[:3000]
            context_text += f"Source {i+1}: [Date: {pub_date}] [Title: {title}] {content} (URL: {url})\n"
            
        return context_text, results, query, is_strict_mode, len(target_domains)
        
    except Exception as e:
        return f"Error: {str(e)}", [], "Error", False, 0

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5), reraise=True)
def call_gemini(system_prompt, user_text, model_name, api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    chain = prompt | llm
    return chain.invoke({"input": user_text}).content

# 深度戰略分析
def run_strategic_analysis(query, context_text, model_name, api_key, mode="FUSION"):
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    if mode == "FUSION":
        system_prompt = f"""
        你是一位極度嚴謹的社會科學研究員。
        
        【⚠️ 時間錨點】：今天是：{today_str}。
        
        【⚠️ 數據結構指令 (重要)】：
        在產生 [DATA_TIMELINE] 時，**不需要** 輸出完整的網址 (URL)。
        請輸出 **來源編號 (Source ID)**，格式為 `Source X` 的數字 `X`。
        
        【分析方法論】：
        1. **資訊檢索**：閱讀大量文本，識別資訊飽和度。
        2. **框架分析**：依據 Entman (1993) 理論，解構不同陣營的敘事框架。
        3. **三角驗證**：交叉比對官方說法、媒體報導與第三方查核。
        
        【輸出格式 (嚴格遵守)】：
        ### [DATA_TIMELINE]
        (格式：YYYY-MM-DD|媒體|標題|Source_ID)
        -> Source_ID 請填寫整數 (例如 1, 5, 20)。
        -> 日期規則：若標示 [Date: Missing]，請嘗試從內文推算；若無法推算，請填「近期」。
        
        ### [REPORT_TEXT]
        (Markdown 報告 - 繁體中文)
        請包含以下章節：
        1. **📊 全域現況摘要 (Situational Analysis)**
        2. **🔍 爭議點事實查核 (Fact-Check)**
        3. **⚖️ 媒體框架光譜分析 (Framing Analysis)**
        4. **🧠 深度識讀與利益分析 (Cui Bono)**
        5. **🤔 結構性反思 (Critical Reflection)**
        """
        
    elif mode == "DEEP_SCENARIO":
        system_prompt = f"""
        你是一位專精於未來學 (Futures Studies) 的戰略顧問。
        【⚠️ 時間錨點】：今天是 {today_str}。
        【⚠️ 最高指令】：使用繁體中文。
        
        【分析方法論】：
        1. **CLA 層次分析**：表象 -> 系統 -> 世界觀 -> 神話。
        2. **可能性圓錐**：推演三種情境。

        【輸出格式】：
        ### [DATA_TIMELINE]
        (留空)
        
        ### [REPORT_TEXT]
        (Markdown 報告 - 繁體中文)
        1. **🎯 CLA 深度解構 (Causal Layered Analysis)**
           - Litany (表象)
           - System (系統)
           - Worldview (世界觀)
           - Myth (神話)
        2. **🔮 未來情境模擬 (Scenario Planning)**
           - 基準 / 轉折 / 極端情境
        3. **💡 綜合戰略建議**
        """
    else:
        system_prompt = f"請針對 {query} 進行分析。"

    return call_gemini(system_prompt, context_text, model_name, api_key)

def parse_gemini_data(text):
    data = {"timeline": [], "report_text": ""}
    
    if not text: return data

    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        
        if "|" in line and len(line.split("|")) >= 3 and (line[0].isdigit() or "20" in line or "Future" in line or "近期" in line):
            parts = line.split("|")
            try:
                date = parts[0].strip()
                name = parts[1].strip()
                title = parts[2].strip()
                source_id_str = "0"
                
                if len(parts) >= 4: 
                    raw_id = parts[3].strip()
                    nums = re.findall(r'\d+', raw_id)
                    if nums: source_id_str = nums[0]
                
                if "XX" in date or "xx" in date: date = "近期"
                
                data["timeline"].append({
                    "date": date,
                    "media": name,
                    "title": title,
                    "source_id": int(source_id_str)
                })
            except: pass

    if "### [REPORT_TEXT]" in text:
        data["report_text"] = text.split("### [REPORT_TEXT]")[1].strip()
    elif "### REPORT_TEXT" in text:
        data["report_text"] = text.split("### REPORT_TEXT")[1].strip()
    else:
        match = re.search(r"(#+\s*.*摘要|1\.\s*.*摘要|#+\s*.*CLA)", text)
        if match:
            data["report_text"] = text[match.start():]
        else:
            data["report_text"] = text

    return data

def render_html_timeline(timeline_data, sources, blind_mode):
    if not timeline_data:
        return

    table_rows = ""
    for item in timeline_data:
        date = item.get('date', '近期')
        media = "*****" if blind_mode else item.get('media', 'Unknown')
        title = item.get('title', 'No Title')
        
        s_id = item.get('source_id', 0)
        real_url = "#"
        if 0 < s_id <= len(sources):
            real_url = sources[s_id-1].get('url', '#')
            if (date == "近期" or "Missing" in date) and 'final_date' in sources[s_id-1]:
                final_d = sources[s_id-1]['final_date']
                if final_d and final_d != "Missing":
                    date = final_d
        
        cat = classify_source(real_url)
        label, _ = get_category_meta(cat)
        emoji = "⚪"
        if "中國" in label: emoji = "🔴"
        elif "泛藍" in label: emoji = "🔵"
        elif "泛綠" in label: emoji = "🟢"
        elif "官方" in label: emoji = "⚪"
        elif "獨立" in label: emoji = "🕵️"
        elif "國際" in label: emoji = "🌏"
        elif "農場" in label: emoji = "⛔"
        
        if real_url and real_url != "#":
            title_html = f'<a href="{real_url}" target="_blank">{title}</a>'
        else:
            title_html = title

        media_display = f"{emoji} {media}"
        row_html = f"<tr><td style='white-space:nowrap;'>{date}</td><td style='white-space:nowrap;'>{media_display}</td><td>{title_html}</td></tr>"
        table_rows += row_html

    full_html = f"""
    <div class="scrollable-table-container">
    <table class="custom-table">
    <thead>
    <tr>
    <th style="width:120px;">日期</th>
    <th style="width:140px;">媒體 (URL分類)</th>
    <th>新聞標題 (點擊閱讀)</th>
    </tr>
    </thead>
    <tbody>
    {table_rows}
    </tbody>
    </table>
    </div>
    """
    
    st.markdown("### 📅 關鍵發展時序")
    st.markdown(full_html, unsafe_allow_html=True)

# 4. 下載功能
def convert_data_to_json(data):
    import json
    return json.dumps(data, indent=2, ensure_ascii=False)

def convert_data_to_md(data):
    return f"""
# 全域觀點分析報告 (Academic Standard)
产生時間: {datetime.now()}

## 1. 深度分析
{data.get('report_text')}

## 2. 時間軸
{pd.DataFrame(data.get('timeline')).to_markdown(index=False)}
    """

# ==========================================
# 5. UI
# ==========================================
with st.sidebar:
    st.title("全域觀點解析 V34.3")
    
    analysis_mode = st.radio(
        "選擇分析引擎：",
        options=["全域深度解析 (Fusion)", "未來發展推演 (Scenario)"],
        captions=["學術框架：框架分析 + 三角驗證", "學術框架：CLA 層次分析 + 未來學"],
        index=0
    )
    st.markdown("---")
    
    blind_mode = st.toggle("🙈 盲測模式 (隱藏媒體名稱)", value=False)
    
    with st.expander("🔑 API 設定", expanded=True):
        if "GOOGLE_API_KEY" in st.secrets:
            st.success("✅ Gemini Key Ready")
            google_key = st.secrets["GOOGLE_API_KEY"]
        else:
            google_key = st.text_input("Gemini Key", type="password")

        if "TAVILY_API_KEY" in st.secrets:
            st.success("✅ Tavily Ready")
            tavily_key = st.secrets["TAVILY_API_KEY"]
        else:
            tavily_key = st.text_input("Tavily Key", type="password")
            
        model_name = st.selectbox(
            "模型 (Gemini 2.5 Series)", 
            ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"], 
            index=0,
            help="建議使用 Pro 版以獲得最佳的邏輯推演與指令遵循能力。"
        )
        
        search_days = st.number_input(
            "搜尋時間範圍 (天數)",
            min_value=1,
            max_value=1825,
            value=30,
            step=1,
            help="請輸入欲搜尋的過去天數，上限為 1825 天 (5年)。"
        )
        
        max_results = st.slider("搜尋篇數上限 (Sample Size)", 10, 100, 30, help="增加篇數可避免小樣本偏誤，但會增加分析時間。")
        
        selected_regions = st.multiselect(
            "搜尋視角 (Region) - 可複選",
            ["🇹🇼 台灣 (Taiwan)", "🌏 亞洲 (Asia)", "🌍 歐洲 (Europe)", "🌎 美洲 (Americas)", "🕵️ 獨立/自媒體 (Indie)"],
            default=["🇹🇼 台灣 (Taiwan)"]
        )

    with st.expander("🧠 學術分析方法論 (Research Methodology)", expanded=True):
        st.markdown("""
        <div class="methodology-text">
        <div class="methodology-header">1. 資訊檢索與樣本檢定 (Information Retrieval & Sampling)</div>
        本系統採用 <b>開源情報 (OSINT)</b> 標準進行資料探勘。
        <ul>
            <li><b>三軌平行搜尋</b>：同時針對「事實/時序」、「觀點/爭議」、「深度/懶人包」三條軌道進行搜尋，確保資訊完整性。</li>
            <li><b>網域圍籬</b>：嚴格執行白名單機制，確保資訊來源可靠。</li>
            <li><b>智慧日期提取</b>：結合 API 元數據、URL 規則與 AI 內文推斷，最大化還原事件時間。</li>
        </ul>

        <div class="methodology-header">2. 框架分析與立場判定 (Framing & Stance)</div>
        本研究採用 <b>Entman (1993) 的框架理論 (Framing Theory)</b> 與 <b>批判話語分析 (CDA)</b>。
        <ul>
            <li><b>語意層次</b>：分析文本中的修辭 (Rhetoric)、隱喻 (Metaphor) 與標籤化 (Labeling) 策略。</li>
            <li><b>機構層次</b>：結合媒體所有權結構 (Ownership) 與過往政治傾向資料庫，進行雙重驗證 (Triangulation)。</li>
        </ul>

        <div class="methodology-header">3. 可信度與查核 (Verification)</div>
        採用史丹佛大學歷史教育群 (SHEG) 提倡之 <b>水平閱讀法 (Lateral Reading)</b>。
        <ul>
            <li><b>交叉比對</b>：將媒體報導與 <b>Cofacts 謠言查核資料庫</b> 及官方原始文件進行比對。</li>
        </ul>

        <div class="methodology-header">4. 戰略推演模型 (Futures Framework)</div>
        僅應用於「未來發展推演」模式。
        <ul>
            <li><b>第一性原理 (First Principles)</b>：解構議題至最基礎的物理或經濟限制。</li>
            <li><b>層次分析法 (CLA)</b>：由表象 (Litany) 深入至系統結構 (System) 與社會神話 (Myth)。</li>
            <li><b>可能性圓錐 (Cone of Plausibility)</b>：區分基準情境 (Probable)、轉折情境 (Plausible) 與極端情境 (Possible)。</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📚 監測資料庫清單", expanded=False):
        for key, domains in DB_MAP.items():
            label, color = get_category_meta(key)
            st.markdown(f"**{label}**")
            st.markdown(f"`{', '.join(domains[:3])}...`")

    with st.expander("📂 匯入舊情報", expanded=False):
        past_report_input = st.text_area("貼上舊報告 Markdown：", height=100)
        
    st.markdown("### 📥 報告匯出")
    if st.session_state.get('result') or st.session_state.get('scenario_result'):
        export_data = st.session_state.get('result').copy()
        if st.session_state.get('scenario_result'):
            export_data['report_text'] += "\n\n# 未來發展推演報告\n" + st.session_state.get('scenario_result')['report_text']
            
        st.download_button("下載 JSON", convert_data_to_json(export_data), "report.json", "application/json")
        st.download_button("下載 Markdown", convert_data_to_md(export_data), "report.md", "text/markdown")

st.title(f"{analysis_mode.split(' ')[0]}")
query = st.text_input("輸入議題關鍵字", placeholder="例如：台積電美國設廠爭議")
search_btn = st.button("🚀 啟動全域掃描", type="primary")

# 初始化 session state
if 'result' not in st.session_state: st.session_state.result = None
if 'scenario_result' not in st.session_state: st.session_state.scenario_result = None
if 'sources' not in st.session_state: st.session_state.sources = None

if search_btn and query and google_key and tavily_key:
    st.session_state.result = None
    st.session_state.scenario_result = None
    
    with st.status("🚀 啟動全域掃描引擎 (V34.3 樣式修復版)...", expanded=True) as status:
        
        days_label = f"近 {search_days} 天"
        regions_label = ", ".join([r.split(" ")[1] for r in selected_regions])
        
        st.write(f"📡 1. 連線 Tavily 搜尋 (視角: {regions_label} / 時間: {days_label})...")
        st.write(f"   ↳ 目標樣本數: {max_results} 篇 (三軌搜尋 + 網域圍籬)")
        
        context_text, sources, actual_query, is_strict_tw, domain_count = get_search_context(query, tavily_key, search_days, selected_regions, max_results, past_report_input)
        
        if is_strict_tw:
            st.write(f"🛡️ 網域圍籬已啟動：已鎖定監測資料庫內的 {domain_count} 個權威來源。")
        else:
            st.write("⚠️ 未選定區域，執行全網搜尋 (已排除知乎等農場)。")
        
        st.session_state.sources = sources
        
        st.write("🛡️ 2. 查詢 Cofacts 謠言資料庫 (API)...")
        cofacts_txt = search_cofacts(query)
        if cofacts_txt: context_text += f"\n{cofacts_txt}\n"
        
        st.write("🧠 3. AI 進行深度戰略分析 (學術框架應用 + 樣本檢定)...")
        
        # 預設執行 FUSION 模式
        raw_report = run_strategic_analysis(query, context_text, model_name, google_key, mode="FUSION")
        st.session_state.result = parse_gemini_data(raw_report)
            
        status.update(label="✅ 分析完成", state="complete", expanded=False)
        
    st.rerun()

# 顯示區域
if st.session_state.result:
    data = st.session_state.result
    # [V34.1] 傳入 sources 供 ID 映射使用
    render_html_timeline(data.get("timeline"), st.session_state.sources, blind_mode)

    # 2. 顯示第一階段：綜合戰略分析報告
    st.markdown("---")
    st.markdown("### 📝 綜合戰略分析報告")
    formatted_text = format_citation_style(data.get("report_text", ""))
    st.markdown(f'<div class="report-paper">{formatted_text}</div>', unsafe_allow_html=True)
    
    # [V33.4] 資訊滾動按鈕
    if "未來" not in analysis_mode and not st.session_state.scenario_result:
        st.markdown("---")
        if st.button("🚀 將此結果餵給未來發展推演 (資訊滾動)", type="secondary"):
            with st.spinner("🔮 正在讀取前次情報，啟動 CLA 層次分析與未來推演..."):
                current_report = data.get("report_text", "")
                raw_text = run_strategic_analysis(query, current_report, model_name, google_key, mode="DEEP_SCENARIO")
                st.session_state.scenario_result = parse_gemini_data(raw_text) 
                st.rerun()

# [V33.4] 顯示第二階段：未來發展推演報告
if st.session_state.scenario_result:
    st.markdown("---")
    st.markdown("### 🔮 未來發展推演報告")
    scenario_data = st.session_state.scenario_result
    formatted_scenario = format_citation_style(scenario_data.get("report_text", ""))
    st.markdown(f'<div class="report-paper">{formatted_scenario}</div>', unsafe_allow_html=True)

if st.session_state.sources:
    st.markdown("---")
    st.markdown("### 📚 引用文獻列表")
    md_table = "| 編號 | 媒體/網域 | 標題摘要 | 連結 |\n|:---:|:---|:---|:---|\n"
    for i, s in enumerate(st.session_state.sources):
        domain = get_domain_name(s.get('url'))
        if blind_mode: domain = "*****"
        
        title = s.get('title', 'No Title')
        if len(title) > 60: title = title[:60] + "..."
        url = s.get('url')
        md_table += f"| **{i+1}** | `{domain}` | {title} | [點擊]({url}) |\n"
    st.markdown(md_table)
