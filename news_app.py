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
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from tavily import TavilyClient

# ==========================================
# 1. 基礎設定與 CSS樣式
# ==========================================
st.set_page_config(page_title="全域觀點解析 V30.1", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    .stButton button[kind="secondary"] { border: 2px solid #673ab7; color: #673ab7; font-weight: bold; }
    
    .report-paper {
        background-color: #fdfbf7; 
        color: #2c3e50; 
        padding: 30px; 
        border-radius: 4px; 
        margin-bottom: 15px; 
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        font-family: "Georgia", "Cambria", "Times New Roman", serif;
        line-height: 1.8;
        font-size: 1.05rem;
    }
    
    .citation {
        font-size: 0.85em; color: #757575; background-color: #f0f0f0;
        padding: 2px 6px; border-radius: 4px; margin: 0 2px;
        font-family: sans-serif; border: 1px solid #e0e0e0; font-weight: 500;
    }

    /* V30 極簡卷軸表格 */
    .scrollable-table-container {
        height: 500px; 
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
# 2. 資料庫與共用常數
# ==========================================
TAIWAN_WHITELIST = [
    "udn.com", "ltn.com.tw", "chinatimes.com", "cna.com.tw", 
    "storm.mg", "setn.com", "ettoday.net", "tvbs.com.tw", 
    "mirrormedia.mg", "thenewslens.com", "upmedia.mg", 
    "rwnews.tw", "news.pts.org.tw", "ctee.com.tw", "businessweekly.com.tw",
    "news.yahoo.com.tw"
]

INDIE_WHITELIST = [
    "twreporter.org", "theinitium.com", "thenewslens.com", 
    "mindiworldnews.com", "vocus.cc", "matters.town", 
    "plainlaw.me", "whogovernstw.org", "rightplus.org", 
    "biosmonthly.com", "storystudio.tw", "womany.net", "dq.yam.com"
]

DB_MAP = {
    "CHINA": ["xinhuanet.com", "people.com.cn", "huanqiu.com", "cctv.com", "chinadaily.com.cn", "taiwan.cn", "gwytb.gov.cn", "guancha.cn"],
    "GREEN": ["ltn.com.tw", "ftvnews.com.tw", "setn.com", "rti.org.tw", "newtalk.tw", "mirrormedia.mg", "dpp.org.tw"],
    "BLUE": ["udn.com", "chinatimes.com", "tvbs.com.tw", "cti.com.tw", "nownews.com", "ctee.com.tw", "kmt.org.tw"],
    "OFFICIAL": ["cna.com.tw", "pts.org.tw", "mnd.gov.tw", "mac.gov.tw", "tfc-taiwan.org.tw"],
    "INDIE": ["twreporter.org", "theinitium.com", "thenewslens.com", "upmedia.mg", "storm.mg", "mindiworldnews.com", "vocus.cc", "matters.town"],
    "INTL": ["bbc.com", "cnn.com", "reuters.com", "apnews.com", "bloomberg.com", "wsj.com", "nytimes.com", "dw.com", "voanews.com"],
    "FARM": ["kknews.cc", "read01.com", "ppfocus.com", "buzzhand.com", "bomb01.com", "qiqi.news", "inf.news", "toutiao.com"]
}

NAME_KEYWORDS = {
    "CHINA": ["新華", "人民日報", "環球", "央視", "國台辦", "中評", "解放軍", "陸媒", "北京", "宋濤", "xinhuanet", "huanqiu"],
    "GREEN": ["自由", "三立", "民視", "新頭殼", "鏡週刊", "民進黨", "賴清德", "綠營", "獨派", "抗中保台", "ltn", "setn", "ftv"],
    "BLUE": ["聯合", "中國時報", "中時", "TVBS", "中天", "工商時報", "旺旺", "國民黨", "KMT", "侯友宜", "藍營", "統派", "udn", "chinatimes"],
    "FARM": ["網傳", "謠言", "爆料", "內容農場", "PTT", "Dcard", "爆料公社"],
    "OFFICIAL": ["中央社", "公視", "cna", "pts", "gov"],
    "VIDEO": ["YouTube", "YouTuber", "網紅", "TikTok", "抖音", "館長", "直播"]
}

def get_domain_name(url):
    try: return urlparse(url).netloc.replace("www.", "")
    except: return ""

def classify_source(url):
    if not url or url == "#": return "OTHER"
    try:
        domain = urlparse(url).netloc.lower()
        clean_domain = domain.replace("www.", "")
    except: return "OTHER"

    for cat, domains in DB_MAP.items():
        for d in domains:
            if d in clean_domain:
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

def format_citation_style(text):
    if not text: return ""
    def compress_match(match):
        nums = re.findall(r'\d+', match.group(0))
        unique_nums = sorted(list(set(nums)), key=int)
        return f'<span class="citation">Source {",".join(unique_nums)}</span>'
    pattern_compress = r'(\[Source \d+\](?:[,;]?\s*\[Source \d+\])*)'
    text = re.sub(pattern_compress, compress_match, text)
    return text

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

def get_search_context(query, api_key_tavily, days_back, selected_regions, max_results, context_report=None):
    try:
        tavily = TavilyClient(api_key=api_key_tavily)
        
        search_params = {
            "search_depth": "advanced",
            "topic": "general",
            "days": days_back,
            "max_results": max_results
        }

        suffixes = []
        target_domains = [] 
        
        has_taiwan = False
        has_indie = False
        has_intl = False
        
        if not isinstance(selected_regions, list): selected_regions = [selected_regions]

        for r in selected_regions:
            if "台灣" in r: 
                has_taiwan = True
                suffixes.append("台灣 新聞" if is_chinese(query) else "Taiwan News")
                target_domains.extend(TAIWAN_WHITELIST)
            
            if "獨立" in r:
                has_indie = True
                suffixes.append("評論 深度報導") 
                target_domains.extend(INDIE_WHITELIST)
                
            if "亞洲" in r: has_intl = True; suffixes.append("Asia News")
            if "歐洲" in r: has_intl = True; suffixes.append("Europe News")
            if "美洲" in r: has_intl = True; suffixes.append("US Americas News")
        
        if not suffixes: suffixes.append("News")
        
        search_q = f"{query} {' '.join(suffixes)}"
        if context_report: search_q += " analysis"
        
        search_params["query"] = search_q

        if (has_taiwan or has_indie) and not has_intl:
            search_params["include_domains"] = list(set(target_domains))
        else:
            search_params["exclude_domains"] = [
                "daum.net", "naver.com", "tistory.com",
                "espn.com", "bleacherreport.com", "cbssports.com", 
                "pinterest.com", "amazon.com", "tripadvisor.com"
            ]
        
        actual_query = search_params["query"]
        
        response = tavily.search(**search_params)
        results = response.get('results', [])
        context_text = ""
        
        for i, res in enumerate(results):
            title = res.get('title', 'No Title')
            url = res.get('url', '#')
            pub_date = res.get('published_date')
            if pub_date:
                pub_date = pub_date[:10]
            else:
                pub_date = "----" 
            
            content = res.get('content', '')[:1200]
            context_text += f"Source {i+1}: [Date: {pub_date}] [Title: {title}] {content} (URL: {url})\n"
            
        return context_text, results, actual_query, (has_taiwan or has_indie) and not has_intl
        
    except Exception as e:
        return f"Error: {str(e)}", [], "Error", False

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5), reraise=True)
def call_gemini(system_prompt, user_text, model_name, api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    chain = prompt | llm
    return chain.invoke({"input": user_text}).content

# 深度戰略分析
def run_strategic_analysis(query, context_text, model_name, api_key, mode="FUSION"):
    if mode == "FUSION":
        system_prompt = f"""
        你是一位社會科學研究員與情報分析師。請針對「{query}」進行【全域深度解析】，並嚴格遵循學術方法論。
        
        【方法論框架 (Methodological Framework)】：
        1. **資訊檢索 (Information Retrieval)**：基於提供的 Context 進行證據權重評估。
        2. **框架分析 (Framing Analysis)**：依據 Entman (1993) 理論，分析各方媒體如何「選擇」與「凸顯」特定事實。
        3. **三角驗證 (Triangulation)**：交叉比對官方說法、媒體報導與第三方查核(Cofacts)。
        
        【輸出格式 (嚴格遵守)】：
        ### [DATA_TIMELINE]
        (格式：YYYY-MM-DD|媒體|標題|網址) 
        -> 網址請務必對應 Context 中的 Source Link。
        -> 日期請從 Context [Date:...] 提取，若無則依據內文推斷。
        
        ### [REPORT_TEXT]
        (Markdown 報告 - 請使用 [Source X] 格式引用)
        請包含以下章節：
        1. **📊 全域現況摘要 (Situational Analysis)**：整合主要事實。
        2. **🔍 爭議點事實查核 (Fact-Check Matrix)**：列出爭議陳述與驗證結果。
        3. **⚖️ 媒體框架光譜 (Media Framing Spectrum)**：分析不同陣營的敘事框架差異。
        4. **🧠 深度識讀與利益分析 (Cui Bono)**：分析誰是受益者，誰是受害者。
        5. **🤔 關鍵反思 (Critical Reflection)**：對議題的結構性思考。
        """
        
    else: # SCENARIO (Futures Studies)
        system_prompt = f"""
        你是一位未來學家 (Futurist)。請針對「{query}」應用未來學方法論進行戰略推演。
        
        【方法論框架 (Futures Methodology)】：
        1. **第一性原理 (First Principles)**：回歸議題的最基本事實與驅動力 (Drivers)。
        2. **層次分析法 (Causal Layered Analysis, CLA)**：從表象 (Litany) 深入到系統(System)與世界觀(Worldview)。
        3. **可能性圓錐 (Cone of Plausibility)**：推演三種不同機率的未來路徑。

        【輸出格式】：
        ### [DATA_TIMELINE]
        (格式：YYYY-MM-DD|媒體|標題|網址)
        
        ### [REPORT_TEXT]
        (Markdown 報告)
        1. **🎯 第一性原理拆解 (First Principles Decomposition)**
           - 核心驅動力分析
        2. **🔮 未來情境模擬 (Scenario Planning)**
           - 基準情境 (Baseline): 延續現狀 (Business as Usual)
           - 轉折情境 (Alternative): 關鍵變數改變
           - 極端情境 (Wild Card): 黑天鵝事件
        3. **💡 綜合戰略建議 (Strategic Recommendations)**
        """

    return call_gemini(system_prompt, context_text, model_name, api_key)

def parse_gemini_data(text):
    data = {"timeline": [], "report_text": ""}
    
    if not text: return data

    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        
        if "|" in line and len(line.split("|")) >= 3 and (line[0].isdigit() or "20" in line or "Future" in line):
            parts = line.split("|")
            try:
                date = parts[0].strip()
                name = parts[1].strip()
                title = parts[2].strip()
                url = "#"
                
                if len(parts) >= 6: url = parts[5].strip()
                elif len(parts) >= 4: url = parts[3].strip()
                
                url = url.rstrip(")").rstrip("]").strip()
                
                data["timeline"].append({
                    "date": date,
                    "media": name,
                    "title": title,
                    "url": url
                })
            except: pass

    if "### [REPORT_TEXT]" in text:
        data["report_text"] = text.split("### [REPORT_TEXT]")[1].strip()
    elif "### REPORT_TEXT" in text:
        data["report_text"] = text.split("### REPORT_TEXT")[1].strip()
    else:
        match = re.search(r"(#+\s*.*摘要|1\.\s*.*摘要|#+\s*.*第一性原理)", text)
        if match:
            data["report_text"] = text[match.start():]
        else:
            data["report_text"] = text

    return data

# 渲染 HTML 表格
def render_html_timeline(timeline_data, blind_mode):
    if not timeline_data:
        st.info("無時間軸資料。")
        return

    table_rows = ""
    for item in timeline_data:
        date = item.get('date', 'Unknown')
        media = "*****" if blind_mode else item.get('media', 'Unknown')
        title = item.get('title', 'No Title')
        url = item.get('url', '#')
        
        cat = classify_source(url)
        label, _ = get_category_meta(cat)
        emoji = "⚪"
        if "中國" in label: emoji = "🔴"
        elif "泛藍" in label: emoji = "🔵"
        elif "泛綠" in label: emoji = "🟢"
        elif "官方" in label: emoji = "⚪"
        elif "獨立" in label: emoji = "🕵️"
        elif "國際" in label: emoji = "🌏"
        elif "農場" in label: emoji = "⛔"
        
        if url and url != "#":
            title_html = f'<a href="{url}" target="_blank">{title}</a>'
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
    st.title("全域觀點解析 V30.1")
    
    analysis_mode = st.radio(
        "選擇分析引擎：",
        options=["全域深度解析 (Fusion)", "未來發展推演 (Scenario)"],
        captions=["學術框架：框架分析 + 三角驗證", "學術框架：第一性原理 + CLA + 未來學"],
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
            
        model_name = st.selectbox("模型", ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"], index=0)
        
        search_days = st.selectbox(
            "搜尋時間範圍",
            options=[3, 7, 14, 30, 90, 1825],
            format_func=lambda x: "📅 不限時間 (All Time)" if x == 1825 else f"近 {x} 天",
            index=2
        )
        
        max_results = st.slider("搜尋篇數上限", 10, 60, 20)
        
        selected_regions = st.multiselect(
            "搜尋視角 (Region) - 可複選",
            ["🇹🇼 台灣 (Taiwan)", "🌏 亞洲 (Asia)", "🌍 歐洲 (Europe)", "🌎 美洲 (Americas)", "🕵️ 獨立/自媒體 (Indie)"],
            default=["🇹🇼 台灣 (Taiwan)"]
        )

    with st.expander("🧠 學術分析方法論 (Research Methodology)", expanded=True):
        st.markdown("""
        <div class="methodology-text">
        <div class="methodology-header">1. 資訊檢索與篩選 (OSINT Strategy)</div>
        本系統採用 <b>開源情報 (OSINT)</b> 標準進行資料探勘。
        <ul>
            <li><b>搜尋廣度</b>：整合 Tavily API，進行多維度關鍵字排列組合 (Permutations) 搜尋。</li>
            <li><b>來源驗證</b>：採用白名單機制優先鎖定具公信力之主流媒體與獨立媒體，並排除內容農場 (Content Farms)。</li>
            <li><b>時序重構</b>：若 Metadata 缺失，系統會針對內文進行自然語言處理 (NLP) 以推斷確切事件時間。</li>
        </ul>

        <div class="methodology-header">2. 框架分析與立場判定 (Framing & Stance)</div>
        本研究採用 <b>Entman (1993) 的框架理論 (Framing Theory)</b> 與 <b>批判話語分析 (CDA)</b>。
        <ul>
            <li><b>語意層次</b>：分析文本中的修辭 (Rhetoric)、隱喻 (Metaphor) 與標籤化 (Labeling) 策略。</li>
            <li><b>機構層次</b>：結合媒體所有權結構 (Ownership) 與過往政治傾向資料庫，進行雙重驗證 (Triangulation)。</li>
            <li><b>光譜定義</b>：
                <ul><li><b>批判/挑戰 (Critical)</b>：挑戰現狀或執政當局。</li>
                <li><b>體制/護航 (Establishment)</b>：維護現狀或政策辯護。</li></ul>
            </li>
        </ul>

        <div class="methodology-header">3. 可信度與查核 (Verification)</div>
        採用史丹佛大學歷史教育群 (SHEG) 提倡之 <b>水平閱讀法 (Lateral Reading)</b>。
        <ul>
            <li><b>交叉比對</b>：將媒體報導與 <b>Cofacts 謠言查核資料庫</b> 及官方原始文件進行比對。</li>
            <li><b>證據權重</b>：評估消息來源是否具名、數據是否具備統計顯著性。</li>
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
    if st.session_state.get('result') or st.session_state.get('wargame_result'):
        active_data = st.session_state.get('wargame_result') if "Scenario" in analysis_mode else st.session_state.get('result')
        if active_data:
            st.download_button("下載 JSON", convert_data_to_json(active_data), "report.json", "application/json")
            st.download_button("下載 Markdown", convert_data_to_md(active_data), "report.md", "text/markdown")

st.title(f"{analysis_mode.split(' ')[0]}")
query = st.text_input("輸入議題關鍵字", placeholder="例如：台積電美國設廠爭議")
search_btn = st.button("🚀 啟動全域掃描", type="primary")

if 'result' not in st.session_state: st.session_state.result = None
if 'sources' not in st.session_state: st.session_state.sources = None

if search_btn and query and google_key and tavily_key:
    st.session_state.result = None
    
    with st.status("🚀 啟動全域掃描引擎 (V30.1)...", expanded=True) as status:
        
        days_label = "不限時間" if search_days == 1825 else f"近 {search_days} 天"
        regions_label = ", ".join([r.split(" ")[1] for r in selected_regions])
        st.write(f"📡 1. 連線 Tavily 搜尋 (視角: {regions_label} / 時間: {days_label})...")
        
        context_text, sources, actual_query, is_strict_tw = get_search_context(query, tavily_key, search_days, selected_regions, max_results, past_report_input)
        st.session_state.sources = sources
        
        st.write("🛡️ 2. 查詢 Cofacts 謠言資料庫 (API)...")
        cofacts_txt = search_cofacts(query)
        if cofacts_txt: context_text += f"\n{cofacts_txt}\n"
        
        st.write("🧠 3. AI 進行深度戰略分析 (學術框架應用)...")
        
        mode_code = "V205" if "未來" in analysis_mode else "FUSION"
        raw_report = run_strategic_analysis(query, context_text, model_name, google_key, mode=mode_code)
        st.session_state.result = parse_gemini_data(raw_report)
            
        status.update(label="✅ 分析完成", state="complete", expanded=False)
        
    st.rerun()

if st.session_state.result:
    data = st.session_state.result
    
    # 1. 顯示卷軸表格 (V30 極簡版)
    render_html_timeline(data.get("timeline"), blind_mode)

    # 2. 顯示深度報告
    st.markdown("---")
    st.markdown("### 📝 綜合戰略分析報告")
    formatted_text = format_citation_style(data.get("report_text", ""))
    st.markdown(f'<div class="report-paper">{formatted_text}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    # [V30.1] 資訊滾動按鈕實作
    if "未來" not in analysis_mode:
        if st.button("🚀 將此結果餵給未來發展推演 (資訊滾動)", type="secondary"):
            with st.spinner("🔮 正在讀取前次情報，啟動第一性原理推演..."):
                # 將當前報告作為 Context 餵給 Scenario 模式
                current_report = data.get("report_text", "")
                raw_text = run_strategic_analysis(query, current_report, model_name, google_key, mode="V205")
                st.session_state.result = parse_gemini_data(raw_text)
                st.rerun()

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
