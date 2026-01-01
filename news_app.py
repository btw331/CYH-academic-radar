# ==========================================
# 0. 優先執行：警告屏蔽與套件設定
# ==========================================
import warnings
import os
import json
import re
import pandas as pd
import time
import requests
import concurrent.futures
import random
import markdown
from urllib.parse import urlparse
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
from tavily import TavilyClient

warnings.filterwarnings("ignore")
os.environ["on_bad_lines"] = "skip"

# ==========================================
# 1. 基礎設定與 CSS樣式
# ==========================================
st.set_page_config(page_title="全域觀點解析 V38.0", page_icon="⚖️", layout="wide")

CSS_STYLE = """
<style>
    body { font-family: "Microsoft JhengHei", "Georgia", sans-serif; line-height: 1.6; color: #333; }
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
    
    .citation {
        font-size: 0.75em;          
        color: #777777;             
        background-color: #f4f4f4;  
        padding: 2px 6px;           
        border-radius: 4px;         
        margin: 0 4px;              
        font-family: sans-serif; 
        border: 1px solid #e0e0e0;  
        font-weight: 400;           
        vertical-align: 1px;        
        display: inline-block;      
    }

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
    
    @media print {
        .scrollable-table-container { height: auto; overflow: visible; }
        body { font-size: 12pt; }
        a { text-decoration: none; color: #000; }
        .report-paper { box-shadow: none; border: none; padding: 0; }
    }
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# ==========================================
# 2. 資料庫與共用常數 (Config)
# ==========================================
BLUE_WHITELIST = ["udn.com", "chinatimes.com", "tvbs.com.tw", "cti.com.tw", "nownews.com", "ctee.com.tw", "storm.mg"]
GREEN_WHITELIST = ["ltn.com.tw", "ftvnews.com.tw", "setn.com", "rti.org.tw", "newtalk.tw", "mirrormedia.mg", "upmedia.mg"]
OFFICIAL_WHITELIST = ["cna.com.tw", "pts.org.tw", "mnd.gov.tw", "mac.gov.tw", "tfc-taiwan.org.tw", "gov.tw"]
FULL_TAIWAN_WHITELIST = BLUE_WHITELIST + GREEN_WHITELIST + OFFICIAL_WHITELIST + ["yahoo.com.tw", "ettoday.net", "businessweekly.com.tw"]

INDIE_WHITELIST = ["twreporter.org", "theinitium.com", "thenewslens.com", "mindiworldnews.com", "vocus.cc", "matters.town", "plainlaw.me"]
INTL_WHITELIST = ["bbc.com", "cnn.com", "reuters.com", "apnews.com", "bloomberg.com", "wsj.com", "nytimes.com", "dw.com", "voanews.com", "nikkei.com", "nhk.or.jp"]

DOMAIN_NAME_MAP = {
    "udn.com": "聯合報", "chinatimes.com": "中國時報", "tvbs.com.tw": "TVBS", "cti.com.tw": "中天新聞",
    "nownews.com": "NOWnews", "ctee.com.tw": "工商時報", "storm.mg": "風傳媒",
    "ltn.com.tw": "自由時報", "ftvnews.com.tw": "民視新聞", "setn.com": "三立新聞", "rti.org.tw": "央廣",
    "newtalk.tw": "新頭殼", "mirrormedia.mg": "鏡週刊", "upmedia.mg": "上報",
    "cna.com.tw": "中央社", "pts.org.tw": "公視", "twreporter.org": "報導者",
    "theinitium.com": "端傳媒", "thenewslens.com": "關鍵評論網", "mindiworldnews.com": "敏迪選讀",
    "vocus.cc": "方格子", "ptt.cc": "PTT", "dcard.tw": "Dcard",
    "bbc.com": "BBC", "cnn.com": "CNN", "reuters.com": "路透社", "apnews.com": "美聯社",
    "bloomberg.com": "彭博", "wsj.com": "華爾街日報", "nytimes.com": "紐約時報",
    "mobile01.com": "Mobile01", "yahoo.com": "Yahoo新聞", "ettoday.net": "ETtoday",
    "businessweekly.com.tw": "商業周刊", "mygopen.com": "MyGoPen"
}

DB_MAP = {
    "CHINA": ["xinhuanet", "people.com.cn", "huanqiu", "cctv", "chinadaily", "taiwan.cn", "gwytb", "guancha"],
    "GREEN": ["ltn", "ftv", "setn", "rti.org", "newtalk", "mirrormedia", "dpp", "upmedia"],
    "BLUE": ["udn", "chinatimes", "tvbs", "cti", "nownews", "ctee", "kmt", "storm"],
    "OFFICIAL": ["cna.com", "pts.org", "mnd.gov", "mac.gov", "tfc-taiwan", "gov.tw"],
    "INDIE": ["twreporter", "theinitium", "thenewslens", "mindiworld", "vocus", "matters", "plainlaw"],
    "INTL": ["bbc", "cnn", "reuters", "apnews", "bloomberg", "wsj", "nytimes", "dw.com", "voanews", "rfi"],
    "FARM": ["kknews", "read01", "ppfocus", "buzzhand", "bomb01", "qiqi", "inf.news", "toutiao"],
    "SOCIAL": ["ptt.cc", "dcard", "mobile01", "facebook", "youtube"]
}

NOISE_BLACKLIST = ["zhihu.com", "baidu.com", "pinterest.com", "instagram.com", "tiktok.com", "tmall.com", "taobao.com", "163.com", "sohu.com"]

# ==========================================
# 3. 輔助函式 (Helper Functions)
# ==========================================

def get_domain_name(url: str) -> str:
    try: return urlparse(url).netloc.replace("www.", "")
    except: return ""

def classify_source(url: str) -> str:
    if not url or url == "#": return "OTHER"
    try:
        domain = urlparse(url).netloc.lower()
        clean_domain = domain.replace("www.", "")
    except: return "OTHER"
    for cat, keywords in DB_MAP.items():
        for kw in keywords:
            if kw in domain: return cat
    return "OTHER"

def get_category_meta(cat: str) -> Tuple[str, str]:
    meta = {
        "CHINA": ("🇨🇳 中國官媒", "#d32f2f"),
        "FARM": ("⛔ 內容農場", "#ef6c00"),
        "BLUE": ("🔵 泛藍觀點", "#1565c0"),
        "GREEN": ("🟢 泛綠觀點", "#2e7d32"),
        "OFFICIAL": ("⚪ 官方/中立", "#546e7a"),
        "INDIE": ("🕵️ 獨立/深度", "#fbc02d"),
        "INTL": ("🌏 國際媒體", "#f57c00"),
        "VIDEO": ("🟣 影音社群", "#7b1fa2"),
        "SOCIAL": ("⚠️ 社群聲量", "#607d8b"),
        "OTHER": ("📄 其他來源", "#9e9e9e")
    }
    return meta.get(cat, ("📄 其他來源", "#9e9e9e"))

def format_citation_style(text: str) -> str:
    if not text: return ""
    def replacement(match):
        nums = re.findall(r'\d+', match.group(0))
        if not nums: return match.group(0)
        unique_nums = sorted(list(set(nums)), key=int)
        return f'<span class="citation">Source {", ".join(unique_nums)}</span>'
    text = re.sub(r'(\[Source \d+\](?:[,;]?\s*\[Source \d+\])*)', replacement, text)
    text = re.sub(r'([\[\(（]\s*Source\s+[\d,，、\s]+[\]\)）])', replacement, text)
    return text

def extract_date_from_url(url: str) -> Optional[str]:
    if not url: return None
    patterns = [r'/(\d{4})[-/](\d{2})[-/](\d{2})/', r'/(\d{4})(\d{2})(\d{2})/', r'-(\d{4})(\d{2})(\d{2})']
    for p in patterns:
        match = re.search(p, url)
        if match: return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None

# [V37.3 New] 核心渲染邏輯抽取 (DRY Fix)
def process_timeline_rows(timeline_data: List[Dict], sources: List[Dict], blind_mode: bool) -> str:
    """
    處理時間軸數據，執行嚴格清洗、排序與格式化。
    回傳：已排序的 HTML 表格行 (tr/td) 字串。
    """
    if not timeline_data: return ""
    
    valid_rows = []
    
    for item in timeline_data:
        s_id = item.get('source_id', 0)
        # 1. 嚴格過濾：無效來源直接丟棄
        if s_id == 0 or s_id > len(sources): continue
        
        source_data = sources[s_id-1]
        real_url = source_data.get('url', '#')
        if real_url == "#": continue 
        
        # 2. 日期瀑布流
        meta_date = source_data.get('published_date')
        url_date = extract_date_from_url(real_url)
        llm_date = item.get('date')
        
        real_date = "1970-01-01" 
        display_date = "------"
        
        if meta_date and meta_date != "Missing": 
            real_date = meta_date
            display_date = meta_date
        elif url_date: 
            real_date = url_date
            display_date = url_date
        elif llm_date and re.match(r'\d{4}-\d{2}-\d{2}', llm_date) and "XX" not in llm_date:
            real_date = llm_date
            display_date = llm_date
        
        # 3. 媒體名稱模糊匹配
        cat = classify_source(real_url)
        label, _ = get_category_meta(cat)
        domain = get_domain_name(real_url)
        
        media_name = domain
        # Fuzzy match domain name
        for k, v in DOMAIN_NAME_MAP.items():
            if k in domain: media_name = v
        
        emoji = "⚪"
        if "中國" in label: emoji = "🔴"
        elif "泛藍" in label: emoji = "🔵"
        elif "泛綠" in label: emoji = "🟢"
        elif "官方" in label: emoji = "⚪"
        elif "獨立" in label: emoji = "🕵️"
        elif "國際" in label: emoji = "🌏"
        elif "農場" in label: emoji = "⛔"
        elif "社群" in label: emoji = "⚠️"
        
        display_media = f"{emoji} {media_name}"
        if blind_mode: display_media = "*****"
        
        title = item.get('title', 'No Title')
        title_html = f'<a href="{real_url}" target="_blank">{title}</a>'
        
        valid_rows.append({
            "sort_date": real_date,
            "html": f"<tr><td style='white-space:nowrap;'>{display_date}</td><td style='white-space:nowrap;'>{display_media}</td><td>{title_html}</td></tr>"
        })

    # 4. 強制按日期排序 (最新的在上面)
    valid_rows.sort(key=lambda x: x['sort_date'], reverse=True)
    
    return "".join([r['html'] for r in valid_rows])

# ==========================================
# 4. 業務邏輯 (Business Logic)
# ==========================================

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
def generate_dynamic_keywords(query: str, api_key: str) -> List[str]:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)
        prompt = f"""
        請針對議題「{query}」，生成 3 組最具情報價值的搜尋關鍵字，分別對應以下三個維度：
        1. [事實軌]：針對事件發展、時間軸、新聞報導。
        2. [觀點軌]：針對爭議、正反評論、社論。
        3. [深度軌]：針對懶人包、影響分析、法規細節。
        
        請直接輸出 3 個字串，用逗號分隔，不要標號。
        範例："{query} 事件進度, {query} 正反爭議, {query} 懶人包重點"
        """
        resp = llm.invoke(prompt).content
        keywords = [k.strip() for k in resp.split(',') if k.strip()]
        return keywords[:3] if len(keywords) >= 3 else [f"{query} 新聞 事件", f"{query} 爭議 評論", f"{query} 懶人包 分析"]
    except:
        return [f"{query} 新聞 事件", f"{query} 爭議 評論", f"{query} 懶人包 分析"] 

def search_cofacts(query: str) -> str:
    url = "https://cofacts-api.g0v.tw/graphql"
    graphql_query = """query ListArticles($text: String!) { ListArticles(filter: {q: $text}, orderBy: [{_score: DESC}], first: 3) { edges { node { text articleReplies(status: NORMAL) { reply { text type } } } } } }"""
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

def execute_hybrid_search(query: str, api_key_tavily: str, search_params: Dict, is_strict_mode: bool, dynamic_keywords: List[str], selected_regions: List[str]) -> List[Dict]:
    tavily = TavilyClient(api_key=api_key_tavily)
    seen_urls = set()
    tasks = []
    
    # 1. 通用熱度搜尋 (Tri-Track)
    general_domains = []
    if "台灣" in str(selected_regions): general_domains.extend(FULL_TAIWAN_WHITELIST)
    if "獨立" in str(selected_regions): general_domains.extend(INDIE_WHITELIST)
    if "亞洲" in str(selected_regions): general_domains.extend(INTL_WHITELIST)
    
    general_params = search_params.copy()
    general_params['max_results'] = 10 
    if is_strict_mode and general_domains:
        general_params['include_domains'] = list(set(general_domains))
    
    tasks.append({"name": "General_Main", "query": query, "params": general_params})
    tasks.append({"name": "General_Fact", "query": dynamic_keywords[0], "params": general_params})
    tasks.append({"name": "General_Opn", "query": dynamic_keywords[1], "params": general_params})
    tasks.append({"name": "General_Deep", "query": dynamic_keywords[2], "params": general_params})
    
    # 2. 分眾保底搜尋 (Hybrid Weighted - Standard Guard)
    if "台灣" in str(selected_regions):
        blue_params = search_params.copy()
        blue_params['max_results'] = 5 
        blue_params['include_domains'] = BLUE_WHITELIST
        tasks.append({"name": "Blue_Guard", "query": f"{query}", "params": blue_params})
        
        green_params = search_params.copy()
        green_params['max_results'] = 5 
        green_params['include_domains'] = GREEN_WHITELIST
        tasks.append({"name": "Green_Guard", "query": f"{query}", "params": green_params})
        
        official_params = search_params.copy()
        official_params['max_results'] = 5
        official_params['include_domains'] = OFFICIAL_WHITELIST
        tasks.append({"name": "Official_Guard", "query": f"{query} 聲明 新聞稿", "params": official_params})

    def fetch(task):
        try:
            return tavily.search(query=task['query'], **task['params']).get('results', [])
        except: return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch, t): t['name'] for t in tasks}
        results_map = {}
        for future in concurrent.futures.as_completed(futures):
            t_name = futures[future]
            results_map[t_name] = future.result()
            
    final_list = []
    
    # A. 優先加入保底
    guards = ["Blue_Guard", "Green_Guard", "Official_Guard"]
    for guard_name in guards:
        if guard_name in results_map:
            for item in results_map[guard_name]:
                if item['url'] not in seen_urls:
                    seen_urls.add(item['url'])
                    final_list.append(item)
    
    # B. 再加入通用 (Tri-Track)
    general_keys = ["General_Fact", "General_Opn", "General_Deep", "General_Main"]
    max_len = max([len(results_map.get(k, [])) for k in general_keys]) if general_keys else 0
    
    for i in range(max_len):
        for key in general_keys:
            if key in results_map and i < len(results_map[key]):
                item = results_map[key][i]
                if item['url'] not in seen_urls:
                    seen_urls.add(item['url'])
                    final_list.append(item)
                
    return final_list

def get_search_context(query: str, api_key_tavily: str, days_back: int, selected_regions: List[str], max_results: int, dynamic_keywords: List[str]):
    try:
        active_blacklist = NOISE_BLACKLIST

        search_params = {
            "search_depth": "advanced",
            "topic": "general",
            "days": days_back,
            "exclude_domains": active_blacklist
        }

        is_strict_mode = bool(selected_regions)
        results = execute_hybrid_search(query, api_key_tavily, search_params, is_strict_mode, dynamic_keywords, selected_regions)
        
        results.sort(key=lambda x: x.get('published_date') or "", reverse=True)
        results = results[:max_results]
        
        context_text = ""
        for i, res in enumerate(results):
            title = res.get('title', 'No Title')
            url = res.get('url', '#')
            
            pub_date = res.get('published_date')
            if not pub_date:
                url_date = extract_date_from_url(url)
                pub_date = url_date if url_date else "Missing"
            else:
                pub_date = pub_date[:10]
            
            res['final_date'] = pub_date
            content = res.get('content', '')[:3000]
            context_text += f"Source {i+1}: [Date: {pub_date}] [Title: {title}] {content} (URL: {url})\n"
            
        return context_text, results, query, is_strict_mode
        
    except Exception as e:
        return f"Error: {str(e)}", [], "Error", False

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5), reraise=True)
def call_gemini(system_prompt: str, user_text: str, model_name: str, api_key: str) -> str:
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    chain = prompt | llm
    return chain.invoke({"input": user_text}).content

def run_strategic_analysis(query: str, context_text: str, model_name: str, api_key: str, mode: str="FUSION") -> str:
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    tone_instruction = """
    【⚠️ 語氣風格指令】：
    1. **極度審慎**：嚴禁臆測。若證據不足，請直接標示「目前資訊不足」。
    2. **去軍事化**：嚴禁使用軍事隱喻。
    3. **中性專業**：使用社會科學術語。
    """

    if mode == "FUSION":
        system_prompt = f"""
        你是一位極度嚴謹的情報分析師。
        
        【⚠️ 時間錨點】：今天是 {today_str}。
        {tone_instruction}
        
        【⚠️ 數據結構指令】：輸出 Source ID (如 Source 1)。
        
        【分析方法論】：
        1. **邏輯謬誤偵測**：指出滑坡謬誤、稻草人論證。
        2. **證據強度分級**：評估證據力（強/弱）。
        3. **聲量權重校正**：識別複讀機，挖掘長尾觀點。
        
        【輸出格式 (嚴格遵守)】：
        ### [DATA_TIMELINE]
        (格式：YYYY-MM-DD|媒體|標題|Source_ID)
        *請注意：只能列出 Context 中實際存在的 Source，嚴禁捏造 Source ID。若無 Source ID 則不列出。*
        
        ### [REPORT_TEXT]
        (Markdown 報告 - 繁體中文)
        1. **📊 全域現況摘要 (Situational Analysis)**
           - 請以 **Markdown 表格** 呈現關鍵事件時間軸 (欄位包含：日期 | 事件摘要 | 關鍵影響)。
        2. **🔍 爭議點與事實查核 (Fact-Check & Logic Scan)**
           - *包含：邏輯謬誤偵測、證據強度評估*
        3. **⚖️ 媒體框架光譜分析 (Framing Analysis)**
           - *請應用聲量權重校正，指出話語權是否失衡*
        4. **🧠 深度識讀與利益分析 (Cui Bono)**
        5. **🤔 結構性反思 (Structural Reflection)**
        """
        
    elif mode == "DEEP_SCENARIO":
        system_prompt = f"""
        你是一位專精於未來學 (Futures Studies) 的戰略顧問。
        
        【⚠️ 時間錨點】：今天是 {today_str}。
        {tone_instruction}
        
        【分析任務】：
        1. **早期預警指標**：列出監測訊號。
        2. **驗屍分析**：反推失敗變數。

        【輸出格式】：
        ### [DATA_TIMELINE]
        (留空)
        
        ### [REPORT_TEXT]
        (Markdown 報告 - 繁體中文)
        1. **🎯 CLA 深度解構 (Causal Layered Analysis)**
           - Litany / System / Worldview / Myth
        2. **🔮 未來趨勢路徑模擬 (Scenario Planning)**
           - **基準路徑 (Baseline)** + 🚩 預警指標
           - **轉折路徑 (Alternative)** + 🚩 預警指標
           - **極端路徑 (Wild Card)** + 🚩 預警指標
        3. **💀 驗屍分析 (Pre-mortem Analysis)**
        4. **💡 綜合發展與因應建議**
        """
    else:
        system_prompt = f"請針對 {query} 進行分析。"

    return call_gemini(system_prompt, context_text, model_name, api_key)

def parse_gemini_data(text: str) -> Dict[str, Any]:
    """
    解析 Gemini AI 返回的文本，提取時間軸和報告內容
    
    Args:
        text: AI 返回的原始文本
        
    Returns:
        包含 timeline 和 report_text 的字典
    """
    data = {"timeline": [], "report_text": ""}
    if not text or not text.strip():
        return data

    # 先提取時間軸數據（從 [DATA_TIMELINE] 區塊）
    timeline_section = ""
    if "### [DATA_TIMELINE]" in text:
        parts = text.split("### [DATA_TIMELINE]")
        if len(parts) > 1:
            timeline_section = parts[1].split("### [REPORT_TEXT]")[0] if "### [REPORT_TEXT]" in parts[1] else parts[1]
    elif "[DATA_TIMELINE]" in text:
        parts = text.split("[DATA_TIMELINE]")
        if len(parts) > 1:
            timeline_section = parts[1].split("[REPORT_TEXT]")[0] if "[REPORT_TEXT]" in parts[1] else parts[1]
    
    # 解析時間軸
    if timeline_section:
        lines = timeline_section.split('\n')
        for line in lines:
            line = line.strip()
            # 檢查是否為時間軸行（包含 | 分隔符且至少有 3 個部分）
            if "|" in line and len(line.split("|")) >= 3:
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
                    if "XX" in date or "xx" in date: 
                        date = "近期"
                    
                    # 驗證日期格式（YYYY-MM-DD 或 "近期"）
                    if re.match(r'^\d{4}-\d{2}-\d{2}$', date) or date == "近期":
                        data["timeline"].append({
                            "date": date,
                            "media": name,
                            "title": title,
                            "source_id": int(source_id_str)
                        })
                except Exception as e:
                    # 靜默跳過無效行
                    continue

    # 提取報告文本（優先順序：REPORT_TEXT 標記 > 摘要標記 > 全部文本）
    report_text = ""
    
    # 方法 1: 查找 [REPORT_TEXT] 標記
    if "### [REPORT_TEXT]" in text:
        parts = text.split("### [REPORT_TEXT]")
        if len(parts) > 1:
            report_text = parts[1].strip()
    elif "### REPORT_TEXT" in text:
        parts = text.split("### REPORT_TEXT")
        if len(parts) > 1:
            report_text = parts[1].strip()
    elif "[REPORT_TEXT]" in text:
        parts = text.split("[REPORT_TEXT]")
        if len(parts) > 1:
            report_text = parts[1].strip()
    
    # 方法 2: 如果沒有 REPORT_TEXT 標記，查找摘要或分析部分
    if not report_text:
        # 查找包含「摘要」、「分析」、「CLA」等關鍵字的標題
        patterns = [
            r"(#+\s*.*?摘要.*?\n.*?)(?=#+\s*|$)",
            r"(#+\s*.*?分析.*?\n.*?)(?=#+\s*|$)",
            r"(1\.\s*.*?摘要.*?\n.*?)(?=\d+\.\s*|$)",
            r"(#+\s*.*?CLA.*?\n.*?)(?=#+\s*|$)",
            r"(📊\s*.*?全域現況.*?\n.*?)(?=#+\s*|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                report_text = match.group(1).strip()
                break
    
    # 方法 3: 如果還是沒有，移除時間軸部分後，使用剩餘文本
    if not report_text:
        # 移除 [DATA_TIMELINE] 區塊
        if "### [DATA_TIMELINE]" in text:
            parts = text.split("### [DATA_TIMELINE]")
            if len(parts) > 1:
                remaining = parts[1].split("### [REPORT_TEXT]")[-1] if "### [REPORT_TEXT]" in parts[1] else ""
                if remaining.strip():
                    report_text = remaining.strip()
        else:
            # 如果沒有明確的標記，使用整個文本（但排除時間軸行）
            lines = text.split('\n')
            report_lines = []
            for line in lines:
                # 跳過時間軸行（包含 | 且格式為日期|媒體|標題）
                if "|" in line and len(line.split("|")) >= 3:
                    # 檢查是否為時間軸格式
                    parts = line.split("|")
                    if len(parts) >= 3 and (re.match(r'^\d{4}-\d{2}-\d{2}', parts[0].strip()) or "近期" in parts[0]):
                        continue
                report_lines.append(line)
            report_text = '\n'.join(report_lines).strip()
    
    # 清理報告文本：移除多餘的標記和空行
    if report_text:
        # 移除開頭的標記行
        report_text = re.sub(r'^###?\s*\[?REPORT_TEXT\]?\s*\n*', '', report_text, flags=re.MULTILINE)
        # 移除開頭的空行
        report_text = report_text.lstrip('\n').strip()
        
        # 清理過多的破折號（可能是 Markdown 渲染問題）
        # 移除連續超過 10 個破折號的行
        report_text = re.sub(r'^-{10,}\s*$', '', report_text, flags=re.MULTILINE)
        # 移除連續超過 5 個破折號但保留表格分隔符
        lines = report_text.split('\n')
        cleaned_lines = []
        for line in lines:
            # 保留表格分隔符（|:---:| 格式）
            if re.match(r'^\|[\s:---]+\|', line):
                cleaned_lines.append(line)
            # 移除只有破折號的行（超過 5 個）
            elif re.match(r'^-{5,}\s*$', line):
                continue  # 跳過這行
            else:
                cleaned_lines.append(line)
        report_text = '\n'.join(cleaned_lines)
    
    data["report_text"] = report_text if report_text else text  # 如果還是空的，使用原始文本
    
    return data

def create_full_html_report(data_result, scenario_result, sources, blind_mode) -> str:
    # [V37.3] 使用重構後的邏輯
    table_rows = process_timeline_rows(data_result.get("timeline", []), sources, blind_mode)
    
    timeline_html = ""
    if table_rows:
        timeline_html = f"""
        <h3>📅 關鍵發展時序</h3>
        <table class="custom-table" border="1" cellspacing="0" cellpadding="5" style="width:100%; border-collapse:collapse;">
            <thead><tr><th width="120">日期</th><th width="180">媒體來源 (Code Verified)</th><th>新聞標題 (點擊閱讀)</th></tr></thead>
            <tbody>{table_rows}</tbody>
        </table>
        <hr>
        """

    report_html_1 = ""
    if data_result:
        raw_md = data_result.get("report_text", "")
        html_content = markdown.markdown(raw_md, extensions=['tables'])
        final_html = format_citation_style(html_content)
        report_html_1 = f'<div class="report-paper"><h3>📝 平衡報導分析</h3>{final_html}</div>'

    report_html_2 = ""
    if scenario_result:
        raw_md_2 = scenario_result.get("report_text", "")
        html_content_2 = markdown.markdown(raw_md_2, extensions=['tables'])
        final_html_2 = format_citation_style(html_content_2)
        report_html_2 = f'<div class="report-paper"><h3>🔮 未來發展推演報告</h3>{final_html_2}</div>'

    sources_html = ""
    if sources:
        s_rows = ""
        for i, s in enumerate(sources):
            domain = get_domain_name(s.get('url'))
            media_name = domain
            for k, v in DOMAIN_NAME_MAP.items():
                if k in domain: media_name = v
                
            title = s.get('title', 'No Title')
            url = s.get('url')
            s_rows += f"<li><b>[{i+1}]</b> {media_name} - <a href='{url}' target='_blank'>{title}</a></li>"
        sources_html = f"<hr><h3>📚 引用文獻列表</h3><ul>{s_rows}</ul>"

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>全域觀點分析報告</title>
        {CSS_STYLE}
    </head>
    <body style="padding: 20px; max-width: 900px; margin: 0 auto;">
        <h1>全域觀點分析報告 (V38.0)</h1>
        <p>生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        {timeline_html}
        {report_html_1}
        {report_html_2}
        {sources_html}
    </body>
    </html>
    """
    return full_html

def render_html_timeline(timeline_data, sources, blind_mode):
    # [V37.3] 直接呼叫重構後的邏輯
    table_rows = process_timeline_rows(timeline_data, sources, blind_mode)
    if not table_rows: return

    full_html = f"""
    <div class="scrollable-table-container">
    <table class="custom-table">
    <thead>
    <tr>
    <th style="width:120px;">日期</th>
    <th style="width:180px;">媒體</th>
    <th>新聞標題</th>
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

def export_full_state():
    data = {
        "result": st.session_state.result,
        "scenario_result": st.session_state.scenario_result,
        "sources": st.session_state.sources
    }
    return json.dumps(data, indent=2, ensure_ascii=False)

def convert_data_to_md(data):
    return f"""
# 全域觀點分析報告 (V38.0)
产生時間: {datetime.now()}

## 1. 平衡報導分析
{data.get('report_text')}

## 2. 時間軸
{pd.DataFrame(data.get('timeline')).to_markdown(index=False)}
    """

# ==========================================
# 5. UI
# ==========================================
with st.sidebar:
    st.title("全域觀點解析 V38.0")
    st.caption("✨ 新增：Tavily 搜尋 UI + 學術方法論詳解")
    
    analysis_mode = st.radio(
        "選擇分析引擎：",
        options=["全域深度解析 (Fusion)", "未來發展推演 (Scenario)"],
        captions=["學術框架：框架 + 邏輯偵錯", "學術框架：CLA + 預警指標"],
        index=0
    )
    st.markdown("---")
    
    blind_mode = st.toggle("🙈 盲測模式", value=False)
    
    with st.expander("🔑 API 設定", expanded=True):
        st.info("⚠️ 請輸入您的 API Key (不會儲存，重新整理後需再次輸入)")
        google_key = st.text_input("Gemini Key", type="password", help="用於 AI 分析的 Google Gemini API 金鑰")
        tavily_key = st.text_input("Tavily Key", type="password", help="用於新聞搜尋的 Tavily API 金鑰（必需）")
        
        # 顯示 Tavily 搜尋狀態
        if tavily_key:
            st.success("✅ Tavily 搜尋已啟用")
        else:
            st.warning("⚠️ 請輸入 Tavily Key 以啟用新聞搜尋功能")
            
        model_name = st.selectbox(
            "模型 (Gemini 2.5 Series)", 
            ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"], 
            index=0,
            help="選擇用於分析的 Gemini 模型版本"
        )
        
        st.markdown("---")
        st.markdown("#### 🔍 Tavily 搜尋設定")
        search_days = st.number_input("搜尋時間範圍 (天數)", min_value=1, max_value=1825, value=30, step=1, help="設定要搜尋多少天內的新聞")
        max_results = st.slider("搜尋篇數上限", 10, 100, 30, help="設定最多搜尋多少篇新聞")
        
        selected_regions = st.multiselect(
            "搜尋視角 (Region) - 可複選",
            ["🇹🇼 台灣 (Taiwan)", "🌏 亞洲 (Asia)", "🌍 歐洲 (Europe)", "🌎 美洲 (Americas)", "🕵️ 獨立/自媒體 (Indie)"],
            default=["🇹🇼 台灣 (Taiwan)"],
            help="選擇要搜尋的區域，可多選"
        )

    with st.expander("📂 匯入舊情報 (JSON還原 / 文字貼上)", expanded=False):
        uploaded_file = st.file_uploader("上傳檔案", type=["json", "md", "txt"])
        default_text = ""
        is_json_upload = False
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".json"):
                    is_json_upload = True
                    st.success(f"✅ 完整存檔: {uploaded_file.name}")
                else:
                    default_text = uploaded_file.getvalue().decode("utf-8")
                    st.success(f"✅ 文字檔: {uploaded_file.name}")
            except: pass

        past_report_input = st.text_area("或貼上內容：", value=default_text, height=150)
        
        if uploaded_file and st.button("🔄 確認載入/還原"):
            if is_json_upload:
                try:
                    state_data = json.load(uploaded_file)
                    st.session_state.result = state_data.get("result")
                    st.session_state.scenario_result = state_data.get("scenario_result")
                    st.session_state.sources = state_data.get("sources")
                    st.rerun()
                except: st.error("JSON 解析失敗")
            else:
                st.toast("✅ 文字已匯入")

    # ==================== 學術方法論詳解 ====================
    st.markdown("---")
    st.markdown("### 🧠 情報分析方法論詳解")
    st.caption("📚 點擊下方區塊查看詳細的學術理論與方法說明")
    st.markdown("")  # 添加空行增加可讀性
    
    # 強制顯示：確保這些區塊一定會顯示
    # 這些 expander 區塊包含完整的學術方法論說明
    
    with st.expander("1. 資訊檢索：混和權重與三軌搜尋 (Hybrid Weighted Search)", expanded=False):
        st.markdown("""
        **核心機制：混和權重搜尋**
        - **分眾保底 (Safety Net)**：強制開啟專用通道，確保藍營、綠營、官方至少各抓取 5 篇代表性文章，保障弱勢觀點入場。
        - **熱度補完 (Volume Fill)**：剩餘名額開放給全網熱度排序，反映真實輿論聲量。
        
        **三軌搜尋架構 (Tri-Track via Dynamic Keywords)**
        將「通用搜尋 (General)」任務拆解為三組不同目的的指令，確保抓取內容的維度完整：
        1. **事實與時序 (Facts & Timeline)**
           - 指令：`{query} 新聞 事件 時間軸`
           - 目標：確保報告的骨架（人、事、時、地、物）是準確的。
        2. **觀點與爭議 (Opinions & Controversy)**
           - 指令：`{query} 評論 觀點 爭議 分析`
           - 目標：捕捉不同陣營（正方/反方）的論述邏輯，這是 Entman 框架分析的原料。
        3. **深度與結構 (Deep Dive)**
           - 指令：`{query} 懶人包 重點 影響`
           - 目標：快速獲取議題的全貌、背景知識與結構化資訊（如法規比較）。
        """)
        
    with st.expander("2. 框架分析：Entman 理論與立場判定 (Framing)"):
        st.markdown("""
        **Entman 框架理論 (Framing Theory)**
        我們分析文本如何透過「選擇 (Selection)」與「凸顯 (Salience)」來建構現實。
        - **問題定義**：不同陣營如何定義問題的核心？
        - **歸因分析**：將責任歸咎於誰？
        - **道德評價**：使用什麼樣的形容詞來進行道德審判？
        
        **機構層次驗證**
        結合媒體所有權結構 (Ownership) 與過往政治傾向資料庫 (DB_MAP)，對文章立場進行雙重驗證。
        """)
        
    with st.expander("3. 可信度驗證：水平閱讀與邏輯偵錯 (Verification)"):
        st.markdown("""
        **水平閱讀法 (Lateral Reading)**
        採用史丹佛歷史教育群 (SHEG) 提倡之方法，不只深讀單一來源，而是橫向比對多個來源以確認事實。
        
        **邏輯偵錯 (Logic Scan)**
        AI 會自動掃描文本中的邏輯謬誤：
        - **滑坡謬誤**：誇大微小行動的災難性後果。
        - **稻草人論證**：扭曲對手觀點以便攻擊。
        
        **Cofacts 協作查核**
        即時串接 g0v Cofacts 謠言資料庫，標註已被社群查核為錯誤的資訊。
        """)
        
    with st.expander("4. 戰略推演：CLA 層次分析與預警 (Futures)"):
        st.markdown("""
        **CLA 層次分析法 (Causal Layered Analysis)**
        深入挖掘議題的四個層次：
        1. **表象 (Litany)**：公眾看到的事件與數據。
        2. **系統 (System)**：造成事件的社會結構與政策成因。
        3. **世界觀 (Worldview)**：利益相關者的深層價值觀與意識形態。
        4. **神話/隱喻 (Myth)**：潛意識中的集體焦慮或故事原型。
        
        **早期預警指標 (Signposts)**
        為每個未來情境設定具體的監測訊號。
        
        **驗屍分析 (Pre-mortem)**
        假設預測失敗，反推可能的隱蔽變數。
        """)
        
    st.markdown("### 📥 報告匯出")
    if st.session_state.get('result') or st.session_state.get('scenario_result'):
        html_report = create_full_html_report(st.session_state.result, st.session_state.scenario_result, st.session_state.sources, blind_mode)
        st.download_button("📥 列印用檔案 (HTML)", html_report, "Printable_Report.html", "text/html")
        full_state_json = export_full_state()
        st.download_button("📥 完整狀態 (JSON)", full_state_json, "Full_State.json", "application/json")
        
        export_data = st.session_state.get('result').copy()
        if st.session_state.get('scenario_result'):
            export_data['report_text'] += "\n\n# 未來發展推演報告\n" + st.session_state.get('scenario_result')['report_text']
        st.download_button("📥 純文字 (Markdown)", convert_data_to_md(export_data), "report.md", "text/markdown")

st.title(f"{analysis_mode.split(' ')[0]}")
query = st.text_input("輸入議題關鍵字", placeholder="例如：台積電美國設廠爭議")
search_btn = st.button("🚀 啟動全域掃描", type="primary")

if 'result' not in st.session_state: st.session_state.result = None
if 'scenario_result' not in st.session_state: st.session_state.scenario_result = None
if 'sources' not in st.session_state: st.session_state.sources = None

if search_btn and query and google_key and tavily_key:
    st.session_state.result = None
    st.session_state.scenario_result = None
    
    with st.status("🚀 啟動 V38.0 平衡報導分析引擎...", expanded=True) as status:
        
        st.write("🧠 1. 生成動態搜尋策略...")
        dynamic_keywords = generate_dynamic_keywords(query, google_key)
        st.write(f"   ↳ 鎖定戰略關鍵字: {', '.join(dynamic_keywords)}")
        
        regions_label = ", ".join([r.split(" ")[1] for r in selected_regions])
        st.write(f"📡 2. 執行混和權重搜尋 (視角: {regions_label})...")
        st.write("   ↳ 啟動機制：分眾保底 (藍/綠/官方) + 熱度補完 (動態三軌)")
        
        # 驗證 Tavily Key
        if not tavily_key:
            st.error("❌ 錯誤：未提供 Tavily API Key，無法執行搜尋")
            status.update(label="❌ 搜尋失敗", state="error", expanded=False)
            st.stop()
        
        context_text, sources, actual_query, is_strict_tw = get_search_context(
            query, tavily_key, search_days, selected_regions, max_results, dynamic_keywords
        )
        
        st.write(f"   ↳ 搜尋完成：共獲取 {len(sources)} 篇資料 (已去重)。")
        if is_strict_tw:
            st.write(f"🛡️ 網域圍籬已啟動。")
        
        st.session_state.sources = sources
        
        st.write("🛡️ 3. 查詢 Cofacts 謠言資料庫...")
        cofacts_txt = search_cofacts(query)
        if cofacts_txt: context_text += f"\n{cofacts_txt}\n"
        
        st.write("🧠 4. AI 進行深度戰略分析 (ACH 競爭假設 + 邏輯偵錯)...")
        
        mode_code = "DEEP_SCENARIO" if "未來" in analysis_mode else "FUSION"
        analysis_context = past_report_input if (mode_code == "DEEP_SCENARIO" and past_report_input) else context_text

        raw_report = run_strategic_analysis(query, analysis_context, model_name, google_key, mode=mode_code)
        
        # 解析報告數據
        parsed_data = parse_gemini_data(raw_report)
        
        # 驗證解析結果
        if not parsed_data.get("report_text") or not parsed_data.get("report_text").strip():
            st.warning("⚠️ AI 返回的報告格式可能不符合預期，嘗試使用備用解析方法...")
            # 備用方法：如果沒有找到 REPORT_TEXT，使用整個文本（排除時間軸）
            if raw_report:
                # 移除 [DATA_TIMELINE] 區塊
                if "### [DATA_TIMELINE]" in raw_report:
                    parts = raw_report.split("### [DATA_TIMELINE]")
                    if len(parts) > 1:
                        remaining = parts[1]
                        if "### [REPORT_TEXT]" in remaining:
                            parsed_data["report_text"] = remaining.split("### [REPORT_TEXT]")[1].strip()
                        else:
                            # 移除時間軸行
                            lines = remaining.split('\n')
                            report_lines = []
                            for line in lines:
                                if "|" in line and len(line.split("|")) >= 3:
                                    parts_line = line.split("|")
                                    if len(parts_line) >= 3 and (re.match(r'^\d{4}-\d{2}-\d{2}', parts_line[0].strip()) or "近期" in parts_line[0]):
                                        continue
                                report_lines.append(line)
                            parsed_data["report_text"] = '\n'.join(report_lines).strip()
                else:
                    # 如果完全沒有標記，使用整個文本
                    parsed_data["report_text"] = raw_report.strip()
        
        st.session_state.result = parsed_data
        
        # 顯示解析統計
        timeline_count = len(parsed_data.get("timeline", []))
        report_length = len(parsed_data.get("report_text", ""))
        st.write(f"   ↳ 解析完成：時間軸 {timeline_count} 筆，報告長度 {report_length} 字元")
            
        status.update(label="✅ 分析完成", state="complete", expanded=False)
        
    st.rerun()

if st.session_state.result:
    data = st.session_state.result
    render_html_timeline(data.get("timeline"), st.session_state.sources, blind_mode)

    st.markdown("---")
    st.markdown("### 📝 平衡報導分析")
    
    report_text = data.get("report_text", "")
    
    # 檢查報告內容是否為空
    if not report_text or not report_text.strip():
        st.warning("⚠️ 報告內容為空。可能的原因：")
        st.info("""
        1. AI 返回的格式不符合預期
        2. 解析過程中出現錯誤
        3. 請檢查終端機/控制台的錯誤訊息
        
        **建議：**
        - 嘗試重新執行分析
        - 檢查 API 金鑰是否正確
        - 查看原始返回數據（可在調試模式下）
        """)
        
        # 顯示調試信息
        with st.expander("🔍 調試信息", expanded=False):
            st.write("**原始數據結構：**")
            st.json({
                "has_timeline": len(data.get("timeline", [])) > 0,
                "timeline_count": len(data.get("timeline", [])),
                "report_text_length": len(report_text),
                "report_text_preview": report_text[:500] if report_text else "（空）"
            })
    else:
        # 清理報告文本中的過多破折號
        cleaned_report = report_text
        
        # 移除連續超過 10 個破折號的行
        cleaned_report = re.sub(r'^-{10,}\s*$', '', cleaned_report, flags=re.MULTILINE)
        
        # 移除只有破折號的行（保留表格分隔符）
        lines = cleaned_report.split('\n')
        cleaned_lines = []
        for line in lines:
            # 保留表格分隔符（|:---:| 格式）
            if re.match(r'^\|[\s:---]+\|', line):
                cleaned_lines.append(line)
            # 跳過只有破折號的行（超過 5 個）
            elif re.match(r'^-{5,}\s*$', line):
                continue
            else:
                cleaned_lines.append(line)
        cleaned_report = '\n'.join(cleaned_lines)
        
        formatted_text = format_citation_style(cleaned_report)
        html_content = markdown.markdown(formatted_text, extensions=['tables'])
        st.markdown(f'<div class="report-paper">{html_content}</div>', unsafe_allow_html=True)
    
    if "未來" not in analysis_mode and not st.session_state.scenario_result:
        st.markdown("---")
        if st.button("🚀 將此結果餵給未來發展推演 (資訊滾動)", type="secondary"):
            with st.spinner("🔮 正在讀取前次情報，啟動 CLA 層次分析與未來推演..."):
                current_report = data.get("report_text", "")
                raw_text = run_strategic_analysis(query, current_report, model_name, google_key, mode="DEEP_SCENARIO")
                st.session_state.scenario_result = parse_gemini_data(raw_text) 
                st.rerun()

if st.session_state.scenario_result:
    st.markdown("---")
    st.markdown("### 🔮 未來發展推演報告")
    scenario_data = st.session_state.scenario_result
    formatted_scenario = format_citation_style(scenario_data.get("report_text", ""))
    html_scenario = markdown.markdown(formatted_scenario, extensions=['tables'])
    st.markdown(f'<div class="report-paper">{html_scenario}</div>', unsafe_allow_html=True)

if st.session_state.sources:
    st.markdown("---")
    st.markdown("### 📚 引用文獻列表")
    md_table = "| 編號 | 媒體/網域 | 標題摘要 | 連結 |\n|:---:|:---|:---|:---|\n"
    for i, s in enumerate(st.session_state.sources):
        domain = get_domain_name(s.get('url'))
        
        media_name = domain
        for k, v in DOMAIN_NAME_MAP.items():
            if k in domain: media_name = v
            
        if blind_mode: media_name = "*****"
        
        title = s.get('title', 'No Title')
        if len(title) > 60: title = title[:60] + "..."
        url = s.get('url')
        md_table += f"| **{i+1}** | `{media_name}` | {title} | [點擊]({url}) |\n"
    st.markdown(md_table)
