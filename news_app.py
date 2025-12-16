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
import markdown
from urllib.parse import urlparse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from tavily import TavilyClient

# ==========================================
# 1. 基礎設定與 CSS樣式
# ==========================================
st.set_page_config(page_title="全域觀點解析 V35.3", page_icon="⚖️", layout="wide")

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
# 2. 資料庫與共用常數
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

GRAY_WHITELIST = [
    "ptt.cc", "dcard.tw", "mobile01.com"
]

DB_MAP = {
    "CHINA": ["xinhuanet", "people.com.cn", "huanqiu", "cctv", "chinadaily", "taiwan.cn", "gwytb", "guancha"],
    "GREEN": ["ltn", "ftv", "setn", "rti.org", "newtalk", "mirrormedia", "dpp.org", "libertytimes"],
    "BLUE": ["udn", "chinatimes", "tvbs", "cti", "nownews", "ctee", "kmt.org", "uniteddaily"],
    "OFFICIAL": ["cna.com", "pts.org", "mnd.gov", "mac.gov", "tfc-taiwan", "gov.tw"],
    "INDIE": ["twreporter", "theinitium", "thenewslens", "upmedia", "storm.mg", "mindiworld", "vocus", "matters", "plainlaw"],
    "INTL": ["bbc", "cnn", "reuters", "apnews", "bloomberg", "wsj", "nytimes", "dw.com", "voanews", "rfi.fr"],
    "FARM": ["kknews", "read01", "ppfocus", "buzzhand", "bomb01", "qiqi", "inf.news", "toutiao"],
    "SOCIAL": ["ptt.cc", "dcard.tw", "mobile01.com", "facebook.com", "youtube.com"]
}

NOISE_BLACKLIST = [
    "zhihu.com", "baidu.com", "pinterest.com", "instagram.com", 
    "tiktok.com", "tmall.com", "taobao.com", "163.com", "sohu.com"
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
        "SOCIAL": ("⚠️ 社群聲量", "#607d8b"),
        "OTHER": ("📄 其他來源", "#9e9e9e")
    }
    return meta.get(cat, ("📄 其他來源", "#9e9e9e"))

def format_citation_style(text):
    if not text: return ""
    def replacement(match):
        nums = re.findall(r'\d+', match.group(0))
        if not nums: return match.group(0)
        unique_nums = sorted(list(set(nums)), key=int)
        return f'<span class="citation">Source {", ".join(unique_nums)}</span>'
    text = re.sub(r'(\[Source \d+\](?:[,;]?\s*\[Source \d+\])*)', replacement, text)
    text = re.sub(r'([\[\(（]\s*Source\s+[\d,，、\s]+[\]\)）])', replacement, text)
    return text

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

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
def generate_dynamic_keywords(query, api_key):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)
        prompt = f"""
        你是專業的情報分析師。請針對議題「{query}」，生成 3 組「最具情報價值」的搜尋關鍵字。
        策略：核心爭議、數據事實、深度分析。
        請直接輸出 3 個關鍵字字串，用逗號分隔。例如："{query} 爭議, {query} 懶人包, {query} 影響"
        """
        resp = llm.invoke(prompt).content
        keywords = [k.strip() for k in resp.split(',') if k.strip()]
        return keywords[:3] if keywords else [f"{query} 爭議", f"{query} 分析", f"{query} 懶人包"]
    except:
        return [f"{query} 爭議", f"{query} 分析", f"{query} 懶人包"] 

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

def execute_swarm_search(query, api_key_tavily, search_params, is_strict_mode, dynamic_queries):
    tavily = TavilyClient(api_key=api_key_tavily)
    queries = [query] + dynamic_queries
    sub_params = search_params.copy()
    sub_params['max_results'] = 20 
    all_results = []
    seen_urls = set()
    
    def fetch(q):
        try:
            return tavily.search(query=q, **sub_params).get('results', [])
        except: return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, q) for q in queries]
        for future in concurrent.futures.as_completed(futures):
            res_list = future.result()
            for item in res_list:
                url = item.get('url')
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(item)
    return all_results

def get_search_context(query, api_key_tavily, days_back, selected_regions, max_results, enable_outpost, dynamic_keywords):
    try:
        active_blacklist = [d for d in NOISE_BLACKLIST if d not in ["ptt.cc", "dcard.tw"]] if enable_outpost else NOISE_BLACKLIST

        search_params = {
            "search_depth": "advanced",
            "topic": "general",
            "days": days_back,
            "max_results": max_results,
            "exclude_domains": active_blacklist
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
        
        if enable_outpost:
            target_domains.extend(GRAY_WHITELIST)
            if not is_strict_mode: is_strict_mode = True 

        if is_strict_mode and target_domains:
            target_domains = list(set(target_domains))
            search_params["include_domains"] = target_domains

        results = execute_swarm_search(query, api_key_tavily, search_params, is_strict_mode, dynamic_keywords)
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

def run_strategic_analysis(query, context_text, model_name, api_key, mode="FUSION"):
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    tone_instruction = """
    【⚠️ 語氣風格指令】：
    1. **去軍事化**：嚴禁使用軍事隱喻。
    2. **中性專業**：使用社會科學術語。
    3. **建設性**：側重問題解決。
    """

    if mode == "FUSION":
        system_prompt = f"""
        你是一位極度嚴謹的情報分析師。
        
        【⚠️ 時間錨點】：今天是 {today_str}。
        {tone_instruction}
        
        【⚠️ 數據結構指令】：輸出 Source ID (如 Source 1)。
        
        【分析任務升級】：
        1. **邏輯謬誤偵測**：指出滑坡謬誤、稻草人論證。
        2. **證據強度分級**：評估證據力（強/弱）。
        
        【輸出格式 (嚴格遵守)】：
        ### [DATA_TIMELINE]
        (格式：YYYY-MM-DD|媒體|標題|Source_ID)
        
        ### [REPORT_TEXT]
        (Markdown 報告 - 繁體中文)
        1. **📊 全域現況摘要 (Situational Analysis)**
        2. **🔍 爭議點與事實查核 (Fact-Check & Logic Scan)**
           - *包含：邏輯謬誤偵測、證據強度評估*
        3. **⚖️ 媒體框架光譜分析 (Framing Analysis)**
        4. **🧠 深度識讀與利益分析 (Cui Bono)**
        5. **🤔 結構性反思 (Structural Reflection)**
        """
        
    elif mode == "DEEP_SCENARIO":
        system_prompt = f"""
        你是一位專精於未來學 (Futures Studies) 的戰略顧問。
        
        【⚠️ 時間錨點】：今天是 {today_str}。
        {tone_instruction}
        
        【分析任務升級】：
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

def create_full_html_report(data_result, scenario_result, sources, blind_mode):
    timeline_html = ""
    if data_result and data_result.get("timeline"):
        rows = ""
        for item in data_result["timeline"]:
            date = item.get('date', '近期')
            media = "*****" if blind_mode else item.get('media', 'Unknown')
            title = item.get('title', 'No Title')
            s_id = item.get('source_id', 0)
            real_url = "#"
            if sources and 0 < s_id <= len(sources):
                real_url = sources[s_id-1].get('url', '#')
                if (date == "近期" or "Missing" in date) and 'final_date' in sources[s_id-1]:
                    final_d = sources[s_id-1]['final_date']
                    if final_d and final_d != "Missing": date = final_d
            
            cat = classify_source(real_url)
            label, _ = get_category_meta(cat)
            emoji = "⚪"
            if "中國" in label: emoji = "🔴"
            elif "泛藍" in label: emoji = "🔵"
            elif "泛綠" in label: emoji = "🟢"
            elif "社群" in label: emoji = "⚠️"
            
            title_html = f'<a href="{real_url}" target="_blank">{title}</a>' if real_url != "#" else title
            rows += f"<tr><td>{date}</td><td>{emoji} {media}</td><td>{title_html}</td></tr>"
        
        timeline_html = f"""
        <h3>📅 關鍵發展時序</h3>
        <table class="custom-table" border="1" cellspacing="0" cellpadding="5" style="width:100%; border-collapse:collapse;">
            <thead><tr><th width="120">日期</th><th width="140">媒體</th><th>標題</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        <hr>
        """

    report_html_1 = ""
    if data_result:
        raw_md = data_result.get("report_text", "")
        raw_md = format_citation_style(raw_md)
        html_content = markdown.markdown(raw_md, extensions=['tables'])
        report_html_1 = f'<div class="report-paper"><h3>📝 平衡報導分析</h3>{html_content}</div>'

    report_html_2 = ""
    if scenario_result:
        raw_md_2 = scenario_result.get("report_text", "")
        raw_md_2 = format_citation_style(raw_md_2)
        html_content_2 = markdown.markdown(raw_md_2, extensions=['tables'])
        report_html_2 = f'<div class="report-paper"><h3>🔮 未來發展推演報告</h3>{html_content_2}</div>'

    sources_html = ""
    if sources:
        s_rows = ""
        for i, s in enumerate(sources):
            domain = get_domain_name(s.get('url'))
            title = s.get('title', 'No Title')
            url = s.get('url')
            s_rows += f"<li><b>[{i+1}]</b> {domain} - <a href='{url}' target='_blank'>{title}</a></li>"
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
        <h1>全域觀點分析報告 (V35.3)</h1>
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
                if final_d and final_d != "Missing": date = final_d
        
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
        elif "社群" in label: emoji = "⚠️"
        
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

def export_full_state():
    data = {
        "result": st.session_state.result,
        "scenario_result": st.session_state.scenario_result,
        "sources": st.session_state.sources
    }
    return json.dumps(data, indent=2, ensure_ascii=False)

def convert_data_to_md(data):
    return f"""
# 全域觀點分析報告 (V35.3)
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
    st.title("全域觀點解析 V35.3")
    
    analysis_mode = st.radio(
        "選擇分析引擎：",
        options=["全域深度解析 (Fusion)", "未來發展推演 (Scenario)"],
        captions=["學術框架：框架 + 邏輯偵錯", "學術框架：CLA + 預警指標"],
        index=0
    )
    st.markdown("---")
    
    enable_outpost = st.toggle("📡 前哨站模式 (納入 PTT/Dcard)", value=False)
    blind_mode = st.toggle("🙈 盲測模式", value=False)
    
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
            index=0
        )
        
        search_days = st.number_input("搜尋時間範圍 (天數)", min_value=1, max_value=1825, value=30, step=1)
        max_results = st.slider("搜尋篇數上限", 10, 100, 30)
        
        selected_regions = st.multiselect(
            "搜尋視角 (Region) - 可複選",
            ["🇹🇼 台灣 (Taiwan)", "🌏 亞洲 (Asia)", "🌍 歐洲 (Europe)", "🌎 美洲 (Americas)", "🕵️ 獨立/自媒體 (Indie)"],
            default=["🇹🇼 台灣 (Taiwan)"]
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

    with st.expander("🧠 V35.3 情報分析方法論 (完整版)", expanded=False):
        st.markdown("""
        <div class="methodology-text">
        <div class="methodology-header">1. 資訊檢索與樣本檢定 (Information Retrieval & Sampling)</div>
        本系統採用 <b>開源情報 (OSINT)</b> 標準進行資料探勘。
        <ul>
            <li><b>三軌平行搜尋 (Tri-Track)</b>：同時針對「事實/時序」、「觀點/爭議」、「深度/懶人包」三條軌道進行搜尋，確保資訊完整性。</li>
            <li><b>網域圍籬 (Domain Fencing)</b>：嚴格執行白名單機制，確保資訊來源可靠。</li>
            <li><b>前哨站模式 (Outpost)</b>：可選監測 PTT/Dcard 等社群論壇，獲取早期預警。</li>
            <li><b>動態關鍵字 (Dynamic Query)</b>：AI 自動生成衍生搜尋詞，精準打擊爭議點。</li>
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
            <li><b>邏輯偵錯 (Logic Scan)</b>：AI 自動識別滑坡謬誤、稻草人論證。</li>
            <li><b>證據分級</b>：評估新聞來源的證據強度（強/弱）。</li>
        </ul>

        <div class="methodology-header">4. 戰略推演模型 (Futures Framework)</div>
        僅應用於「未來發展推演」模式。
        <ul>
            <li><b>第一性原理 (First Principles)</b>：解構議題至最基礎的物理或經濟限制。</li>
            <li><b>層次分析法 (CLA)</b>：由表象 (Litany) 深入至系統結構 (System) 與社會神話 (Myth)。</li>
            <li><b>可能性圓錐 (Cone of Plausibility)</b>：區分基準情境 (Probable)、轉折情境 (Plausible) 與極端情境 (Possible)。</li>
            <li><b>驗屍分析 (Pre-mortem)</b>：反向推演預測失敗的可能原因。</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
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
    
    with st.status("🚀 啟動 V35.3 平衡報導分析引擎...", expanded=True) as status:
        
        st.write("🧠 1. 生成動態搜尋策略...")
        dynamic_keywords = generate_dynamic_keywords(query, google_key)
        
        regions_label = ", ".join([r.split(" ")[1] for r in selected_regions])
        st.write(f"📡 2. 執行蜂群搜尋 (視角: {regions_label})...")
        
        context_text, sources, actual_query, is_strict_tw, domain_count = get_search_context(
            query, tavily_key, search_days, selected_regions, max_results, enable_outpost, dynamic_keywords
        )
        
        if is_strict_tw:
            st.write(f"🛡️ 網域圍籬已啟動 (鎖定 {domain_count} 個來源)。")
        if enable_outpost:
            st.write("⚠️ 前哨站模式已開啟：納入 PTT/Dcard 社群聲量監測。")
        
        st.session_state.sources = sources
        
        st.write("🛡️ 3. 查詢 Cofacts 謠言資料庫...")
        cofacts_txt = search_cofacts(query)
        if cofacts_txt: context_text += f"\n{cofacts_txt}\n"
        
        st.write("🧠 4. AI 進行深度戰略分析...")
        
        mode_code = "DEEP_SCENARIO" if "未來" in analysis_mode else "FUSION"
        analysis_context = past_report_input if (mode_code == "DEEP_SCENARIO" and past_report_input) else context_text

        raw_report = run_strategic_analysis(query, analysis_context, model_name, google_key, mode=mode_code)
        st.session_state.result = parse_gemini_data(raw_report)
            
        status.update(label="✅ 分析完成", state="complete", expanded=False)
        
    st.rerun()

if st.session_state.result:
    data = st.session_state.result
    render_html_timeline(data.get("timeline"), st.session_state.sources, blind_mode)

    st.markdown("---")
    # [V35.3] 使用 markdown 套件將文字轉為 HTML，並注入 CSS
    st.markdown("### 📝 平衡報導分析")
    
    # 1. 處理引用格式 (正則表達式)
    formatted_md = format_citation_style(data.get("report_text", ""))
    
    # 2. 將 Markdown 轉換為 HTML (解決瀏覽器直接顯示源代碼的問題)
    html_content = markdown.markdown(formatted_md, extensions=['tables'])
    
    # 3. 渲染
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
    
    # 同樣的渲染邏輯
    formatted_md_2 = format_citation_style(scenario_data.get("report_text", ""))
    html_content_2 = markdown.markdown(formatted_md_2, extensions=['tables'])
    st.markdown(f'<div class="report-paper">{html_content_2}</div>', unsafe_allow_html=True)

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
