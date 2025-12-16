# ==========================================
# 0. 優先執行：警告屏蔽與套件設定
# ==========================================
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["on_bad_lines"] = "skip"

import streamlit as st
import re
import pandas as pd
import time
import requests
import json
import concurrent.futures
import random
from urllib.parse import urlparse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
import streamlit.components.v1 as components
from tavily import TavilyClient

# ==========================================
# 1. 基礎設定與 CSS樣式
# ==========================================
st.set_page_config(page_title="全域觀點解析 V18.1", page_icon="⚖️", layout="wide")

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
        font-size: 0.85em; 
        color: #757575; 
        background-color: #f0f0f0;
        padding: 2px 6px; 
        border-radius: 4px; 
        margin: 0 2px;
        font-family: sans-serif; 
        border: 1px solid #e0e0e0;
        font-weight: 500;
    }

    .perspective-box {
        padding: 15px; border-radius: 8px; margin-bottom: 10px; font-size: 0.95em;
        border-left-width: 4px; border-left-style: solid; background-color: #fff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .box-green { border-left-color: #2e7d32; }
    .box-blue { border-left-color: #1565c0; }
    .box-neutral { border-left-color: #616161; }
    
    .mermaid-box {
        background-color: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin-top: 15px;
    }
    
    .table-header-green { color: #2e7d32; font-weight: bold; font-size: 1.1em; border-bottom: 2px solid #2e7d32; margin-bottom: 10px; padding-bottom: 5px; }
    .table-header-blue { color: #1565c0; font-weight: bold; font-size: 1.1em; border-bottom: 2px solid #1565c0; margin-bottom: 10px; padding-bottom: 5px; }
    .table-header-neutral { color: #616161; font-weight: bold; font-size: 1.1em; border-bottom: 2px solid #616161; margin-bottom: 10px; padding-bottom: 5px; }
    
    .legend-box {
        background-color: #e3f2fd; border-radius: 8px; padding: 10px 15px; font-size: 0.9em; margin-bottom: 15px; border: 1px solid #bbdefb; color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 資料庫與共用常數
# ==========================================
TAIWAN_DOMAINS = [
    "udn.com", "ltn.com.tw", "chinatimes.com", "cna.com.tw", 
    "storm.mg", "setn.com", "ettoday.net", "tvbs.com.tw", 
    "mirrormedia.mg", "thenewslens.com", "upmedia.mg", 
    "rwnews.tw", "news.pts.org.tw", "ctee.com.tw", "businessweekly.com.tw",
    "news.yahoo.com.tw"
]

CAMP_KEYWORDS = {
    "GREEN": ["自由", "三立", "民視", "新頭殼", "鏡週刊", "放言", "賴清德", "民進黨", "青鳥", "中央社"],
    "BLUE": ["聯合", "中時", "中國時報", "TVBS", "中天", "風傳媒", "國民黨", "藍營", "赵少康"],
    "RED": ["新華", "人民日報", "環球", "央視", "中評", "国台办"]
}

def get_domain_name(url):
    try: return urlparse(url).netloc.replace("www.", "")
    except: return ""

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

# [V18.1] 搜尋核心：自訂數量 + 日期補救
def get_search_context(query, api_key_tavily, days_back, region_mode, max_results, context_report=None):
    try:
        tavily = TavilyClient(api_key=api_key_tavily)
        
        search_params = {
            "search_depth": "advanced",
            "topic": "general",
            "days": days_back,
            "max_results": max_results # 使用自訂數量
        }

        if "台灣" in region_mode:
            search_params["query"] = f"{query}" 
            search_params["include_domains"] = TAIWAN_DOMAINS 
        else:
            if "亞洲" in region_mode: suffix = "Asia News"
            elif "歐洲" in region_mode: suffix = "Europe News"
            elif "美洲" in region_mode: suffix = "Americas News"
            else: suffix = "news"
            
            search_params["query"] = f"{query} {suffix}"
            search_params["exclude_domains"] = ["daum.net", "naver.com", "espn.com", "pinterest.com"]

        if context_report: search_params["query"] += " analysis"
        actual_query = search_params["query"]
        
        response = tavily.search(**search_params)
        results = response.get('results', [])
        context_text = ""
        
        for i, res in enumerate(results):
            title = res.get('title', 'No Title')
            url = res.get('url', '#')
            # [V18.1] 日期補救：若 published_date 為空，嘗試抓 content 開頭
            raw_date = res.get('published_date', '')
            if not raw_date:
                raw_date = "Recent" # 標記為近期
            else:
                raw_date = raw_date[:10] # 只取 YYYY-MM-DD
                
            content = res.get('content', '')[:800]
            # 傳給 AI 時明確標註 Date
            context_text += f"Source {i+1}: [Date: {raw_date}] [Title: {title}] {content} (URL: {url})\n"
            
        return context_text, results, actual_query, ("台灣" in region_mode)
        
    except Exception as e:
        return f"Error: {str(e)}", [], "Error", False

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5), reraise=True)
def call_gemini(system_prompt, user_text, model_name, api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    chain = prompt | llm
    return chain.invoke({"input": user_text}).content

def sanitize_mermaid_code(code):
    code = re.sub(r'```mermaid', '', code)
    code = re.sub(r'```', '', code)
    code = code.strip()
    lines = code.split('\n')
    clean_lines = []
    if not any(l.strip().startswith('graph') for l in lines):
        clean_lines.append("graph TD")
    for line in lines:
        if not line.strip(): continue
        def clean_label(match):
            text = match.group(1)
            safe_text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text) 
            return f'["{safe_text}"]'
        line = re.sub(r'\["(.*?)"\]', clean_label, line)
        line = re.sub(r'\[(.*?)\]', clean_label, line)
        line = re.sub(r'\((.*?)\)', clean_label, line)
        clean_lines.append(line)
    return "\n".join(clean_lines)

def render_mermaid(code):
    clean_code = sanitize_mermaid_code(code)
    html_code = f"""
    <div class="mermaid" style="text-align: center;">
    {clean_code}
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' }});
    </script>
    """
    components.html(html_code, height=600, scrolling=True)

# 3.3 核心邏輯：數位戰情室
def run_council_of_rivals(query, context_text, model_name, api_key):
    prompts = {
        "A_SIDE": "你是一位【體制內/現狀分析師】。請找出支持現狀、政策合理性或官方解釋的證據。必須引用來源 [Source ID]。",
        "B_SIDE": "你是一位【改革/批判派分析師】。請找出質疑現狀、結構性問題或反對意見的證據。必須引用來源 [Source ID]。",
        "CONTEXT": "你是一位【脈絡歷史學家】。請分析爭議背後的深層歷史成因、經濟結構或地緣政治因素。必須引用來源 [Source ID]。"
    }
    
    opinions = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_role = {
            executor.submit(call_gemini, prompt, context_text, model_name, api_key): role 
            for role, prompt in prompts.items()
        }
        for future in concurrent.futures.as_completed(future_to_role):
            role = future_to_role[future]
            try: opinions[role] = future.result()
            except Exception as e: opinions[role] = f"分析失敗: {e}"

    editor_prompt = f"""
    你是一位堅持「平衡報導」的總編輯。針對「{query}」，請產出一份深度全解讀。
    
    【輸入素材】：
    A觀點 (體制): {opinions.get('A_SIDE')}
    B觀點 (批判): {opinions.get('B_SIDE')}
    脈絡: {opinions.get('CONTEXT')}
    
    【任務指令】：
    1. **引用壓縮**：若連續引用多個來源，請寫成 `[Source 1, 2, 3]` 的格式。
    2. **Mermaid 製圖**：請生成 Mermaid `graph TD` 代碼，展示「變數 A 如何導致 變數 B」的因果鏈。
       - 嚴格規定：節點名稱請使用 **純文字**，不要包含括號、問號或其他符號。
       - 代碼請包在 ```mermaid ... ``` 區塊中。
    3. **未來情境**：推導 3 種可能的發展劇本。
    
    【輸出格式】：
    ### [REPORT_TEXT]
    (Markdown 報告內容...)
    """
    final_report = call_gemini(editor_prompt, context_text, model_name, api_key)
    return opinions, final_report

# 3.4 核心邏輯：輿情光譜
def run_spectrum_analysis(query, context_text, model_name, api_key):
    system_prompt = f"""
    你是一位媒體識讀專家。請針對「{query}」進行媒體框架分析。
    
    【評分嚴格規定】：
    1. **立場分數 (Stance)**：
       - **負數 (-10 到 -1)**：批判/反對/泛綠/獨派。
       - **零 (0)**：中立/純事實。
       - **正數 (1 到 10)**：支持/體制/泛藍/統派。
    2. **可信度 (Credibility)**：0-3 (農場/極端) ... 8-10 (權威/查核)。
    
    【輸出格式 (請保持格式整潔，每行一筆，使用 | 分隔)】：
    ### [DATA_TIMELINE]
    YYYY-MM-DD|媒體|標題
    
    ### [DATA_SPECTRUM]
    (重要：必須包含 6 個欄位，日期請務必從 Context 中的 [Date: ...] 提取，若無則填 Recent)
    來源名稱|日期|新聞標題|立場(-10~10)|可信度(0~10)|網址
    
    ### [REPORT_TEXT]
    (Markdown 報告，請使用 `[Source 1, 3]` 格式引用)
    請包含：全域現況摘要、媒體框架分析、識讀建議。
    """
    return call_gemini(system_prompt, context_text, model_name, api_key)

# 3.5 資料解析器
def parse_gemini_data(text):
    data = {"timeline": [], "spectrum": [], "mermaid": "", "report_text": ""}
    
    mermaid_match = re.search(r"```mermaid\n(.*?)\n```", text, re.DOTALL)
    if mermaid_match:
        data["mermaid"] = mermaid_match.group(1)
        text = text.replace(mermaid_match.group(0), "")

    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Timeline
        if "|" in line and len(line.split("|")) >= 3 and (line[0].isdigit() or "20" in line):
            parts = line.split("|")
            data["timeline"].append({"date": parts[0].strip(), "media": parts[1].strip(), "event": parts[2].strip()})
            
        # Spectrum
        if "|" in line and len(line.split("|")) >= 4 and not line.startswith("###") and not "YYYY" in line:
            parts = line.split("|")
            try:
                name = parts[0].strip()
                date = "Recent" # [V18.1] 預設為 Recent
                title = "點擊閱讀報導"
                base_stance = 0
                base_cred = 0
                url = "#"
                
                # 彈性解析
                if len(parts) >= 6:
                    date = parts[1].strip()
                    title = parts[2].strip()
                    base_stance = float(parts[3].strip())
                    base_cred = float(parts[4].strip())
                    url = parts[5].strip()
                elif len(parts) == 5:
                    title = parts[1].strip()
                    base_stance = float(parts[2].strip())
                    base_cred = float(parts[3].strip())
                    url = parts[4].strip()
                else:
                    base_stance = float(parts[1].strip())
                    base_cred = float(parts[2].strip())
                    url = parts[3].strip()

                final_stance = base_stance
                if any(k in name for k in CAMP_KEYWORDS["GREEN"]):
                    if final_stance > 0: final_stance = final_stance * -1
                    if final_stance == 0: final_stance = -5
                elif any(k in name for k in CAMP_KEYWORDS["BLUE"] + CAMP_KEYWORDS["RED"]):
                    if final_stance < 0: final_stance = final_stance * -1
                    if final_stance == 0: final_stance = 5
                
                data["spectrum"].append({
                    "source": name,
                    "date": date,
                    "title": title,
                    "stance": int(final_stance),
                    "credibility": int(base_cred), 
                    "url": url
                })
            except: pass

    report_split = re.split(r'###\s*\[?REPORT_TEXT\]?', text)
    if len(report_split) > 1:
        data["report_text"] = report_split[-1].strip()
    else:
        data["report_text"] = text

    return data

def render_spectrum_split(spectrum_data):
    if not spectrum_data: return
    
    st.markdown("""
    <div class="legend-box">
        <b>📊 燈號與數值說明：</b><br>
        • <b>政治立場 (Stance)</b>：🟢 負分 (-10 ~ -1) 代表批判/泛綠；🔵 正分 (+1 ~ +10) 代表體制/泛藍；⚪ 0 代表中立。<br>
        • <b>可信度 (Credibility)</b>：🟢 高 (7-10)；🟡 中 (4-6)；🔴 低 (0-3)。
    </div>
    """, unsafe_allow_html=True)
    
    green_list = []
    blue_list = []
    neutral_list = []
    
    for item in spectrum_data:
        if item['stance'] < 0: green_list.append(item)
        elif item['stance'] > 0: blue_list.append(item)
        else: neutral_list.append(item)
        
    green_list.sort(key=lambda x: x['credibility'], reverse=True)
    blue_list.sort(key=lambda x: x['credibility'], reverse=True)
    neutral_list.sort(key=lambda x: x['credibility'], reverse=True)
    
    def make_md_table(items):
        if not items: return "_無相關資料_"
        md = "| 日期 | 媒體 | 新聞標題 (點擊閱讀) | 立場 | 可信度 |\n|:---:|:---|:---|:---:|:---:|\n"
        for i in items:
            s = i['stance']
            if s < 0: s_txt = f"🟢 {s}"
            elif s > 0: s_txt = f"🔵 +{s}"
            else: s_txt = "⚪ 0"
            
            c = i['credibility']
            if c >= 7: c_txt = f"🟢 {c}"
            elif c >= 4: c_txt = f"🟡 {c}"
            else: c_txt = f"🔴 {c}"
            
            t_text = i.get('title', '點擊閱讀報導')
            if len(t_text) > 25: t_text = t_text[:25] + "..."
            t_url = i.get('url', '#')
            t_date = i.get('date', 'Recent')
            
            title_link = f"[{t_text}]({t_url})"
            md += f"| {t_date} | {i['source']} | {title_link} | {s_txt} | {c_txt} |\n"
        return md

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="table-header-green">🟢 泛綠 / 批判陣營 (Green/Critical)</div>', unsafe_allow_html=True)
        st.markdown(make_md_table(green_list))
    with c2:
        st.markdown('<div class="table-header-blue">🔵 泛藍 / 體制陣營 (Blue/Establishment)</div>', unsafe_allow_html=True)
        st.markdown(make_md_table(blue_list))
        
    if neutral_list:
        st.markdown("---")
        st.markdown('<div class="table-header-neutral">⚪ 中立 / 其他觀點 (Neutral/Other)</div>', unsafe_allow_html=True)
        st.markdown(make_md_table(neutral_list))

# 4. 下載功能
def convert_data_to_json(data):
    return json.dumps(data, indent=2, ensure_ascii=False)

def convert_data_to_md(data):
    return f"""
# 全域觀點分析報告
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
    st.title("全域觀點解析 V18.1")
    analysis_mode = st.radio("選擇模式：", options=["🛡️ 輿情光譜 (Spectrum)", "🔮 未來發展推演 (Scenario)"], index=0)
    st.markdown("---")
    
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
            "搜尋時間範圍 (Time Range)",
            options=[3, 7, 14, 30, 90, 1825],
            format_func=lambda x: "📅 不限時間 (All Time)" if x == 1825 else f"近 {x} 天",
            index=2
        )
        
        # [V18.1] 自訂搜尋數量
        max_results = st.slider("搜尋篇數上限", min_value=10, max_value=50, value=20, step=5, help="增加篇數可提升完整度，但會增加等待時間。")
        
        region_mode = st.selectbox(
            "搜尋視角 (Region)",
            ["🇹🇼 台灣限定 (Taiwan Only)", "🌏 亞洲視角 (Asia)", "🌍 歐洲視角 (Europe)", "🌎 美洲視角 (Americas)"]
        )

    with st.expander("🧠 系統邏輯說明 (Transparency)", expanded=False):
        st.markdown("""
        **1. 搜尋優化 (Search Strategy)**
        * **台灣模式**: 啟用「白名單機制」，僅搜尋聯合、自由、中時、中央社等 15 家主流台媒。
        * **日期補救**: 自動修復 Tavily 回傳的空日期欄位。
        
        **2. 政治光譜校正 (Calibration)**
        * **🟢 泛綠/批判區**：自由、三立、民視... (強制負分)
        * **🔵 泛藍/體制區**：中時、聯合、TVBS... (強制正分)
        """)

    with st.expander("📂 匯入舊情報", expanded=False):
        past_report_input = st.text_area("貼上舊報告 Markdown：", height=100)
        
    st.markdown("### 📥 報告匯出")
    if st.session_state.get('spectrum_result') or st.session_state.get('wargame_result'):
        active_data = st.session_state.get('wargame_result') if "Scenario" in analysis_mode else st.session_state.get('spectrum_result')
        if active_data:
            st.download_button("下載 JSON", convert_data_to_json(active_data), "report.json", "application/json")
            st.download_button("下載 Markdown", convert_data_to_md(active_data), "report.md", "text/markdown")

st.title(f"{analysis_mode.split(' ')[1]}")
query = st.text_input("輸入議題關鍵字", placeholder="例如：台積電美國設廠爭議")
search_btn = st.button("🚀 啟動分析引擎", type="primary")

if 'spectrum_result' not in st.session_state: st.session_state.spectrum_result = None
if 'wargame_result' not in st.session_state: st.session_state.wargame_result = None
if 'wargame_opinions' not in st.session_state: st.session_state.wargame_opinions = None
if 'sources' not in st.session_state: st.session_state.sources = None
if 'full_context' not in st.session_state: st.session_state.full_context = ""

if search_btn and query and google_key and tavily_key:
    st.session_state.spectrum_result = None
    st.session_state.wargame_result = None
    st.session_state.wargame_opinions = None
    
    with st.status("🚀 啟動全域掃描引擎 (V18.1)...", expanded=True) as status:
        
        days_label = "不限時間" if search_days == 1825 else f"近 {search_days} 天"
        st.write(f"📡 1. 連線 Tavily 搜尋 (視角: {region_mode} / 時間: {days_label} / 數量: {max_results})...")
        
        # [V18.1] 傳入自訂 max_results
        context_text, sources, actual_query, is_tw_only = get_search_context(query, tavily_key, search_days, region_mode, max_results, past_report_input)
        st.session_state.sources = sources
        
        if is_tw_only:
             st.info(f"🔍 已啟用台灣媒體白名單鎖定 (Whitelist Mode)")
        else:
             st.info(f"🔍 實際搜尋關鍵字: {actual_query}")
        
        st.write("🛡️ 2. 查詢 Cofacts 謠言資料庫 (API)...")
        cofacts_txt = search_cofacts(query)
        if cofacts_txt:
            context_text += f"\n{cofacts_txt}\n"
        st.session_state.full_context = context_text
        
        st.write("🧠 3. AI 進行深度閱讀與分析...")
        
        if "Spectrum" in analysis_mode:
            raw_report = run_spectrum_analysis(query, context_text, model_name, google_key)
            st.session_state.spectrum_result = parse_gemini_data(raw_report)
        else:
            st.write("⚔️ 4. 召開虛擬戰情會議 (多代理人辯論)...")
            opinions, raw_report = run_council_of_rivals(query, context_text, model_name, google_key)
            st.session_state.wargame_opinions = opinions
            st.session_state.wargame_result = parse_gemini_data(raw_report)
            
        status.update(label="✅ 分析完成", state="complete", expanded=False)
        
    st.rerun()

if st.session_state.spectrum_result and "Spectrum" in analysis_mode:
    data = st.session_state.spectrum_result
    
    if data.get("spectrum"):
        st.markdown("### 📊 輿論陣地分析表 (Spectrum Table)")
        render_spectrum_split(data["spectrum"])

    st.markdown("### 📝 媒體識讀報告")
    formatted_text = format_citation_style(data.get("report_text", ""))
    st.markdown(f'<div class="report-paper">{formatted_text}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("覺得議題需要更深度推演？請點擊下方按鈕。")
    if st.button("🚀 基於此情報啟動未來發展推演 (Scenario)", type="primary"):
        if st.session_state.full_context:
            with st.status("⚔️ 進行多視角推演...", expanded=True) as status:
                st.write("1. 啟動虛擬幕僚群...")
                opinions, raw_report = run_council_of_rivals(query, st.session_state.full_context, model_name, google_key)
                st.session_state.wargame_opinions = opinions
                st.session_state.wargame_result = parse_gemini_data(raw_report)
                status.update(label="✅ 推演完成", state="complete", expanded=False)
                st.rerun()

if st.session_state.wargame_result:
    st.divider()
    st.markdown(f"<h2 style='text-align: center;'>⚔️ 未來發展推演：{query}</h2>", unsafe_allow_html=True)
    
    ops = st.session_state.wargame_opinions
    if ops:
        c_a, c_b, c_ctx = st.columns(3)
        with c_a:
            st.markdown(f'<div class="perspective-box box-blue"><b>🔵 體制/現狀視角</b><br>{ops.get("A_SIDE")[:150]}...</div>', unsafe_allow_html=True)
            with st.popover("查看完整論述"): 
                st.markdown(format_citation_style(ops.get("A_SIDE")), unsafe_allow_html=True)
        with c_b:
            st.markdown(f'<div class="perspective-box box-green"><b>🟢 批判/改革視角</b><br>{ops.get("B_SIDE")[:150]}...</div>', unsafe_allow_html=True)
            with st.popover("查看完整論述"): 
                st.markdown(format_citation_style(ops.get("B_SIDE")), unsafe_allow_html=True)
        with c_ctx:
            st.markdown(f'<div class="perspective-box box-neutral"><b>📜 深層脈絡分析</b><br>{ops.get("CONTEXT")[:150]}...</div>', unsafe_allow_html=True)
            with st.popover("查看完整論述"): 
                st.markdown(format_citation_style(ops.get("CONTEXT")), unsafe_allow_html=True)

    data_wg = st.session_state.wargame_result
    
    if data_wg.get("mermaid"):
        st.markdown("### 🕸️ 系統因果迴路圖 (System Dynamics)")
        st.markdown('<div class="mermaid-box">', unsafe_allow_html=True)
        render_mermaid(data_wg["mermaid"])
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ 系統未能生成有效的因果圖代碼。")

    st.markdown("### 📝 總編輯深度決策報告")
    formatted_report = format_citation_style(data_wg.get("report_text", ""))
    st.markdown(f'<div class="report-paper">{formatted_report}</div>', unsafe_allow_html=True)

if st.session_state.sources:
    st.markdown("---")
    st.markdown("### 📚 引用文獻列表")
    md_table = "| 編號 | 媒體/網域 | 標題摘要 | 連結 |\n|:---:|:---|:---|:---|\n"
    for i, s in enumerate(st.session_state.sources):
        domain = get_domain_name(s.get('url'))
        title = s.get('title', 'No Title')
        if len(title) > 60: title = title[:60] + "..."
        url = s.get('url')
        md_table += f"| **{i+1}** | `{domain}` | {title} | [點擊]({url}) |\n"
    st.markdown(md_table)
