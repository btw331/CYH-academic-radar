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
# 1. 基礎設定與 CSS樣式 (融合舊版美學)
# ==========================================
st.set_page_config(page_title="全域觀點解析 V19.0", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    .stButton button[kind="secondary"] { border: 2px solid #673ab7; color: #673ab7; font-weight: bold; }
    
    /* 舊版指標卡片樣式 - 回歸！ */
    .metric-container {
        text-align: center; padding: 10px; background-color: #ffffff;
        border-radius: 8px; border: 1px solid #f0f0f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); transition: transform 0.2s;
        margin-bottom: 10px;
    }
    .metric-container:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .metric-score { font-size: 1.8em; font-weight: 700; margin: 0; line-height: 1.2; color: #1565c0; }
    .metric-label { font-size: 0.9em; font-weight: 500; margin-top: 5px; color: #666; }

    /* 報告紙張風格 */
    .report-paper {
        background-color: #fdfbf7; color: #2c3e50; padding: 30px; 
        border-radius: 4px; margin-bottom: 15px; border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        font-family: "Georgia", "Cambria", "Times New Roman", serif;
        line-height: 1.8; font-size: 1.05rem;
    }
    
    .citation {
        font-size: 0.85em; color: #757575; background-color: #f0f0f0;
        padding: 2px 6px; border-radius: 4px; margin: 0 2px;
        font-family: sans-serif; border: 1px solid #e0e0e0; font-weight: 500;
    }

    .table-header-green { color: #2e7d32; font-weight: bold; font-size: 1.1em; border-bottom: 2px solid #2e7d32; margin-bottom: 10px; padding-bottom: 5px; }
    .table-header-blue { color: #1565c0; font-weight: bold; font-size: 1.1em; border-bottom: 2px solid #1565c0; margin-bottom: 10px; padding-bottom: 5px; }
    
    .mermaid-box { background-color: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 資料庫擴充 (整合舊版 DB_MAP)
# ==========================================
# 擴充後的台灣白名單 (含網媒)
TAIWAN_WHITELIST = [
    "udn.com", "ltn.com.tw", "chinatimes.com", "cna.com.tw", 
    "storm.mg", "setn.com", "ettoday.net", "tvbs.com.tw", 
    "mirrormedia.mg", "thenewslens.com", "upmedia.mg", 
    "rwnews.tw", "news.pts.org.tw", "ctee.com.tw", "businessweekly.com.tw",
    "news.yahoo.com.tw", "twreporter.org", "theinitium.com", "mindiworldnews.com", "vocus.cc"
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

# V18.2 搜尋核心 + V19 白名單擴充
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
        is_strict_taiwan = False
        
        # 區域判斷邏輯
        if len(selected_regions) == 1 and "台灣" in selected_regions[0]:
            is_strict_taiwan = True
            suffixes.append("台灣 新聞" if is_chinese(query) else "Taiwan News")
        else:
            for r in selected_regions:
                if "台灣" in r: suffixes.append("台灣 新聞")
                if "亞洲" in r: suffixes.append("Asia News")
                if "歐洲" in r: suffixes.append("Europe News")
                if "美洲" in r: suffixes.append("US Americas News")
        
        if not suffixes: suffixes.append("News")
        search_q = f"{query} {' '.join(suffixes)}"
        if context_report: search_q += " analysis"
        
        search_params["query"] = search_q

        if is_strict_taiwan:
            search_params["include_domains"] = TAIWAN_WHITELIST
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
        
        if context_report:
            context_text += f"【歷史背景】\n{context_report[:800]}...\n\n"
            
        context_text += "【最新網路情報】(請嚴格使用 [Source ID] 引用)\n"
        
        for i, res in enumerate(results):
            title = res.get('title', 'No Title')
            url = res.get('url', '#')
            pub_date = res.get('published_date', '')
            if not pub_date: pub_date = "Recent"
            else: pub_date = pub_date[:10]
            content = res.get('content', '')[:800]
            context_text += f"Source {i+1}: [Date: {pub_date}] [Title: {title}] {content} (URL: {url})\n"
            
        return context_text, results, actual_query, is_strict_taiwan
        
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

# 3.3 核心邏輯：數位戰情室 (融合舊版未來學架構)
def run_council_of_rivals(query, context_text, model_name, api_key):
    prompts = {
        "A_SIDE": "你是一位【體制內/現狀分析師】。請找出支持現狀、政策合理性或官方解釋的證據。必須引用來源 [Source ID]。",
        "B_SIDE": "你是一位【改革/批判派分析師】。請找出質疑現狀、結構性問題或反對意見的證據。必須引用來源 [Source ID]。",
        "CONTEXT": "你是一位【脈絡歷史學家】。請分析爭議背後的深層歷史成因、經濟結構或地緣政治因素。必須引用來源 [Source ID]。",
        "FUTURIST": "你是一位【未來趨勢預測師】。請應用第一性原理與可能性圓錐，推演三種未來情境：基準(Baseline)、轉折(Plausible)、極端(Wild Card)。"
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
    未來推演: {opinions.get('FUTURIST')}
    
    【任務指令】：
    1. **整合分析**：請融合上述觀點，產出一份結構完整的報告。
    2. **Mermaid 製圖**：請生成 Mermaid `graph TD` 代碼，展示因果鏈。
    
    【輸出格式】：
    ### [REPORT_TEXT]
    (Markdown 報告內容，請包含「🔮 未來情境模擬」章節)
    """
    final_report = call_gemini(editor_prompt, context_text, model_name, api_key)
    return opinions, final_report

# 3.4 核心邏輯：輿情光譜 (V18.2 版本)
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
    (YYYY-MM-DD|媒體|事件標題)
    
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
        
        if "|" in line and len(line.split("|")) >= 3 and (line[0].isdigit() or "20" in line):
            parts = line.split("|")
            if len(parts) == 3: 
                data["timeline"].append({"date": parts[0].strip(), "media": parts[1].strip(), "event": parts[2].strip()})
            
        if "|" in line and len(line.split("|")) >= 4 and not line.startswith("###") and not "YYYY" in line:
            parts = line.split("|")
            try:
                name = parts[0].strip()
                date = "Recent" 
                title = "點擊閱讀報導"
                base_stance = 0
                base_cred = 0
                url = "#"
                
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

# [V19.0] 渲染表格 (支援盲測模式)
def render_spectrum_split(spectrum_data, blind_mode):
    if not spectrum_data: return
    
    # 復刻舊版卡片風格的指標顯示 (Mockup)
    c1, c2, c3 = st.columns(3)
    avg_cred = sum(i['credibility'] for i in spectrum_data) / len(spectrum_data) if spectrum_data else 0
    polarization = len([i for i in spectrum_data if abs(i['stance']) > 5])
    
    with c1: st.markdown(f'<div class="metric-container"><p class="metric-score" style="color:#2e7d32">{len(spectrum_data)}</p><p class="metric-label">分析篇數</p></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-container"><p class="metric-score" style="color:#1565c0">{avg_cred:.1f}</p><p class="metric-label">平均可信度</p></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-container"><p class="metric-score" style="color:#d32f2f">{polarization}</p><p class="metric-label">高對立文章</p></div>', unsafe_allow_html=True)

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
            
            # [V19.0] 盲測模式處理
            display_source = "*****" if blind_mode else i['source']
            
            title_link = f"[{t_text}]({t_url})"
            md += f"| {t_date} | {display_source} | {title_link} | {s_txt} | {c_txt} |\n"
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

def render_timeline_markdown(timeline_data):
    if not timeline_data: return
    md = "| 日期 | 媒體 | 事件/標題 |\n|:---:|:---|:---|\n"
    for item in timeline_data:
        md += f"| {item.get('date','')} | {item.get('media','')} | {item.get('event','')} |\n"
    st.markdown(md)

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
    st.title("全域觀點解析 V19.0")
    analysis_mode = st.radio("選擇模式：", options=["🛡️ 輿情光譜 (Spectrum)", "🔮 未來發展推演 (Scenario)"], index=0)
    st.markdown("---")
    
    # [V19.0] 恢復盲測模式
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
            "搜尋時間範圍 (Time Range)",
            options=[3, 7, 14, 30, 90, 1825],
            format_func=lambda x: "📅 不限時間 (All Time)" if x == 1825 else f"近 {x} 天",
            index=2
        )
        
        max_results = st.slider("搜尋篇數上限", 10, 50, 20)
        selected_regions = st.multiselect(
            "搜尋視角 (Region) - 可複選",
            ["🇹🇼 台灣 (Taiwan)", "🌏 亞洲 (Asia)", "🌍 歐洲 (Europe)", "🌎 美洲 (Americas)"],
            default=["🇹🇼 台灣 (Taiwan)"]
        )

    with st.expander("🧠 系統邏輯說明", expanded=False):
        st.markdown("""
        **1. 搜尋優化**
        * **台灣模式**: 啟用擴充版白名單 (含數位網媒)。
        * **盲測模式**: 遮蔽來源，專注內容。
        
        **2. 未來推演 (Scenario)**
        * 引入舊版「第一性原理」與「可能性圓錐」架構。
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
    
    with st.status("🚀 啟動全域掃描引擎 (V19.0)...", expanded=True) as status:
        
        days_label = "不限時間" if search_days == 1825 else f"近 {search_days} 天"
        regions_label = ", ".join([r.split(" ")[1] for r in selected_regions])
        st.write(f"📡 1. 連線 Tavily 搜尋 (視角: {regions_label} / 時間: {days_label})...")
        
        context_text, sources, actual_query, is_strict_tw = get_search_context(query, tavily_key, search_days, selected_regions, max_results, past_report_input)
        st.session_state.sources = sources
        
        if is_strict_tw:
             st.info(f"🔍 已啟用擴充版台灣媒體白名單 (Enhanced Whitelist)")
        else:
             st.info(f"🔍 混選模式：啟用垃圾過濾 (Smart Blacklist)")
        
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
            st.write("⚔️ 4. 召開虛擬戰情會議 (加入未來學推演)...")
            opinions, raw_report = run_council_of_rivals(query, context_text, model_name, google_key)
            st.session_state.wargame_opinions = opinions
            st.session_state.wargame_result = parse_gemini_data(raw_report)
            
        status.update(label="✅ 分析完成", state="complete", expanded=False)
        
    st.rerun()

if st.session_state.spectrum_result and "Spectrum" in analysis_mode:
    data = st.session_state.spectrum_result
    
    # [V19.0] 傳入盲測狀態
    if data.get("spectrum"):
        st.markdown("### 📊 輿論陣地分析表 (Spectrum Table)")
        render_spectrum_split(data["spectrum"], blind_mode)
    
    if data.get("timeline"):
        st.markdown("### 📅 議題發展時間軸 (News Timeline)")
        render_timeline_markdown(data["timeline"])

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
        # 配合盲測模式隱藏來源
        if blind_mode: domain = "*****"
        
        title = s.get('title', 'No Title')
        if len(title) > 60: title = title[:60] + "..."
        url = s.get('url')
        md_table += f"| **{i+1}** | `{domain}` | {title} | [點擊]({url}) |\n"
    st.markdown(md_table)
