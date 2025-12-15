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
import random
import concurrent.futures
from urllib.parse import urlparse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
import plotly.express as px
import streamlit.components.v1 as components

# ==========================================
# 1. 基礎設定與 CSS樣式
# ==========================================
st.set_page_config(page_title="全域觀點解析 V15", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    .stButton button[kind="secondary"] { border: 2px solid #673ab7; color: #673ab7; font-weight: bold; }
    
    /* 報告區塊風格 - 紙張質感 */
    .report-paper {
        background-color: #fdfbf7; 
        color: #2c3e50; 
        padding: 30px; 
        border-radius: 4px; 
        margin-bottom: 15px; 
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        font-family: "Georgia", serif;
        line-height: 1.8;
    }
    
    /* 觀點對照盒 */
    .perspective-box {
        padding: 15px; border-radius: 8px; margin-bottom: 10px; font-size: 0.95em;
        border-left-width: 4px; border-left-style: solid;
    }
    .box-green { background-color: #e8f5e9; border-left-color: #2e7d32; color: #1b5e20; }
    .box-blue { background-color: #e3f2fd; border-left-color: #1565c0; color: #0d47a1; }
    .box-neutral { background-color: #f5f5f5; border-left-color: #616161; color: #424242; }
    
    .mermaid-box {
        background-color: #ffffff; padding: 10px; border-radius: 8px; border: 1px solid #ddd; margin-top: 10px;
    }
    
    /* 標題樣式 */
    .section-title {
        font-size: 1.3em; font-weight: bold; color: #37474f; margin-top: 20px; margin-bottom: 10px; border-bottom: 2px solid #eceff1; padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 資料庫與共用常數
# ==========================================
DB_MAP = {
    "CHINA": ["xinhuanet.com", "people.com.cn", "huanqiu.com", "cctv.com", "chinadaily.com.cn", "cgtn.com", "taiwan.cn", "gwytb.gov.cn", "guancha.cn", "thepaper.cn"],
    "INTL": ["reuters.com", "apnews.com", "bloomberg.com", "wsj.com", "ft.com", "economist.com", "bbc.com", "dw.com", "voanews.com", "thediplomat.com"],
    "OFFICIAL": ["cna.com.tw", "pts.org.tw", "mnd.gov.tw", "indsr.org.tw", "tfc-taiwan.org.tw", "mygopen.com"],
    "GREEN": ["ltn.com.tw", "ftvnews.com.tw", "setn.com", "newtalk.tw", "mirrormedia.mg"],
    "BLUE": ["udn.com", "chinatimes.com", "tvbs.com.tw", "cti.com.tw", "ctee.com.tw"],
}
NAME_KEYWORDS = { "CHINA": ["新華", "人民", "環球"], "GREEN": ["自由", "三立", "民視"], "BLUE": ["聯合", "中時", "TVBS"] }

def get_domain_name(url):
    try: return urlparse(url).netloc.replace("www.", "")
    except: return ""

def classify_media_name(name):
    n = name.lower()
    for cat, keywords in NAME_KEYWORDS.items():
        if any(k in n for k in keywords): return cat
    return "OTHER"

def get_category_meta(cat):
    meta = {
        "CHINA": ("🇨🇳 中國官媒", "#d32f2f"), "BLUE": ("🔵 泛藍觀點", "#1565c0"),
        "GREEN": ("🟢 泛綠觀點", "#2e7d32"), "OFFICIAL": ("⚪ 官方/中立", "#546e7a"),
        "INTL": ("🌏 國際媒體", "#f57c00"), "OTHER": ("📄 其他來源", "#9e9e9e")
    }
    return meta.get(cat, ("其他", "#9e9e9e"))

# ==========================================
# 3. 核心功能模組
# ==========================================

# 3.1 基礎工具：搜尋與 Cofacts
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
        response = requests.post(url, json={'query': graphql_query, 'variables': {'text': query}}, timeout=5)
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

def get_search_context(query, api_key_tavily, context_report=None):
    os.environ["TAVILY_API_KEY"] = api_key_tavily
    search = TavilySearchResults(max_results=15)
    
    search_q = f"{query} 2025 最新發展"
    if context_report:
        search_q += " analysis"
    
    try:
        results = search.invoke(search_q)
        context_text = ""
        
        # 加 Cofacts
        cofacts_txt = search_cofacts(query)
        if cofacts_txt: context_text += f"{cofacts_txt}\n{'-'*20}\n"
        
        if context_report:
            context_text += f"【歷史背景摘要】\n{context_report[:1000]}...\n\n"
            
        context_text += "【最新網路情報】\n"
        for i, res in enumerate(results):
            context_text += f"Source {i+1}: {res.get('url')} | {str(res.get('content'))[:1000]}\n"
            
        return context_text, results, cofacts_txt
    except Exception as e:
        return f"Error: {str(e)}", [], ""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def call_gemini(system_prompt, user_text, model_name, api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    chain = prompt | llm
    return chain.invoke({"input": user_text}).content

# 3.2 進階功能：Mermaid 渲染器
def render_mermaid(code):
    html_code = f"""
    <div class="mermaid">
    {code}
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
    </script>
    """
    components.html(html_code, height=400, scrolling=True)

# 3.3 核心邏輯：數位戰情室 (Council of Rivals)
def run_council_of_rivals(query, context_text, model_name, api_key):
    # 定義三個 Agent 的人設 (平衡報導風格)
    prompts = {
        "A_SIDE": """你是一位【視角 A 分析師】(通常代表既有體制/官方/保守觀點)。
        你的任務：分析現行政策的合理性、強調穩定與秩序、指出改變可能帶來的風險。
        請從情報中找出支持「維持現狀」或「官方立場」的論述證據。""",
        
        "B_SIDE": """你是一位【視角 B 分析師】(通常代表挑戰者/改革/批判觀點)。
        你的任務：分析現狀的結構性問題、強調改變的必要性、指出官方論述的盲點。
        請從情報中找出支持「質疑現狀」或「反方立場」的論述證據。""",
        
        "CONTEXT": """你是一位【脈絡分析師】(Contextualizer)。
        你的任務：不選邊站，而是分析「為什麼現在會吵這個？」。
        請從歷史背景、經濟結構、或國際局勢的角度，解釋這個爭議發生的深層原因。"""
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

    # 綜合研判：總編輯 (Editor in Chief)
    editor_prompt = f"""
    你是一位堅持「平衡報導」的資深總編輯。
    你收到了三份針對「{query}」的分析稿件。
    
    【視角 A】: {opinions.get('A_SIDE')}
    【視角 B】: {opinions.get('B_SIDE')}
    【深層脈絡】: {opinions.get('CONTEXT')}
    
    請產出一份「深度全解讀」報告，包含：
    1. **核心爭議點**：雙方到底在吵什麼？(Key Conflicts)
    2. **資訊落差 (Information Gap)**：雙方各自隱瞞或忽略了什麼？
    3. **因果迴路圖 (Mermaid)**：生成一段 Mermaid `graph TD` 代碼，展示事件的因果關係。請將代碼包在 ```mermaid ... ``` 區塊中。
    4. **未來情境推演**：若爭議持續，可能發展出的 3 種走向。
    
    【輸出格式】：
    ### [REPORT_TEXT]
    (Markdown 報告內容...)
    """
    
    final_report = call_gemini(editor_prompt, context_text, model_name, api_key)
    return opinions, final_report

# 3.4 核心邏輯：輿情光譜 (原 V13 功能，移除分數)
def run_spectrum_analysis(query, context_text, model_name, api_key):
    system_prompt = f"""
    你是一位媒體識讀專家。請針對「{query}」進行媒體框架分析 (Framing Analysis)。
    
    【任務】：
    1. 識別每個來源的「敘事立場」(支持哪一方?) 與 「可信度」。
    2. **重要**：請給出具體座標，以便繪製光譜圖。
       - 立場 (X軸): -10(強烈反對/批判/A方) <-> 0(中立/事實描述) <-> 10(強烈支持/護航/B方)
       - 可信度 (Y軸): 0(內容農場/謠言) <-> 10(權威機構/數據詳實)
    
    【輸出格式】：
    ### [DATA_TIMELINE]
    YYYY-MM-DD|媒體|標題
    
    ### [DATA_SPECTRUM]
    來源|立場|可信度|網址
    
    ### [REPORT_TEXT]
    (Markdown 報告)
    請包含：
    1. **📊 全域現況摘要** (含 Cofacts 查核結果)
    2. **⚖️ 媒體框架分析** (不同媒體如何「包裝」這個事件？)
    3. **🧠 識讀建議** (民眾該如何解讀這些資訊？)
    """
    return call_gemini(system_prompt, context_text, model_name, api_key)

# 3.5 資料解析器 (加入 Jitter 機制)
def parse_gemini_data(text):
    data = {"timeline": [], "spectrum": [], "mermaid": "", "report_text": ""}
    
    # 提取 Mermaid
    mermaid_match = re.search(r"```mermaid\n(.*?)\n```", text, re.DOTALL)
    if mermaid_match:
        data["mermaid"] = mermaid_match.group(1)
        text = text.replace(mermaid_match.group(0), "")

    for line in text.split('\n'):
        line = line.strip()
        
        # Parse Timeline
        if "|" in line and len(line.split("|")) >= 3 and (line[0].isdigit() or "Future" in line):
            parts = line.split("|")
            data["timeline"].append({"date": parts[0], "media": parts[1], "event": parts[2]})
            
        # Parse Spectrum (加入 Jitter 防止重疊)
        if "|" in line and len(line.split("|")) >= 4 and not line.startswith("###") and not "日期" in line:
            parts = line.split("|")
            try:
                # [V15 UPDATE] Jitter Logic: Add random noise to separate dots
                base_stance = float(parts[1])
                base_cred = float(parts[2])
                
                # 加入 -0.5 ~ 0.5 的隨機擾動
                jitter_x = random.uniform(-0.6, 0.6)
                jitter_y = random.uniform(-0.4, 0.4)
                
                data["spectrum"].append({
                    "source": parts[0], 
                    "stance": base_stance + jitter_x, 
                    "credibility": base_cred + jitter_y, 
                    "url": parts[3]
                })
            except: pass

    # 提取報告本文
    if "### [REPORT_TEXT]" in text:
        data["report_text"] = text.split("### [REPORT_TEXT]")[1].strip()
    else:
        data["report_text"] = text 

    return data

def render_spectrum_chart(spectrum_data):
    if not spectrum_data: return None
    df = pd.DataFrame(spectrum_data)
    
    # [V15 UPDATE] 優化圖表設計
    fig = px.scatter(
        df, 
        x="stance", 
        y="credibility", 
        hover_name="source", 
        text="source", 
        size=[20]*len(df), # 點變大
        color="stance", 
        color_continuous_scale=["#2e7d32", "#eeeeee", "#1565c0"], # 綠 -> 白 -> 藍 (更柔和)
        range_x=[-12, 12], # 擴大範圍讓點不要貼邊
        range_y=[-1, 12],
        opacity=0.85, # 透明度
        labels={"stance": "觀點光譜 (左:批判/反方 --- 右:支持/正方)", "credibility": "資訊可信度"}
    )
    
    # 象限背景
    fig.add_shape(type="rect", x0=-12, y0=6, x1=0, y1=12, fillcolor="rgba(46, 125, 50, 0.05)", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=6, x1=12, y1=12, fillcolor="rgba(21, 101, 192, 0.05)", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=-12, y0=-1, x1=12, y1=5, fillcolor="rgba(255, 167, 38, 0.05)", layer="below", line_width=0) # 低可信區
    
    fig.update_layout(
        xaxis_title="◀ 觀點 A (批判/改革) --------- 中立 --------- 觀點 B (支持/體制) ▶",
        yaxis_title="資訊品質 (低 -> 高)",
        showlegend=False,
        height=550,
        font=dict(size=14)
    )
    fig.update_traces(textposition='top center', textfont_size=12)
    return fig

# ==========================================
# 4. 介面 (UI)
# ==========================================
with st.sidebar:
    st.title("全域觀點解析 V15")
    
    # 模式選擇
    analysis_mode = st.radio(
        "選擇分析模式：",
        options=["🛡️ 輿情光譜 (Spectrum)", "🔮 深度戰情室 (Deep Dive)"],
        captions=["即時：媒體框架 + 查核", "深度：多視角辯證 + 系統思考"],
        index=0
    )
    
    st.markdown("---")
    
    # Secrets 管理
    with st.expander("🔑 系統權限", expanded=True):
        if "GOOGLE_API_KEY" in st.secrets:
            st.success("✅ Gemini Ready")
            google_key = st.secrets["GOOGLE_API_KEY"]
        else:
            google_key = st.text_input("Gemini Key", type="password")

        if "TAVILY_API_KEY" in st.secrets:
            st.success("✅ Tavily Ready")
            tavily_key = st.secrets["TAVILY_API_KEY"]
        else:
            tavily_key = st.text_input("Tavily Key", type="password")
            
        model_name = st.selectbox("模型", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)

    with st.expander("📂 匯入舊情報 (滾動分析)", expanded=False):
        past_report_input = st.text_area("貼上舊 Markdown 報告：", height=100)

# 主畫面
st.title(f"{analysis_mode.split(' ')[1]}")
query = st.text_input("輸入議題關鍵字", placeholder="例如：台積電美國設廠爭議")
search_btn = st.button("🚀 啟動分析引擎", type="primary")

# Session State 初始化
if 'spectrum_result' not in st.session_state: st.session_state.spectrum_result = None
if 'wargame_result' not in st.session_state: st.session_state.wargame_result = None
if 'wargame_opinions' not in st.session_state: st.session_state.wargame_opinions = None
if 'sources' not in st.session_state: st.session_state.sources = None
if 'full_context' not in st.session_state: st.session_state.full_context = ""

# 1. 執行第一階段：輿情搜尋
if search_btn and query and google_key and tavily_key:
    st.session_state.spectrum_result = None
    st.session_state.wargame_result = None
    st.session_state.wargame_opinions = None
    
    with st.spinner("📡 正在進行全網情報蒐集 (Tavily + Cofacts)..."):
        context_text, sources, cofacts_txt = get_search_context(query, tavily_key, past_report_input)
        st.session_state.sources = sources
        st.session_state.full_context = context_text
        
        # 執行光譜分析 (預設先跑這個)
        if "Spectrum" in analysis_mode:
            raw_report = run_spectrum_analysis(query, context_text, model_name, google_key)
            st.session_state.spectrum_result = parse_gemini_data(raw_report)
        else:
            # 直接跑戰情室
            with st.status("⚔️ 召開多視角分析會議...", expanded=True) as status:
                st.write("1. 正在傳喚不同觀點分析師...")
                opinions, raw_report = run_council_of_rivals(query, context_text, model_name, google_key)
                st.session_state.wargame_opinions = opinions
                st.session_state.wargame_result = parse_gemini_data(raw_report)
                status.update(label="✅ 分析完成", state="complete", expanded=False)
    
    st.rerun() # 強制刷新 UI

# 2. 顯示結果：輿情光譜模式
if st.session_state.spectrum_result and "Spectrum" in analysis_mode:
    data = st.session_state.spectrum_result
    
    # 光譜圖 (優化版)
    if data.get("spectrum"):
        st.markdown("### 🗺️ 輿論陣地光譜 (AI 動態識別)")
        st.caption("透過 Jitter 技術分散重疊點，X軸代表觀點傾向，Y軸代表資訊詳實度。")
        fig = render_spectrum_chart(data["spectrum"])
        st.plotly_chart(fig, use_container_width=True)

    # 分析報告
    st.markdown("### 📝 媒體識讀報告")
    st.markdown(f'<div class="report-paper">{data.get("report_text")}</div>', unsafe_allow_html=True)
    
    # 參考來源
    with st.expander("📚 原始情報來源列表"):
        if st.session_state.sources:
            for s in st.session_state.sources:
                st.markdown(f"- [{s.get('url')}]({s.get('url')})")

    st.markdown("---")
    
    # 轉接戰情室按鈕
    st.markdown("### 🔮 深度透視")
    st.info("覺得議題太複雜？啟動「深度戰情室」進行多視角辯證與因果分析。")
    
    if st.button("🚀 基於此情報啟動深度戰情室 (Deep Dive)", type="primary", use_container_width=True):
        if st.session_state.full_context:
            with st.status("⚔️ 正在召集數位幕僚...", expanded=True) as status:
                st.write("1. 傳送情報給三位分析師進行平行辯論...")
                opinions, raw_report = run_council_of_rivals(query, st.session_state.full_context, model_name, google_key)
                st.session_state.wargame_opinions = opinions
                st.session_state.wargame_result = parse_gemini_data(raw_report)
                status.update(label="✅ 推演完成", state="complete", expanded=False)
        else:
            st.error("❌ 找不到情報上下文，請先執行搜尋。")

# 3. 顯示結果：戰情室模式 (Deep Dive)
if st.session_state.wargame_result and st.session_state.wargame_opinions:
    st.divider()
    st.markdown(f"<h2 style='text-align: center;'>⚔️ 深度戰情室：{query}</h2>", unsafe_allow_html=True)
    
    # 幕僚辯論
    st.markdown("### 🗣️ 多視角觀點交鋒")
    ops = st.session_state.wargame_opinions
    c_a, c_b, c_ctx = st.columns(3)
    with c_a:
        st.markdown(f'<div class="perspective-box box-blue"><b>🔵 視角 A (現狀/體制)</b><br>{ops.get("A_SIDE")[:200]}...</div>', unsafe_allow_html=True)
        with st.popover("查看完整論述"): st.markdown(ops.get("A_SIDE"))
    with c_b:
        st.markdown(f'<div class="perspective-box box-green"><b>🟢 視角 B (挑戰/改革)</b><br>{ops.get("B_SIDE")[:200]}...</div>', unsafe_allow_html=True)
        with st.popover("查看完整論述"): st.markdown(ops.get("B_SIDE"))
    with c_ctx:
        st.markdown(f'<div class="perspective-box box-neutral"><b>📜 深層脈絡</b><br>{ops.get("CONTEXT")[:200]}...</div>', unsafe_allow_html=True)
        with st.popover("查看完整論述"): st.markdown(ops.get("CONTEXT"))

    # Mermaid 圖表
    data_wg = st.session_state.wargame_result
    if data_wg.get("mermaid"):
        st.markdown("### 🕸️ 系統因果迴路圖 (System Dynamics)")
        render_mermaid(data_wg["mermaid"])

    # 最終報告
    st.markdown("### 📝 總編輯深度全解讀")
    st.markdown(f'<div class="report-paper">{data_wg.get("report_text")}</div>', unsafe_allow_html=True)
