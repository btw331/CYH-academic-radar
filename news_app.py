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
st.set_page_config(page_title="全域觀點解析 V15.1", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    .stButton button[kind="secondary"] { border: 2px solid #673ab7; color: #673ab7; font-weight: bold; }
    
    /* 報告區塊風格 */
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
    
    /* 引用標記樣式 */
    .report-paper code {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 2px 4px;
        border-radius: 4px;
        font-size: 0.9em;
        font-family: monospace;
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
        background-color: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 資料庫與共用常數
# ==========================================
NAME_KEYWORDS = { "CHINA": ["新華", "人民", "環球"], "GREEN": ["自由", "三立", "民視"], "BLUE": ["聯合", "中時", "TVBS"] }

def get_domain_name(url):
    try: return urlparse(url).netloc.replace("www.", "")
    except: return ""

def classify_media_name(name):
    n = name.lower()
    for cat, keywords in NAME_KEYWORDS.items():
        if any(k in n for k in keywords): return cat
    return "OTHER"

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
    if context_report: search_q += " analysis"
    
    try:
        results = search.invoke(search_q)
        context_text = ""
        
        cofacts_txt = search_cofacts(query)
        if cofacts_txt: context_text += f"{cofacts_txt}\n{'-'*20}\n"
        
        if context_report:
            context_text += f"【歷史背景】\n{context_report[:1000]}...\n\n"
            
        context_text += "【最新網路情報】(請嚴格使用 [Source ID] 引用)\n"
        for i, res in enumerate(results):
            # [V15.1] 強調 Source ID
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

# 3.2 [V15.1 修復] Mermaid 渲染器與清洗器
def sanitize_mermaid_code(code):
    """修復常見的 Mermaid 語法錯誤"""
    lines = code.split('\n')
    clean_lines = []
    if not any(l.strip().startswith('graph') for l in lines):
        clean_lines.append("graph TD")
    
    for line in lines:
        # 移除 markdown 標記
        line = line.replace("```mermaid", "").replace("```", "")
        # 修復節點名稱包含括號但未加引號的問題 (簡單版)
        # 例如: A(開始) -> A["開始"]
        if "(" in line and ")" in line and '"' not in line:
            # 這是個粗略的修復，對於簡單圖表有效
            line = line.replace("(", '["').replace(")", '"]')
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
    components.html(html_code, height=500, scrolling=True)

# 3.3 核心邏輯：數位戰情室
def run_council_of_rivals(query, context_text, model_name, api_key):
    prompts = {
        "A_SIDE": "你是一位【官方/體制派分析師】。請找出支持現狀、政策合理性或官方解釋的證據。必須引用來源 [Source ID]。",
        "B_SIDE": "你是一位【批判/改革派分析師】。請找出質疑現狀、結構性問題或反對意見的證據。必須引用來源 [Source ID]。",
        "CONTEXT": "你是一位【脈絡分析師】。請分析爭議背後的歷史成因、經濟結構或地緣政治因素。必須引用來源 [Source ID]。"
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

    # [V15.1 更新 Prompt] 強制引用標註與 Mermaid 格式
    editor_prompt = f"""
    你是一位堅持「平衡報導」的總編輯。針對「{query}」，請產出一份深度全解讀。
    
    【輸入素材】：
    A觀點: {opinions.get('A_SIDE')}
    B觀點: {opinions.get('B_SIDE')}
    脈絡: {opinions.get('CONTEXT')}
    
    【任務指令】：
    1. **嚴格引用**：報告中的每一個論點，都必須標註來源編號，格式為 `[Source X]`。如果沒有來源，請勿瞎編。
    2. **Mermaid 製圖**：請生成 Mermaid `graph TD` 代碼，展示「變數 A 如何導致 變數 B」的因果鏈。
       - 節點名稱請盡量簡短，例如 `A[經濟制裁]`。
       - 節點內若有標點符號，請務必使用引號，例如 `B["民怨(高漲)"]`。
       - 代碼請包在 ```mermaid ... ``` 區塊中。
    
    【輸出格式】：
    ### [REPORT_TEXT]
    (Markdown 報告內容...)
    """
    
    final_report = call_gemini(editor_prompt, context_text, model_name, api_key)
    return opinions, final_report

# 3.4 核心邏輯：輿情光譜
def run_spectrum_analysis(query, context_text, model_name, api_key):
    system_prompt = f"""
    媒體識讀專家請注意：針對「{query}」進行框架分析。
    
    【引用要求】：報告內文請務必標註 `[Source X]`。
    
    【輸出格式】：
    ### [DATA_TIMELINE]
    YYYY-MM-DD|媒體|標題
    
    ### [DATA_SPECTRUM]
    來源|立場(-10~10)|可信度(0~10)|網址
    
    ### [REPORT_TEXT]
    (Markdown 報告，需包含引用)
    """
    return call_gemini(system_prompt, context_text, model_name, api_key)

# 3.5 資料解析器
def parse_gemini_data(text):
    data = {"timeline": [], "spectrum": [], "mermaid": "", "report_text": ""}
    
    # 提取 Mermaid
    mermaid_match = re.search(r"```mermaid\n(.*?)\n```", text, re.DOTALL)
    if mermaid_match:
        data["mermaid"] = mermaid_match.group(1)
        text = text.replace(mermaid_match.group(0), "")

    for line in text.split('\n'):
        line = line.strip()
        if "|" in line and len(line.split("|")) >= 3 and (line[0].isdigit() or "Future" in line):
            parts = line.split("|")
            data["timeline"].append({"date": parts[0], "media": parts[1], "event": parts[2]})
            
        if "|" in line and len(line.split("|")) >= 4 and not line.startswith("###") and not "日期" in line:
            parts = line.split("|")
            try:
                # Jitter
                base_stance = float(parts[1])
                base_cred = float(parts[2])
                jitter_x = random.uniform(-0.6, 0.6)
                jitter_y = random.uniform(-0.4, 0.4)
                data["spectrum"].append({
                    "source": parts[0], "stance": base_stance + jitter_x, 
                    "credibility": base_cred + jitter_y, "url": parts[3]
                })
            except: pass

    if "### [REPORT_TEXT]" in text:
        data["report_text"] = text.split("### [REPORT_TEXT]")[1].strip()
    else:
        data["report_text"] = text 

    return data

def render_spectrum_chart(spectrum_data):
    if not spectrum_data: return None
    df = pd.DataFrame(spectrum_data)
    fig = px.scatter(
        df, x="stance", y="credibility", hover_name="source", text="source", size=[20]*len(df),
        color="stance", color_continuous_scale=["#2e7d32", "#eeeeee", "#1565c0"],
        range_x=[-12, 12], range_y=[-1, 12], opacity=0.85,
        labels={"stance": "觀點光譜", "credibility": "資訊可信度"}
    )
    # Backgrounds
    fig.add_shape(type="rect", x0=-12, y0=6, x1=0, y1=12, fillcolor="rgba(46, 125, 50, 0.05)", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=6, x1=12, y1=12, fillcolor="rgba(21, 101, 192, 0.05)", layer="below", line_width=0)
    fig.update_layout(xaxis_title="◀ 反方/批判 --- 中立 --- 正方/支持 ▶", yaxis_title="資訊品質", showlegend=False, height=550)
    fig.update_traces(textposition='top center')
    return fig

# ==========================================
# 4. 介面 (UI)
# ==========================================
with st.sidebar:
    st.title("全域觀點解析 V15.1")
    analysis_mode = st.radio("模式選擇：", options=["🛡️ 輿情光譜", "🔮 深度戰情室"], index=0)
    st.markdown("---")
    
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

    with st.expander("📂 匯入舊情報", expanded=False):
        past_report_input = st.text_area("貼上舊報告：", height=100)

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
    
    with st.spinner("📡 正在進行全網情報蒐集..."):
        context_text, sources, cofacts_txt = get_search_context(query, tavily_key, past_report_input)
        st.session_state.sources = sources
        st.session_state.full_context = context_text
        
        if "輿情" in analysis_mode:
            raw_report = run_spectrum_analysis(query, context_text, model_name, google_key)
            st.session_state.spectrum_result = parse_gemini_data(raw_report)
        else:
            with st.status("⚔️ 召開多視角分析會議...", expanded=True) as status:
                st.write("1. 傳送情報給三位分析師...")
                opinions, raw_report = run_council_of_rivals(query, context_text, model_name, google_key)
                st.session_state.wargame_opinions = opinions
                st.session_state.wargame_result = parse_gemini_data(raw_report)
                status.update(label="✅ 分析完成", state="complete", expanded=False)
    st.rerun()

# 渲染結果：輿情光譜
if st.session_state.spectrum_result and "輿情" in analysis_mode:
    data = st.session_state.spectrum_result
    
    if data.get("spectrum"):
        st.markdown("### 🗺️ 輿論陣地光譜")
        fig = render_spectrum_chart(data["spectrum"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📝 媒體識讀報告")
    st.markdown(f'<div class="report-paper">{data.get("report_text")}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("覺得議題太複雜？點擊下方按鈕啟動深度推演。")
    if st.button("🚀 啟動深度戰情室 (基於此情報)", type="primary"):
        if st.session_state.full_context:
            with st.status("⚔️ 召開多視角分析會議...", expanded=True) as status:
                st.write("1. 啟動數位幕僚群...")
                opinions, raw_report = run_council_of_rivals(query, st.session_state.full_context, model_name, google_key)
                st.session_state.wargame_opinions = opinions
                st.session_state.wargame_result = parse_gemini_data(raw_report)
                status.update(label="✅ 推演完成", state="complete", expanded=False)
                st.rerun()

# 渲染結果：戰情室
if st.session_state.wargame_result:
    st.divider()
    st.markdown(f"<h2 style='text-align: center;'>⚔️ 深度戰情室：{query}</h2>", unsafe_allow_html=True)
    
    ops = st.session_state.wargame_opinions
    if ops:
        c_a, c_b, c_ctx = st.columns(3)
        with c_a:
            st.markdown(f'<div class="perspective-box box-blue"><b>🔵 體制/現狀視角</b><br>{ops.get("A_SIDE")[:150]}...</div>', unsafe_allow_html=True)
            with st.popover("完整論述"): st.markdown(ops.get("A_SIDE"))
        with c_b:
            st.markdown(f'<div class="perspective-box box-green"><b>🟢 批判/改革視角</b><br>{ops.get("B_SIDE")[:150]}...</div>', unsafe_allow_html=True)
            with st.popover("完整論述"): st.markdown(ops.get("B_SIDE"))
        with c_ctx:
            st.markdown(f'<div class="perspective-box box-neutral"><b>📜 脈絡分析</b><br>{ops.get("CONTEXT")[:150]}...</div>', unsafe_allow_html=True)
            with st.popover("完整論述"): st.markdown(ops.get("CONTEXT"))

    data_wg = st.session_state.wargame_result
    if data_wg.get("mermaid"):
        st.markdown("### 🕸️ 因果迴路圖 (System Dynamics)")
        st.markdown('<div class="mermaid-box">', unsafe_allow_html=True)
        render_mermaid(data_wg["mermaid"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📝 總編輯深度全解讀")
    st.markdown(f'<div class="report-paper">{data_wg.get("report_text")}</div>', unsafe_allow_html=True)

# [V15.1 改進] 文獻列表：Markdown 表格化
if st.session_state.sources:
    st.markdown("---")
    st.markdown("### 📚 引用文獻列表")
    
    # 建立 Markdown 表格字串
    md_table = "| ID | 媒體/網域 | 標題摘要 | 連結 |\n|:---:|:---|:---|:---|\n"
    for i, s in enumerate(st.session_state.sources):
        domain = get_domain_name(s.get('url'))
        title = s.get('content', '')[:60].replace("\n", " ").replace("|", " ") + "..."
        url = s.get('url')
        md_table += f"| **{i+1}** | `{domain}` | {title} | [點擊]({url}) |\n"
    
    st.markdown(md_table)
