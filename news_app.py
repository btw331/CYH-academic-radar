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
st.set_page_config(page_title="全域觀點解析 V15.4", page_icon="⚖️", layout="wide")

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
        font-family: "Georgia", serif;
        line-height: 1.8;
    }
    
    .perspective-box {
        padding: 15px; border-radius: 8px; margin-bottom: 10px; font-size: 0.95em;
        border-left-width: 4px; border-left-style: solid;
        background-color: #fff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .box-green { border-left-color: #2e7d32; }
    .box-blue { border-left-color: #1565c0; }
    .box-neutral { border-left-color: #616161; }
    
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
                result_text += "【Cofacts 真的假的 - 查核資料庫】\n"
                for i, art in enumerate(articles):
                    node = art.get('node', {})
                    rumor = node.get('text', '')[:50]
                    replies = node.get('articleReplies', [])
                    if replies:
                        r_type = replies[0].get('reply', {}).get('type')
                        type_map = {"RUMOR": "❌ 含有不實資訊", "NOT_ARTICLE": "⭕ 查無不實/個人意見", "OPINION": "💬 純屬意見"}
                        display_type = type_map.get(r_type, r_type)
                        result_text += f"- 網傳謠言: {rumor}... (查核判定: {display_type})\n"
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

# 3.2 Mermaid 清洗與渲染
def sanitize_mermaid_code(code):
    """
    修復 Mermaid 語法錯誤：
    1. 移除 markdown 標記
    2. 將節點名稱中的括號 () 替換為安全字符，避免語法崩潰
    """
    code = re.sub(r'```mermaid', '', code)
    code = re.sub(r'```', '', code)
    code = code.strip()
    
    lines = code.split('\n')
    clean_lines = []
    
    if not any(l.strip().startswith('graph') for l in lines):
        clean_lines.append("graph TD")
        
    for line in lines:
        if not line.strip(): continue
        
        # 處理 A[Label] 格式，防止 Label 內有 ()
        if '[' in line and ']' in line:
            parts = line.split('[', 1)
            node_id = parts[0]
            rest = parts[1].rsplit(']', 1)
            label = rest[0]
            edge = rest[1] if len(rest) > 1 else ""
            safe_label = label.replace('(', ' ').replace(')', ' ').replace('"', "'")
            clean_lines.append(f'{node_id}["{safe_label}"]{edge}')
            
        # 處理 A(Label) 格式 -> 轉為 A["Label"]
        elif '(' in line and ')' in line and '>"' not in line:
            parts = line.split('(', 1)
            node_id = parts[0]
            rest = parts[1].rsplit(')', 1)
            label = rest[0]
            edge = rest[1] if len(rest) > 1 else ""
            safe_label = label.replace('(', ' ').replace(')', ' ').replace('"', "'")
            clean_lines.append(f'{node_id}["{safe_label}"]{edge}')
        else:
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

# 3.3 核心邏輯：數位戰情室 (Council of Rivals)
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
    1. **嚴格引用**：報告中的每一個論點，都必須標註來源編號，格式為 `[Source X]`。
    2. **Mermaid 製圖**：請生成 Mermaid `graph TD` 代碼，展示「變數 A 如何導致 變數 B」的系統動力因果鏈。
       - 關鍵：節點名稱請使用方括號 `[]`，例如 `A["政策X"] --> B["民怨上升"]`。不要在名稱中使用圓括號。
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
    你是一位媒體識讀專家。請針對「{query}」進行框架分析 (Framing Analysis)。
    
    【任務】：
    1. 識別每個來源的「敘事立場」與「可信度」。
    2. **重要**：請給出具體座標，以便繪製光譜圖。
       - 立場 (X軸): -10(強烈反對/批判/A方) <-> 0(中立/事實描述) <-> 10(強烈支持/護航/B方)
       - 可信度 (Y軸): 0(內容農場/謠言) <-> 10(權威機構/數據詳實)
    
    【輸出格式】：
    ### [DATA_TIMELINE]
    YYYY-MM-DD|媒體|標題
    
    ### [DATA_SPECTRUM]
    來源名稱|立場(-10~10)|可信度(0~10)|網址
    
    ### [REPORT_TEXT]
    (Markdown 報告，需包含 [Source X] 引用)
    請包含：全域現況摘要、媒體框架分析、識讀建議。
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
                # Jitter 機制：加入隨機擾動防止重疊
                base_stance = float(parts[1])
                base_cred = float(parts[2])
                jitter_x = random.uniform(-0.8, 0.8) 
                jitter_y = random.uniform(-0.5, 0.5)
                data["spectrum"].append({
                    "source": parts[0].strip(), 
                    "stance": base_stance + jitter_x, 
                    "credibility": base_cred + jitter_y, 
                    "url": parts[3].strip()
                })
            except: pass

    if "### [REPORT_TEXT]" in text:
        data["report_text"] = text.split("### [REPORT_TEXT]")[1].strip()
    else:
        data["report_text"] = text 

    return data

# 優化版光譜圖
def render_spectrum_chart(spectrum_data):
    if not spectrum_data: return None
    df = pd.DataFrame(spectrum_data)
    
    fig = px.scatter(
        df, x="stance", y="credibility", hover_name="source", text="source", size=[25]*len(df),
        color="stance", color_continuous_scale=["#2e7d32", "#eeeeee", "#1565c0"],
        range_x=[-15, 15], # 拉大 X 軸範圍
        range_y=[-2, 13],  # 拉大 Y 軸範圍
        opacity=0.9,
        labels={"stance": "觀點光譜", "credibility": "資訊可信度"}
    )
    # 背景象限
    fig.add_shape(type="rect", x0=-15, y0=6, x1=0, y1=13, fillcolor="rgba(46, 125, 50, 0.05)", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=6, x1=15, y1=13, fillcolor="rgba(21, 101, 192, 0.05)", layer="below", line_width=0)
    
    fig.update_layout(
        xaxis_title="◀ 批判/改革 (綠) ------- 中立 ------- 體制/支持 (藍) ▶",
        yaxis_title="資訊品質 (低 -> 高)",
        showlegend=False,
        height=600,
        font=dict(size=14)
    )
    fig.update_traces(textposition='top center', textfont_size=13)
    return fig

# 4. 下載功能
def convert_data_to_json(data):
    return json.dumps(data, indent=2, ensure_ascii=False)

def convert_data_to_md(data):
    return f"""
# 全域觀點分析報告
產生時間: {datetime.now()}

## 1. 深度分析
{data.get('report_text')}

## 2. 時間軸
{pd.DataFrame(data.get('timeline')).to_markdown(index=False)}
    """

# ==========================================
# 5. UI
# ==========================================
with st.sidebar:
    st.title("全域觀點解析 V15.4")
    analysis_mode = st.radio("選擇模式：", options=["🛡️ 輿情光譜 (Spectrum)", "🔮 未來戰棋 (War Game)"], index=0)
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
            
        # [V15.4 Update] Added gemini-2.5-flash-lite
        model_name = st.selectbox("模型", ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"], index=0)

    with st.expander("📂 匯入舊情報", expanded=False):
        past_report_input = st.text_area("貼上舊報告 Markdown：", height=100)
        
    # 下載按鈕
    st.markdown("### 📥 報告匯出")
    if st.session_state.get('spectrum_result') or st.session_state.get('wargame_result'):
        active_data = st.session_state.get('wargame_result') if "War" in analysis_mode else st.session_state.get('spectrum_result')
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

# 邏輯執行
if search_btn and query and google_key and tavily_key:
    st.session_state.spectrum_result = None
    st.session_state.wargame_result = None
    st.session_state.wargame_opinions = None
    
    with st.spinner("📡 正在進行全網情報蒐集 (Tavily + Cofacts)..."):
        context_text, sources, cofacts_txt = get_search_context(query, tavily_key, past_report_input)
        st.session_state.sources = sources
        st.session_state.full_context = context_text
        
        if "Spectrum" in analysis_mode:
            raw_report = run_spectrum_analysis(query, context_text, model_name, google_key)
            st.session_state.spectrum_result = parse_gemini_data(raw_report)
        else:
            with st.status("⚔️ 召開多視角戰情會議...", expanded=True) as status:
                st.write("1. 正在傳喚不同觀點分析師...")
                opinions, raw_report = run_council_of_rivals(query, context_text, model_name, google_key)
                st.session_state.wargame_opinions = opinions
                st.session_state.wargame_result = parse_gemini_data(raw_report)
                status.update(label="✅ 分析完成", state="complete", expanded=False)
    st.rerun()

# 顯示：輿情光譜
if st.session_state.spectrum_result and "Spectrum" in analysis_mode:
    data = st.session_state.spectrum_result
    
    if data.get("spectrum"):
        st.markdown("### 🗺️ 輿論陣地光譜 (Spectrum Map)")
        fig = render_spectrum_chart(data["spectrum"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📝 媒體識讀報告")
    st.markdown(f'<div class="report-paper">{data.get("report_text")}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("覺得議題需要更深度推演？請點擊下方按鈕。")
    if st.button("🚀 基於此情報啟動未來戰棋 (War Game)", type="primary"):
        if st.session_state.full_context:
            with st.status("⚔️ 啟動數位戰情室...", expanded=True) as status:
                st.write("1. 啟動虛擬幕僚群...")
                opinions, raw_report = run_council_of_rivals(query, st.session_state.full_context, model_name, google_key)
                st.session_state.wargame_opinions = opinions
                st.session_state.wargame_result = parse_gemini_data(raw_report)
                status.update(label="✅ 推演完成", state="complete", expanded=False)
                st.rerun()

# 顯示：未來戰棋
if st.session_state.wargame_result:
    st.divider()
    st.markdown(f"<h2 style='text-align: center;'>⚔️ 未來發展推演：{query}</h2>", unsafe_allow_html=True)
    
    ops = st.session_state.wargame_opinions
    if ops:
        c_a, c_b, c_ctx = st.columns(3)
        with c_a:
            st.markdown(f'<div class="perspective-box box-blue"><b>🔵 體制/現狀視角</b><br>{ops.get("A_SIDE")[:150]}...</div>', unsafe_allow_html=True)
            with st.popover("查看完整論述"): st.markdown(ops.get("A_SIDE"))
        with c_b:
            st.markdown(f'<div class="perspective-box box-green"><b>🟢 批判/改革視角</b><br>{ops.get("B_SIDE")[:150]}...</div>', unsafe_allow_html=True)
            with st.popover("查看完整論述"): st.markdown(ops.get("B_SIDE"))
        with c_ctx:
            st.markdown(f'<div class="perspective-box box-neutral"><b>📜 深層脈絡分析</b><br>{ops.get("CONTEXT")[:150]}...</div>', unsafe_allow_html=True)
            with st.popover("查看完整論述"): st.markdown(ops.get("CONTEXT"))

    data_wg = st.session_state.wargame_result
    
    if data_wg.get("mermaid"):
        st.markdown("### 🕸️ 系統因果迴路圖 (System Dynamics)")
        st.markdown('<div class="mermaid-box">', unsafe_allow_html=True)
        render_mermaid(data_wg["mermaid"])
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ 系統未能生成有效的因果圖代碼。")

    st.markdown("### 📝 總編輯深度決策報告")
    st.markdown(f'<div class="report-paper">{data_wg.get("report_text")}</div>', unsafe_allow_html=True)

# 文獻列表
if st.session_state.sources:
    st.markdown("---")
    st.markdown("### 📚 引用文獻列表")
    md_table = "| 編號 | 媒體/網域 | 標題摘要 | 連結 |\n|:---:|:---|:---|:---|\n"
    for i, s in enumerate(st.session_state.sources):
        domain = get_domain_name(s.get('url'))
        title = s.get('content', '')[:60].replace("\n", " ").replace("|", " ") + "..."
        url = s.get('url')
        md_table += f"| **{i+1}** | `{domain}` | {title} | [點擊]({url}) |\n"
    st.markdown(md_table)
