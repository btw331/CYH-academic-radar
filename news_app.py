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
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
import streamlit.components.v1 as components

# ==========================================
# 1. 基礎設定與 CSS樣式
# ==========================================
st.set_page_config(page_title="全域觀點解析 V16.4", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    .stButton button[kind="secondary"] { border: 2px solid #673ab7; color: #673ab7; font-weight: bold; }
    
    /* 報告紙張風格 */
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
    
    /* 引用標記樣式 */
    .citation {
        font-size: 0.75em; color: #9e9e9e; background-color: #f5f5f5;
        padding: 1px 4px; border-radius: 4px; vertical-align: super;
        font-family: sans-serif; border: 1px solid #eeeeee;
    }

    /* 觀點對照盒 */
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
    
    /* 表格標題 */
    .table-header-green { color: #2e7d32; font-weight: bold; font-size: 1.1em; border-bottom: 2px solid #2e7d32; margin-bottom: 10px; padding-bottom: 5px; }
    .table-header-blue { color: #1565c0; font-weight: bold; font-size: 1.1em; border-bottom: 2px solid #1565c0; margin-bottom: 10px; padding-bottom: 5px; }
    .table-header-neutral { color: #616161; font-weight: bold; font-size: 1.1em; border-bottom: 2px solid #616161; margin-bottom: 10px; padding-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 資料庫與共用常數 (硬邏輯校正用)
# ==========================================
CAMP_KEYWORDS = {
    "GREEN": ["自由", "三立", "民視", "新頭殼", "鏡週刊", "放言", "賴清德", "民進黨", "青鳥"],
    "BLUE": ["聯合", "中時", "中國時報", "TVBS", "中天", "風傳媒", "國民黨", "藍營"],
    "RED": ["新華", "人民日報", "環球", "央視", "中評", "国台办"]
}

def get_domain_name(url):
    try: return urlparse(url).netloc.replace("www.", "")
    except: return ""

def format_citation_style(text):
    if not text: return ""
    pattern = r'(\[Source[^\]]*\])'
    styled_text = re.sub(pattern, r'<span class="citation">\1</span>', text)
    return styled_text

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
    
    search_q = f"{query} 2025 news analysis"
    if context_report: search_q += " history context"
    
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

# 3.2 Mermaid 強力清洗器
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
    1. **嚴格引用**：報告中的每一個論點，都必須標註來源編號，格式為 `[Source X]`。
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

# 3.4 核心邏輯：輿情光譜 (新增：請求 AI 提供標題)
def run_spectrum_analysis(query, context_text, model_name, api_key):
    system_prompt = f"""
    你是一位媒體識讀專家。請針對「{query}」進行媒體框架分析。
    
    【評分嚴格規定】：
    1. **立場分數 (Stance)**：必須區分正負！
       - **負數 (-10 到 -1)**：批判/反對/泛綠/獨派。
       - **零 (0)**：中立/純事實。
       - **正數 (1 到 10)**：支持/體制/泛藍/統派。
    
    2. **可信度 (Credibility)**：
       - 0-3：農場/極端。
       - 4-7：一般媒體。
       - 8-10：權威/查核。
    
    【輸出格式 (請保持格式整潔，每行一筆，使用 | 分隔)】：
    ### [DATA_TIMELINE]
    YYYY-MM-DD|媒體|標題
    
    ### [DATA_SPECTRUM]
    來源名稱|新聞標題|立場(-10~10)|可信度(0~10)|網址
    
    ### [REPORT_TEXT]
    (Markdown 報告，需包含 [Source X] 引用)
    請包含：全域現況摘要、媒體框架分析、識讀建議。
    """
    return call_gemini(system_prompt, context_text, model_name, api_key)

# 3.5 資料解析器 (含硬邏輯校正 + 標題解析)
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
            
        # Spectrum (更新：解析 5 個欄位)
        if "|" in line and len(line.split("|")) >= 5 and not line.startswith("###") and not "日期" in line:
            parts = line.split("|")
            try:
                name = parts[0].strip()
                title = parts[1].strip() # [V16.4] 新增標題
                base_stance = float(parts[2].strip())
                base_cred = float(parts[3].strip())
                url = parts[4].strip()
                
                # 硬邏輯校正
                final_stance = base_stance
                if any(k in name for k in CAMP_KEYWORDS["GREEN"]):
                    if final_stance > 0: final_stance = final_stance * -1
                    if final_stance == 0: final_stance = -5
                elif any(k in name for k in CAMP_KEYWORDS["BLUE"] + CAMP_KEYWORDS["RED"]):
                    if final_stance < 0: final_stance = final_stance * -1
                    if final_stance == 0: final_stance = 5
                
                data["spectrum"].append({
                    "source": name,
                    "title": title, # [V16.4] 儲存標題
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

# [V16.4] 渲染含標題的表格
def render_spectrum_split(spectrum_data):
    if not spectrum_data: return
    
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
        # [V16.4] 新增「新聞標題」欄位
        md = "| 媒體 | 新聞標題 (點擊閱讀) | 立場 | 可信度 |\n|:---|:---|:---:|:---:|\n"
        for i in items:
            s = i['stance']
            if s < 0: s_txt = f"🟢 {s}"
            elif s > 0: s_txt = f"🔵 +{s}"
            else: s_txt = "⚪ 0"
            
            c = i['credibility']
            if c >= 7: c_txt = f"🟢 {c}"
            elif c >= 4: c_txt = f"🟡 {c}"
            else: c_txt = f"🔴 {c}"
            
            # [V16.4] 標題即連結
            title_link = f"[{i['title']}]({i['url']})"
            
            md += f"| {i['source']} | {title_link} | {s_txt} | {c_txt} |\n"
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
    st.title("全域觀點解析 V16.4")
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

    with st.expander("🧠 系統邏輯說明 (Transparency)", expanded=False):
        st.markdown("""
        **1. 政治光譜校正機制 (Calibration)**
        * **🟢 泛綠/批判區**：
          - 包含：自由、三立、民視、鏡週刊...
          - 邏輯：強制歸類為負分，防止 AI 幻覺。
        * **🔵 泛藍/體制區**：
          - 包含：中時、聯合、TVBS、風傳媒...
          - 邏輯：強制歸類為正分。
        
        **2. 深度報告生成邏輯 (Report Logic)**
        * **媒體框架分析 (Framing)**:
          - **理論基礎**: 使用傳播學 Framing Theory。
          - **AI指令**: 要求偵測來源是否使用「衝突框架(Conflict)」、「歸責框架(Attribution)」或「經濟後果框架」。
        * **識讀建議 (Literacy)**:
          - **生成依據**: 基於「資訊落差 (Information Gap)」與「情緒渲染度」。
          - **AI指令**: 若偵測到高分歧，建議讀者「暫停轉發」並「交叉比對」相反立場報導。

        **3. 數位戰情室設定 (Scenario)**
        * **🦅 鷹派**: 專注衝突升級與敵意螺旋。
        * **🕊️ 鴿派**: 專注經濟互依與現狀維持。
        * **📜 歷史學家**: 尋找過去 50 年的相似歷史案例 (Historical Analogy)。
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
            with st.status("⚔️ 進行多視角推演...", expanded=True) as status:
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

# 顯示：未來戰棋
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
