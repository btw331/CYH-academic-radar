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
st.set_page_config(page_title="全域戰情室 V14", page_icon="🏯", layout="wide")

st.markdown("""
<style>
    .metric-container {
        text-align: center;
        padding: 15px;
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #f0f0f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .metric-score { font-size: 2.8em; font-weight: 700; margin: 0; line-height: 1.2;}
    .metric-label { font-size: 1.0em; font-weight: 500; margin-top: 5px; color: #666; letter-spacing: 1px; }
    
    .stButton button[kind="secondary"] { border: 2px solid #673ab7; color: #673ab7; font-weight: bold; }
    
    /* 戰情室風格 */
    .war-room-box {
        background-color: #263238; color: #eceff1; padding: 20px; border-radius: 8px; margin-bottom: 15px; border-left: 5px solid #ffca28;
    }
    .agent-box {
        padding: 15px; border-radius: 8px; margin-bottom: 10px; font-size: 0.95em;
    }
    .agent-hawk { background-color: #ffebee; border-left: 4px solid #d32f2f; color: #b71c1c; }
    .agent-dove { background-color: #e8f5e9; border-left: 4px solid #2e7d32; color: #1b5e20; }
    .agent-history { background-color: #fff3e0; border-left: 4px solid #ef6c00; color: #e65100; }
    
    .mermaid-box {
        background-color: #ffffff; padding: 10px; border-radius: 8px; border: 1px solid #ddd; margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 資料庫與共用常數
# ==========================================
# (保留原有的 DB_MAP, NAME_KEYWORDS 等設定，為節省篇幅省略，實際執行時請保留)
# ... [此處與 V13 相同，若需完整代碼請參考上一版，這裡假設您會保留該區塊] ...
# 為了確保代碼完整可執行，我還是把它貼上：
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

def get_score_text_color(score):
    if score >= 80: return "#d32f2f"
    if score >= 60: return "#e65100"
    if score >= 40: return "#f57f17"
    if score >= 20: return "#388e3c"
    return "#757575"

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
            context_text += f"【歷史情報背景】\n{context_report[:1000]}...\n\n"
            
        context_text += "【最新網路情報】\n"
        for i, res in enumerate(results):
            context_text += f"Source {i+1}: {res.get('url')} | {str(res.get('content'))[:1000]}\n"
            
        return context_text, results, cofacts_txt
    except Exception as e:
        return f"Error: {str(e)}", [], ""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def call_gemini(system_prompt, user_text, model_name, api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2) # 降低溫度以求穩定
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    chain = prompt | llm
    return chain.invoke({"input": user_text}).content

# 3.2 進階功能：Mermaid 渲染器
def render_mermaid(code):
    """使用 HTML Component 渲染 Mermaid 圖表"""
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
    # 定義三個 Agent 的人設
    prompts = {
        "HAWK": """你是一位【鷹派戰略家】(The Hawk)。
        你的性格：悲觀、警惕、專注於最壞情況。
        你的任務：分析衝突升級的可能性、對手的惡意動機、軍事/強制手段的風險。
        請找出情報中所有顯示「局勢惡化」的訊號。""",
        
        "DOVE": """你是一位【鴿派外交官】(The Dove)。
        你的性格：理性、務實、專注於共同利益。
        你的任務：分析經濟依賴、外交緩衝機制、維持現狀的強大慣性。
        請找出情報中所有顯示「局勢可控」或「雙方克制」的訊號。""",
        
        "HISTORIAN": """你是一位【冷靜的歷史學家】(The Historian)。
        你的性格：客觀、宏觀、不受當下情緒影響。
        你的任務：忽略短期雜訊，從過去 50 年的國際關係史中找到最相似的 1-2 個案例 (Historical Analogy)。
        告訴我們：以前發生類似狀況時，最後結局通常是如何？"""
    }
    
    # 1. 平行運算：三位幕僚同時思考
    opinions = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_role = {
            executor.submit(call_gemini, prompt, context_text, model_name, api_key): role 
            for role, prompt in prompts.items()
        }
        for future in concurrent.futures.as_completed(future_to_role):
            role = future_to_role[future]
            try:
                opinions[role] = future.result()
            except Exception as e:
                opinions[role] = f"分析失敗: {e}"

    # 2. 綜合研判：國家安全顧問 (NSA)
    nsa_prompt = f"""
    你是由總統任命的【國家安全顧問】。
    你剛剛聽取了三位幕僚針對「{query}」的簡報，他們的觀點南轅北轍。
    
    【鷹派觀點 (Hawk)】: {opinions.get('HAWK')}
    【鴿派觀點 (Dove)】: {opinions.get('DOVE')}
    【歷史借鏡 (Historian)】: {opinions.get('HISTORIAN')}
    
    請執行以下任務，產出最終決策報告：
    
    1. **STEEP 結構化掃描**：請從 Social, Tech, Economic, Environmental, Political 五個維度，列出關鍵驅動力。
    2. **交叉衝擊 (Cross-Impact)**：分析關鍵變數的交互作用 (例如：若 A 發生，會強化還是削弱 B？)。
    3. **因果迴路圖 (Mermaid)**：請生成一段 Mermaid JS 的 `graph TD` 代碼，畫出事件的系統動力圖。請務必將代碼包在 ```mermaid ... ``` 區塊中。
    4. **未來情境推演**：基於上述分析，推導 3 種情境 (基準/轉折/極端)。
    
    【輸出格式】：
    ### [DATA_SCORES]
    Threat: [0-100]
    Attack: [0-100]
    Impact: [0-100]
    Division: [0-100]
    Resilience: [0-100]
    
    ### [REPORT_TEXT]
    (Markdown 報告內容...)
    """
    
    final_report = call_gemini(nsa_prompt, context_text, model_name, api_key)
    return opinions, final_report

# 3.4 核心邏輯：輿情光譜 (原 V13 功能)
def run_spectrum_analysis(query, context_text, model_name, api_key):
    system_prompt = f"""
    你是一位全域情報分析師。請針對「{query}」進行深度解析。
    請分析每個來源的「政治立場 (-10~10)」與「可信度 (0~10)」。
    
    【輸出格式】：
    ### [DATA_SCORES]
    Threat: [分數] ... (略)
    
    ### [DATA_TIMELINE]
    YYYY-MM-DD|媒體|標題
    
    ### [DATA_SPECTRUM]
    來源|立場|可信度|網址
    
    ### [REPORT_TEXT]
    (Markdown 報告)
    """
    return call_gemini(system_prompt, context_text, model_name, api_key)

# 3.5 資料解析器
def parse_gemini_data(text):
    data = {"scores": {}, "timeline": [], "spectrum": [], "mermaid": "", "report_text": ""}
    
    # 提取 Mermaid
    mermaid_match = re.search(r"```mermaid\n(.*?)\n```", text, re.DOTALL)
    if mermaid_match:
        data["mermaid"] = mermaid_match.group(1)
        # 移除報告中的 mermaid 代碼，避免重複顯示
        text = text.replace(mermaid_match.group(0), "")

    # 提取分數與其他
    for line in text.split('\n'):
        line = line.strip()
        if "Threat:" in line: 
            try: data["scores"]["Threat"] = int(re.search(r'\d+', line).group())
            except: pass
        if "Attack:" in line: 
            try: data["scores"]["Attack"] = int(re.search(r'\d+', line).group())
            except: pass
        if "Impact:" in line: 
            try: data["scores"]["Impact"] = int(re.search(r'\d+', line).group())
            except: pass
        if "Division:" in line: 
            try: data["scores"]["Division"] = int(re.search(r'\d+', line).group())
            except: pass
        if "Resilience:" in line: 
            try: data["scores"]["Resilience"] = int(re.search(r'\d+', line).group())
            except: pass
            
        if "|" in line and len(line.split("|")) >= 3 and (line[0].isdigit() or "Future" in line):
            parts = line.split("|")
            data["timeline"].append({"date": parts[0], "media": parts[1], "event": parts[2]})
            
        if "|" in line and len(line.split("|")) >= 4 and not line.startswith("###") and not "日期" in line:
            parts = line.split("|")
            try: data["spectrum"].append({"source": parts[0], "stance": float(parts[1]), "credibility": float(parts[2]), "url": parts[3]})
            except: pass

    # 提取報告本文
    if "### [REPORT_TEXT]" in text:
        data["report_text"] = text.split("### [REPORT_TEXT]")[1].strip()
    else:
        data["report_text"] = text # Fallback

    return data

def render_spectrum_chart(spectrum_data):
    if not spectrum_data: return None
    df = pd.DataFrame(spectrum_data)
    fig = px.scatter(df, x="stance", y="credibility", hover_name="source", text="source", size=[15]*len(df),
                     color="stance", color_continuous_scale=["#2e7d32", "#eeeeee", "#d32f2f"], range_x=[-11, 11], range_y=[-1, 11],
                     labels={"stance": "立場 (綠 <-> 藍/紅)", "credibility": "可信度"})
    fig.add_shape(type="rect", x0=-11, y0=5, x1=0, y1=11, fillcolor="rgba(46, 125, 50, 0.1)", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=5, x1=11, y1=11, fillcolor="rgba(21, 101, 192, 0.1)", layer="below", line_width=0)
    fig.update_layout(xaxis_title="◀ 泛綠 --- 中立 --- 泛藍/紅 ▶", yaxis_title="可信度", showlegend=False, height=450)
    fig.update_traces(textposition='top center')
    return fig

# ==========================================
# 4. 介面 (UI)
# ==========================================
with st.sidebar:
    st.title("全域戰情室 V14")
    
    # 模式選擇 (決定是否啟動 Council of Rivals)
    analysis_mode = st.radio(
        "選擇分析模式：",
        options=["🛡️ 全域輿情監測 (Spectrum)", "🔮 未來戰棋推演 (War Game)"],
        captions=["即時：Cofacts查核 + 輿論光譜", "深度：紅隊演練 + 系統思考圖"],
        index=1
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
query = st.text_input("輸入戰略議題", placeholder="例如：台海封鎖情境推演")
search_btn = st.button("🚀 啟動分析引擎", type="primary")

if 'result' not in st.session_state: st.session_state.result = None
if 'opinions' not in st.session_state: st.session_state.opinions = None
if 'sources' not in st.session_state: st.session_state.sources = None

if search_btn and query and google_key and tavily_key:
    st.session_state.result = None
    st.session_state.opinions = None
    
    # 1. 獲取情報 (通用)
    with st.spinner("📡 正在進行全網情報蒐集 (Tavily + Cofacts)..."):
        context_text, sources, cofacts_txt = get_search_context(query, tavily_key, past_report_input)
        st.session_state.sources = sources
    
    # 2. 分流處理
    if "戰棋" in analysis_mode:
        with st.status("⚔️ 召開數位戰情會議 (Council of Rivals)...", expanded=True) as status:
            st.write("1. 正在傳喚 🦅 鷹派、🕊️ 鴿派、📜 歷史學家...")
            opinions, raw_report = run_council_of_rivals(query, context_text, model_name, google_key)
            st.session_state.opinions = opinions
            
            st.write("2. 國家安全顧問 (NSA) 正在進行 STEEP 綜合研判...")
            st.write("3. 繪製因果迴路圖 (Causal Loop Diagram)...")
            parsed = parse_gemini_data(raw_report)
            st.session_state.result = parsed
            status.update(label="✅ 推演完成", state="complete", expanded=False)
            
    else: # 輿情監測模式
        with st.spinner("⚖️ 正在繪製輿論光譜..."):
            raw_report = run_spectrum_analysis(query, context_text, model_name, google_key)
            parsed = parse_gemini_data(raw_report)
            st.session_state.result = parsed

# 渲染結果
if st.session_state.result:
    data = st.session_state.result
    
    # 1. 核心指標
    scores = data.get("scores", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        ("威脅指數", scores.get("Threat", 0)), 
        ("攻擊熱度", scores.get("Attack", 0)),
        ("分歧程度", scores.get("Division", 0)),
        ("影響深遠", scores.get("Impact", 0)),
        ("系統韌性", scores.get("Resilience", 0))
    ]
    for col, (lbl, val) in zip([c1, c2, c3, c4, c5], metrics):
        col.markdown(f"""<div class="metric-container"><p class="metric-score" style="color:{get_score_text_color(val)}">{val}</p><p class="metric-label">{lbl}</p></div>""", unsafe_allow_html=True)

    # 2. 戰棋模式專屬：幕僚辯論 & Mermaid
    if st.session_state.opinions:
        st.markdown("### 🗣️ 數位戰情室辯論紀錄")
        ops = st.session_state.opinions
        c_hawk, c_dove, c_hist = st.columns(3)
        with c_hawk:
            st.markdown(f'<div class="agent-box agent-hawk"><b>🦅 鷹派 (Hawk)</b><br>{ops.get("HAWK")[:300]}...</div>', unsafe_allow_html=True)
            with st.popover("查看鷹派完整報告"): st.markdown(ops.get("HAWK"))
        with c_dove:
            st.markdown(f'<div class="agent-box agent-dove"><b>🕊️ 鴿派 (Dove)</b><br>{ops.get("DOVE")[:300]}...</div>', unsafe_allow_html=True)
            with st.popover("查看鴿派完整報告"): st.markdown(ops.get("DOVE"))
        with c_hist:
            st.markdown(f'<div class="agent-box agent-history"><b>📜 歷史學家</b><br>{ops.get("HISTORIAN")[:300]}...</div>', unsafe_allow_html=True)
            with st.popover("查看歷史借鏡"): st.markdown(ops.get("HISTORIAN"))

        if data.get("mermaid"):
            st.markdown("### 🕸️ 系統因果迴路圖 (Causal Loop)")
            st.caption("AI 自動生成的系統動力學圖表，展示變數間的回饋關係。")
            render_mermaid(data["mermaid"])

    # 3. 輿情模式專屬：光譜圖
    if data.get("spectrum"):
        st.markdown("### 🗺️ 輿論陣地光譜")
        fig = render_spectrum_chart(data["spectrum"])
        st.plotly_chart(fig, use_container_width=True)

    # 4. 完整報告
    st.markdown("### 📝 綜合情報判讀")
    st.markdown(f'<div class="war-room-box">{data.get("report_text")}</div>', unsafe_allow_html=True)
    
    # 5. 時間軸與來源
    with st.expander("📅 發展時序與情報來源"):
        if data.get("timeline"):
            st.dataframe(pd.DataFrame(data["timeline"]), use_container_width=True)
        if st.session_state.sources:
            for s in st.session_state.sources:
                st.markdown(f"- [{s.get('url')}]({s.get('url')})")
