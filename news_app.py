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
from urllib.parse import urlparse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. 基礎設定與 CSS樣式
# ==========================================
st.set_page_config(page_title="全域觀點搜尋 V13", page_icon="⚖️", layout="wide")

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
    
    .fact-check-box {
        background-color: #e8f5e9; border: 1px solid #c8e6c9; border-radius: 8px; padding: 15px; margin-bottom: 20px;
    }
    .fact-check-title { color: #2e7d32; font-weight: bold; font-size: 1.1em; display: flex; align-items: center; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 資料庫與共用常數 (保留原有 DB_MAP)
# ==========================================
DB_MAP = {
    "CHINA": ["xinhuanet.com", "people.com.cn", "huanqiu.com", "cctv.com", "chinadaily.com.cn", "cgtn.com", "taiwan.cn", "gwytb.gov.cn", "guancha.cn", "thepaper.cn", "sina.com.cn", "163.com", "sohu.com", "ifeng.com", "crntt.com", "hk01.com"],
    "JAPAN": ["nhk.or.jp", "asia.nikkei.com", "yomiuri.co.jp", "asahi.com", "japantimes.co.jp", "mainichi.jp", "sankei.com"],
    "INTL": ["reuters.com", "apnews.com", "bloomberg.com", "wsj.com", "ft.com", "economist.com", "bbc.com", "dw.com", "voanews.com", "thediplomat.com", "foreignpolicy.com", "guardian.co.uk", "aljazeera.com", "rfi.fr", "nytimes.com", "cnn.com", "csis.org"],
    "DIGITAL": ["twreporter.org", "theinitium.com", "storm.mg", "upmedia.mg", "mindiworldnews.com", "allsides.com", "ground.news", "thenewslens.com", "readr.tw", "vocus.cc"],
    "OFFICIAL": ["cna.com.tw", "pts.org.tw", "mnd.gov.tw", "indsr.org.tw", "tfc-taiwan.org.tw", "mygopen.com", "cofacts.tw", "mac.gov.tw"],
    "GREEN": ["ltn.com.tw", "ftvnews.com.tw", "setn.com", "rti.org.tw", "newtalk.tw", "peoplenews.tw", "mirrormedia.mg", "dpp.org.tw"],
    "BLUE": ["udn.com", "chinatimes.com", "tvbs.com.tw", "cti.com.tw", "coolloud.org.tw", "nownews.com", "ctee.com.tw", "want-daily.com", "kmt.org.tw"],
    "FARM": ["kknews.cc", "read01.com", "ppfocus.com", "buzzhand.com", "bomb01.com", "qiqi.news", "lackk.com", "mission-tw.com", "hottopic.com", "weibo.com", "xuehua.us", "inf.news", "toutiao.com", "baidu.com", "ptt.cc", "dcard.tw", "mobile01.com"],
    "VIDEO": ["youtube.com", "youtu.be", "tiktok.com", "douyin.com", "bilibili.com", "ixigua.com"],
    "AGGREGATOR": ["yahoo.com", "msn.com", "linetoday.com", "google.com", "ettoday.net"]
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

def classify_media_name(name):
    n = name.lower()
    for cat, keywords in NAME_KEYWORDS.items():
        if any(k in n for k in keywords): return cat
    return "OTHER"

def get_category_meta(cat):
    meta = {
        "CHINA": ("🇨🇳 中國官媒", "#d32f2f"),
        "FARM": ("⛔ 內容農場", "#ef6c00"),
        "BLUE": ("🔵 泛藍觀點", "#1565c0"),
        "GREEN": ("🟢 泛綠觀點", "#2e7d32"),
        "OFFICIAL": ("⚪ 官方/中立", "#546e7a"),
        "AGGREGATOR": ("🌐 入口網站", "#607d8b"),
        "JAPAN": ("🇯🇵 日本觀點", "#f57c00"),
        "INTL": ("🌏 國際媒體", "#f57c00"),
        "DIGITAL": ("🟡 數位/網媒", "#fbc02d"),
        "VIDEO": ("🟣 影音社群", "#7b1fa2"),
        "OTHER": ("📄 其他來源", "#9e9e9e")
    }
    return meta.get(cat, ("其他", "#9e9e9e"))

def get_score_text_color(score):
    if score >= 80: return "#d32f2f"
    if score >= 60: return "#e65100"
    if score >= 40: return "#f57f17"
    if score >= 20: return "#388e3c"
    return "#757575"

# ==========================================
# 3. 功能模組：Cofacts & Gemini
# ==========================================

# [NEW] Cofacts API 查詢功能
def search_cofacts(query):
    url = "https://cofacts-api.g0v.tw/graphql"
    # GraphQL 查詢語法
    graphql_query = """
    query ListArticles($text: String!) {
      ListArticles(filter: {q: $text}, orderBy: [{_score: DESC}], first: 3) {
        edges {
          node {
            text
            articleReplies(status: NORMAL) {
              reply {
                text
                type
              }
            }
          }
        }
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
                result_text += "【Cofacts 真的假的 - 查核資料庫結果】\n"
                for i, art in enumerate(articles):
                    node = art.get('node', {})
                    rumor_text = node.get('text', '')[:100]
                    replies = node.get('articleReplies', [])
                    if replies:
                        for rep in replies:
                            r = rep.get('reply', {})
                            r_type = r.get('type')
                            r_text = r.get('text', '')[:200]
                            # 轉換類型為易讀文字
                            type_map = {"RUMOR": "❌ 含有不實資訊", "NOT_ARTICLE": "⭕ 查無不實/個人意見", "OPINION": "💬 純屬意見"}
                            display_type = type_map.get(r_type, r_type)
                            result_text += f"- 網傳謠言: {rumor_text}...\n  -> 查核結果: {display_type} | 說明: {r_text}...\n"
            return result_text
    except Exception as e:
        return f"Cofacts 查詢失敗: {str(e)}"
    return ""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def call_gemini_with_retry(chain, input_data):
    return chain.invoke(input_data)

def run_fusion_analysis(query, api_key_google, api_key_tavily, model_name, mode="FUSION", context_report=None):
    os.environ["GOOGLE_API_KEY"] = api_key_google
    os.environ["TAVILY_API_KEY"] = api_key_tavily
    
    try:
        # 1. 搜尋階段 (Tavily + Cofacts)
        results = None
        context_text = ""
        
        # [NEW] 同步執行 Cofacts 查詢
        cofacts_result = search_cofacts(query)
        if cofacts_result:
            context_text += f"\n{cofacts_result}\n" + "="*30 + "\n"

        search = TavilySearchResults(max_results=20) 
        
        if context_report and len(context_report) > 50:
            context_text += f"【前次分析報告】\n{context_report}\n\n【今日最新情報】\n"
            q_mix = f"{query} 2025 最新發展"
        else:
            q_mix = f"{query} 2025 最新 台灣新聞 爭議 懶人包 評論"

        results = search.invoke(q_mix)
        for i, res in enumerate(results):
            context_text += f"Source {i+1}: {res.get('url')} | {str(res.get('content'))[:2000]}\n"

        # 2. 思考階段 (Prompt Engineering)
        # [NEW] 加入輿論光譜 (Spectrum) 與 動態來源分類 指令
        system_prompt = f"""
        你是一位全域情報分析師。請針對「{query}」進行深度解析。
        
        【任務 1: 真實性驗證】
        請優先參考提供的【Cofacts 查核資料】。若有明確查核報告，請在報告開頭以「⚠️ 查核警示」標註。
        
        【任務 2: 輿論光譜定位 (Spectrum Mapping)】
        請分析每個來源的「政治立場」與「可信度」。
        - 立場 (X軸): -10(深綠/獨派) <-> 0(中立) <-> 10(深藍/統派/紅)
        - 可信度 (Y軸): 0(內容農場/假訊息) <-> 10(權威媒體/官方數據)
        
        【任務 3: 動態來源識別】
        若遇到非知名網域，請根據其標題風格（如標題殺人法、農場文體）進行動態標記。

        【輸出格式 (嚴格遵守)】：
        ### [DATA_SCORES]
        Threat: [0-100]
        Attack: [0-100]
        Impact: [0-100]
        Division: [0-100]
        Resilience: [0-100]
        
        ### [DATA_TIMELINE]
        (格式：YYYY-MM-DD|媒體|標題)
        
        ### [DATA_SPECTRUM]
        (格式：來源名稱|立場分數(-10~10)|可信度分數(0~10)|網址)
        Example:
        新華社|10|4|http://...
        自由時報|-8|7|http://...
        PTT網友|0|2|http://...

        ### [REPORT_TEXT]
        (Markdown 報告)
        請包含：
        1. **📊 全域現況摘要** (含 Cofacts 查核結果)
        2. **⚖️ 輿論光譜分析** (解讀光譜分佈的意義：是極化對立還是共識？)
        3. **🔍 深度識讀與利益分析**
        """

        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{context_text}")])
        chain = prompt | llm
        response = call_gemini_with_retry(chain, {"context_text": context_text})
        return response.content, results, cofacts_result

    except Exception as e:
        if "429" in str(e): return "API_LIMIT_ERROR", None, None
        return f"ERROR: {str(e)}", None, None

def parse_gemini_data(text):
    data = {"scores": {"Threat":0, "Attack":0, "Impact":0, "Division":0, "Resilience":0}, 
            "timeline": [], "spectrum": [], "report_text": ""}
    
    if not text or text.startswith("ERROR"):
        data["report_text"] = text
        return data

    for line in text.split('\n'):
        line = line.strip()
        # Parse Scores
        for key in data["scores"]:
            if f"{key}:" in line:
                try: data["scores"][key] = int(re.search(r'\d+', line).group())
                except: pass
        
        # Parse Timeline
        if "|" in line and len(line.split("|")) >= 3 and (line[0].isdigit() or "Future" in line):
            parts = line.split("|")
            data["timeline"].append({"date": parts[0].strip(), "media": parts[1].strip(), "event": parts[2].strip()})
            
        # [NEW] Parse Spectrum
        # Logic: 排除 timeline (通常有日期) 和 header，抓取 4 個欄位的資料
        if "|" in line and len(line.split("|")) >= 4 and not line.startswith("###") and not "日期" in line:
            parts = line.split("|")
            try:
                data["spectrum"].append({
                    "source": parts[0].strip(),
                    "stance": float(parts[1].strip()),
                    "credibility": float(parts[2].strip()),
                    "url": parts[3].strip()
                })
            except: pass

    # Parse Report Text
    if "### [REPORT_TEXT]" in text:
        data["report_text"] = text.split("### [REPORT_TEXT]")[1].strip()
    elif "### REPORT_TEXT" in text:
        data["report_text"] = text.split("### REPORT_TEXT")[1].strip()
    else:
        # Fallback extraction
        match = re.search(r"(#+\s*.*摘要|1\.\s*.*摘要)", text)
        if match: data["report_text"] = text[match.start():]
        else: data["report_text"] = text

    return data

def render_spectrum_chart(spectrum_data):
    if not spectrum_data: return None
    
    df = pd.DataFrame(spectrum_data)
    
    # 建立散布圖
    fig = px.scatter(
        df, 
        x="stance", 
        y="credibility", 
        hover_name="source",
        text="source",
        size=[15]*len(df), # 固定點大小
        color="stance",
        color_continuous_scale=["#2e7d32", "#eeeeee", "#d32f2f"], # 綠 -> 白 -> 紅
        range_x=[-11, 11],
        range_y=[-1, 11],
        labels={"stance": "政治立場 (綠 <-> 藍/紅)", "credibility": "可信度/專業度"},
        title="輿論光譜分佈圖 (AI 動態判定)"
    )
    
    # 加入象限背景與標註
    fig.add_shape(type="rect", x0=-11, y0=5, x1=0, y1=11, fillcolor="rgba(46, 125, 50, 0.1)", layer="below", line_width=0) # 綠營權威
    fig.add_shape(type="rect", x0=0, y0=5, x1=11, y1=11, fillcolor="rgba(21, 101, 192, 0.1)", layer="below", line_width=0) # 藍營權威
    fig.add_shape(type="rect", x0=-11, y0=-1, x1=11, y1=5, fillcolor="rgba(255, 167, 38, 0.1)", layer="below", line_width=0) # 農場/低可信
    
    fig.update_layout(
        xaxis_title="◀ 泛綠觀點 --------- 中立 --------- 泛藍/官方觀點 ▶",
        yaxis_title="可信度 (低 -> 高)",
        showlegend=False,
        height=500
    )
    fig.update_traces(textposition='top center')
    
    return fig

# ==========================================
# 4. 介面 (UI)
# ==========================================
with st.sidebar:
    st.title("全域情報中心 V13")
    st.caption("核心：Cofacts 查核 + 輿論光譜 + 動態識別")
    
    with st.expander("🔑 系統設定", expanded=True):
        if "GOOGLE_API_KEY" in st.secrets:
            st.success("✅ Gemini Key Ready")
            google_key = st.secrets["GOOGLE_API_KEY"]
        else:
            google_key = st.text_input("Gemini Key", type="password")

        if "TAVILY_API_KEY" in st.secrets:
            st.success("✅ Tavily Key Ready")
            tavily_key = st.secrets["TAVILY_API_KEY"]
        else:
            tavily_key = st.text_input("Tavily Key", type="password")
            
        model_name = st.selectbox("模型", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)

    # 歷史報告匯入區
    with st.expander("📂 滾動式追蹤 (匯入舊報告)", expanded=False):
        past_report_input = st.text_area("貼上之前的 Markdown 報告：", height=100)

# 主畫面
st.title("⚖️ 全域觀點搜尋 (Full Spectrum)")
query = st.text_input("輸入議題關鍵字", placeholder="例如：台積電美國設廠爭議")
search_btn = st.button("🚀 啟動全域掃描", type="primary")

if 'result' not in st.session_state: st.session_state.result = None
if 'cofacts' not in st.session_state: st.session_state.cofacts = None

if search_btn and query:
    st.session_state.result = None 
    st.session_state.cofacts = None
    
    with st.spinner("🕵️‍♂️ 正在調閱 Cofacts 查核資料庫 & 掃描全網輿論..."):
        report_context = past_report_input if past_report_input.strip() else None
        
        raw_text, sources, cofacts_txt = run_fusion_analysis(query, google_key, tavily_key, model_name, context_report=report_context)
        
        parsed_data = parse_gemini_data(raw_text)
        st.session_state.result = parsed_data
        st.session_state.cofacts = cofacts_txt
        st.rerun()

if st.session_state.result:
    data = st.session_state.result
    
    # 1. 顯示查核結果 (如果有)
    if st.session_state.cofacts:
        st.markdown(f"""
        <div class="fact-check-box">
            <div class="fact-check-title">🛡️ Cofacts 真的假的 - 自動查核警示</div>
            <div style="white-space: pre-wrap; margin-top: 10px; font-size: 0.9em;">{st.session_state.cofacts}</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. 核心指標
    scores = data.get("scores", {})
    c1, c2, c3, c4 = st.columns(4)
    metrics = [("傳播熱度", scores.get("Attack", 0)), ("觀點分歧", scores.get("Division", 0)),
               ("影響潛力", scores.get("Impact", 0)), ("資訊透明", scores.get("Resilience", 0))]
    
    for col, (label, score) in zip([c1, c2, c3, c4], metrics):
        col.markdown(f"""
        <div class="metric-container">
            <p class="metric-score" style="color: {get_score_text_color(score)};">{score}</p>
            <p class="metric-label">{label}</p>
        </div>
        """, unsafe_allow_html=True)

    # 3. [NEW] 輿論光譜圖 (Plotly)
    st.markdown("---")
    st.subheader("🗺️ 輿論陣地光譜 (AI 動態識別)")
    st.caption("X軸：政治立場 (左綠/右藍) | Y軸：資訊可信度 (上高/下低)")
    if data["spectrum"]:
        fig = render_spectrum_chart(data["spectrum"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("尚無足夠資料繪製光譜圖。")

    # 4. 分析報告
    st.markdown("---")
    st.subheader("📝 深度分析報告")
    st.markdown(data.get("report_text", "無分析報告。"))
    
    # 5. 時間軸
    st.markdown("---")
    with st.expander("📅 關鍵發展時序表"):
        if data["timeline"]:
            st.dataframe(pd.DataFrame(data["timeline"]), width='stretch', hide_index=True)
