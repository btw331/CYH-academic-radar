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
from urllib.parse import urlparse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from tavily import TavilyClient

# ==========================================
# 1. 基礎設定與 CSS樣式 (融合舊版卡片風格)
# ==========================================
st.set_page_config(page_title="全域觀點解析 V25.0", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    /* 舊版經典指標卡片回歸 */
    .metric-container {
        text-align: center;
        padding: 15px;
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #f0f0f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        margin-bottom: 10px;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .metric-score { font-size: 2.5em; font-weight: 700; margin: 0; line-height: 1.2;}
    .metric-label { font-size: 1.0em; font-weight: 500; margin-top: 5px; color: #666; letter-spacing: 1px; }
    
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
        font-size: 0.85em; color: #757575; background-color: #f0f0f0;
        padding: 2px 6px; border-radius: 4px; margin: 0 2px;
        font-family: sans-serif; border: 1px solid #e0e0e0; font-weight: 500;
    }

    .perspective-box {
        padding: 15px; border-radius: 8px; margin-bottom: 10px; font-size: 0.95em;
        border-left-width: 4px; border-left-style: solid; background-color: #fff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .box-green { border-left-color: #2e7d32; }
    .box-blue { border-left-color: #1565c0; }
    .box-neutral { border-left-color: #616161; }
    
    .stDataFrame { border: 1px solid #f0f0f0; border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 資料庫與共用常數
# ==========================================
TAIWAN_WHITELIST = [
    "udn.com", "ltn.com.tw", "chinatimes.com", "cna.com.tw", 
    "storm.mg", "setn.com", "ettoday.net", "tvbs.com.tw", 
    "mirrormedia.mg", "thenewslens.com", "upmedia.mg", 
    "rwnews.tw", "news.pts.org.tw", "ctee.com.tw", "businessweekly.com.tw",
    "news.yahoo.com.tw"
]

INDIE_WHITELIST = [
    "twreporter.org", "theinitium.com", "thenewslens.com", 
    "mindiworldnews.com", "vocus.cc", "matters.town", 
    "plainlaw.me", "whogovernstw.org", "rightplus.org", 
    "biosmonthly.com", "storystudio.tw", "womany.net", "dq.yam.com"
]

# 完整資料庫分類 (用於分類標籤)
DB_MAP = {
    "CHINA": ["xinhuanet.com", "people.com.cn", "huanqiu.com", "cctv.com", "chinadaily.com.cn", "taiwan.cn", "gwytb.gov.cn", "guancha.cn"],
    "GREEN": ["ltn.com.tw", "ftvnews.com.tw", "setn.com", "rti.org.tw", "newtalk.tw", "mirrormedia.mg", "dpp.org.tw"],
    "BLUE": ["udn.com", "chinatimes.com", "tvbs.com.tw", "cti.com.tw", "nownews.com", "ctee.com.tw", "kmt.org.tw"],
    "OFFICIAL": ["cna.com.tw", "pts.org.tw", "mnd.gov.tw", "mac.gov.tw", "tfc-taiwan.org.tw"],
    "INDIE": ["twreporter.org", "theinitium.com", "thenewslens.com", "upmedia.mg", "storm.mg", "mindiworldnews.com", "vocus.cc"],
    "INTL": ["bbc.com", "cnn.com", "reuters.com", "apnews.com", "bloomberg.com", "wsj.com", "nytimes.com", "dw.com", "voanews.com"],
    "FARM": ["kknews.cc", "read01.com", "ppfocus.com", "buzzhand.com", "bomb01.com", "qiqi.news", "inf.news", "toutiao.com"]
}

NAME_KEYWORDS = {
    "CHINA": ["新華", "人民日報", "環球", "央視", "國台辦", "中評", "解放軍", "陸媒", "北京", "宋濤", "xinhuanet", "huanqiu"],
    "GREEN": ["自由", "三立", "民視", "新頭殼", "鏡週刊", "民進黨", "賴清德", "綠營", "獨派", "抗中保台", "ltn", "setn", "ftv"],
    "BLUE": ["聯合", "中國時報", "中時", "TVBS", "中天", "工商時報", "旺旺", "國民黨", "KMT", "侯友宜", "藍營", "統派", "udn", "chinatimes"],
    "FARM": ["網傳", "謠言", "爆料", "內容農場", "PTT", "Dcard", "爆料公社"],
    "OFFICIAL": ["中央社", "公視", "cna", "pts", "gov"],
    "VIDEO": ["YouTube", "YouTuber", "網紅", "TikTok", "抖音", "館長", "直播"]
}

# 舊版戰略模組 (Narratives)
NARRATIVE_MODULES_STR = """
1. 疑美論/國際孤立論
2. 戰爭恐懼/兩岸緊張
3. 施政爭議/治理能力
4. 文化認同/民族情感
5. 軍力懸殊/投降主義
6. 經濟依賴/惠台措施
7. 法律戰/主權爭議
8. 體制優越論
9. 內部協力/政治攻防
"""

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
        "INDIE": ("🕵️ 獨立/深度", "#fbc02d"),
        "INTL": ("🌏 國際媒體", "#f57c00"),
        "VIDEO": ("🟣 影音社群", "#7b1fa2"),
        "OTHER": ("📄 其他來源", "#9e9e9e")
    }
    return meta.get(cat, ("📄 其他", "#9e9e9e"))

def get_score_text_color(score):
    if score >= 80: return "#d32f2f"
    if score >= 60: return "#e65100"
    if score >= 40: return "#f57f17"
    if score >= 20: return "#388e3c"
    return "#757575"

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

# V24 的精準搜尋邏輯 (保留)
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
        target_domains = [] 
        
        has_taiwan = False
        has_indie = False
        has_intl = False
        
        if not isinstance(selected_regions, list): selected_regions = [selected_regions]

        for r in selected_regions:
            if "台灣" in r: 
                has_taiwan = True
                suffixes.append("台灣 新聞" if is_chinese(query) else "Taiwan News")
                target_domains.extend(TAIWAN_WHITELIST)
            
            if "獨立" in r:
                has_indie = True
                suffixes.append("評論 深度報導") 
                target_domains.extend(INDIE_WHITELIST)
                
            if "亞洲" in r: has_intl = True; suffixes.append("Asia News")
            if "歐洲" in r: has_intl = True; suffixes.append("Europe News")
            if "美洲" in r: has_intl = True; suffixes.append("US Americas News")
        
        if not suffixes: suffixes.append("News")
        
        search_q = f"{query} {' '.join(suffixes)}"
        if context_report: search_q += " analysis"
        
        search_params["query"] = search_q

        if (has_taiwan or has_indie) and not has_intl:
            search_params["include_domains"] = list(set(target_domains))
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
        
        for i, res in enumerate(results):
            title = res.get('title', 'No Title')
            url = res.get('url', '#')
            pub_date = res.get('published_date')
            if pub_date:
                pub_date = pub_date[:10]
            else:
                pub_date = "----" 
            
            content = res.get('content', '')[:1200]
            context_text += f"Source {i+1}: [Date: {pub_date}] [Title: {title}] {content} (URL: {url})\n"
            
        return context_text, results, actual_query
        
    except Exception as e:
        return f"Error: {str(e)}", [], "Error"

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5), reraise=True)
def call_gemini(system_prompt, user_text, model_name, api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    chain = prompt | llm
    return chain.invoke({"input": user_text}).content

# 3.4 核心邏輯：深度戰略分析 (V25.0 恢復舊版強大邏輯)
def run_strategic_analysis(query, context_text, model_name, api_key, mode="FUSION"):
    # 模式 A: 全域深度解析 (Fusion) - 側重現狀與事實查核
    if mode == "FUSION":
        system_prompt = f"""
        你是一位集「深度調查記者」與「媒體識讀專家」於一身的情報分析師。
        請針對議題「{query}」進行【全域深度解析】，整合事實查核與觀點分析。
        
        【評分指標 (0-100)】(請根據 Context 內容進行量化評估)：
        1. Attack (傳播熱度): 討論密度與情緒強烈度。
        2. Division (觀點分歧): 陣營間的對立程度。
        3. Impact (影響潛力): 對政策或社會的潛在影響。
        4. Resilience (資訊透明): 官方資料與查核的完整度。
        *Threat (綜合爭議指數): 綜合上述指標的加權評分。

        【輸出格式 (嚴格遵守)】：
        ### [DATA_SCORES]
        Threat: [分數]
        Attack: [分數]
        Impact: [分數]
        Division: [分數]
        Resilience: [分數]
        
        ### [DATA_TIMELINE]
        (格式：YYYY-MM-DD|媒體|標題) -> 務必從 Context [Date:...] 提取
        
        ### [REPORT_TEXT]
        (Markdown 報告 - 請使用 [Source X] 引用來源)
        請包含以下章節：
        1. **📊 全域現況摘要 (Situation)**
        2. **🔍 爭議點事實查核矩陣 (Fact-Check)**
        3. **⚖️ 媒體觀點光譜對照 (藍/綠/紅/獨)**
        4. **🧠 深度識讀與利益分析 (Cui Bono)**
        5. **🤔 關鍵反思**
        """
        
    # 模式 B: 未來發展推演 (Scenario) - 側重戰略預測
    else:
        system_prompt = f"""
        你是一位資深的趨勢預測分析師。請針對「{query}」進行戰略推演。
        
        【分析核心 (Foresight Framework)】：
        1. **第一性原理 (First Principles)**：剖析議題背後的底層驅動力 (如：人口結構、經濟誘因、地緣政治)。
        2. **可能性圓錐 (Cone of Plausibility)**：推演三種未來發展路徑。
            - **基準情境 (Baseline)**: 若現狀持續，最可能的結果。
            - **轉折情境 (Plausible)**: 關鍵變數改變後的合理發展。
            - **極端情境 (Wild Card)**: 低機率但高衝擊的黑天鵝事件。

        【評分定義 (趨勢版)】：
        1. Attack -> 影響顯著性 (Significance)
        2. Division -> 發展不確定性 (Uncertainty)
        3. Impact -> 時間緊迫度 (Urgency)
        4. Resilience -> 系統複雜度 (Complexity)
        *Threat -> 綜合影響力 (Impact Index)

        【輸出格式】：
        ### [DATA_SCORES]
        Threat: [分數]
        Attack: [分數]
        Impact: [分數]
        Division: [分數]
        Resilience: [分數]
        
        ### [DATA_TIMELINE]
        (格式：YYYY-MM-DD|媒體|標題)
        
        ### [REPORT_TEXT]
        (Markdown 報告)
        1. **🎯 第一性原理拆解 (底層邏輯)**
        2. **🔮 未來情境模擬 (可能性圓錐)**
           - 基準情境
           - 轉折情境
           - 極端情境
        3. **💡 綜合戰略建議**
        """

    return call_gemini(system_prompt, context_text, model_name, api_key)

# 3.5 資料解析器 (支援分數提取)
def parse_gemini_data(text):
    data = {"scores": {"Threat":0, "Attack":0, "Impact":0, "Division":0, "Resilience":0}, 
            "timeline": [], "report_text": ""}
    
    if not text: return data

    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        
        # 解析分數
        for key in data["scores"]:
            if f"{key}:" in line:
                try: 
                    score_match = re.search(r'\d+', line)
                    if score_match:
                        data["scores"][key] = int(score_match.group())
                except: pass
        
        # 解析時間軸
        if "|" in line and len(line.split("|")) >= 3 and (line[0].isdigit() or "20" in line or "Future" in line):
            parts = line.split("|")
            try:
                data["timeline"].append({
                    "date": parts[0].strip(),
                    "media": parts[1].strip(),
                    "title": parts[2].strip()
                })
            except: pass

    # 解析報告內文
    if "### [REPORT_TEXT]" in text:
        data["report_text"] = text.split("### [REPORT_TEXT]")[1].strip()
    elif "### REPORT_TEXT" in text:
        data["report_text"] = text.split("### REPORT_TEXT")[1].strip()
    else:
        # Fallback: 嘗試找第一個標題
        match = re.search(r"(#+\s*.*摘要|1\.\s*.*摘要|#+\s*.*第一性原理)", text)
        if match:
            data["report_text"] = text[match.start():]
        else:
            data["report_text"] = text

    return data

# 4. 下載功能
def convert_data_to_json(data):
    import json
    return json.dumps(data, indent=2, ensure_ascii=False)

def convert_data_to_md(data):
    return f"""
# 全域觀點分析報告
产生時間: {datetime.now()}

## 1. 核心指標
Threat: {data['scores']['Threat']} | Attack: {data['scores']['Attack']} | Division: {data['scores']['Division']}

## 2. 深度分析
{data.get('report_text')}

## 3. 時間軸
{pd.DataFrame(data.get('timeline')).to_markdown(index=False)}
    """

# ==========================================
# 5. UI
# ==========================================
with st.sidebar:
    st.title("全域觀點解析 V25.0")
    
    # [V25.0] 恢復雙模式選擇
    analysis_mode = st.radio(
        "選擇分析引擎：",
        options=["全域深度解析 (Fusion)", "未來發展推演 (Scenario)"],
        captions=["側重：事實查核 + 利益分析", "側重：第一性原理 + 可能性圓錐"],
        index=0
    )
    st.markdown("---")
    
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
            "搜尋時間範圍",
            options=[3, 7, 14, 30, 90, 1825],
            format_func=lambda x: "📅 不限時間 (5年)" if x == 1825 else f"近 {x} 天",
            index=2
        )
        
        max_results = st.slider("搜尋篇數上限", 10, 60, 20)
        
        selected_regions = st.multiselect(
            "搜尋視角 (Region)",
            ["🇹🇼 台灣 (Taiwan)", "🌏 亞洲 (Asia)", "🌍 歐洲 (Europe)", "🌎 美洲 (Americas)", "🕵️ 獨立/自媒體 (Indie)"],
            default=["🇹🇼 台灣 (Taiwan)"]
        )

    # [V25.0] 指標定義回歸
    with st.expander("📖 評分指標定義 (Metrics)", expanded=False):
        if "未來" in analysis_mode:
            st.markdown("""
- **1. 影響顯著性 (Significance)**: 議題權重 + 影響層級
- **2. 發展不確定性 (Uncertainty)**: 未知變數 / 總變數
- **3. 時間緊迫度 (Urgency)**: 1 / (剩餘反應時間)
- **4. 系統複雜度 (Complexity)**: 利害關係人數量 * 耦合度
            """)
        else:
            st.markdown("""
- **1. 傳播熱度 (Attack)**: 媒體報導量 + 情緒強度
- **2. 觀點分歧 (Division)**: 陣營對立度 + 模糊度
- **3. 影響潛力 (Impact)**: 受眾規模 + 政策連動
- **4. 資訊透明 (Resilience)**: 官方資料 + 第三方查核
            """)

st.title(f"{analysis_mode.split(' ')[0]}")
query = st.text_input("輸入議題關鍵字", placeholder="例如：台積電美國設廠爭議")
search_btn = st.button("🚀 啟動全域掃描", type="primary")

if 'result' not in st.session_state: st.session_state.result = None
if 'sources' not in st.session_state: st.session_state.sources = None

if search_btn and query and google_key and tavily_key:
    st.session_state.result = None
    
    with st.status("🚀 啟動全域掃描引擎 (V25.0)...", expanded=True) as status:
        
        days_label = "不限時間" if search_days == 1825 else f"近 {search_days} 天"
        regions_label = ", ".join([r.split(" ")[1] for r in selected_regions])
        st.write(f"📡 1. 連線 Tavily 搜尋 (視角: {regions_label} / 時間: {days_label})...")
        
        context_text, sources, actual_query = get_search_context(query, tavily_key, search_days, selected_regions, max_results)
        st.session_state.sources = sources
        
        st.write("🛡️ 2. 查詢 Cofacts 謠言資料庫 (API)...")
        cofacts_txt = search_cofacts(query)
        if cofacts_txt: context_text += f"\n{cofacts_txt}\n"
        
        st.write("🧠 3. AI 進行戰略深度分析 (模式: {})...".format("未來推演" if "未來" in analysis_mode else "深度解析"))
        
        mode_code = "V205" if "未來" in analysis_mode else "FUSION"
        raw_report = run_strategic_analysis(query, context_text, model_name, google_key, mode=mode_code)
        st.session_state.result = parse_gemini_data(raw_report)
            
        status.update(label="✅ 分析完成", state="complete", expanded=False)
        
    st.rerun()

if st.session_state.result:
    data = st.session_state.result
    scores = data.get("scores", {})
    
    # 1. 顯示指標卡片 (Metric Cards) - 舊版靈魂
    c1, c2, c3, c4 = st.columns(4)
    
    if "未來" in analysis_mode:
        metrics = [("影響顯著性", scores.get("Attack", 0)), ("發展不確定性", scores.get("Division", 0)),
                   ("時間緊迫度", scores.get("Impact", 0)), ("系統複雜度", scores.get("Resilience", 0))]
    else:
        metrics = [("傳播熱度", scores.get("Attack", 0)), ("觀點分歧", scores.get("Division", 0)),
                   ("影響潛力", scores.get("Impact", 0)), ("資訊透明", scores.get("Resilience", 0))]
    
    for col, (label, score) in zip([c1, c2, c3, c4], metrics):
        text_color = get_score_text_color(score)
        col.markdown(f"""
        <div class="metric-container">
            <p class="metric-score" style="color: {text_color};">{score}</p>
            <p class="metric-label">{label}</p>
        </div>
        """, unsafe_allow_html=True)

    # 2. 顯示時間軸 (V23 精華)
    if data.get("timeline"):
        st.markdown("---")
        st.markdown("### 📅 關鍵發展時序")
        
        timeline_data = []
        for item in data["timeline"]:
            media_name = item['media']
            # 盲測模式
            if blind_mode: media_name = "*****"
            
            # 分類燈號
            cat = classify_media_name(media_name)
            emoji = "⚪"
            if cat == "CHINA": emoji = "🔴"
            elif cat == "BLUE": emoji = "🔵"
            elif cat == "GREEN": emoji = "🟢"
            elif cat == "FARM": emoji = "🟠"
            
            timeline_data.append({
                "日期": item['date'],
                "來源": f"{emoji} {media_name}",
                "事件摘要": item['title']
            })
        
        st.dataframe(pd.DataFrame(timeline_data), use_container_width=True, hide_index=True)

    # 3. 顯示深度報告 (V-Legacy 靈魂)
    st.markdown("---")
    st.markdown("### 📝 綜合戰略分析報告")
    formatted_text = format_citation_style(data.get("report_text", ""))
    st.markdown(f'<div class="report-paper">{formatted_text}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("📥 下載報告 (JSON)", type="secondary"):
        st.download_button("Download", convert_data_to_json(data), "report.json", "application/json")

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
