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
from urllib.parse import urlparse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from tavily import TavilyClient

# ==========================================
# 1. 基礎設定與 CSS樣式
# ==========================================
st.set_page_config(page_title="全域觀點搜尋 V22.1", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    /* 舊版經典指標卡片 */
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
    
    /* 滾動按鈕樣式 */
    .stButton button[kind="secondary"] {
        border: 2px solid #673ab7;
        color: #673ab7;
        font-weight: bold;
    }
    
    /* 來源連結樣式 */
    .source-link { 
        color: #1565c0 !important; 
        text-decoration: none; 
        font-weight: bold;
    }
    .source-link:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 資料庫與共用常數 (新增 INDIE)
# ==========================================
# [V22.0] 台灣主流媒體
TAIWAN_WHITELIST = [
    "udn.com", "ltn.com.tw", "chinatimes.com", "cna.com.tw", 
    "setn.com", "ettoday.net", "tvbs.com.tw", "ctee.com.tw", 
    "businessweekly.com.tw", "news.yahoo.com.tw", "mirrormedia.mg"
]

# [V22.1] 獨立/自媒體/深度媒體白名單
INDIE_WHITELIST = [
    "twreporter.org",       # 報導者
    "theinitium.com",       # 端傳媒
    "thenewslens.com",      # 關鍵評論網
    "storm.mg",             # 風傳媒 (深度區)
    "upmedia.mg",           # 上報
    "mindiworldnews.com",   # 敏迪選讀
    "vocus.cc",             # 方格子 (自媒體)
    "matters.town",         # Matters (Web3 自媒體)
    "plainlaw.me",          # 法律白話文
    "whogovernstw.org",     # 菜市場政治學
    "rightplus.org",        # 多多益善
    "biosmonthly.com",      # BIOS monthly
    "storystudio.tw",       # 故事 StoryStudio
    "womany.net",           # 女人迷
    "dq.yam.com"            # 地球圖輯隊
]

# 舊版分類對照表 (用於分類標籤)
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

NARRATIVE_MODULES_LIST = [
    "疑美論/國際孤立論", "戰爭恐懼/兩岸緊張", "施政爭議/治理能力", "文化認同/民族情感",
    "軍力懸殊/投降主義", "經濟依賴/惠台措施", "法律戰/主權爭議", "體制優越論", "內部協力/政治攻防"
]
NARRATIVE_MODULES_STR = "\n".join([f"{i+1}. {m}" for i, m in enumerate(NARRATIVE_MODULES_LIST)])

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

def classify_source(url):
    url_str = url.lower()
    for cat, keywords in DB_MAP.items():
        for kw in keywords:
            if kw in url_str: return cat
    return "OTHER"

def get_score_text_color(score):
    if score >= 80: return "#d32f2f"
    if score >= 60: return "#e65100"
    if score >= 40: return "#f57f17"
    if score >= 20: return "#388e3c"
    return "#757575"

def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

# ==========================================
# 3. 雙核融合分析引擎 (V22.1 邏輯)
# ==========================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def call_gemini_with_retry(chain, input_data):
    return chain.invoke(input_data)

# [V22.1] 搜尋邏輯升級：支援白名單混合模式
def get_search_results(query, api_key_tavily, days_back, selected_regions, max_results):
    try:
        tavily = TavilyClient(api_key=api_key_tavily)
        
        search_params = {
            "search_depth": "advanced",
            "topic": "general",
            "days": days_back,
            "max_results": max_results
        }

        suffixes = []
        target_domains = [] # 收集白名單網域
        
        # 1. 處理區域選擇
        has_taiwan = False
        has_indie = False
        has_intl = False
        
        for r in selected_regions:
            if "台灣" in r: 
                has_taiwan = True
                suffixes.append("台灣 新聞" if is_chinese(query) else "Taiwan News")
                target_domains.extend(TAIWAN_WHITELIST)
            
            if "獨立" in r:
                has_indie = True
                # 獨立媒體通常也是中文，不需要特別加英文後綴，除非想搜 global indie
                suffixes.append("評論 深度報導") 
                target_domains.extend(INDIE_WHITELIST)
                
            if "亞洲" in r: 
                has_intl = True
                suffixes.append("Asia News")
            if "歐洲" in r: 
                has_intl = True
                suffixes.append("Europe News")
            if "美洲" in r: 
                has_intl = True
                suffixes.append("US Americas News")
        
        if not suffixes: suffixes.append("News")
        search_q = f"{query} {' '.join(suffixes)}"
        search_params["query"] = search_q

        # 2. 決定是否啟用白名單 (include_domains)
        # 邏輯：如果只選了「台灣」或「獨立媒體」（沒有選國際區域），則啟用嚴格白名單
        if (has_taiwan or has_indie) and not has_intl:
            search_params["include_domains"] = list(set(target_domains)) # 去重
        else:
            # 如果混選了國際區域，不能用 include_domains (會把國際新聞濾掉)
            # 改用 exclude_domains 排除垃圾
            search_params["exclude_domains"] = [
                "daum.net", "naver.com", "tistory.com",
                "espn.com", "bleacherreport.com", "cbssports.com", 
                "pinterest.com", "amazon.com", "tripadvisor.com"
            ]
        
        # 執行搜尋
        response = tavily.search(**search_params)
        results = response.get('results', [])
        
        context_text = ""
        for i, res in enumerate(results):
            pub_date = res.get('published_date', 'Recent')[:10]
            context_text += f"Source {i+1}: [Date: {pub_date}] [Title: {res.get('title')}] {str(res.get('content'))[:2000]} (URL: {res.get('url')})\n"
            
        return context_text, results

    except Exception as e:
        return f"SEARCH_ERROR: {str(e)}", []

def run_fusion_analysis(query, api_key_google, api_key_tavily, model_name, days_back, selected_regions, max_results, mode="FUSION", context_report=None):
    os.environ["GOOGLE_API_KEY"] = api_key_google
    
    # [V22.1] 呼叫新的搜尋邏輯
    context_text, results = get_search_results(query, api_key_tavily, days_back, selected_regions, max_results)
    
    if context_report and len(context_report) > 50:
        full_context = f"【前次分析報告 (歷史背景)】\n{context_report}\n\n【本次搜尋情報】\n{context_text}"
        task_instruction = f"你已收到一份歷史分析報告。請以此為基礎，結合今日最新情報，進行「滾動式」的未來情境模擬。"
    else:
        full_context = context_text
        task_instruction = f"請針對議題「{query}」進行【全域深度解析】，整合事實查核與觀點分析。"

    if mode == "V205":
        system_prompt = f"""
        你是一位資深的趨勢預測分析師。{task_instruction}
        
        【分析核心 (Foresight Framework)】：
        1. **第一性原理 (First Principles)**：剖析議題背後的底層驅動力。
        2. **可能性圓錐 (Cone of Plausibility)**：推演三種未來發展路徑 (基準、轉折、極端)。

        【評分定義】：
        1. Attack -> 影響顯著性
        2. Division -> 發展不確定性
        3. Impact -> 時間緊迫度
        4. Resilience -> 系統複雜度
        *Threat -> 綜合影響力

        【輸出格式】：
        ### [DATA_SCORES]
        Threat: [分數]
        Attack: [分數]
        Impact: [分數]
        Division: [分數]
        Resilience: [分數]
        
        ### [DATA_TIMELINE]
        (格式：Future-Date|預測|事件)
        
        ### [DATA_NARRATIVES]
        (第一性原理,5)

        ### [REPORT_TEXT]
        (Markdown 報告)
        # 🎯 第一性原理拆解
        # 🔮 未來情境模擬
        # 💡 綜合建議
        """
    else:
        system_prompt = f"""
        你是一位集「深度調查記者」與「媒體識讀專家」於一身的情報分析師。
        請針對議題「{query}」進行【全域深度解析】。
        
        【評分指標 (0-100)】：
        1. Attack (傳播熱度)
        2. Division (觀點分歧)
        3. Impact (影響潛力)
        4. Resilience (資訊透明)
        *Threat (爭議指數)

        【輸出格式 (嚴格遵守)】：
        ### [DATA_SCORES]
        Threat: [分數]
        Attack: [分數]
        Impact: [分數]
        Division: [分數]
        Resilience: [分數]
        
        ### [DATA_TIMELINE]
        (格式：YYYY-MM-DD|媒體|標題) -> 請務必從 Context [Date:...] 提取日期
        
        ### [DATA_NARRATIVES]
        (格式：劇本名稱,強度1-5)
        {NARRATIVE_MODULES_STR}

        ### [REPORT_TEXT]
        (Markdown 報告 - 請使用 [Source X] 引用來源)
        請包含以下章節：
        1. **📊 全域現況摘要**
        2. **🔍 爭議點事實查核矩陣 (Fact-Check)**
        3. **⚖️ 媒體觀點光譜對照 (包含獨立/自媒體觀點)**
        4. **🧠 深度識讀與利益分析 (Cui Bono)**
        5. **🤔 關鍵反思**
        """

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{full_context}")])
    chain = prompt | llm
    response = call_gemini_with_retry(chain, {"full_context": full_context})
    return response.content, results

def parse_gemini_data(text):
    data = {"scores": {"Threat":0, "Attack":0, "Impact":0, "Division":0, "Resilience":0}, 
            "narratives": {}, "timeline": [], "report_text": ""}
    
    if not text or text.startswith("ERROR") or text == "API_LIMIT_ERROR":
        data["report_text"] = text
        return data

    for line in text.split('\n'):
        line = line.strip()
        for key in data["scores"]:
            if f"{key}:" in line:
                try: data["scores"][key] = int(re.search(r'\d+', line).group())
                except: pass
        
        if "|" in line and (line[0].isdigit() or "Future" in line):
            parts = line.split("|")
            if len(parts) >= 3:
                data["timeline"].append({
                    "date": parts[0].strip(),
                    "media": parts[1].strip(),
                    "event": parts[2].strip()
                })

    if "### [REPORT_TEXT]" in text:
        data["report_text"] = text.split("### [REPORT_TEXT]")[1].strip()
    elif "### REPORT_TEXT" in text:
        data["report_text"] = text.split("### REPORT_TEXT")[1].strip()
    else:
        match = re.search(r"(#+\s*.*摘要|1\.\s*.*摘要)", text)
        if match:
            data["report_text"] = text[match.start():]
        else:
            data["report_text"] = text

    return data

def generate_download_content(query, data, sources):
    timeline_str = "| 日期 | 媒體 | 事件 |\n|---|---|---|\n"
    for item in data.get('timeline', []):
        timeline_str += f"| {item['date']} | {item['media']} | {item['event']} |\n"

    source_str = ""
    if sources:
        for i, s in enumerate(sources):
            source_str += f"{i+1}. [{s.get('content')[:60]}...]({s.get('url')})\n"

    full_content = f"""
# 分析報告：{query}

**分析時間**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 1. 核心指標評估
* **綜合指數**: {data['scores'].get('Threat', 0)}
* **指標 A**: {data['scores'].get('Attack', 0)}
* **指標 B**: {data['scores'].get('Division', 0)}
* **指標 C**: {data['scores'].get('Impact', 0)}
* **指標 D**: {data['scores'].get('Resilience', 0)}

## 2. 關鍵發展時序
{timeline_str}

## 3. 深度分析內容
{data.get('report_text', '無報告內容')}

## 4. 參考文獻
{source_str}
"""
    return full_content

# ==========================================
# 4. 介面 (UI)
# ==========================================
with st.sidebar:
    st.title("全域觀點搜尋 V22.1")
    
    analysis_mode = st.radio(
        "選擇分析引擎：",
        options=["全域深度解析 (Fusion)", "未來發展推演"],
        captions=["融合：媒體識讀 + 事實查核 + 利益分析", "推演：第一性原理 + 可能性圓錐"],
        index=0
    )
    st.markdown("---")

    with st.expander("📂 匯入前次報告 (持續追蹤用)", expanded=False):
        past_report_input = st.text_area(
            "請貼上之前的分析報告內容 (Markdown)：", 
            height=150, 
            placeholder="在此貼上舊報告，系統將結合今日新情報進行滾動分析..."
        )

    st.markdown("---")
    blind_mode = st.toggle("🙈 盲測模式", value=False)
    
    with st.expander("🔑 設定 & 參數", expanded=True):
        if "GOOGLE_API_KEY" in st.secrets:
            google_key = st.secrets["GOOGLE_API_KEY"]
            st.success("✅ Gemini Key Auto-filled")
        else:
            google_key = st.text_input("Gemini Key", value="", type="password")

        if "TAVILY_API_KEY" in st.secrets:
            tavily_key = st.secrets["TAVILY_API_KEY"]
            st.success("✅ Tavily Key Auto-filled")
        else:
            tavily_key = st.text_input("Tavily Key", value="", type="password")
            
        model_options = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]
        selected_model = st.selectbox("模型選擇", model_options, index=0)
        
        # [V22.0] 時間範圍 (含無限時間)
        search_days = st.selectbox(
            "搜尋時間範圍",
            options=[3, 7, 14, 30, 90, 1825],
            format_func=lambda x: "📅 不限時間 (5年)" if x == 1825 else f"近 {x} 天",
            index=2
        )
        
        # [V22.0] 搜尋篇數 (自訂)
        max_results = st.slider("搜尋篇數上限", 20, 100, 20)
        
        # [V22.1] 區域複選 (含獨立媒體)
        selected_regions = st.multiselect(
            "搜尋視角 (Region) - 可複選",
            ["🇹🇼 台灣 (Taiwan)", "🌏 亞洲 (Asia)", "🌍 歐洲 (Europe)", "🌎 美洲 (Americas)", "🕵️ 獨立/自媒體 (Indie)"],
            default=["🇹🇼 台灣 (Taiwan)"]
        )

    with st.expander("📖 評分指標定義 (含公式)", expanded=False):
        if "未來" in analysis_mode:
            st.markdown("""
- **1. 影響顯著性 (Significance)**: 議題權重 + 影響層級
- **2. 發展不確定性 (Uncertainty)**: 未知變數 / 總變數
- **3. 時間緊迫度 (Urgency)**: 1 / (剩餘反應時間)
- **4. 系統複雜度 (Complexity)**: 利害關係人數量 * 耦合度
            """)
        else:
            st.markdown("""
- **1. 傳播熱度 (Attack)**: 媒體報導量 + 社群聲量
- **2. 觀點分歧 (Division)**: 陣營對立度 + 模糊度
- **3. 影響潛力 (Impact)**: 受眾規模 + 時間
- **4. 資訊透明 (Resilience)**: 官方資料 + 第三方查核
            """)

# 主畫面
st.title(f"全域觀點搜尋 - {analysis_mode.split(' ')[0]}")
query = st.text_input("輸入新聞議題", placeholder="例如：川普對台言論")
search_btn = st.button("🔍 啟動全域掃描", type="primary")

if 'result' not in st.session_state: st.session_state.result = None
if 'sources' not in st.session_state: st.session_state.sources = None
if 'previous_report' not in st.session_state: st.session_state.previous_report = None

if search_btn and query:
    st.session_state.result = None 
    st.session_state.previous_report = None
    
    with st.spinner("🕵️‍♂️ 正在進行全網搜尋與智慧分析..."):
        mode_code = "V205" if "未來" in analysis_mode else "FUSION"
        
        report_context = past_report_input if past_report_input.strip() else None
        
        # [V22.1] 傳遞新參數
        raw_text, sources = run_fusion_analysis(
            query, google_key, tavily_key, selected_model, 
            days_back=search_days, 
            selected_regions=selected_regions,
            max_results=max_results,
            mode=mode_code, 
            context_report=report_context
        )
        
        parsed_data = parse_gemini_data(raw_text)
        st.session_state.result = parsed_data
        st.session_state.sources = sources
        st.rerun()

if st.session_state.result:
    data = st.session_state.result
    sources = st.session_state.sources
    
    if data["report_text"].startswith("ERROR") or data["report_text"] == "API_LIMIT_ERROR":
        st.error(f"⚠️ 系統訊息：{data['report_text']}")
        if data["report_text"] == "API_LIMIT_ERROR":
            st.info("💡 建議：請稍等 30 秒後再試。")
    else:
        # 1. 彩色文字指標
        scores = data.get("scores", {})
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
            
        # 2. 時間軸表格 (加入燈號)
        st.markdown("---")
        st.subheader("📅 關鍵發展時序")
        if data["timeline"]:
            processed_data = []
            for item in data["timeline"]:
                media_name = item['media']
                # 盲測模式處理
                display_media = "*****" if blind_mode else media_name
                
                cat = classify_media_name(media_name)
                emoji = "⚪"
                if cat == "CHINA": emoji = "🔴"
                elif cat == "BLUE": emoji = "🔵"
                elif cat == "GREEN": emoji = "🟢"
                elif cat == "FARM": emoji = "🟠"
                elif cat == "VIDEO": emoji = "🟣"
                elif cat == "INTL": emoji = "🌏"
                
                processed_data.append({
                    "日期": item['date'],
                    "來源": f"{emoji} {display_media}",
                    "事件摘要": item['event']
                })
            
            df = pd.DataFrame(processed_data)
            st.dataframe(df, width=1200, hide_index=True, use_container_width=True)
        else:
            st.info("無時間軸資料。")

        # 3. 綜合分析報告
        st.markdown("---")
        st.subheader("📝 綜合分析報告")
        
        # 資訊滾動按鈕
        if "未來" not in analysis_mode:
            st.info("💡 戰略升級：您可以將此分析結果作為基礎，進行「可能性圓錐」推演。")
            
            def on_roll_click(current_report):
                st.session_state.previous_report = current_report
                
            if st.button("🚀 進行資訊滾動：將此結果餵給未來發展推演", type="secondary", on_click=on_roll_click, args=(data["report_text"],)):
                with st.spinner("🔮 正在讀取前次情報，啟動第一性原理推演..."):
                    raw_text, _ = run_fusion_analysis(
                        query, google_key, tavily_key, selected_model, 
                        days_back=search_days, selected_regions=selected_regions, max_results=max_results,
                        mode="V205", context_report=st.session_state.previous_report
                    )
                    st.session_state.result = parse_gemini_data(raw_text)
                    st.rerun()

        st.markdown(data.get("report_text", "無分析報告。"))
        
        # 4. 下載
        st.markdown("---")
        download_content = generate_download_content(query, data, sources)
        st.download_button(label="📥 下載報告", data=download_content, file_name="Report.md", type="primary")

        # 5. 參考文獻 (純文字表格)
        st.markdown("---")
        st.subheader("📚 參考文獻")
        if sources:
            df_data = []
            for i, s in enumerate(sources):
                domain = get_domain_name(s.get('url'))
                display_domain = "******" if blind_mode else domain
                title = s.get('title', 'No Title') 
                if not title: title = s.get('content', '')[:30] + "..."
                
                df_data.append({"編號": i+1, "媒體/網域": display_domain, "標題摘要": title, "原始連結": s.get('url')})
            
            df = pd.DataFrame(df_data)
            st.dataframe(
                df, 
                column_config={"原始連結": st.column_config.LinkColumn("點擊前往")},
                hide_index=True,
                use_container_width=True
            )
