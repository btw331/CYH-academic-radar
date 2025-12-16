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
# 1. 基礎設定與 CSS樣式
# ==========================================
st.set_page_config(page_title="全域觀點解析 V26.2", page_icon="🔗", layout="wide")

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

    /* 表格優化 */
    .stDataFrame { border: 1px solid #f0f0f0; border-radius: 8px; overflow: hidden; }
    
    /* 連結樣式優化 */
    a { text-decoration: none; color: #0366d6; }
    a:hover { text-decoration: underline; }
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

CAMP_KEYWORDS = {
    "GREEN": ["自由", "三立", "民視", "新頭殼", "鏡週刊", "放言", "賴清德", "民進黨", "青鳥", "中央社", "Liberty Times"],
    "BLUE": ["聯合", "中時", "中國時報", "TVBS", "中天", "風傳媒", "國民黨", "藍營", "赵少康", "United Daily", "China Times"],
    "RED": ["新華", "人民日報", "環球", "央視", "中評", "国台办", "China Daily"]
}

DB_MAP = {
    "CHINA": ["xinhuanet.com", "people.com.cn", "huanqiu.com"],
    "GREEN": ["ltn.com.tw", "ftvnews.com.tw", "setn.com"],
    "BLUE": ["udn.com", "chinatimes.com", "tvbs.com.tw"],
    "OFFICIAL": ["cna.com.tw", "pts.org.tw", "mnd.gov.tw"],
    "INDIE": ["twreporter.org", "theinitium.com", "thenewslens.com"],
    "INTL": ["bbc.com", "cnn.com", "reuters.com"]
}

def get_domain_name(url):
    try: return urlparse(url).netloc.replace("www.", "")
    except: return ""

def get_category_meta(cat):
    meta = {
        "CHINA": ("🇨🇳 中國官媒", "#d32f2f"),
        "GREEN": ("🟢 泛綠觀點", "#2e7d32"),
        "BLUE": ("🔵 泛藍觀點", "#1565c0"),
        "OFFICIAL": ("⚪ 官方/中立", "#546e7a"),
        "INDIE": ("🕵️ 獨立/深度", "#fbc02d"),
        "INTL": ("🌏 國際媒體", "#f57c00")
    }
    return meta.get(cat, ("📄 其他", "#9e9e9e"))

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
            
        return context_text, results, actual_query, (has_taiwan or has_indie) and not has_intl
        
    except Exception as e:
        return f"Error: {str(e)}", [], "Error", False

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5), reraise=True)
def call_gemini(system_prompt, user_text, model_name, api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    chain = prompt | llm
    return chain.invoke({"input": user_text}).content

# 深度戰略分析 (保留 V26.0 核心)
def run_strategic_analysis(query, context_text, model_name, api_key, mode="FUSION"):
    if mode == "FUSION":
        system_prompt = f"""
        你是一位集「深度調查記者」與「媒體識讀專家」於一身的情報分析師。
        請針對議題「{query}」進行【全域深度解析】，整合事實查核與觀點分析。
        
        【任務重點】：
        1. **時間軸建立**: 從 Context 中提取正確的日期與事件順序。
        2. **立場判定**: 請根據「語意分析」與「媒體背景」判斷立場 (-10~+10)。
        3. **深度分析**: 執行「Cui Bono (誰獲益)」利益分析與事實查核。

        【輸出格式 (嚴格遵守)】：
        ### [DATA_TIMELINE]
        (格式：YYYY-MM-DD|媒體|標題|立場(-10~10)|可信度(0-10)|網址) 
        -> **網址 (URL)** 必須對應到 Context 中的 Source Link，不可留白。
        -> 日期若無，請根據內文推斷或標示 "Recent"。
        
        ### [REPORT_TEXT]
        (Markdown 報告 - 請使用 [Source X] 引用來源)
        請包含以下章節：
        1. **📊 全域現況摘要 (Situation)**
        2. **🔍 爭議點事實查核矩陣 (Fact-Check)**
        3. **⚖️ 媒體觀點光譜對照 (藍/綠/紅/獨)**
        4. **🧠 深度識讀與利益分析 (Cui Bono)**
        5. **🤔 關鍵反思**
        """
        
    else: # SCENARIO
        system_prompt = f"""
        你是一位資深的趨勢預測分析師。請針對「{query}」進行戰略推演。
        
        【分析核心 (Foresight Framework)】：
        1. **第一性原理 (First Principles)**：剖析議題背後的底層驅動力。
        2. **可能性圓錐 (Cone of Plausibility)**：推演三種未來發展路徑。

        【輸出格式】：
        ### [DATA_TIMELINE]
        (格式：YYYY-MM-DD|媒體|標題|立場(0)|可信度(5)|網址)
        -> **網址 (URL)** 必須保留，以便使用者點擊查證。
        
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

# 強制校正邏輯 (保留)
def calibrate_stance(media_name, ai_score):
    name_clean = media_name.replace("新聞", "").replace("報導", "").replace("網", "")
    
    if any(k in name_clean for k in CAMP_KEYWORDS["GREEN"]):
        if ai_score > 0: return ai_score * -1
        if ai_score == 0: return -3
        return ai_score

    if any(k in name_clean for k in CAMP_KEYWORDS["BLUE"] + CAMP_KEYWORDS["RED"]):
        if ai_score < 0: return ai_score * -1
        if ai_score == 0: return 3
        return ai_score
        
    return ai_score

# 資料解析器 (含網址清洗)
def parse_gemini_data(text):
    data = {"timeline": [], "report_text": ""}
    
    if not text: return data

    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        
        if "|" in line and len(line.split("|")) >= 4 and not line.startswith("###") and not "YYYY" in line:
            parts = line.split("|")
            try:
                date = parts[0].strip()
                name = parts[1].strip()
                title = parts[2].strip()
                base_stance = 0
                base_cred = 0
                url = "#"
                
                if len(parts) >= 6:
                    base_stance = float(parts[3].strip())
                    base_cred = float(parts[4].strip())
                    url = parts[5].strip()
                    url = url.rstrip(")").rstrip("]").strip()
                elif len(parts) == 5:
                    base_cred = float(parts[3].strip())
                    url = parts[4].strip()
                    url = url.rstrip(")").rstrip("]").strip()

                final_stance = calibrate_stance(name, base_stance)
                
                data["timeline"].append({
                    "date": date,
                    "media": name,
                    "title": title,
                    "stance": int(final_stance),
                    "credibility": int(base_cred), 
                    "url": url
                })
            except: pass

    if "### [REPORT_TEXT]" in text:
        data["report_text"] = text.split("### [REPORT_TEXT]")[1].strip()
    elif "### REPORT_TEXT" in text:
        data["report_text"] = text.split("### REPORT_TEXT")[1].strip()
    else:
        match = re.search(r"(#+\s*.*摘要|1\.\s*.*摘要|#+\s*.*第一性原理)", text)
        if match:
            data["report_text"] = text[match.start():]
        else:
            data["report_text"] = text

    return data

# [V26.2] 渲染時間軸 (樣式不變，加入超連結)
def render_timeline_enhanced(timeline_data):
    if not timeline_data: 
        st.warning("⚠️ 無法生成時間軸：可能是搜尋結果不足，或 AI 無法解析日期。")
        return
    
    st.markdown("### 📅 議題發展時間軸 (News Timeline)")
    
    # 保持 V26.0 的圖例樣式
    st.markdown("""
    <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; font-size:0.9em; margin-bottom:15px;">
        <b>💡 燈號說明：</b><br>
        • <b>政治立場 (Stance)</b>：🟢 負分 (批判/泛綠)；🔵 正分 (體制/泛藍)；⚪ 0 (中立)。<br>
        • <b>可信度 (Credibility)</b>：🟢 高 (7-10)；🟡 中 (4-6)；🔴 低 (0-3)。
    </div>
    """, unsafe_allow_html=True)
    
    md = "| 日期 | 媒體 | 新聞標題 (點擊閱讀) | 立場 | 可信度 |\n|:---:|:---|:---|:---:|:---:|\n"
    for item in timeline_data:
        c = item['credibility']
        if c >= 8: c_txt = f"🟢 高 ({c})"
        elif c >= 5: c_txt = f"🟡 中 ({c})"
        else: c_txt = f"🔴 低 ({c})"
        
        s = item['stance']
        if s < -2: s_txt = f"🟢 {s}"
        elif s > 2: s_txt = f"🔵 +{s}"
        else: s_txt = "⚪ 0"
        
        t_text = item['title']
        if len(t_text) > 35: t_text = t_text[:35] + "..."
        t_url = item['url']
        
        # [V26.2 Feature] 加入超連結
        if t_url and t_url != "#":
            title_link = f"[{t_text}]({t_url})"
        else:
            title_link = t_text
            
        md += f"| {item['date']} | {item['media']} | {title_link} | {s_txt} | {c_txt} |\n"
    
    st.markdown(md)

# 4. 下載功能
def convert_data_to_json(data):
    import json
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
    st.title("全域觀點解析 V26.2")
    analysis_mode = st.radio("選擇模式：", options=["📰 議題時序分析 (Timeline)", "🔮 未來發展推演 (Scenario)"], index=0)
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
            format_func=lambda x: "📅 不限時間 (All Time)" if x == 1825 else f"近 {x} 天",
            index=2
        )
        
        max_results = st.slider("搜尋篇數上限", 10, 60, 20)
        
        selected_regions = st.multiselect(
            "搜尋視角 (Region)",
            ["🇹🇼 台灣 (Taiwan)", "🌏 亞洲 (Asia)", "🌍 歐洲 (Europe)", "🌎 美洲 (Americas)", "🕵️ 獨立/自媒體 (Indie)"],
            default=["🇹🇼 台灣 (Taiwan)"]
        )

    with st.expander("🧠 詳細分析方法論 (Methodology)", expanded=False):
        st.markdown("""
        **1. 議題時間軸 (Timeline)**
        * **來源**: Tavily API 搜尋結果。
        * **排序**: 依據新聞發布日期由舊至新。
        * **日期補救**: 若 metadata 缺失，AI 閱讀內文推算。

        **2. 政治立場判定 (Hybrid Stance)**
        * **採用「雙重驗證機制」**：
        * **Step A (AI 語意)**：分析標題與內文的情緒強弱 (-10~+10)。
        * **Step B (資料庫校正)**：
          - **🟢 泛綠/批判**: 自由、三立、民視 (強制歸類為負分)。
          - **🔵 泛藍/體制**: 中時、聯合、TVBS (強制歸類為正分)。
        
        **3. 可信度評估 (Credibility)**
        * **權威度**: 考量媒體聲譽 (如中央社 vs 農場)。
        * **完整性**: 檢視是否包含消息來源、數據佐證。
        """)

    with st.expander("📚 監測資料庫清單", expanded=False):
        for key, domains in DB_MAP.items():
            label, color = get_category_meta(key)
            st.markdown(f"**{label}**")
            st.markdown(f"`{', '.join(domains[:3])}...`")

    with st.expander("📂 匯入舊情報", expanded=False):
        past_report_input = st.text_area("貼上舊報告 Markdown：", height=100)
        
    st.markdown("### 📥 報告匯出")
    if st.session_state.get('result') or st.session_state.get('wargame_result'):
        active_data = st.session_state.get('wargame_result') if "Scenario" in analysis_mode else st.session_state.get('result')
        if active_data:
            st.download_button("下載 JSON", convert_data_to_json(active_data), "report.json", "application/json")
            st.download_button("下載 Markdown", convert_data_to_md(active_data), "report.md", "text/markdown")

st.title(f"{analysis_mode.split(' ')[0]}")
query = st.text_input("輸入議題關鍵字", placeholder="例如：台積電美國設廠爭議")
search_btn = st.button("🚀 啟動全域掃描", type="primary")

if 'result' not in st.session_state: st.session_state.result = None
if 'sources' not in st.session_state: st.session_state.sources = None

if search_btn and query and google_key and tavily_key:
    st.session_state.result = None
    
    with st.status("🚀 啟動全域掃描引擎 (V26.2)...", expanded=True) as status:
        
        days_label = "不限時間" if search_days == 1825 else f"近 {search_days} 天"
        regions_label = ", ".join([r.split(" ")[1] for r in selected_regions])
        st.write(f"📡 1. 連線 Tavily 搜尋 (視角: {regions_label} / 時間: {days_label})...")
        
        context_text, sources, actual_query, is_strict_tw = get_search_context(query, tavily_key, search_days, selected_regions, max_results, past_report_input)
        st.session_state.sources = sources
        
        st.write("🛡️ 2. 查詢 Cofacts 謠言資料庫 (API)...")
        cofacts_txt = search_cofacts(query)
        if cofacts_txt: context_text += f"\n{cofacts_txt}\n"
        
        st.write("🧠 3. AI 進行深度戰略分析...")
        
        mode_code = "V205" if "未來" in analysis_mode else "FUSION"
        raw_report = run_strategic_analysis(query, context_text, model_name, google_key, mode=mode_code)
        st.session_state.result = parse_gemini_data(raw_report)
            
        status.update(label="✅ 分析完成", state="complete", expanded=False)
        
    st.rerun()

if st.session_state.result:
    data = st.session_state.result
    
    # 1. 顯示時間軸 (V26.2 含超連結)
    if data.get("timeline"):
        render_timeline_enhanced(data["timeline"])

    # 2. 顯示深度報告
    st.markdown("---")
    st.markdown("### 📝 綜合戰略分析報告")
    formatted_text = format_citation_style(data.get("report_text", ""))
    st.markdown(f'<div class="report-paper">{formatted_text}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    if "未來" not in analysis_mode:
        if st.button("🚀 將此結果餵給未來發展推演 (資訊滾動)", type="secondary"):
            pass 

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
