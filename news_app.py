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
st.set_page_config(page_title="全域觀點解析 V23.0", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
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
    
    /* 表格樣式優化 */
    .stDataFrame { border: 1px solid #f0f0f0; border-radius: 8px; overflow: hidden; }
    
    /* 側邊欄說明文字 */
    .sidebar-text { font-size: 0.9em; color: #555; line-height: 1.5; }
    .sidebar-header { font-weight: bold; color: #333; margin-top: 10px; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 資料庫定義 (用於側邊欄顯示與搜尋)
# ==========================================
# 台灣主流媒體白名單 (搜尋用)
TAIWAN_WHITELIST = [
    "udn.com", "ltn.com.tw", "chinatimes.com", "cna.com.tw", 
    "storm.mg", "setn.com", "ettoday.net", "tvbs.com.tw", 
    "mirrormedia.mg", "thenewslens.com", "upmedia.mg", 
    "rwnews.tw", "news.pts.org.tw", "ctee.com.tw", "businessweekly.com.tw",
    "news.yahoo.com.tw"
]

# 獨立媒體白名單
INDIE_WHITELIST = [
    "twreporter.org", "theinitium.com", "thenewslens.com", 
    "mindiworldnews.com", "vocus.cc", "matters.town", 
    "plainlaw.me", "whogovernstw.org", "rightplus.org", 
    "biosmonthly.com", "storystudio.tw", "womany.net", "dq.yam.com"
]

# 完整資料庫分類 (側邊欄展示用)
DB_MAP = {
    "CHINA": ["xinhuanet.com", "people.com.cn", "huanqiu.com", "cctv.com", "chinadaily.com.cn", "taiwan.cn", "gwytb.gov.cn", "guancha.cn", "thepaper.cn", "ifeng.com", "crntt.com"],
    "GREEN": ["ltn.com.tw", "ftvnews.com.tw", "setn.com", "rti.org.tw", "newtalk.tw", "mirrormedia.mg", "dpp.org.tw"],
    "BLUE": ["udn.com", "chinatimes.com", "tvbs.com.tw", "cti.com.tw", "nownews.com", "ctee.com.tw", "kmt.org.tw"],
    "OFFICIAL": ["cna.com.tw", "pts.org.tw", "mnd.gov.tw", "mac.gov.tw", "tfc-taiwan.org.tw"],
    "INDIE": ["twreporter.org", "theinitium.com", "thenewslens.com", "upmedia.mg", "storm.mg", "mindiworldnews.com", "vocus.cc"],
    "INTL": ["bbc.com", "cnn.com", "reuters.com", "apnews.com", "bloomberg.com", "wsj.com", "nytimes.com", "dw.com", "voanews.com", "rfi.fr"],
    "FARM": ["kknews.cc", "read01.com", "ppfocus.com", "buzzhand.com", "bomb01.com", "qiqi.news", "inf.news", "toutiao.com"]
}

# 關鍵字對照 (用於各種判斷)
CAMP_KEYWORDS = {
    "GREEN": ["自由", "三立", "民視", "新頭殼", "鏡週刊", "放言", "賴清德", "民進黨", "青鳥", "中央社"],
    "BLUE": ["聯合", "中時", "中國時報", "TVBS", "中天", "風傳媒", "國民黨", "藍營", "赵少康"],
    "RED": ["新華", "人民日報", "環球", "央視", "中評", "国台办"]
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
        "INTL": ("🌏 國際媒體", "#f57c00"),
        "FARM": ("⛔ 內容農場", "#ef6c00")
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
        
        # [Debug] Ensure list type
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

        # 網域邏輯：只有在「未選國際」且「有選台灣/獨立」時才啟用白名單
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
            # 日期處理
            pub_date = res.get('published_date')
            if pub_date:
                pub_date = pub_date[:10]
            else:
                pub_date = "----" # 留空給 AI 判讀
            
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
    1. **引用壓縮**：若連續引用多個來源，請寫成 `[Source 1, 2, 3]` 的格式。
    2. **Mermaid 製圖**：請生成 Mermaid `graph TD` 代碼，展示「變數 A 如何導致 變數 B」的因果鏈。
       - 嚴格規定：節點名稱請使用 **純文字**，不要包含括號、問號或其他符號。
       - 代碼請包在 ```mermaid ... ``` 區塊中。
    3. **未來情境**：推導 3 種可能的發展劇本。
    
    【輸出格式】：
    ### [REPORT_TEXT]
    (Markdown 報告內容，請包含「🔮 未來情境模擬」章節)
    """
    final_report = call_gemini(editor_prompt, context_text, model_name, api_key)
    return opinions, final_report

# 3.4 核心邏輯：時間軸分析 (V23.0 修正日期抓取)
def run_timeline_analysis(query, context_text, model_name, api_key):
    system_prompt = f"""
    你是一位專業的調查記者與數據分析師。請針對「{query}」整理一份詳盡的【議題發展時間軸】。
    
    【任務目標】：
    1. **時間序排列**：將新聞事件依照日期先後排序，從最早到最新。
    2. **資料提取**：從 Context 中提取 日期、媒體、標題、連結 URL。
    3. **可信度評估**：針對該來源的可信度給予 0-10 分評價 (0=農場/極端, 10=權威/查核)。
    4. **立場判讀**：判斷該媒體對此議題的立場 (-10=強烈批判/反對, 0=中立, +10=強烈支持/體制)。
    
    【日期提取鐵律 (Date Extraction Rules)】：
    1. 若 Context 中已有 [Date: YYYY-MM-DD]，優先使用。
    2. 若 Date 為 '----'，你必須閱讀新聞標題或內文前段，推斷發生日期。例如看到 "昨天", "週三" 配合 Context 時間推算。
    3. 若完全無法推斷，請填寫 '2025-??-??'。
    
    【輸出格式 (請保持格式整潔，每行一筆，使用 | 分隔)】：
    ### [DATA_TIMELINE]
    (重要：必須包含 6 個欄位)
    日期(YYYY-MM-DD)|媒體|新聞標題|立場(-10~10)|可信度(0-10)|網址
    
    ### [REPORT_TEXT]
    (Markdown 報告，請使用 `[Source 1, 3]` 格式引用)
    請包含：
    1. 📊 全域現況摘要
    2. 🗓️ 關鍵時間節點解析
    3. 💡 媒體識讀與爭議點分析
    """
    return call_gemini(system_prompt, context_text, model_name, api_key)

# 3.5 資料解析器
def parse_gemini_data(text):
    data = {"timeline": [], "mermaid": "", "report_text": ""}
    
    mermaid_match = re.search(r"```mermaid\n(.*?)\n```", text, re.DOTALL)
    if mermaid_match:
        data["mermaid"] = mermaid_match.group(1)
        text = text.replace(mermaid_match.group(0), "")

    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Timeline Parsing
        if "|" in line and len(line.split("|")) >= 4 and not line.startswith("###") and not "YYYY" in line:
            parts = line.split("|")
            try:
                name = parts[1].strip()
                date = parts[0].strip()
                title = parts[2].strip()
                base_stance = 0
                base_cred = 0
                url = "#"
                
                # 6 欄位解析
                if len(parts) >= 6:
                    base_stance = float(parts[3].strip())
                    base_cred = float(parts[4].strip())
                    url = parts[5].strip()
                elif len(parts) == 5:
                    base_cred = float(parts[3].strip())
                    url = parts[4].strip()

                final_stance = base_stance
                if any(k in name for k in CAMP_KEYWORDS["GREEN"]):
                    if final_stance > 0: final_stance = final_stance * -1
                    if final_stance == 0: final_stance = -5
                elif any(k in name for k in CAMP_KEYWORDS["BLUE"] + CAMP_KEYWORDS["RED"]):
                    if final_stance < 0: final_stance = final_stance * -1
                    if final_stance == 0: final_stance = 5
                
                data["timeline"].append({
                    "date": date,
                    "media": name,
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

# [V23.0] 渲染時間軸表格 (含可信度與立場)
def render_timeline_enhanced(timeline_data):
    if not timeline_data: 
        st.warning("⚠️ 無法生成時間軸：可能是搜尋結果不足，或 AI 無法解析日期。")
        return
    
    st.markdown("### 📅 議題發展時間軸 (News Timeline)")
    
    md = "| 日期 | 媒體 | 新聞標題 (點擊閱讀) | 立場 | 可信度 |\n|:---:|:---|:---|:---:|:---:|\n"
    for item in timeline_data:
        # 可信度燈號
        c = item['credibility']
        if c >= 8: c_txt = f"🟢 高 ({c})"
        elif c >= 5: c_txt = f"🟡 中 ({c})"
        else: c_txt = f"🔴 低 ({c})"
        
        # 立場燈號
        s = item['stance']
        if s < -2: s_txt = f"🟢 泛綠/批判 ({s})"
        elif s > 2: s_txt = f"🔵 泛藍/體制 (+{s})"
        else: s_txt = "⚪ 中立"
        
        t_text = item['title']
        if len(t_text) > 35: t_text = t_text[:35] + "..."
        t_url = item['url']
        
        title_link = f"[{t_text}]({t_url})"
        md += f"| {item['date']} | {item['media']} | {title_link} | {s_txt} | {c_txt} |\n"
    
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
    st.title("全域觀點解析 V23.0")
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
            "搜尋時間範圍 (Time Range)",
            options=[3, 7, 14, 30, 90, 1825],
            format_func=lambda x: "📅 不限時間 (All Time)" if x == 1825 else f"近 {x} 天",
            index=2
        )
        
        max_results = st.slider("搜尋篇數上限 (Max Results)", 10, 50, 20, help="增加篇數可獲得更完整觀點，但分析時間會變長。")
        
        selected_regions = st.multiselect(
            "搜尋視角 (Region) - 可複選",
            ["🇹🇼 台灣 (Taiwan)", "🌏 亞洲 (Asia)", "🌍 歐洲 (Europe)", "🌎 美洲 (Americas)", "🕵️ 獨立/自媒體 (Indie)"],
            default=["🇹🇼 台灣 (Taiwan)"]
        )

    # [V23.0] 監測資料庫清單
    with st.expander("📚 監測資料庫 (Monitoring DB)", expanded=False):
        for key, domains in DB_MAP.items():
            label, color = get_category_meta(key)
            st.markdown(f"**{label}**")
            st.markdown(f"`{', '.join(domains[:5])}...`")

    # [V23.0] 方法論說明
    with st.expander("🧠 分析方法論 (Methodology)", expanded=False):
        st.markdown("""
        **1. 議題時間軸 (Timeline Construction)**
        * **來源**: 使用 Tavily API 搜尋並提取新聞發布時間。
        * **排序**: 依照事件發生順序或報導時間，由舊至新排列，還原事件脈絡。
        * **日期補救**: 若 metadata 缺失，AI 會閱讀內文前段 (如 '昨日', '週三') 進行推算。

        **2. 政治立場判定 (Stance Classification)**
        * **AI 語意分析**: 分析標題與內文的修辭強弱 (Sentiment Analysis)。
        * **光譜校正**: 
          - **負分 (-10 ~ -1)**: 批判現狀、泛綠觀點。
          - **正分 (+1 ~ +10)**: 支持體制、泛藍觀點。
          - **0 分**: 純事實報導或中立觀點。
        
        **3. 可信度評估 (Credibility Assessment)**
        * **權威度**: 考量媒體聲譽 (如中央社 vs 內容農場)。
        * **完整性**: 檢視是否包含消息來源、數據佐證。
        * **查核狀態**: 與 Cofacts 謠言資料庫比對。
        """)

    with st.expander("📂 匯入舊情報", expanded=False):
        past_report_input = st.text_area("貼上舊報告 Markdown：", height=100)
        
    st.markdown("### 📥 報告匯出")
    if st.session_state.get('result') or st.session_state.get('wargame_result'):
        active_data = st.session_state.get('wargame_result') if "Scenario" in analysis_mode else st.session_state.get('result')
        if active_data:
            st.download_button("下載 JSON", convert_data_to_json(active_data), "report.json", "application/json")
            st.download_button("下載 Markdown", convert_data_to_md(active_data), "report.md", "text/markdown")

st.title(f"{analysis_mode.split(' ')[1]}")
query = st.text_input("輸入議題關鍵字", placeholder="例如：台積電美國設廠爭議")
search_btn = st.button("🚀 啟動分析引擎", type="primary")

if 'result' not in st.session_state: st.session_state.result = None
if 'wargame_result' not in st.session_state: st.session_state.wargame_result = None
if 'wargame_opinions' not in st.session_state: st.session_state.wargame_opinions = None
if 'sources' not in st.session_state: st.session_state.sources = None
if 'full_context' not in st.session_state: st.session_state.full_context = ""

if search_btn and query and google_key and tavily_key:
    st.session_state.result = None
    st.session_state.wargame_result = None
    st.session_state.wargame_opinions = None
    
    with st.status("🚀 啟動全域掃描引擎 (V23.0)...", expanded=True) as status:
        
        days_label = "不限時間" if search_days == 1825 else f"近 {search_days} 天"
        regions_label = ", ".join([r.split(" ")[1] for r in selected_regions])
        st.write(f"📡 1. 連線 Tavily 搜尋 (視角: {regions_label} / 時間: {days_label})...")
        
        context_text, sources, actual_query, is_strict_tw = get_search_context(query, tavily_key, search_days, selected_regions, max_results, past_report_input)
        st.session_state.sources = sources
        
        if is_strict_tw:
             st.info(f"🔍 已啟用台灣媒體白名單鎖定 (Whitelist Mode)")
        else:
             st.info(f"🔍 實際搜尋關鍵字: {actual_query}")
        
        st.write("🛡️ 2. 查詢 Cofacts 謠言資料庫 (API)...")
        cofacts_txt = search_cofacts(query)
        if cofacts_txt:
            context_text += f"\n{cofacts_txt}\n"
        st.session_state.full_context = context_text
        
        st.write("🧠 3. AI 進行深度閱讀與分析...")
        
        if "Timeline" in analysis_mode:
            raw_report = run_timeline_analysis(query, context_text, model_name, google_key)
            st.session_state.result = parse_gemini_data(raw_report)
        else:
            st.write("⚔️ 4. 召開虛擬戰情會議 (加入未來學推演)...")
            opinions, raw_report = run_council_of_rivals(query, context_text, model_name, google_key)
            st.session_state.wargame_opinions = opinions
            st.session_state.wargame_result = parse_gemini_data(raw_report)
            
        status.update(label="✅ 分析完成", state="complete", expanded=False)
        
    st.rerun()

# 顯示結果：Timeline 模式
if st.session_state.result and "Timeline" in analysis_mode:
    data = st.session_state.result
    
    # [V23.0] 主角是時間軸表格
    render_timeline_enhanced(data.get("timeline"))

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

# 顯示結果：Scenario 模式
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
