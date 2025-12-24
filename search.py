import streamlit as st
import google.generativeai as genai
from tavily import TavilyClient

# --- 頁面設定 ---
st.set_page_config(page_title="Gemini x Tavily 超級搜尋引擎", page_icon="🔍", layout="wide")

# --- 標題與簡介 ---
st.title("🔍 Gemini x Tavily 即時搜尋引擎")
st.markdown("""
這是一個 RAG (檢索增強生成) 搜尋工具。
1. **Tavily** 負責搜尋網路並爬取最新內容。
2. **Gemini** 負責閱讀這些內容並整理成報告。
""")

# --- 側邊欄：設定 API Key ---
with st.sidebar:
    st.header("🔑 API 金鑰設定")
    
    # 嘗試從 st.secrets 讀取，如果沒有則顯示輸入框
    gemini_key = st.text_input("Gemini API Key", type="password", help="請至 Google AI Studio 申請")
    tavily_key = st.text_input("Tavily API Key", type="password", help="請至 Tavily 官網申請")
    
    st.divider()
    st.markdown("### ⚙️ 搜尋設定")
    search_depth = st.radio("搜尋深度", ["basic", "advanced"], index=1, help="Basic 較快，Advanced 資訊較完整")
    max_results = st.slider("參考資料數量", 3, 10, 5)

# --- 核心功能函數 ---

def get_tavily_search(query, api_key, depth="advanced", max_results=5):
    """使用 Tavily 搜尋網路資料"""
    tavily = TavilyClient(api_key=api_key)
    response = tavily.search(
        query=query,
        search_depth=depth,
        max_results=max_results,
        include_answer=True, # 讓 Tavily 也嘗試給一個簡短答案
        include_raw_content=False # 我們只需要處理過的乾淨 context
    )
    return response

def generate_gemini_response(query, search_results, api_key):
    """將搜尋結果餵給 Gemini 進行總結"""
    genai.configure(api_key=api_key)
    
    # 這裡我們使用 1.5 Flash，因為速度快且便宜，適合處理大量文字
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # 組合 Context
    context_text = ""
    for i, result in enumerate(search_results.get('results', [])):
        context_text += f"\n--- 資料來源 {i+1}: {result['title']} ---\n"
        context_text += f"網址: {result['url']}\n"
        context_text += f"內容摘要: {result['content']}\n"

    # Prompt Engineering (針對您的偏好：可信度高、資訊多)
    prompt = f"""
    你是一個專業的高級研究員。使用者的問題是："{query}"
    
    以下是從網路上搜尋到的最新資料（Context）：
    {context_text}
    
    請根據上述資料，回答使用者的問題。
    
    回答要求：
    1. **資訊豐富且詳盡**：不要只給簡短答案，請提供深度分析。
    2. **結構清晰**：使用 Markdown 標題、列點。
    3. **標註來源**：在提到的事實後方，用 [來源 1]、[來源 2] 的方式標註。
    4. **保持客觀**：如果資料中有衝突，請列出不同觀點。
    5. **繁體中文**：請使用台灣繁體中文回答。
    
    請開始你的分析：
    """
    
    # 生成內容 (使用 stream 讓體驗更好)
    response = model.generate_content(prompt, stream=True)
    return response

# --- 主介面邏輯 ---

query = st.text_input("請輸入你想知道的問題 (例如：最新的 Garmin 健力訓練功能分析)", placeholder="在這裡輸入搜尋關鍵字...")
search_btn = st.button("開始搜尋", type="primary")

if search_btn and query:
    if not gemini_key or not tavily_key:
        st.error("❌ 請先在側邊欄輸入 API Keys！")
    else:
        # 1. Tavily 搜尋階段
        with st.status("🕵️‍♂️ 正在網海上搜尋資料...", expanded=True) as status:
            st.write("正在呼叫 Tavily API...")
            try:
                search_data = get_tavily_search(query, tavily_key, search_depth, max_results)
                st.write(f"✅ 找到 {len(search_data['results'])} 筆相關資料")
                status.update(label="搜尋完成！正在請 Gemini 閱讀與撰寫報告...", state="running", expanded=False)
            except Exception as e:
                st.error(f"搜尋失敗: {e}")
                st.stop()

        # 2. 顯示搜尋到的來源 (給使用者看它參考了哪裡)
        with st.expander("📚 查看原始搜尋來源 (點擊展開)"):
            for res in search_data['results']:
                st.markdown(f"**[{res['title']}]({res['url']})**")
                st.caption(res['content'][:200] + "...")
                st.divider()

        # 3. Gemini 生成階段
        st.subheader("💡 Gemini 的研究報告")
        result_container = st.empty()
        full_response = ""
        
        try:
            response_stream = generate_gemini_response(query, search_data, gemini_key)
            
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    result_container.markdown(full_response + "▌") # 打字機效果
            
            result_container.markdown(full_response) # 最後顯示完整版
            
        except Exception as e:
            st.error(f"生成失敗: {e}")

# --- 頁尾 ---
st.markdown("---")
st.caption("Powered by Gemini 1.5 Flash & Tavily Search API")