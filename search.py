import streamlit as st
import google.generativeai as genai
from tavily import TavilyClient

# --- 頁面設定 ---
st.set_page_config(
    page_title="Gemini 2.5 x Tavily 終極搜尋引擎", 
    page_icon="🚀", 
    layout="wide"
)

# --- 標題與簡介 ---
st.title("🚀 Gemini 2.5 x Tavily 即時搜尋引擎")
st.markdown("""
這是一個結合 **Google 最新 Gemini 2.5 模型** 與 **Tavily 聯網搜尋** 的 RAG 工具。
能為您從網路上抓取 2025 最新資訊，並整理成深度報告。
""")

# --- 側邊欄：設定 API Key 與 模型 ---
with st.sidebar:
    st.header("⚙️ 核心設定")
    
    # 1. API Keys
    with st.expander("🔑 API 金鑰 (點此展開)", expanded=True):
        gemini_key = st.text_input("Gemini API Key", type="password", help="請至 Google AI Studio 申請")
        tavily_key = st.text_input("Tavily API Key", type="password", help="請至 Tavily 官網申請")
    
    st.divider()
    
    # 2. 模型選擇器 (Gemini 2.5 全系列)
    st.subheader("🧠 模型選擇 (Model)")
    selected_model = st.selectbox(
        "請選擇 Gemini 版本：",
        [
            "gemini-2.5-pro", 
            "gemini-2.5-flash", 
            "gemini-2.5-flash-lite"
        ],
        index=1, # 預設選 Flash (平衡)
        help="Pro: 最聰明但較慢 | Flash: 平衡 | Lite: 最快"
    )
    
    # 顯示模型特性提示
    if "pro" in selected_model:
        st.info("🔥 **Pro 版**：適合複雜推理、寫程式、深度報告。")
    elif "lite" in selected_model:
        st.success("⚡ **Lite 版**：極速回應，適合簡單查詢。")
    else:
        st.info("⚖️ **Flash 版**：速度與品質的最佳平衡 (推薦)。")

    st.divider()
    
    # 3. 搜尋參數
    st.subheader("🌍 搜尋設定")
    search_depth = st.radio("搜尋深度", ["basic", "advanced"], index=1)
    max_results = st.slider("參考資料數量", 3, 10, 5)

# --- 核心功能函數 ---

def get_tavily_search(query, api_key, depth="advanced", max_results=5):
    """使用 Tavily 搜尋網路資料"""
    tavily = TavilyClient(api_key=api_key)
    response = tavily.search(
        query=query,
        search_depth=depth,
        max_results=max_results,
        include_answer=True,
        include_raw_content=False
    )
    return response

def generate_gemini_response(query, search_results, api_key, model_name):
    """將搜尋結果餵給指定的 Gemini 模型進行總結"""
    genai.configure(api_key=api_key)
    
    # 使用使用者選擇的模型 (例如 gemini-2.5-pro)
    model = genai.GenerativeModel(model_name)
    
    # 組合 Context
    context_text = ""
    for i, result in enumerate(search_results.get('results', [])):
        context_text += f"\n--- 來源 {i+1}: {result['title']} ---\n"
        context_text += f"網址: {result['url']}\n"
        context_text += f"內容: {result['content']}\n"

    # Prompt Engineering
    prompt = f"""
    你是一個專業的高級研究員，正在協助使用者進行深度調查。
    
    【使用者問題】："{query}"
    
    【搜尋到的最新資料】：
    {context_text}
    
    【任務指令】：
    請根據上述資料，撰寫一份**詳盡、結構清晰且無錯誤**的回答。
    1. **深度優先**：請挖掘資料中的細節，不要只給表面答案。
    2. **標註來源**：引用數據或觀點時，請用 [來源X] 標註。
    3. **模型身分**：你現在使用的是 {model_name} 模型，請發揮你的長處。
    4. **語言**：請使用台灣繁體中文。
    
    請開始撰寫報告：
    """
    
    # 生成內容 (Stream 模式)
    response = model.generate_content(prompt, stream=True)
    return response

# --- 主介面邏輯 ---

query = st.text_input("💬 請輸入您的問題：", placeholder="例如：2025年最新的 SBD 訓練科學研究有哪些？")
search_btn = st.button("🚀 開始深度搜尋", type="primary")

if search_btn and query:
    if not gemini_key or not tavily_key:
        st.error("❌ 請先在側邊欄填入 API Keys 才能運作喔！")
    else:
        # 1. 搜尋階段
        with st.status(f"🕵️‍♂️ 正在呼叫 Tavily 搜尋 (深度: {search_depth})...", expanded=True) as status:
            try:
                search_data = get_tavily_search(query, tavily_key, search_depth, max_results)
                st.write(f"✅ 成功找到 {len(search_data['results'])} 筆資料，正在下載內容...")
                status.update(label=f"搜尋完成！正在呼叫 {selected_model} 進行分析...", state="running", expanded=False)
            except Exception as e:
                st.error(f"搜尋發生錯誤: {e}")
                st.stop()

        # 2. 顯示來源 (可折疊)
        with st.expander("📚 點此查看搜尋到的原始來源"):
            for res in search_data['results']:
                st.markdown(f"**[{res['title']}]({res['url']})**")
                st.caption(res['content'][:250] + "...")
                st.divider()

        # 3. 生成階段
        st.subheader(f"💡 {selected_model} 的深度報告")
        result_container = st.empty()
        full_response = ""
        
        try:
            # 傳入 selected_model
            response_stream = generate_gemini_response(query, search_data, gemini_key, selected_model)
            
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    result_container.markdown(full_response + "▌")
            
            result_container.markdown(full_response)
            
        except Exception as e:
            st.error(f"生成失敗: {e}\n(請確認您的 API Key 是否有權限存取 2.5 模型)")

# --- 頁尾 ---
st.markdown("---")
st.caption("Designed for Advanced Research | Powered by Gemini 2.5 Series & Tavily")
