# ==========================================
# 0. Priority: Warnings & Environment
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
# 1. Config & CSS
# ==========================================
st.set_page_config(page_title="Global View V15.2", page_icon="⚖️", layout="wide")

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
# 2. Constants & Helpers
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
# 3. Core Modules
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
        response = requests.post(url, json={'query': graphql_query, 'variables': {'text': query}}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            articles = data.get('data', {}).get('ListArticles', {}).get('edges', [])
            result_text = ""
            if articles:
                result_text += "【Cofacts Fact-Check】\n"
                for i, art in enumerate(articles):
                    node = art.get('node', {})
                    rumor = node.get('text', '')[:50]
                    replies = node.get('articleReplies', [])
                    if replies:
                        r_type = replies[0].get('reply', {}).get('type')
                        result_text += f"- Rumor: {rumor}... (Type: {r_type})\n"
            return result_text
    except: return ""
    return ""

def get_search_context(query, api_key_tavily, context_report=None):
    os.environ["TAVILY_API_KEY"] = api_key_tavily
    search = TavilySearchResults(max_results=15)
    
    search_q = f"{query} 2025 news analysis"
    
    try:
        results = search.invoke(search_q)
        context_text = ""
        
        cofacts_txt = search_cofacts(query)
        if cofacts_txt: context_text += f"{cofacts_txt}\n{'-'*20}\n"
        
        if context_report:
            context_text += f"【History】\n{context_report[:1000]}...\n\n"
            
        context_text += "【Latest News】(Use [Source X])\n"
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

# [V15.2 Fix] Stronger Mermaid Sanitizer
def sanitize_mermaid_code(code):
    """
    Aggressively fixes Mermaid syntax errors.
    1. Removes markdown tags.
    2. Replaces () inside node names with safe characters to prevent syntax errors.
    """
    # 1. Remove Markdown block
    code = re.sub(r'```mermaid', '', code)
    code = re.sub(r'```', '', code)
    code = code.strip()
    
    lines = code.split('\n')
    clean_lines = []
    
    # Ensure header exists
    if not any(l.strip().startswith('graph') for l in lines):
        clean_lines.append("graph TD")
        
    for line in lines:
        # Skip empty lines
        if not line.strip(): continue
        
        # [Fix] Node Label Cleaning
        # Pattern: finds content inside [] or () or {}
        # We want to keep the brackets but sanitize the content inside
        
        # If line defines a node like: A[Text with (brackets)]
        # We need to ensure internal () are removed or escaped
        
        # Simple approach: Replace ( and ) with space if they are inside the label
        # Ideally, we just tell Mermaid to be loose, but cleaning helps
        
        # Check for A[Label] pattern
        if '[' in line and ']' in line:
            parts = line.split('[', 1)
            node_id = parts[0]
            rest = parts[1].rsplit(']', 1)
            label = rest[0]
            edge = rest[1] if len(rest) > 1 else ""
            
            # Clean label: remove characters that break mermaid
            safe_label = label.replace('(', ' ').replace(')', ' ').replace('"', "'")
            clean_lines.append(f'{node_id}["{safe_label}"]{edge}')
            
        elif '(' in line and ')' in line and '>"' not in line:
             # Handle A(Label) style -> convert to A["Label"]
            parts = line.split('(', 1)
            node_id = parts[0]
            rest = parts[1].rsplit(')', 1)
            label = rest[0]
            edge = rest[1] if len(rest) > 1 else ""
            
            safe_label = label.replace('(', ' ').replace(')', ' ').replace('"', "'")
            clean_lines.append(f'{node_id}["{safe_label}"]{edge}')
        else:
            # Leave simple lines (like subgraph or styling) alone, but remove raw ( )
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

# 3.3 Core: Council of Rivals (War Game)
def run_council_of_rivals(query, context_text, model_name, api_key):
    prompts = {
        "A_SIDE": "You are a [Status Quo/Establishment Analyst]. Analyze evidence supporting the current policy or official stance. Cite sources with [Source X].",
        "B_SIDE": "You are a [Reform/Critical Analyst]. Analyze evidence questioning the status quo or supporting alternative views. Cite sources with [Source X].",
        "CONTEXT": "You are a [Context Historian]. Analyze deep historical, economic, or geopolitical causes. Cite sources with [Source X]."
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
            except Exception as e: opinions[role] = f"Error: {e}"

    editor_prompt = f"""
    You are an Editor-in-Chief. Compile a deep analysis report on "{query}".
    
    Inputs:
    A_View: {opinions.get('A_SIDE')}
    B_View: {opinions.get('B_SIDE')}
    Context: {opinions.get('CONTEXT')}
    
    Tasks:
    1. **Citations**: STRICTLY use `[Source X]` for every claim.
    2. **Mermaid Diagram**: Generate a Mermaid `graph TD` showing causal loops (Variable A -> Variable B). 
       - KEY REQUIREMENT: Use `[]` for labels. Do NOT use `()` inside labels. 
       - Example: `A["Policy X"] --> B["Public Anger"]`.
       - Wrap code in ```mermaid ... ```.
    3. **Future Scenarios**: Deduce 3 possible outcomes.
    
    Output Format:
    ### [REPORT_TEXT]
    (Markdown report...)
    """
    
    final_report = call_gemini(editor_prompt, context_text, model_name, api_key)
    return opinions, final_report

# 3.4 Core: Spectrum Analysis
def run_spectrum_analysis(query, context_text, model_name, api_key):
    system_prompt = f"""
    Media Literacy Expert. Analyze "{query}".
    
    Task:
    1. Identify 'Stance' (-10 Anti/Critical <-> 0 Neutral <-> 10 Pro/Support) and 'Credibility' (0-10) for sources.
    2. Be DIVERSE in scoring. Don't clump everyone in the middle.
    
    Output:
    ### [DATA_TIMELINE]
    YYYY-MM-DD|Media|Title
    
    ### [DATA_SPECTRUM]
    Source Name|Stance(-10 to 10)|Credibility(0 to 10)|URL
    
    ### [REPORT_TEXT]
    (Markdown report with [Source X] citations)
    """
    return call_gemini(system_prompt, context_text, model_name, api_key)

# 3.5 Parser
def parse_gemini_data(text):
    data = {"timeline": [], "spectrum": [], "mermaid": "", "report_text": ""}
    
    # Extract Mermaid
    mermaid_match = re.search(r"```mermaid\n(.*?)\n```", text, re.DOTALL)
    if mermaid_match:
        data["mermaid"] = mermaid_match.group(1)
        text = text.replace(mermaid_match.group(0), "")

    for line in text.split('\n'):
        line = line.strip()
        if "|" in line and len(line.split("|")) >= 3 and (line[0].isdigit() or "Future" in line):
            parts = line.split("|")
            data["timeline"].append({"date": parts[0], "media": parts[1], "event": parts[2]})
            
        if "|" in line and len(line.split("|")) >= 4 and not line.startswith("###") and not "Date" in line:
            parts = line.split("|")
            try:
                # [V15.2 Fix] Parse spectrum with jitter
                base_stance = float(parts[1])
                base_cred = float(parts[2])
                jitter_x = random.uniform(-0.8, 0.8) # More jitter
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

# [V15.2 Fix] Improved Chart Scaling
def render_spectrum_chart(spectrum_data):
    if not spectrum_data: return None
    df = pd.DataFrame(spectrum_data)
    
    fig = px.scatter(
        df, x="stance", y="credibility", hover_name="source", text="source", size=[25]*len(df),
        color="stance", color_continuous_scale=["#2e7d32", "#eeeeee", "#1565c0"],
        range_x=[-15, 15], # Widen X to push extremes out
        range_y=[-2, 13],  # Widen Y to prevent overlapping text
        opacity=0.9,
        labels={"stance": "Political Spectrum", "credibility": "Credibility"}
    )
    # Background Quadrants
    fig.add_shape(type="rect", x0=-15, y0=6, x1=0, y1=13, fillcolor="rgba(46, 125, 50, 0.05)", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=6, x1=15, y1=13, fillcolor="rgba(21, 101, 192, 0.05)", layer="below", line_width=0)
    
    fig.update_layout(
        xaxis_title="◀ Critical/Reform (Green) ------- Neutral ------- Establishment/Pro (Blue) ▶",
        yaxis_title="Information Quality (Low -> High)",
        showlegend=False,
        height=650, # Taller chart
        font=dict(size=14)
    )
    fig.update_traces(textposition='top center', textfont_size=13)
    return fig

# 4. Generate Download Data
def convert_data_to_json(data):
    return json.dumps(data, indent=2, ensure_ascii=False)

def convert_data_to_md(data):
    return f"""
# Global View Analysis Report
Date: {datetime.now()}

## 1. Analysis Content
{data.get('report_text')}

## 2. Timeline
{pd.DataFrame(data.get('timeline')).to_markdown(index=False)}
    """

# ==========================================
# 5. UI
# ==========================================
with st.sidebar:
    st.title("Global View V15.2")
    analysis_mode = st.radio("Mode:", options=["🛡️ Public Opinion (Spectrum)", "🔮 Future War Game"], index=0)
    st.markdown("---")
    
    with st.expander("🔑 API Settings", expanded=True):
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
            
        model_name = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"], index=0)

    with st.expander("📂 Import Old Report", expanded=False):
        past_report_input = st.text_area("Paste Markdown:", height=100)
        
    # [V15.2 Fix] Download Buttons in Sidebar
    st.markdown("### 📥 Export")
    if st.session_state.get('spectrum_result') or st.session_state.get('wargame_result'):
        active_data = st.session_state.get('wargame_result') if "War" in analysis_mode else st.session_state.get('spectrum_result')
        if active_data:
            st.download_button("Download JSON", convert_data_to_json(active_data), "report.json", "application/json")
            st.download_button("Download Markdown", convert_data_to_md(active_data), "report.md", "text/markdown")

st.title(f"{analysis_mode.split(' ')[1]}")
query = st.text_input("Enter Topic", placeholder="e.g., TSMC US Factory Debate")
search_btn = st.button("🚀 Start Analysis", type="primary")

if 'spectrum_result' not in st.session_state: st.session_state.spectrum_result = None
if 'wargame_result' not in st.session_state: st.session_state.wargame_result = None
if 'wargame_opinions' not in st.session_state: st.session_state.wargame_opinions = None
if 'sources' not in st.session_state: st.session_state.sources = None
if 'full_context' not in st.session_state: st.session_state.full_context = ""

# Logic
if search_btn and query and google_key and tavily_key:
    st.session_state.spectrum_result = None
    st.session_state.wargame_result = None
    st.session_state.wargame_opinions = None
    
    with st.spinner("📡 Gathering Intelligence (Tavily + Cofacts)..."):
        context_text, sources, cofacts_txt = get_search_context(query, tavily_key, past_report_input)
        st.session_state.sources = sources
        st.session_state.full_context = context_text
        
        if "Spectrum" in analysis_mode:
            raw_report = run_spectrum_analysis(query, context_text, model_name, google_key)
            st.session_state.spectrum_result = parse_gemini_data(raw_report)
        else:
            with st.status("⚔️ Convening Council of Rivals...", expanded=True) as status:
                st.write("1. Agents Debating...")
                opinions, raw_report = run_council_of_rivals(query, context_text, model_name, google_key)
                st.session_state.wargame_opinions = opinions
                st.session_state.wargame_result = parse_gemini_data(raw_report)
                status.update(label="✅ Analysis Complete", state="complete", expanded=False)
    st.rerun()

# Render: Spectrum
if st.session_state.spectrum_result and "Spectrum" in analysis_mode:
    data = st.session_state.spectrum_result
    
    if data.get("spectrum"):
        st.markdown("### 🗺️ Public Opinion Map (Spectrum)")
        fig = render_spectrum_chart(data["spectrum"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📝 Media Literacy Report")
    st.markdown(f'<div class="report-paper">{data.get("report_text")}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("Need deeper strategic foresight? Click below.")
    if st.button("🚀 Launch Future War Game (Using this Data)", type="primary"):
        if st.session_state.full_context:
            with st.status("⚔️ Convening War Room...", expanded=True) as status:
                st.write("1. Activating Agents...")
                opinions, raw_report = run_council_of_rivals(query, st.session_state.full_context, model_name, google_key)
                st.session_state.wargame_opinions = opinions
                st.session_state.wargame_result = parse_gemini_data(raw_report)
                status.update(label="✅ Done", state="complete", expanded=False)
                st.rerun()

# Render: War Game
if st.session_state.wargame_result:
    st.divider()
    st.markdown(f"<h2 style='text-align: center;'>⚔️ Future Development Deduction: {query}</h2>", unsafe_allow_html=True)
    
    ops = st.session_state.wargame_opinions
    if ops:
        c_a, c_b, c_ctx = st.columns(3)
        with c_a:
            st.markdown(f'<div class="perspective-box box-blue"><b>🔵 Status Quo (A)</b><br>{ops.get("A_SIDE")[:150]}...</div>', unsafe_allow_html=True)
            with st.popover("Full Text"): st.markdown(ops.get("A_SIDE"))
        with c_b:
            st.markdown(f'<div class="perspective-box box-green"><b>🟢 Reform/Critical (B)</b><br>{ops.get("B_SIDE")[:150]}...</div>', unsafe_allow_html=True)
            with st.popover("Full Text"): st.markdown(ops.get("B_SIDE"))
        with c_ctx:
            st.markdown(f'<div class="perspective-box box-neutral"><b>📜 Context</b><br>{ops.get("CONTEXT")[:150]}...</div>', unsafe_allow_html=True)
            with st.popover("Full Text"): st.markdown(ops.get("CONTEXT"))

    data_wg = st.session_state.wargame_result
    
    # [V15.2 Fix] Mermaid Diagram Display
    if data_wg.get("mermaid"):
        st.markdown("### 🕸️ System Dynamics (Causal Loop)")
        st.markdown('<div class="mermaid-box">', unsafe_allow_html=True)
        render_mermaid(data_wg["mermaid"])
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Fallback if AI fails to generate valid mermaid
        st.warning("⚠️ System Dynamics diagram could not be generated from AI response.")

    st.markdown("### 📝 Editor-in-Chief's Deep Analysis")
    st.markdown(f'<div class="report-paper">{data_wg.get("report_text")}</div>', unsafe_allow_html=True)

# Sources Table
if st.session_state.sources:
    st.markdown("---")
    st.markdown("### 📚 Reference List")
    md_table = "| ID | Domain | Title | Link |\n|:---:|:---|:---|:---|\n"
    for i, s in enumerate(st.session_state.sources):
        domain = get_domain_name(s.get('url'))
        title = s.get('content', '')[:60].replace("\n", " ").replace("|", " ") + "..."
        url = s.get('url')
        md_table += f"| **{i+1}** | `{domain}` | {title} | [Link]({url}) |\n"
    st.markdown(md_table)
