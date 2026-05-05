# ==========================================
# 0. 優先執行：警告屏蔽與套件設定
# ==========================================
import warnings
import os
import json
import re
import pandas as pd
import time
import requests
import concurrent.futures
import random
import markdown
import logging
import hashlib
import pickle
import sqlite3
import asyncio
import aiohttp
import sys
from io import BytesIO
from html import unescape
from pathlib import Path
from urllib.parse import urlparse, quote, unquote, urljoin
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from collections import Counter
import math

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
from tavily import TavilyClient
warnings.filterwarnings("ignore")
os.environ["on_bad_lines"] = "skip"

# ==========================================
# 日誌設定
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# 常數定義
# ==========================================
MAX_WORKERS = 12  # 並行處理數
MAX_SEARCH_WORKERS = 3  # 搜尋 API 並發數（大幅降低以避免超過 API 速率限制，從 6 降到 3）
MAX_CONTENT_LENGTH = 3000  # 內容最大長度（重視正確度，不縮減）
TITLE_TRUNCATE_LENGTH = 60
TIMEOUT_COFACTS = 3
TIMEOUT_FACT_CHECK = 5  # Fact Check API 超時時間
SEARCH_REQUEST_DELAY = 1.0  # 搜尋請求之間的延遲（秒），大幅增加以避免 429 錯誤（從 0.2 增加到 1.0）
SEARCH_RETRY_DELAY = 5.0  # 429 錯誤後的重試延遲（秒）
MAX_SEARCH_RETRIES = 2  # 搜尋請求的最大重試次數
SIMILARITY_THRESHOLD = 0.8  # 聲量權重校正：標題相似度閾值
CACHE_EXPIRY_HOURS = 24  # 快取過期時間（小時）
SUMMARY_THRESHOLD = 2000  # 超過此長度將進行摘要
CACHE_DIR = Path(".cache")
CACHE_DB_PATH = CACHE_DIR / "search_cache.db"
LAST_PDF_EXPORT_ERROR = ""
GEMINI_MODEL_OPTIONS = [
    "gemini-3.1-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
]
DEFAULT_GEMINI_MODEL = GEMINI_MODEL_OPTIONS[0]
GEMINI_FALLBACK_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
]

# ==========================================
# 語言風格分析閾值常數（優化：抽取魔法數字）
# ==========================================
CLICKBAIT_MULTIPLIER = 0.3  # Clickbait 分數乘數
CLICKBAIT_THRESHOLD = 0.3  # Clickbait 警示閾值
SENSATIONALISM_MULTIPLIER = 0.15  # 聳動性分數乘數
SENSATIONALISM_THRESHOLD = 0.5  # 聳動性警示閾值
CAPS_RATIO_THRESHOLD = 0.3  # 大寫字母比例警示閾值
TITLE_CONTENT_MISMATCH_THRESHOLD = 0.6  # 標題內容不匹配警示閾值
TITLE_CONTENT_UNKNOWN_SCORE = 0.5  # 無法判斷時的不匹配分數
EMOTIONAL_MANIPULATION_THRESHOLD = 0.5  # 情感操控警示閾值
EMOTIONAL_WORDS_DIVISOR = 8.0  # 情感詞彙計算除數
EMOTIONAL_TITLE_BONUS = 0.2  # 標題包含情感詞彙的加權

# ==========================================
# 證據強度分級閾值常數（GRADE 對應；已微調使扎實文本易達強/中強）
# ==========================================
EVIDENCE_LEVEL_A_PLUS = 0.85  # A+ 極強
EVIDENCE_LEVEL_A = 0.65      # A 強（原 0.70，略降以反映內容權重上調）
EVIDENCE_LEVEL_B_PLUS = 0.50  # B+ 中強（原 0.55）
EVIDENCE_LEVEL_B = 0.35      # B 中等（原 0.40）
EVIDENCE_LEVEL_C = 0.22      # C 中弱（原 0.25）

# ==========================================
# 協調行為偵測閾值常數
# ==========================================
COORDINATION_DUPLICATE_RATIO_THRESHOLD = 0.3  # 重複內容比例閾值
COORDINATION_DOMAIN_CONCENTRATION_THRESHOLD = 0.5  # 域名集中度閾值
COORDINATION_DATE_CONCENTRATION_THRESHOLD = 0.4  # 時間集中度閾值
COORDINATION_HIGH_RISK_SCORE = 0.6  # 高風險協調行為分數閾值
COORDINATION_DUPLICATE_PENALTY = 0.4  # 重複內容扣分
COORDINATION_DOMAIN_PENALTY = 0.3  # 域名集中扣分
COORDINATION_TIME_PENALTY = 0.2  # 時間集中扣分

# ==========================================
# 證據權重評估閾值常數
# ==========================================
EVIDENCE_WEIGHT_STRONG_CONSENSUS_EXPERT_RATIO = 0.7  # 強共識：權威來源比例
EVIDENCE_WEIGHT_STRONG_CONSENSUS_QUALITY = 0.7  # 強共識：品質分數
EVIDENCE_WEIGHT_MODERATE_EXPERT_RATIO = 0.5  # 中等共識：權威來源比例
EVIDENCE_WEIGHT_MODERATE_QUALITY = 0.6  # 中等共識：品質分數
EVIDENCE_WEIGHT_DIVIDED_EXPERT_RATIO = 0.3  # 分歧觀點：權威來源比例
EVIDENCE_WEIGHT_WEAK_EXPERT_RATIO = 0.15  # 弱證據：權威來源比例

# ==========================================
# 內容品質評估閾值常數
# ==========================================
CONTENT_QUALITY_LONG = 1000  # 長內容閾值
CONTENT_QUALITY_MEDIUM = 500  # 中等內容閾值
CONTENT_QUALITY_SHORT = 200  # 短內容閾值
CONTENT_OVERLAP_HIGH = 0.5  # 標題內容重疊度高閾值
CONTENT_OVERLAP_MEDIUM = 0.2  # 標題內容重疊度中閾值

# ==========================================
# 網站品質評估常數
# ==========================================
WEBSITE_QUALITY_MIN_SCORE = 0.3  # 網站品質最低分數
WEBSITE_QUALITY_PENALTY_PER_ISSUE = 0.15  # 每個問題扣分

# ==========================================
# 相似度計算權重常數
# ==========================================
TITLE_SIMILARITY_CHAR_WEIGHT = 0.3  # 字元相似度權重
TITLE_SIMILARITY_WORD_WEIGHT = 0.7  # 詞級相似度權重
TITLE_SIMILARITY_QUICK_FILTER = 0.5  # 快速過濾閾值

# ==========================================
# 共識分析閾值常數
# ==========================================
CONSENSUS_HIGH_THRESHOLD = 0.7  # 高共識度閾值
CONSENSUS_MEDIUM_THRESHOLD = 0.4  # 中等共識度閾值
CROSS_VALIDATION_HIGH_RATIO = 0.7  # 交叉驗證高比例閾值
CROSS_VALIDATION_MEDIUM_RATIO = 0.4  # 交叉驗證中比例閾值

# ==========================================
# 公信力評分閾值常數
# ==========================================
CREDIBILITY_HIGH_THRESHOLD = 0.8  # 高公信力閾值
CREDIBILITY_MEDIUM_THRESHOLD = 0.6  # 中等公信力閾值

# 確保快取目錄存在
CACHE_DIR.mkdir(exist_ok=True)

# AI 輸出必需的章節標記
REQUIRED_SECTIONS_FUSION = [
    "NBLM 核心導讀指南",
    "ACH 競爭假設分析",
    "整體現況與脈絡",
    "爭議點與事實查核",
    "媒體框架光譜分析",
    "共識與分歧",
    "深層偏見與認知盲區解構",
    "Cui Bono",
    "敘事操縱與資訊操作風險",
    "影響力網絡與預警指標",
    "混合戰威脅建模"
]

REQUIRED_SECTIONS_SCENARIO = [
    "CLA 深度解構",
    "未來趨勢路徑模擬",
    "驗屍分析",
    "綜合發展與因應建議"
]

REQUIRED_SECTIONS_TEXT_ANALYSIS = [
    "新聞核心主張摘要",
    "文本可信度與證據強度",
    "語言與情緒操控檢測",
    "Entman 框架分析",
    "邏輯謬誤與事實查核風險",
    "深層偏見與認知盲區",
    "Cui Bono",
    "資訊不足與橫向查證建議"
]


def _extract_text_from_llm_content(content: Any) -> str:
    """
    從 LLM 回應的 content 正確萃取純文字，過濾 signature 等非文字元數據。
    
    Gemini API 有時回傳 content 為 list，其中可能包含：
    - str：純文字內容
    - dict 含 'text' 鍵：實際文字
    - dict 僅含 'signature' 鍵：元數據，不應顯示給使用者
    
    Args:
        content: LLM 回應的 content（可能為 str、list、或 dict）
    
    Returns:
        str: 萃取後的純文字字串
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                # 跳過僅含 signature 或其他元數據的 dict，不加入輸出
            else:
                # 其他類型（如 AIMessageChunk）：嘗試取得字串內容
                if hasattr(item, "content") and item.content:
                    parts.append(_extract_text_from_llm_content(item.content))
                else:
                    text = str(item)
                    # 排除明顯為 metadata 的輸出（如 {'signature': '...'}）
                    if "'signature'" in text and text.startswith("{"):
                        continue
                    parts.append(text)
        return "".join(parts)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(content)


# ==========================================
# 1. 基礎設定與 CSS樣式
# ==========================================
st.set_page_config(page_title="多元觀點解析", page_icon="⚖️", layout="wide")

CSS_STYLE = """
<style>
    body { font-family: "Microsoft JhengHei", "Georgia", sans-serif; line-height: 1.6; color: #333; }
    .stButton button[kind="secondary"] { border: 2px solid #673ab7; color: #673ab7; font-weight: bold; }
    
    .report-paper {
        background-color: #fdfbf7; 
        color: #2c3e50; 
        padding: 32px 40px; 
        border-radius: 8px; 
        margin-bottom: 15px; 
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        font-family: "Microsoft JhengHei", "Segoe UI", "Georgia", sans-serif;
        line-height: 1.75;
        font-size: 1.02rem;
    }
    .report-paper h1 { margin-top: 2em; margin-bottom: 0.6em; font-size: 1.5em; color: #1a1a2e; padding-bottom: 0.3em; border-bottom: 2px solid #e5e7eb; }
    .report-paper h2 { margin-top: 1.8em; margin-bottom: 0.5em; font-size: 1.28em; color: #252550; }
    .report-paper h3 { margin-top: 1.5em; margin-bottom: 0.4em; font-size: 1.15em; color: #2d3748; }
    .report-paper h4 { margin-top: 1.2em; margin-bottom: 0.35em; font-size: 1.05em; }
    .report-paper p { margin-bottom: 1em; }
    
    /* 報告內表格：卡片式區塊、提升可讀性 */
    .report-paper table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 1.2em 0 1.8em 0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        background: #fff;
        table-layout: fixed;
    }
    .report-paper table thead th {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        color: #334155;
        font-weight: 600;
        padding: 14px 18px;
        text-align: left;
        border-bottom: 2px solid #e2e8f0;
        font-size: 0.95em;
    }
    .report-paper table tbody td {
        padding: 16px 18px;
        border-bottom: 1px solid #f1f5f9;
        vertical-align: top;
        line-height: 1.65;
        word-wrap: break-word;
        overflow-wrap: break-word;
        color: #475569;
    }
    .report-paper table tbody td ul,
    .report-paper table tbody td ol {
        margin: 0.35em 0 0.35em 1.2em;
        padding-left: 0.8em;
    }
    .report-paper table tbody td p {
        margin: 0 0 0.55em 0;
    }
    .report-paper table tbody td p:last-child {
        margin-bottom: 0;
    }
    .report-paper table tbody tr {
        transition: background-color 0.15s ease;
    }
    .report-paper table tbody tr:nth-child(even) { background-color: #fafbfc; }
    .report-paper table tbody tr:hover { background-color: #f1f5f9; }
    .report-paper table tbody tr:last-child td { border-bottom: none; }
    /* 第一欄（假設名稱、日期）強調 */
    .report-paper table tbody td:first-child { font-weight: 600; color: #1e293b; min-width: 100px; max-width: 200px; }
    .report-paper table tbody td:nth-child(4), .report-paper table tbody td:nth-child(5) { font-size: 0.95em; color: #64748b; min-width: 70px; }
    
    .citation {
        font-size: 0.78em;
        color: #64748b;
        background-color: #f1f5f9;
        padding: 3px 8px;
        border-radius: 4px;
        margin: 0 2px;
        font-family: ui-monospace, monospace;
        border: 1px solid #e2e8f0;
        font-weight: 500;
        vertical-align: 1px;
        display: inline-block;
    }

    .scrollable-table-container {
        height: 600px; 
        overflow-y: auto; 
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background-color: white;
        margin-bottom: 20px;
    }
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-family: "Microsoft JhengHei", sans-serif;
        font-size: 0.95em;
    }
    .custom-table th { position: sticky; top: 0; background-color: #f1f3f4; color: #333; font-weight: bold; padding: 12px 15px; text-align: left; border-bottom: 2px solid #ddd; z-index: 2; }
    .custom-table td { padding: 10px 15px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; color: #333; }
    .custom-table tr:hover { background-color: #f8f9fa; }
    .custom-table a { color: #1a73e8; text-decoration: none; font-weight: 500; font-size: 1.05em; }
    .custom-table a:hover { text-decoration: underline; color: #1557b0; }
    
    @media print {
        .scrollable-table-container { height: auto; overflow: visible; }
        body { font-size: 12pt; }
        a { text-decoration: none; color: #000; }
        .report-paper { box-shadow: none; border: none; padding: 0; }
    }
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# ==========================================
# 2. 資料庫與共用常數 (Config)
# ==========================================
# ==========================================
# 擴展媒體資料庫（優化：增加搜尋覆蓋率）
# ==========================================

# 藍營媒體白名單（大幅擴展；與 GREEN 互斥，無重複；2026.02.12 補齊缺失）
BLUE_WHITELIST = [
    # 主要藍營媒體
    "udn.com", "udn.com.tw", "chinatimes.com", "tvbs.com.tw", "cti.com.tw",
    "nownews.com", "ctee.com.tw", "storm.mg",
    # 擴展藍營媒體
    "want-daily.com", "want-china.com", "coolloud.org.tw", "kmt.org.tw",
    "appledaily.com.tw", "nexttv.com.tw", "ebc.net.tw", "ctwant.com",
    "wealth.com.tw", "bwnet.com.tw", "cmmedia.com.tw", "cmoney.tw",
    "technews.tw", "techbang.com", "digitimes.com.tw", "ithome.com.tw",
    # 地方媒體（taiwanhot、taiwannews 已移至 NEUTRAL，避免藍綠重複）
    "merit-times.com.tw",
    # 補齊：Yahoo TW 常轉載藍營內容
    "yahoo.com.tw",
]

# 綠營媒體白名單（大幅擴展；與 BLUE 互斥；獨立媒體歸 INDIE 不在此列；2026.02.12 補齊）
GREEN_WHITELIST = [
    # 主要綠營媒體
    "ltn.com.tw", "ftvnews.com.tw", "setn.com", "rti.org.tw",
    "newtalk.tw", "mirrormedia.mg", "upmedia.mg",
    # 擴展綠營媒體（twreporter/theinitium/thenewslens 為 INDIE，不計入綠營保底）
    "peoplenews.tw", "dpp.org.tw",
    "watchout.tw", "taisounds.com", "taiwanjustice.net", "taiwanjustice.org",
    "mnews.tw",
]

# 官方媒體白名單（大幅擴展；2026.02.12 補齊）
OFFICIAL_WHITELIST = [
    # 主要官方媒體
    "cna.com.tw", "pts.org.tw", "mnd.gov.tw", "mac.gov.tw",
    "tfc-taiwan.org.tw", "gov.tw",
    # 擴展官方機構
    "ey.gov.tw", "moea.gov.tw", "mofa.gov.tw", "moc.gov.tw",
    "mohw.gov.tw", "moj.gov.tw", "moe.gov.tw", "mnd.gov.tw",
    "judicial.gov.tw", "taipei.gov.tw", "newtaipei.gov.tw",
    "taichung.gov.tw", "tainan.gov.tw", "khcc.gov.tw",
    # 事實查核機構
    "mygopen.com", "cofacts.tw", "tfc-taiwan.org.tw", "tfc.org.tw",
    # 學術機構
    "nccu.edu.tw", "ntu.edu.tw", "sinica.edu.tw", "nctu.edu.tw", "ncl.edu.tw",
]

# 中立商業媒體白名單（不參與藍綠保底；yahoo.com.tw 已歸藍營）
NEUTRAL_WHITELIST = [
    "taiwanhot.net", "taiwannews.com.tw",
    "ettoday.net", "businessweekly.com.tw",
    "commonhealth.com.tw", "cw.com.tw", "managertoday.com.tw",
    "bnext.com.tw", "inside.com.tw", "techorange.com",
    "gvm.com.tw", "anue.com",
]

# 完整台灣媒體白名單（擴充：包含社群平台；用於通用搜尋範圍）
FULL_TAIWAN_WHITELIST = (
    BLUE_WHITELIST + GREEN_WHITELIST + OFFICIAL_WHITELIST + NEUTRAL_WHITELIST +
    ["youtube.com", "youtu.be", "ptt.cc", "dcard.tw", "mobile01.com"]
)

# 獨立媒體白名單（大幅擴展：調查報導、法律/科學/人文、環境/社運；2026.02.12 補齊）
INDIE_WHITELIST = [
    # 主要獨立／調查媒體
    "twreporter.org", "theinitium.com", "thenewslens.com", "readr.tw",
    # 法律／科學／人文
    "plainlaw.me", "pansci.asia", "story.gushi.tw", "thinkingtaiwan.com",
    "whogovernstw.org", "voicettank.org", "openbook.org.tw",
    # 環境／社運／農業
    "newsmarket.com.tw", "e-info.org.tw", "eventsinfocus.org",
    "civilmedia.tw", "rightplus.org", "npost.tw", "leaptop.com",
    # 國際視野／部落格平台
    "mindiworldnews.com", "vocus.cc", "matters.town", "medium.com",
    "substack.com", "ghost.org", "wordpress.com",
    # 擴展
    "womany.net", "biosmonthly.com",
    # 學術媒體
    "taiwaninsight.com", "taiwaninsight.org", "new7.com.tw",
    # 自媒體影音平台
    "youtube.com", "youtu.be",
]

# 日本媒體白名單（擴展：週刊、調查、商業深度；2026.02.12 補齊）
INTL_JAPAN_WHITELIST = [
    # 主要日報與 NHK
    "asahi.com", "mainichi.jp", "yomiuri.co.jp", "nikkei.com",
    "sankei.com", "nhk.or.jp", "www3.nhk.or.jp",
    # 通訊社
    "kyodonews.com", "jiji.com",
    # 調查／週刊（醜聞與深度報導）
    "bunshun.jp", "dailyshincho.jp", "friday.kodansha.co.jp",
    "gendai.ismedia.jp", "post.tv-asahi.co.jp",
    # 商業／深度分析
    "toyokeizai.net", "diamond.jp", "jbpress.ismedia.jp",
    "wedge.ismedia.jp", "president.jp", "business.nikkei.com",
    "newspicks.com", "agora-web.jp",
    # 英文版
    "japantimes.co.jp", "asia.nikkei.com", "japannews.yomiuri.co.jp",
    # 國際媒體日本報導
    "reuters.com", "apnews.com", "bloomberg.com",
]

# 國際通訊社／外電白名單（三大通訊社 + 區域主要通訊社）
INTL_WIRES_WHITELIST = [
    # 全球三大通訊社
    "reuters.com", "apnews.com", "afp.com",
    # 歐洲通訊社
    "dpa.com", "efe.com", "ansa.it", "belga.be",
    # 亞太通訊社
    "kyodonews.com", "jiji.com", "yna.co.kr",
    # 其他重要外電
    "upi.com", "dpa-international.com"
]

# 國際智庫與政策研究機構白名單
INTL_THINKTANKS_WHITELIST = [
    # 美國智庫
    "brookings.edu", "csis.org", "cfr.org", "rand.org",
    "carnegieendowment.org", "heritage.org", "atlanticcouncil.org",
    "piie.com", "cnas.org", "stimson.org", "wilsoncenter.org",
    "fpri.org", "newamerica.org",
    # 英國／歐洲智庫
    "chathamhouse.org", "iiss.org", "rusi.org",
    "sipri.org", "ecfr.eu", "ifri.org", "swp-berlin.org",
    "clingendael.org", "bruegel.org",
    # 亞太智庫
    "aspi.org.au", "lowyinstitute.org",
    # 政策分析期刊
    "foreignaffairs.com", "foreignpolicy.com", "thediplomat.com"
]

# 亞洲國際媒體白名單（大幅擴展：含外電、智庫、日本、韓國、東南亞）
INTL_ASIA_WHITELIST = [
    # 主要亞洲媒體
    "bbc.com", "cnn.com", "reuters.com", "apnews.com", "bloomberg.com", 
    "wsj.com", "nytimes.com", "nikkei.com", "nhk.or.jp", "scmp.com", 
    "asia.nikkei.com", "channelnewsasia.com",
    # 國際通訊社（外電）
    "afp.com", "dpa.com", "efe.com", "kyodonews.com", "jiji.com", "yna.co.kr",
    # 國際智庫
    "brookings.edu", "csis.org", "cfr.org", "carnegieendowment.org",
    "chathamhouse.org", "iiss.org", "aspi.org.au", "lowyinstitute.org",
    "foreignaffairs.com", "foreignpolicy.com", "thediplomat.com",
    # 日本媒體（與 INTL_JAPAN_WHITELIST 重疊）
    "japantimes.co.jp", "asahi.com", "yomiuri.co.jp", "mainichi.jp",
    "sankei.com", "www3.nhk.or.jp",
    # 韓國媒體
    "koreaherald.com", "koreatimes.co.kr",
    # 東南亞媒體
    "straitstimes.com", "todayonline.com", "thejakartapost.com",
    "bangkokpost.com", "philstar.com", "manilatimes.net",
    "vietnamnews.vn", "thestar.com.my", "malaysiakini.com",
    # 南亞媒體
    "thehindu.com", "indiatimes.com", "dawn.com"
]

# 歐洲國際媒體白名單（大幅擴展：含外電、智庫）
INTL_EUROPE_WHITELIST = [
    # 主要歐洲媒體
    "bbc.com", "dw.com", "euronews.com", "theguardian.com", 
    "lemonde.fr", "elpais.com", "spiegel.de", "ft.com", "politico.eu",
    # 歐洲通訊社（外電）
    "reuters.com", "apnews.com", "afp.com", "dpa.com", "efe.com", "ansa.it",
    # 歐洲智庫
    "chathamhouse.org", "iiss.org", "rusi.org", "ecfr.eu",
    "sipri.org", "ifri.org", "swp-berlin.org", "bruegel.org", "clingendael.org",
    # 擴展歐洲媒體
    "bloomberg.com", "wsj.com", "nytimes.com", "washingtonpost.com", "latimes.com",
    "faz.net", "welt.de", "zeit.de", "sueddeutsche.de",
    "repubblica.it", "corriere.it", "ilsole24ore.com",
    "lefigaro.fr", "liberation.fr", "france24.com", "rfi.fr",
    "elmund.es", "abc.es", "elmundo.es",
    "telegraph.co.uk", "independent.co.uk", "dailymail.co.uk",
    "express.co.uk", "mirror.co.uk", "standard.co.uk"
]

# 美洲國際媒體白名單（大幅擴展：含外電、智庫）
INTL_AMERICAS_WHITELIST = [
    # 主要美洲媒體
    "cnn.com", "bbc.com", "reuters.com", "apnews.com", "bloomberg.com", 
    "wsj.com", "nytimes.com", "washingtonpost.com", "latimes.com", 
    "nbcnews.com", "abcnews.go.com", "cbsnews.com",
    # 通訊社（外電）
    "afp.com", "upi.com",
    # 美國智庫
    "brookings.edu", "csis.org", "cfr.org", "rand.org",
    "carnegieendowment.org", "heritage.org", "atlanticcouncil.org",
    "piie.com", "cnas.org", "stimson.org", "wilsoncenter.org",
    "fpri.org", "newamerica.org",
    # 政策分析期刊
    "foreignaffairs.com", "foreignpolicy.com", "thediplomat.com",
    # 擴展美洲媒體
    "usatoday.com", "usnews.com", "time.com", "newsweek.com",
    "theatlantic.com", "newyorker.com", "wired.com", "techcrunch.com",
    "theverge.com", "arstechnica.com", "engadget.com", "gizmodo.com",
    "globeandmail.com", "cbc.ca", "nationalpost.com", "thestar.com",
    "folha.uol.com.br", "oglobo.globo.com", "estadao.com.br",
    "clarin.com", "lanacion.com.ar", "eluniversal.com.mx",
    "jornada.com.mx", "reforma.com", "milenio.com"
]

# 西方／全球調查與權威媒體白名單（2026.02.12 新增：調查報導、OSINT、智庫）
INTL_WEST_WHITELIST = [
    # 頂級調查／獨立
    "propublica.org", "publicintegrity.org", "thebureauinvestigates.com",
    "icij.org", "revealnews.org", "theintercept.com", "bellingcat.com",
    "vox.com", "axios.com", "politico.com", "politico.eu",
    "slate.com", "motherjones.com", "democracynow.org",
    # 智庫與地緣
    "foreignaffairs.com", "foreignpolicy.com", "thediplomat.com",
    "csis.org", "rusi.org", "chathamhouse.org", "rand.org",
    "atlanticcouncil.org", "carnegieendowment.org", "crisisgroup.org",
    "project-syndicate.org", "theconversation.com",
    # 主流國際
    "nytimes.com", "wsj.com", "washingtonpost.com", "ft.com",
    "theguardian.com", "economist.com", "bloomberg.com",
    "reuters.com", "apnews.com", "afp.com", "bbc.com", "cnn.com",
    "dw.com", "france24.com", "aljazeera.com", "scmp.com",
    "straitstimes.com", "channelnewsasia.com",
]

# 完整國際媒體白名單（去重；含亞洲、歐美、西方調查、外電與智庫）
INTL_WHITELIST = sorted(list(set(
    INTL_ASIA_WHITELIST + INTL_EUROPE_WHITELIST + INTL_AMERICAS_WHITELIST +
    INTL_WEST_WHITELIST + INTL_WIRES_WHITELIST + INTL_THINKTANKS_WHITELIST
)))

# 擴展域名中文名稱映射（用於顯示）
DOMAIN_NAME_MAP = {
    # 藍營媒體
    "udn.com": "聯合報", "udn.com.tw": "聯合報", "chinatimes.com": "中國時報", 
    "tvbs.com.tw": "TVBS", "cti.com.tw": "中天新聞", "nownews.com": "NOWnews",
    "ctee.com.tw": "工商時報", "storm.mg": "風傳媒", "want-daily.com": "旺報",
    "coolloud.org.tw": "苦勞網", "kmt.org.tw": "國民黨", "appledaily.com.tw": "蘋果日報",
    "nexttv.com.tw": "壹電視", "ebc.net.tw": "東森新聞", "ctwant.com": "CTWANT",
    "wealth.com.tw": "今周刊", "bwnet.com.tw": "商業周刊", "cmmedia.com.tw": "遠見",
    "cmoney.tw": "CMoney", "technews.tw": "科技新報", "techbang.com": "T客邦",
    "digitimes.com.tw": "電子時報", "ithome.com.tw": "iThome",
    # 綠營媒體
    "ltn.com.tw": "自由時報", "ftvnews.com.tw": "民視新聞", "setn.com": "三立新聞",
    "rti.org.tw": "央廣", "newtalk.tw": "新頭殼", "mirrormedia.mg": "鏡週刊",
    "upmedia.mg": "上報", "peoplenews.tw": "民報", "dpp.org.tw": "民進黨",
    "taiwanhot.net": "台灣好新聞", "taiwannews.com.tw": "台灣英文新聞",
    "watchout.tw": "沃草", "taisounds.com": "台灣聲音",
    # 官方媒體
    "cna.com.tw": "中央社", "pts.org.tw": "公視", "mnd.gov.tw": "國防部",
    "mac.gov.tw": "陸委會", "tfc-taiwan.org.tw": "台灣事實查核中心",
    "gov.tw": "政府網站", "ey.gov.tw": "行政院", "moea.gov.tw": "經濟部",
    "mofa.gov.tw": "外交部", "moc.gov.tw": "文化部", "mohw.gov.tw": "衛福部",
    "moj.gov.tw": "法務部", "moe.gov.tw": "教育部", "judicial.gov.tw": "司法院",
    "taipei.gov.tw": "台北市政府", "newtaipei.gov.tw": "新北市政府",
    "taichung.gov.tw": "台中市政府", "tainan.gov.tw": "台南市政府",
    "khcc.gov.tw": "高雄市政府",
    # 獨立媒體
    "twreporter.org": "報導者", "theinitium.com": "端傳媒", 
    "thenewslens.com": "關鍵評論網", "mindiworldnews.com": "敏迪選讀",
    "vocus.cc": "方格子", "matters.town": "Matters", "plainlaw.me": "法律白話文",
    "readr.tw": "READr", "new7.com.tw": "新新聞", "watchout.tw": "沃草",
    "thinkingtaiwan.com": "思想坦克", "taiwaninsight.com": "台灣智庫",
    # 國際媒體
    "bbc.com": "BBC", "cnn.com": "CNN", "reuters.com": "路透社", 
    "apnews.com": "美聯社", "bloomberg.com": "彭博", "wsj.com": "華爾街日報",
    "nytimes.com": "紐約時報", "washingtonpost.com": "華盛頓郵報",
    "theguardian.com": "衛報", "dw.com": "德國之聲", "france24.com": "France 24",
    "aljazeera.com": "半島電視台", "scmp.com": "南華早報",
    "channelnewsasia.com": "亞洲新聞台", "straitstimes.com": "海峽時報",
    "japantimes.co.jp": "日本時報", "asahi.com": "朝日新聞",
    "yomiuri.co.jp": "讀賣新聞", "mainichi.jp": "每日新聞", "nhk.or.jp": "NHK",
    "nikkei.com": "日經新聞", "asia.nikkei.com": "日經亞洲",
    "kyodonews.com": "共同社", "jiji.com": "時事通訊社",
    "koreaherald.com": "韓國先驅報", "koreatimes.co.kr": "韓國時報",
    # 國際通訊社（外電）
    "afp.com": "法新社", "dpa.com": "德新社", "efe.com": "埃菲社",
    "ansa.it": "安莎通訊社", "upi.com": "合眾國際社",
    # 國際智庫
    "brookings.edu": "布魯金斯學會", "csis.org": "CSIS", "cfr.org": "外交關係協會",
    "rand.org": "蘭德公司", "carnegieendowment.org": "卡內基國際和平基金會",
    "heritage.org": "傳統基金會", "atlanticcouncil.org": "大西洋理事會",
    "chathamhouse.org": "皇家國際事務研究所", "iiss.org": "國際戰略研究所",
    "rusi.org": "皇家聯合服務研究所", "piie.com": "彼得森國際經濟研究所",
    "sipri.org": "斯德哥爾摩國際和平研究所", "aspi.org.au": "澳洲戰略政策研究所",
    "lowyinstitute.org": "羅伊研究所", "foreignaffairs.com": "外交事務",
    "foreignpolicy.com": "外交政策", "thediplomat.com": "外交家",
    # 社群媒體
    "ptt.cc": "PTT", "dcard.tw": "Dcard", "mobile01.com": "Mobile01",
    "facebook.com": "Facebook", "youtube.com": "YouTube", "twitter.com": "Twitter",
    "instagram.com": "Instagram", "tiktok.com": "TikTok",
    # 聚合平台
    "yahoo.com": "Yahoo新聞", "yahoo.com.tw": "Yahoo奇摩", 
    "ettoday.net": "ETtoday", "businessweekly.com.tw": "商業周刊",
    "commonhealth.com.tw": "康健", "cw.com.tw": "天下雜誌",
    "managertoday.com.tw": "經理人", "bnext.com.tw": "數位時代",
    "inside.com.tw": "Inside", "techorange.com": "科技報橘",
    # 事實查核
    "mygopen.com": "MyGoPen", "cofacts.tw": "Cofacts", "tfc.org.tw": "台灣事實查核中心",
    # 2026.02.12 補齊：中立、獨立、日本週刊、西方調查
    "gvm.com.tw": "遠見", "anue.com": "鉅亨網", "ncl.edu.tw": "國家圖書館",
    "mnews.tw": "民視新聞",
    "pansci.asia": "泛科學", "openbook.org.tw": "Openbook", "voicettank.org": "想想論壇",
    "whogovernstw.org": "誰來關心", "newsmarket.com.tw": "上下游", "e-info.org.tw": "環境資訊中心",
    "bunshun.jp": "文春", "dailyshincho.jp": "日刊新潮", "toyokeizai.net": "東洋經濟",
    "jbpress.ismedia.jp": "JBpress", "wedge.ismedia.jp": "Wedge", "propublica.org": "ProPublica",
    "bellingcat.com": "Bellingcat", "theintercept.com": "The Intercept", "politico.com": "Politico",
}

# 擴展媒體資料庫映射（用於來源分類）
DB_MAP = {
    "CHINA": [
        # 主要中國媒體
        "xinhuanet", "people.com.cn", "huanqiu", "cctv", "chinadaily", 
        "taiwan.cn", "gwytb", "guancha",
        # 擴展中國媒體
        "thepaper.cn", "sina.com.cn", "163.com", "sohu.com", "ifeng.com",
        "crntt.com", "hk01.com", "wenweipo.com", "takungpao.com",
        "rthk.hk", "singtao.com", "mingpao.com", "appledaily.com.hk"
    ],
    "GREEN": [
        # 主要綠營媒體
        "ltn", "ftv", "setn", "rti.org", "newtalk", "mirrormedia", "dpp", "upmedia",
        # 擴展綠營媒體（taiwanhot/taiwannews→NEUTRAL；twreporter/theinitium/thenewslens→INDIE）
        "peoplenews", "watchout", "taisounds", "taiwanjustice", "mnews",
    ],
    "BLUE": [
        # 主要藍營媒體
        "udn", "chinatimes", "tvbs", "cti", "nownews", "ctee", "kmt", "storm",
        # 擴展藍營媒體
        "want-daily", "want-china", "coolloud", "appledaily", "nexttv",
        "ebc", "ctwant", "wealth", "bwnet", "cmmedia", "cmoney",
        "technews", "techbang", "digitimes", "ithome", "merit-times"
    ],
    "OFFICIAL": [
        # 主要官方媒體
        "cna.com", "pts.org", "mnd.gov", "mac.gov", "tfc-taiwan", "gov.tw",
        # 擴展官方機構
        "ey.gov", "moea.gov", "mofa.gov", "moc.gov", "mohw.gov", "moj.gov",
        "moe.gov", "judicial.gov", "taipei.gov", "newtaipei.gov",
        "taichung.gov", "tainan.gov", "khcc.gov",
        # 事實查核機構
        "mygopen", "cofacts", "tfc.org",
        # 學術機構
        "nccu.edu", "ntu.edu", "sinica.edu", "nctu.edu", "ncl.edu",
    ],
    "NEUTRAL": [
        # 中立商業/綜合媒體（不參與藍綠保底）
        "taiwanhot", "taiwannews", "yahoo.com", "ettoday", "businessweekly",
        "commonhealth", "cw.com", "managertoday", "bnext", "inside", "techorange",
        "gvm", "anue",
    ],
    "INDIE": [
        # 台灣獨立／調查／部落格
        "twreporter", "theinitium", "thenewslens", "mindiworld", "vocus",
        "matters", "plainlaw", "readr", "new7", "watchout", "taisounds",
        "taiwanjustice", "thinkingtaiwan", "taiwaninsight", "whogoverns",
        "story.gushi", "pansci", "newsmarket", "e-info", "eventsinfocus",
        "civilmedia", "rightplus", "npost", "leaptop", "opinion.cw",
    ],
    "INTL": [
        # 主要國際媒體
        "bbc", "cnn", "reuters", "apnews", "bloomberg", "wsj", "nytimes", 
        "dw.com", "voanews", "rfi",
        # 國際通訊社（外電）
        "afp.com", "dpa.com", "efe.com", "ansa", "upi",
        "kyodonews", "jiji", "yna",
        # 國際智庫
        "brookings", "csis.org", "cfr.org", "rand.org",
        "carnegieendowment", "heritage.org", "atlanticcouncil",
        "chathamhouse", "iiss.org", "rusi.org", "piie.com",
        "cnas.org", "stimson", "wilsoncenter", "fpri.org",
        "sipri.org", "ecfr.eu", "ifri.org", "swp-berlin",
        "aspi.org", "lowyinstitute",
        # 政策分析期刊
        "foreignaffairs", "foreignpolicy", "thediplomat",
        # 擴展國際媒體
        "theguardian", "washingtonpost", "latimes", "usatoday", "time",
        "newsweek", "theatlantic", "newyorker", "ft.com", "economist",
        "aljazeera", "france24", "euronews", "spiegel", "faz",
        "lemonde", "elpais", "repubblica", "corriere", "telegraph",
        "independent", "scmp", "channelnewsasia", "straitstimes",
        "japantimes", "asahi", "yomiuri", "mainichi", "nhk",
        "nikkei", "asia.nikkei", "koreaherald", "koreatimes",
        # 調查／國際權威（2026.02.12 補齊）
        "propublica", "publicintegrity", "bellingcat", "intercept", "politico",
        "axios", "vox", "motherjones", "project-syndicate", "conversation",
        # 日本深度分析
        "bunshun", "dailyshincho", "toyokeizai", "diamond.jp", "jbpress", "wedge",
    ],
    "FARM": [
        # 內容農場
        "kknews", "read01", "ppfocus", "buzzhand", "bomb01", "qiqi", 
        "inf.news", "toutiao",
        # 擴展內容農場
        "lackk", "mission-tw", "hottopic", "xuehua", "baidu", "sina",
        "163", "sohu", "ifeng", "thepaper"
    ],
    "SOCIAL": [
        # 社群媒體
        "ptt.cc", "dcard", "mobile01", "facebook", "youtube",
        # 擴展社群媒體
        "twitter.com", "x.com", "instagram.com", "tiktok.com",
        "weibo.com", "douyin.com", "bilibili.com", "reddit.com",
        "linkedin.com", "pinterest.com", "substack", "medium",
    ],
    "VIDEO": [
        # 影音平台
        "youtube.com", "youtu.be", "tiktok.com", "douyin.com",
        "bilibili.com", "ixigua.com", "vimeo.com", "dailymotion.com"
    ],
    "AGGREGATOR": [
        # 聚合平台
        "yahoo.com", "msn.com", "linetoday.com", "google.com", 
        "ettoday.net", "yahoo.com.tw", "msn.com.tw"
    ],
    "ACADEMIC": [
        # 學術來源
        ".edu", ".ac.uk", ".ac.jp", ".ac.tw", ".edu.tw",
        "nccu.edu", "ntu.edu", "sinica.edu", "nctu.edu",
        "harvard.edu", "mit.edu", "stanford.edu", "oxford.ac.uk",
        "cambridge.ac.uk", "tokyo.ac.jp", "kyoto-u.ac.jp"
    ]
}

NOISE_BLACKLIST = ["zhihu.com", "baidu.com", "pinterest.com", "instagram.com", "tiktok.com", "tmall.com", "taobao.com", "163.com", "sohu.com"]

# ==========================================
# 來源公信力評分系統（方案 2）
# ==========================================
AUTHORITY_TIERS = {
    "Tier_1_Academic": {
        "domains": [".edu", ".ac.uk", ".ac.jp", ".ac.tw"],
        "base_score": 0.95,
        "weight_coefficient": 1.5
    },
    "Tier_1_Government": {
        "domains": [".gov.tw", ".gov", ".gov.uk", ".gov.au"],
        "base_score": 0.90,
        "weight_coefficient": 1.4
    },
    "Tier_2_International": {
        "media_list": [
            "bbc.com", "reuters.com", "apnews.com", "afp.com", "bloomberg.com", 
            "wsj.com", "nytimes.com", "theguardian.com", "dw.com",
            "brookings.edu", "csis.org", "cfr.org", "rand.org",
            "carnegieendowment.org", "chathamhouse.org", "iiss.org",
            "foreignaffairs.com", "foreignpolicy.com", "thediplomat.com"
        ],
        "base_score": 0.85,
        "weight_coefficient": 1.3
    },
    "Tier_2_Independent": {
        "media_list": ["twreporter.org", "theinitium.com", "thenewslens.com"],
        "base_score": 0.75,
        "weight_coefficient": 1.2
    },
    "Tier_3_Commercial": {
        "base_score": 0.60,
        "weight_coefficient": 1.0
    },
    "Tier_4_Social": {
        "media_list": ["facebook.com", "twitter.com", "youtube.com", "ptt.cc", "dcard.tw"],
        "base_score": 0.30,
        "weight_coefficient": 0.5
    }
}

# ==========================================
# 3. 輔助函式 (Helper Functions)
# ==========================================

def get_domain_name(url: str) -> str:
    try: 
        return urlparse(url).netloc.replace("www.", "")
    except Exception as e:
        logger.debug(f"域名提取失敗: {url}, 錯誤: {str(e)}")
        return ""

class SourceReputationManager:
    """
    來源公信力評分管理器（方案 2）
    
    功能:
    - 靜態評分：基於域名、機構類型
    - 權重係數：用於 RAG 過程中的資訊加權
    """
    
    def __init__(self):
        self.static_scores_cache = {}  # 靜態評分快取
    
    def calculate_credibility_score(self, url: str, domain: str) -> Tuple[float, str]:
        """
        計算來源公信力分數 (0-1)
        
        Returns:
            Tuple[float, str]: (分數, Tier 名稱)
        """
        if url in self.static_scores_cache:
            return self.static_scores_cache[url]
        
        domain_lower = domain.lower()
        
        # Tier 1: 學術機構
        for domain_suffix in AUTHORITY_TIERS["Tier_1_Academic"]["domains"]:
            if domain_suffix in domain_lower:
                score = AUTHORITY_TIERS["Tier_1_Academic"]["base_score"]
                tier = "Tier_1_Academic"
                self.static_scores_cache[url] = (score, tier)
                return score, tier
        
        # Tier 1: 政府機構
        for domain_suffix in AUTHORITY_TIERS["Tier_1_Government"]["domains"]:
            if domain_suffix in domain_lower:
                score = AUTHORITY_TIERS["Tier_1_Government"]["base_score"]
                tier = "Tier_1_Government"
                self.static_scores_cache[url] = (score, tier)
                return score, tier
        
        # Tier 2: 國際權威媒體
        for media in AUTHORITY_TIERS["Tier_2_International"]["media_list"]:
            if media in domain_lower:
                score = AUTHORITY_TIERS["Tier_2_International"]["base_score"]
                tier = "Tier_2_International"
                self.static_scores_cache[url] = (score, tier)
                return score, tier
        
        # Tier 2: 獨立媒體
        for media in AUTHORITY_TIERS["Tier_2_Independent"]["media_list"]:
            if media in domain_lower:
                score = AUTHORITY_TIERS["Tier_2_Independent"]["base_score"]
                tier = "Tier_2_Independent"
                self.static_scores_cache[url] = (score, tier)
                return score, tier
        
        # Tier 4: 社群媒體
        for media in AUTHORITY_TIERS["Tier_4_Social"]["media_list"]:
            if media in domain_lower:
                score = AUTHORITY_TIERS["Tier_4_Social"]["base_score"]
                tier = "Tier_4_Social"
                self.static_scores_cache[url] = (score, tier)
                return score, tier
        
        # Tier 3: 預設商業媒體
        score = AUTHORITY_TIERS["Tier_3_Commercial"]["base_score"]
        tier = "Tier_3_Commercial"
        self.static_scores_cache[url] = (score, tier)
        return score, tier
    
    def get_weight_coefficient(self, source_category: str, domain: str) -> float:
        """
        獲取權重係數（用於 RAG 加權）
        
        Returns:
            float: 權重係數（0.5-1.5）
        """
        _, tier = self.calculate_credibility_score("", domain)
        return AUTHORITY_TIERS.get(tier, {}).get("weight_coefficient", 1.0)

# 全局實例
_reputation_manager = SourceReputationManager()

def classify_source(url: str) -> str:
    if not url or url == "#": return "OTHER"
    try:
        domain = urlparse(url).netloc.lower()
        clean_domain = domain.replace("www.", "")
    except Exception as e:
        logger.debug(f"來源分類失敗: {url}, 錯誤: {str(e)}")
        return "OTHER"
    for cat, keywords in DB_MAP.items():
        for kw in keywords:
            if kw in domain: return cat
    return "OTHER"

def get_category_meta(cat: str) -> Tuple[str, str]:
    meta = {
        "CHINA": ("🇨🇳 中國官媒", "#d32f2f"),
        "FARM": ("⛔ 內容農場", "#ef6c00"),
        "BLUE": ("🔵 泛藍觀點", "#1565c0"),
        "GREEN": ("🟢 泛綠觀點", "#2e7d32"),
        "OFFICIAL": ("⚪ 官方/公廣", "#546e7a"),
        "NEUTRAL": ("📰 中立商業", "#78909c"),
        "INDIE": ("🕵️ 獨立/深度", "#fbc02d"),
        "INTL": ("🌏 國際媒體", "#f57c00"),
        "VIDEO": ("🟣 影音社群", "#7b1fa2"),
        "SOCIAL": ("⚠️ 社群聲量", "#607d8b"),
        "OTHER": ("📄 其他來源", "#9e9e9e")
    }
    return meta.get(cat, ("📄 其他來源", "#9e9e9e"))

def format_citation_style(text: str) -> str:
    if not text: return ""
    def replacement(match):
        nums = re.findall(r'\d+', match.group(0))
        if not nums: return match.group(0)
        unique_nums = sorted(list(set(nums)), key=int)
        return f'<span class="citation">Source {", ".join(unique_nums)}</span>'
    text = re.sub(r'(\[Source \d+\](?:[,;]?\s*\[Source \d+\])*)', replacement, text)
    text = re.sub(r'([\[\(（]\s*Source\s+[\d,，、\s]+[\]\)）])', replacement, text)
    return text

def validate_date(date_str: str) -> bool:
    """驗證日期字串是否為有效的 YYYY-MM-DD 格式"""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False

def extract_date_from_url(url: str) -> Optional[str]:
    """從 URL 中提取日期，並驗證有效性"""
    if not url: return None
    patterns = [r'/(\d{4})[-/](\d{2})[-/](\d{2})/', r'/(\d{4})(\d{2})(\d{2})/', r'-(\d{4})(\d{2})(\d{2})']
    for p in patterns:
        match = re.search(p, url)
        if match:
            date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            # 驗證日期有效性
            if validate_date(date_str):
                return date_str
    return None

# [V37.3 New] 核心渲染邏輯抽取 (DRY Fix)
def process_timeline_rows(timeline_data: List[Dict], sources: List[Dict], blind_mode: bool) -> str:
    """
    處理時間軸數據，執行嚴格清洗、排序與格式化。
    回傳：已排序的 HTML 表格行 (tr/td) 字串。
    """
    if not timeline_data: return ""
    
    valid_rows = []
    invalid_count = 0
    
    for item in timeline_data:
        s_id = item.get('source_id', 0)
        # 1. 嚴格過濾：無效來源直接丟棄
        if s_id == 0 or s_id > len(sources): 
            invalid_count += 1
            logger.debug(f"時間軸項目無效 Source ID: {s_id} (總來源數: {len(sources)})")
            continue
        
        source_data = sources[s_id-1]
        real_url = source_data.get('url', '#')
        if real_url == "#": continue 
        
        # 2. 日期瀑布流
        meta_date = source_data.get('published_date')
        url_date = extract_date_from_url(real_url)
        llm_date = item.get('date')
        
        real_date = "1970-01-01" 
        display_date = "------"
        
        if meta_date and meta_date != "Missing": 
            real_date = meta_date
            display_date = meta_date
        elif url_date: 
            real_date = url_date
            display_date = url_date
        elif llm_date and re.match(r'\d{4}-\d{2}-\d{2}', llm_date) and "XX" not in llm_date:
            real_date = llm_date
            display_date = llm_date
        
        # 3. 媒體名稱模糊匹配
        cat = classify_source(real_url)
        label, _ = get_category_meta(cat)
        domain = get_domain_name(real_url)
        
        media_name = domain
        # Fuzzy match domain name
        for k, v in DOMAIN_NAME_MAP.items():
            if k in domain: media_name = v
        
        emoji = "⚪"
        if "中國" in label: emoji = "🔴"
        elif "泛藍" in label: emoji = "🔵"
        elif "泛綠" in label: emoji = "🟢"
        elif "官方" in label: emoji = "⚪"
        elif "中立" in label: emoji = "📰"
        elif "獨立" in label: emoji = "🕵️"
        elif "國際" in label: emoji = "🌏"
        elif "農場" in label: emoji = "⛔"
        elif "社群" in label: emoji = "⚠️"
        
        display_media = f"{emoji} {media_name}"
        if blind_mode: display_media = "*****"
        
        title = item.get('title', 'No Title')
        title_html = f'<a href="{real_url}" target="_blank">{title}</a>'
        
        valid_rows.append({
            "sort_date": real_date,
            "html": f"<tr><td style='white-space:nowrap;'>{display_date}</td><td style='white-space:nowrap;'>{display_media}</td><td>{title_html}</td></tr>"
        })

    # 4. 強制按日期排序 (最新的在上面)
    valid_rows.sort(key=lambda x: x['sort_date'], reverse=True)
    
    if invalid_count > 0:
        logger.warning(f"時間軸處理：發現 {invalid_count} 個無效的 Source ID 引用")
    
    return "".join([r['html'] for r in valid_rows])

# ==========================================
# 4. 業務邏輯 (Business Logic)
# ==========================================

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
def get_cached_query_expansion(query: str) -> Optional[Dict[str, Any]]:
    """從快取獲取查詢擴展結果"""
    try:
        conn = init_cache_db()
        cache_key = hashlib.md5(query.encode()).hexdigest()
        cursor = conn.execute(
            "SELECT expanded_queries, balanced_queries, expiry_time FROM query_expansion_cache WHERE cache_key = ?",
            (cache_key,)
        )
        row = cursor.fetchone()
        if row:
            expanded_json, balanced_json, expiry_time = row
            if time.time() < expiry_time:
                result = {}
                # 安全解析 expanded_queries
                if expanded_json:
                    try:
                        result["expanded_queries"] = json.loads(expanded_json) if isinstance(expanded_json, str) else expanded_json
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"快取 expanded_queries 解析失敗: {str(e)}")
                        result["expanded_queries"] = None
                else:
                    result["expanded_queries"] = None
                
                # 安全解析 balanced_queries，確保返回字典類型
                if balanced_json:
                    try:
                        if isinstance(balanced_json, str):
                            parsed = json.loads(balanced_json)
                            # 確保解析結果是字典類型
                            if isinstance(parsed, dict):
                                result["balanced_queries"] = parsed
                            else:
                                logger.warning(f"快取 balanced_queries 不是字典類型: {type(parsed).__name__}，設為 None")
                                result["balanced_queries"] = None
                        elif isinstance(balanced_json, dict):
                            result["balanced_queries"] = balanced_json
                        else:
                            logger.warning(f"快取 balanced_queries 類型不正確: {type(balanced_json).__name__}，設為 None")
                            result["balanced_queries"] = None
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"快取 balanced_queries 解析失敗: {str(e)}")
                        result["balanced_queries"] = None
                else:
                    result["balanced_queries"] = None
                
                conn.close()
                logger.info(f"查詢擴展快取命中: {query[:50]}")
                return result
            else:
                # 快取過期，刪除
                conn.execute("DELETE FROM query_expansion_cache WHERE cache_key = ?", (cache_key,))
                conn.commit()
        conn.close()
        return None
    except Exception as e:
        logger.warning(f"查詢擴展快取讀取失敗: {str(e)}")
        return None

def cache_query_expansion(query: str, expanded_queries: List[Dict], balanced_queries: Optional[Dict] = None):
    """將查詢擴展結果存入快取"""
    try:
        conn = init_cache_db()
        cache_key = hashlib.md5(query.encode()).hexdigest()
        timestamp = time.time()
        expiry_time = timestamp + (CACHE_EXPIRY_HOURS * 3600)
        
        expanded_json = json.dumps(expanded_queries, ensure_ascii=False)
        balanced_json = json.dumps(balanced_queries, ensure_ascii=False) if balanced_queries else None
        
        conn.execute("""
            INSERT OR REPLACE INTO query_expansion_cache 
            (cache_key, query, expanded_queries, balanced_queries, timestamp, expiry_time) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (cache_key, query[:200], expanded_json, balanced_json, timestamp, expiry_time))
        conn.commit()
        conn.close()
        logger.info(f"查詢擴展已快取: {query[:50]}")
    except Exception as e:
        logger.warning(f"查詢擴展快取寫入失敗: {str(e)}")

def generate_expanded_queries(query: str, api_key: str, max_expansions: int = 12, use_cache: bool = True, focus_instruction: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    多層次查詢擴展機制（方案 1.1，優化版：減少 API 調用）
    
    Args:
        focus_instruction: 選用。使用者意圖導向（如 "Focus on economic security, ignore gossip"），會注入提示以引導 LLM 生成方向；有值時不讀快取。
    
    Returns:
        List[Dict]: [{"query": "...", "type": "...", "priority": ...}, ...]
    """
    use_cache_this = use_cache and not (focus_instruction or "").strip()
    if use_cache_this:
        cached = get_cached_query_expansion(query)
        if cached and cached.get("expanded_queries"):
            return cached["expanded_queries"][:max_expansions]
    
    expanded_queries = []
    focus_display = (focus_instruction or "").strip() or "無（請依照標準情報程序分析）"
    
    try:
        llm = ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, google_api_key=api_key, temperature=0.4)
        
        combined_prompt = f"""
        你是極度專業的情報檢索專家。請針對議題「{query}」生成「實體優先」的搜尋關鍵字，**必須產出專有名詞與具體術語**，不可只輸出「議題+泛用詞」。
        
        【使用者特別指令 (Focus Instruction)】：
        {focus_display}

        【嚴格禁止】：
        1. 禁止僅輸出「議題＋泛用詞」組合（如「議題 事件爭議」「議題 制裁影響」），必須產出具體專有名詞或術語。
        2. 禁止使用泛泛詞彙（如「新聞」「總整理」「懶人包」「影響」），除非是專有名詞的一部分。
        3. 禁止重複查詢詞。

        【必須包含實體 (Must Include Entities)】：
        - **專有名詞/術語**：如「存亡危機事態」「台灣有事就是日本有事」「經濟安保法」「日美安保」。
        - **關鍵人物**：如「薛劍」「川普」「高市早苗」。
        - **具體行動**：如「稀土出口管制」「稀土制裁」「撤回言論」。

        請依三個戰略維度各輸出 **1 個關鍵字**（需含上述實體）：
        1. 核心引爆點：具體事件/言論/法案（例：高市早苗 存亡危機事態 台灣）
        2. 具體攻防/代價：制裁或法律行動（例：中國 對日 稀土出口管制）
        3. 地緣連動：第三方或同盟（例：台灣有事就是日本有事 日美安保）

        【輸出格式】請**嚴格依下列兩行**輸出，不要其他說明或換行干擾：
        第一部分：關鍵字1, 關鍵字2, 關鍵字3
        第二部分：關鍵字4, 關鍵字5, 關鍵字6, 關鍵字7, 關鍵字8
        """
        
        combined_resp = _extract_text_from_llm_content(llm.invoke(combined_prompt).content)
        combined_resp = (combined_resp or "").strip().replace("\r\n", "\n")
        # 正規化全角符號，方便解析
        combined_resp_norm = combined_resp.replace("：", ":").replace("，", ",")
        
        def parse_part1(text: str) -> List[str]:
            """解析第一部分：嘗試多種模式，回傳至少 3 個關鍵字或空列表。"""
            # 模式 1：第一部分：key1, key2, key3
            m = re.search(r'第一部分\s*:\s*(.+?)(?=第二部分|$)', text, re.DOTALL | re.IGNORECASE)
            if m:
                part = m.group(1).strip()
                keys = [k.strip().strip('"') for k in re.split(r'[,，]', part) if k.strip()]
                if len(keys) >= 3:
                    return keys[:3]
            # 模式 2：僅找「第一」開頭的那一行
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("第一") and "部分" in line:
                    rest = re.sub(r'^第一\s*部分\s*[：:]\s*', '', line, flags=re.I).strip()
                    keys = [k.strip().strip('"') for k in re.split(r'[,，]', rest) if k.strip()]
                    if len(keys) >= 3:
                        return keys[:3]
            # 模式 3：找第一個看起來像「短句, 短句, 短句」的片段（長度合理）
            for seg in re.split(r'[\n。]', text):
                seg = seg.strip()
                if 10 < len(seg) < 200 and seg.count(',') >= 2:
                    keys = [k.strip().strip('"') for k in re.split(r'[,，]', seg) if 3 <= len(k.strip()) <= 80]
                    if len(keys) >= 3:
                        return keys[:3]
            return []
        
        def parse_part2(text: str) -> List[str]:
            """解析第二部分：語義擴展關鍵字。"""
            m = re.search(r'第二部分\s*:\s*(.+?)$', text, re.DOTALL | re.IGNORECASE)
            if m:
                part = m.group(1).strip()
                return [k.strip().strip('"') for k in re.split(r'[,，]', part) if 2 <= len(k.strip()) <= 80][:8]
            return []
        
        part1_keywords = parse_part1(combined_resp_norm) or parse_part1(combined_resp)
        if part1_keywords:
            expanded_queries.append({"query": part1_keywords[0], "type": "核心引爆點", "priority": 1})
            expanded_queries.append({"query": part1_keywords[1], "type": "具體攻防與代價", "priority": 1})
            expanded_queries.append({"query": part1_keywords[2], "type": "地緣連動", "priority": 1})
        
        part2_keywords = parse_part2(combined_resp_norm) or parse_part2(combined_resp)
        for kw in part2_keywords:
            if kw and kw not in [q["query"] for q in expanded_queries]:
                expanded_queries.append({"query": kw, "type": "語義擴展", "priority": 2})
        
        if len(expanded_queries) < 3:
            logger.warning(f"查詢擴展解析不足 3 筆，LLM 回應前 300 字: {(combined_resp or '')[:300]}")
        
        # 若 LLM 解析失敗，使用降級策略（仍盡量用議題+具體詞）
        if len(expanded_queries) < 3:
            expanded_queries.extend([
                {"query": f"{query} 事件 爭議", "type": "核心引爆點", "priority": 1},
                {"query": f"{query} 制裁 影響", "type": "具體攻防與代價", "priority": 1},
                {"query": f"{query} 地緣 同盟", "type": "地緣連動", "priority": 1}
            ])
        
        # 語境級擴展（時間/觀點維度）- 不需 LLM，避免過於泛化
        contextual_queries = [
            f"{query} 最新發展",
            f"{query} 歷史背景",
            f"{query} 支持觀點",
            f"{query} 反對觀點",
            f"{query} 中立分析",
        ]
        
        for cq in contextual_queries:
            if cq not in [q["query"] for q in expanded_queries]:
                expanded_queries.append({"query": cq, "type": "語境擴展", "priority": 3})
        
        # 按優先級排序並去重
        seen = set()
        unique_queries = []
        for q in sorted(expanded_queries, key=lambda x: x["priority"]):
            query_str = q["query"]
            if query_str not in seen:
                seen.add(query_str)
                unique_queries.append(q)
                if len(unique_queries) >= max_expansions:
                    break
        
        # 存入快取（有 focus_instruction 時不寫入，避免覆蓋通用快取）
        if use_cache_this:
            cache_query_expansion(query, unique_queries)
        
        logger.info(f"查詢擴展完成：生成了 {len(unique_queries)} 個擴展查詢（優化：合併為 1 次 API 調用）")
        return unique_queries
        
    except Exception as e:
        logger.warning(f"查詢擴展失敗，使用基礎關鍵字: {str(e)}")
        fallback = [
            {"query": f"{query} 事件 爭議", "type": "核心引爆點", "priority": 1},
            {"query": f"{query} 制裁 影響", "type": "具體攻防與代價", "priority": 1},
            {"query": f"{query} 地緣 同盟", "type": "地緣連動", "priority": 1}
        ]
        if use_cache_this:
            cache_query_expansion(query, fallback)
        return fallback

# 向後相容的函數
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
def generate_dynamic_keywords(query: str, api_key: str, use_cache: bool = True, focus_instruction: Optional[str] = None) -> List[str]:
    """向後相容：返回前三個關鍵字（優化：使用快取）。focus_instruction 會傳入 generate_expanded_queries 以導向意圖。"""
    expanded = generate_expanded_queries(query, api_key, max_expansions=3, use_cache=use_cache, focus_instruction=focus_instruction)
    return [q["query"] for q in expanded[:3]]

def generate_balanced_queries(query: str, api_key: str, use_cache: bool = True) -> Dict[str, List[str]]:
    """
    生成多維度平衡檢索查詢（方案 3）
    
    Returns:
        Dict: {
            "pro_arguments": ["支持觀點查詢1", ...],
            "con_arguments": ["反對觀點查詢1", ...],
            "neutral_analysis": ["中立分析查詢1", ...],
            "factual_timeline": ["事實時序查詢1", ...]
        }
    """
    # 檢查快取（優先使用快取，避免 API 調用）
    if use_cache:
        cached = get_cached_query_expansion(query)
        if cached and cached.get("balanced_queries"):
            balanced_queries = cached.get("balanced_queries")
            # 確保返回的是字典類型
            if isinstance(balanced_queries, dict):
                logger.info(f"使用快取的平衡查詢: {query[:50]}")
                return balanced_queries
            else:
                logger.warning(f"快取中的 balanced_queries 不是字典類型: {type(balanced_queries).__name__}，重新生成")
    
    try:
        llm = ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, google_api_key=api_key, temperature=0.4)
        
        prompt = f"""
針對爭議議題「{query}」，請生成三組「對抗性」搜尋查詢，使用情感導向詞彙以確保捕捉不同立場：

1. 【正方/支持觀點查詢】（3-4個）：
   - 使用正面詞彙：成功、有效、優勢、證實、支持、贊成、好處、效益
   - 範例：「核電」→ "核電 成功案例"、"核能 有效降低碳排放"、"核電優勢 證據"
   - 範例：「主動投資」→ "主動投資 超越大盤 成功"、"技術分析 有效性 證實"

2. 【反方/反對觀點查詢】（3-4個）：
   - 使用負面詞彙：失敗、無效、風險、質疑、反對、批評、缺點、危害
   - 範例：「核電」→ "核電 事故風險"、"核能 安全疑慮"、"核電缺點 證據"
   - 範例：「主動投資」→ "主動投資 失敗率"、"擇時交易 無效 研究"、"技術分析 被質疑"

3. 【中立/學術分析查詢】（3-4個）：
   - 使用客觀詞彙：研究、數據、分析、比較、實證、學術、統計
   - 範例：「核電」→ "核電研究 學術論文"、"核能 成本效益分析"
   - 範例：「主動投資」→ "主動vs被動投資 實證研究"、"投資策略 績效比較"

**重要**：必須使用強烈的情感詞彙（成功/失敗、有效/無效、優勢/風險），避免中性描述。

請以 JSON 格式輸出：
{{
    "pro": ["查詢1", "查詢2", "查詢3", "查詢4"],
    "con": ["查詢1", "查詢2", "查詢3", "查詢4"],
    "neutral": ["查詢1", "查詢2", "查詢3", "查詢4"]
}}
        """
        resp = _extract_text_from_llm_content(llm.invoke(prompt).content)
        
        # 確保 resp 為字串
        if not isinstance(resp, str):
            resp = str(resp)
        
        # 嘗試解析 JSON
        json_match = re.search(r'\{.*\}', resp, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                # 確保 result 是字典類型
                if isinstance(result, dict):
                    return {
                        "pro_arguments": result.get("pro", [f"{query} 支持 優點", f"{query} 贊成"]),
                        "con_arguments": result.get("con", [f"{query} 反對 缺點", f"{query} 批評"]),
                        "neutral_analysis": result.get("neutral", [f"{query} 研究", f"{query} 數據分析"]),
                        "factual_timeline": [f"{query} 時間軸", f"{query} 發展歷程"]
                    }
                else:
                    logger.warning(f"JSON 解析結果不是字典類型: {type(result).__name__}")
            except json.JSONDecodeError as json_error:
                logger.warning(f"JSON 解析失敗: {str(json_error)}")
            except Exception as parse_error:
                logger.warning(f"解析結果時發生錯誤: {str(parse_error)}")
        else:
            logger.warning(f"未找到 JSON 格式，回應內容: {resp[:200]}")
    except ChatGoogleGenerativeAIError as gemini_error:
        error_str = str(gemini_error)
        # 檢查是否為配額耗盡錯誤
        if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str or "quota" in error_str.lower():
            logger.warning(f"Gemini API 配額已耗盡，使用降級策略: {error_str[:100]}")
        else:
            logger.warning(f"平衡查詢生成失敗: {error_str[:100]}")
    except Exception as e:
        error_str = str(e)
        logger.warning(f"平衡查詢生成失敗: {error_str[:100]}")
    
    # 降級策略（使用共用函數）
    # 確保總是返回字典類型
    fallback = {
        "pro_arguments": [f"{query} 支持 優點", f"{query} 贊成 好處"],
        "con_arguments": [f"{query} 反對 缺點", f"{query} 批評 風險"],
        "neutral_analysis": [f"{query} 研究", f"{query} 數據分析", f"{query} 學術"],
        "factual_timeline": [f"{query} 時間軸", f"{query} 發展歷程"]
    }
    
    # 確保 fallback 是字典類型（防禦性編程）
    if not isinstance(fallback, dict):
        logger.error(f"fallback 不是字典類型: {type(fallback).__name__}，強制轉換為字典")
        fallback = {
            "pro_arguments": [f"{query} 支持 優點", f"{query} 贊成 好處"],
            "con_arguments": [f"{query} 反對 缺點", f"{query} 批評 風險"],
            "neutral_analysis": [f"{query} 研究", f"{query} 數據分析", f"{query} 學術"],
            "factual_timeline": [f"{query} 時間軸", f"{query} 發展歷程"]
        }
    
    # 即使失敗也快取降級結果
    if use_cache:
        try:
            cached = get_cached_query_expansion(query)
            if cached:
                cache_query_expansion(query, cached.get("expanded_queries", []), fallback)
            else:
                cache_query_expansion(query, [], fallback)
        except Exception as cache_error:
            logger.warning(f"快取操作失敗: {str(cache_error)}")
    
    # 最終確認返回的是字典類型
    if not isinstance(fallback, dict):
        logger.error(f"fallback 仍然不是字典類型: {type(fallback).__name__}，強制創建新字典")
        fallback = {
            "pro_arguments": [f"{query} 支持 優點", f"{query} 贊成 好處"],
            "con_arguments": [f"{query} 反對 缺點", f"{query} 批評 風險"],
            "neutral_analysis": [f"{query} 研究", f"{query} 數據分析", f"{query} 學術"],
            "factual_timeline": [f"{query} 時間軸", f"{query} 發展歷程"]
        }
    
    return fallback


def translate_queries_to_english(
    query: str,
    expanded_queries: List[Any],
    api_key: Optional[str] = None,
    max_english_queries: int = 3,
) -> List[str]:
    """
    將主查詢與擴展查詢翻譯／改寫為英文搜尋關鍵字，用於歐洲/美洲非中文檢索。
    
    理論基礎：跨語言資訊檢索 (CLIR) 中，目標區域為歐美時以英文查詢可提升當地媒體覆蓋。
    參考「搜尋視角改進建議」建議一。
    
    Args:
        query: 使用者主查詢（中文）
        expanded_queries: 擴展查詢列表，可為 List[str] 或 List[Dict]（含 'query' 鍵）
        api_key: Google Gemini API Key；若為 None 則不呼叫 LLM，返回 []
        max_english_queries: 最多產出幾條英文查詢（避免任務數過多與 API 負荷）
    
    Returns:
        英文查詢字串列表，長度不超過 max_english_queries；失敗或無 api_key 時返回 []
    """
    if not api_key or max_english_queries <= 0:
        return []
    # 擷取查詢文字（前 5 條以控制 prompt 長度）
    query_texts: List[str] = []
    if isinstance(expanded_queries, list):
        for i, q in enumerate(expanded_queries[:5]):
            if isinstance(q, str):
                query_texts.append(q.strip())
            elif isinstance(q, dict) and q.get("query"):
                query_texts.append(str(q["query"]).strip())
    if not query_texts:
        query_texts = [query.strip()] if query else []
    if not query_texts:
        return []

    try:
        llm = ChatGoogleGenerativeAI(
            model=DEFAULT_GEMINI_MODEL, google_api_key=api_key, temperature=0.2
        )
        prompt = f"""You are a search query translator for news and current affairs.

Given the following user query and/or related search phrases (in Chinese or mixed), output 2 to {max_english_queries} concise English search queries suitable for finding the same topic in English-language news (e.g. Reuters, BBC, NYT). Keep proper nouns and names in their standard English form (e.g. Taiwan, China, NATO). Output ONLY one query per line, no numbering, no explanation.

User query: {query[:200]}

Related phrases:
{chr(10).join(query_texts[:5])}

English search queries (one per line):"""
        raw = _extract_text_from_llm_content(llm.invoke(prompt).content)
        if not raw or not isinstance(raw, str):
            return []
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        # 過濾掉明顯非英文或過長的行
        out = []
        for ln in lines[: max_english_queries + 2]:
            if len(ln) > 350:
                continue
            # 簡單啟發：至少含一個英文字母
            if any(c.isalpha() and ord(c) < 128 for c in ln):
                out.append(ln)
            if len(out) >= max_english_queries:
                break
        logger.info(f"英文查詢翻譯完成: 主查詢={query[:40]}..., 產出 {len(out)} 條: {out[:3]}")
        return out[:max_english_queries]
    except Exception as e:
        logger.warning(f"英文查詢翻譯失敗: {str(e)[:200]}")
        return []


def translate_queries_to_japanese_korean(
    query: str,
    expanded_queries: List[Any],
    api_key: Optional[str] = None,
    max_per_language: int = 2,
) -> Dict[str, List[str]]:
    """
    將主查詢與擴展查詢翻譯為日文與韓文搜尋關鍵字，用於亞洲視角之日韓在地媒體檢索。
    
    理論基礎：跨語言資訊檢索 (CLIR)；日本/韓國媒體以當地語言檢索可提升覆蓋。
    
    Args:
        query: 使用者主查詢（中文）
        expanded_queries: 擴展查詢列表，可為 List[str] 或 List[Dict]（含 'query' 鍵）
        api_key: Google Gemini API Key；若為 None 則不呼叫 LLM，返回 {"ja": [], "ko": []}
        max_per_language: 日文與韓文各自最多產出幾條查詢（建議 2，避免任務過多）
    
    Returns:
        Dict 含 "ja" 與 "ko" 鍵，值為該語言查詢字串列表；失敗時返回 {"ja": [], "ko": []}
    """
    out: Dict[str, List[str]] = {"ja": [], "ko": []}
    if not api_key or max_per_language <= 0:
        return out
    query_texts: List[str] = []
    if isinstance(expanded_queries, list):
        for i, q in enumerate(expanded_queries[:4]):
            if isinstance(q, str):
                query_texts.append(q.strip())
            elif isinstance(q, dict) and q.get("query"):
                query_texts.append(str(q["query"]).strip())
    if not query_texts:
        query_texts = [query.strip()] if query else []
    if not query_texts:
        return out

    try:
        llm = ChatGoogleGenerativeAI(
            model=DEFAULT_GEMINI_MODEL, google_api_key=api_key, temperature=0.2
        )
        prompt = f"""You are a search query translator for news. Output search queries in Japanese and Korean for the same topic.

User query (Chinese or mixed): {query[:200]}

Related phrases:
{chr(10).join(query_texts[:4])}

Output EXACTLY in this format (use natural Japanese and Korean for news search; keep proper nouns in local form):
JAPANESE:
<one query per line, up to {max_per_language} lines>
KOREAN:
<one query per line, up to {max_per_language} lines>

No numbering, no explanation. Write only the queries."""
        raw = _extract_text_from_llm_content(llm.invoke(prompt).content)
        if not raw or not isinstance(raw, str):
            return out
        block = raw.strip()
        ja_lines: List[str] = []
        ko_lines: List[str] = []
        current = "ja"
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("KOREAN"):
                current = "ko"
                rest = line[6:].strip()
                if rest:
                    ko_lines.append(rest[:350])
                continue
            if line.upper().startswith("JAPANESE"):
                current = "ja"
                rest = line[8:].strip()
                if rest:
                    ja_lines.append(rest[:350])
                continue
            if current == "ja" and len(ja_lines) < max_per_language:
                ja_lines.append(line[:350])
            elif current == "ko" and len(ko_lines) < max_per_language:
                ko_lines.append(line[:350])
        out["ja"] = ja_lines[:max_per_language]
        out["ko"] = ko_lines[:max_per_language]
        logger.info(f"日韓查詢翻譯完成: 主查詢={query[:40]}..., ja={len(out['ja'])}, ko={len(out['ko'])}")
        return out
    except Exception as e:
        logger.warning(f"日韓查詢翻譯失敗: {str(e)[:200]}")
        return out


def analyze_consensus(all_sources: Dict[str, List[Dict]], api_key: Optional[str] = None, query: Optional[str] = None) -> Dict[str, Any]:
    """
    分析不同立場間的共識與分歧（方案 3.3 - 共識分析，LLM 增強版）
    
    使用 LLM 深度分析共同事實和分歧點，提升共識分析的準確性和深度。
    
    Args:
        all_sources: {
            "pro_sources": [...],
            "con_sources": [...],
            "neutral_sources": [...],
            "factual_sources": [...]
        }
        api_key: Google Gemini API Key（可選，用於 LLM 分析）
        query: 查詢關鍵字（可選，用於 LLM 分析）
    
    Returns:
        Dict: {
            "common_facts": List[Dict],  # 各方都認同的事實（LLM 分析結果）
            "divergence_points": List[Dict],  # 分歧點（LLM 分析結果）
            "consensus_score": 0.0-1.0,  # 共識度分數
            "perspective_balance": {...}  # 立場平衡度
        }
    """
    pro_sources = all_sources.get("pro_sources", [])
    con_sources = all_sources.get("con_sources", [])
    neutral_sources = all_sources.get("neutral_sources", [])
    factual_sources = all_sources.get("factual_sources", [])
    all_sources_list = pro_sources + con_sources + neutral_sources
    used_factual_fallback = False
    if not all_sources_list:
        all_sources_list = factual_sources
        used_factual_fallback = True
    
    total_sources = len(all_sources_list)
    category_counts = Counter(source.get("source_category", "OTHER") for source in all_sources_list)
    
    if used_factual_fallback:
        perspective_balance = {
            "mode": "source_category_fallback",
            "source_count": total_sources,
            "category_distribution": dict(category_counts),
        }
        # 無人工/LLM 立場分桶時，用來源類型多樣性作為保守的平衡近似值。
        diversity_score = min(1.0, len([c for c in category_counts.values() if c > 0]) / 4)
        volume_score = min(1.0, total_sources / 8)
        pro_con_balance = round(diversity_score * 0.7 + volume_score * 0.3, 2) if total_sources else 0.0
    else:
        perspective_balance = {
            "mode": "stance_buckets",
            "pro_ratio": len(pro_sources) / total_sources if total_sources > 0 else 0,
            "con_ratio": len(con_sources) / total_sources if total_sources > 0 else 0,
            "neutral_ratio": len(neutral_sources) / total_sources if total_sources > 0 else 0,
            "category_distribution": dict(category_counts),
        }
        pro_con_balance = min(perspective_balance["pro_ratio"], perspective_balance["con_ratio"]) / max(
            perspective_balance["pro_ratio"], perspective_balance["con_ratio"]
        ) if max(perspective_balance["pro_ratio"], perspective_balance["con_ratio"]) > 0 else 0
    
    # 使用 LLM 分析共同事實和分歧點（如果提供了 API Key 且有足夠來源）
    common_facts = []
    divergence_points = []
    
    if api_key and all_sources_list and len(all_sources_list) >= 3:
        try:
            # 準備來源摘要（限制長度以節省 token）
            sources_summary = []
            for i, source in enumerate(all_sources_list[:20]):  # 最多分析 20 個來源
                source_id = i + 1
                title = source.get('title', '')[:100]
                content = source.get('content', '')[:300]
                category = source.get('source_category', 'OTHER')
                sources_summary.append(f"Source {source_id} ({category}): {title}\n內容摘要: {content[:200]}...")
            
            sources_text = "\n\n".join(sources_summary)
            
            # 使用 LLM 分析共同事實和分歧點
            llm = ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, google_api_key=api_key, temperature=0.3)
            
            analysis_prompt = f"""
針對議題「{query or '相關議題'}」，以下是一系列不同立場的來源報導：

{sources_text}

請分析這些來源，識別：
1. **共同事實**：所有或大多數來源都認同的事實（至少列出 2-3 項）
2. **分歧點**：不同立場間的主要分歧（至少列出 2-3 項）

請以 JSON 格式輸出：
{{
    "common_facts": [
        {{
            "fact": "事實描述",
            "supporting_sources": ["Source 1", "Source 3"],
            "confidence": "高/中/低"
        }}
    ],
    "divergence_points": [
        {{
            "point": "分歧點描述",
            "pro_position": "支持方的立場",
            "con_position": "反對方的立場",
            "pro_sources": ["Source 2"],
            "con_sources": ["Source 5"]
        }}
    ]
}}
"""
            
            response = _extract_text_from_llm_content(llm.invoke(analysis_prompt).content)
            
            # 嘗試解析 JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group())
                common_facts = analysis_result.get("common_facts", [])
                divergence_points = analysis_result.get("divergence_points", [])
                logger.info(f"LLM 共識分析完成：識別 {len(common_facts)} 項共同事實，{len(divergence_points)} 個分歧點")
            else:
                logger.warning("LLM 共識分析：無法解析 JSON 格式，使用空列表")
        
        except Exception as e:
            logger.warning(f"LLM 共識分析失敗: {str(e)}，使用空列表作為降級策略")
            # 降級策略：如果 LLM 分析失敗，返回空列表
    
    return {
        "common_facts": common_facts,
        "divergence_points": divergence_points,
        "consensus_score": pro_con_balance,
        "perspective_balance": perspective_balance,
        "analysis_status": "llm_analyzed" if common_facts or divergence_points else "fallback_only",
    } 

# ==========================================
# 快取機制
# ==========================================
def init_cache_db():
    """初始化快取資料庫"""
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            cache_key TEXT PRIMARY KEY,
            query TEXT,
            results TEXT,
            timestamp REAL,
            expiry_time REAL
        )
    """)
    # 新增查詢擴展快取表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_expansion_cache (
            cache_key TEXT PRIMARY KEY,
            query TEXT,
            expanded_queries TEXT,
            balanced_queries TEXT,
            timestamp REAL,
            expiry_time REAL
        )
    """)
    conn.commit()
    return conn

def get_cache_key(query: str, params: Dict) -> str:
    """生成快取鍵"""
    cache_data = f"{query}_{json.dumps(params, sort_keys=True)}"
    return hashlib.md5(cache_data.encode()).hexdigest()

def get_cached_results(query: str, params: Dict) -> Optional[List[Dict]]:
    """從快取獲取搜尋結果"""
    try:
        conn = init_cache_db()
        cache_key = get_cache_key(query, params)
        cursor = conn.execute(
            "SELECT results, expiry_time FROM search_cache WHERE cache_key = ?",
            (cache_key,)
        )
        row = cursor.fetchone()
        if row:
            results_json, expiry_time = row
            if time.time() < expiry_time:
                # 快取有效
                results = json.loads(results_json)
                conn.close()
                logger.info(f"快取命中: {query[:50]}")
                return results
            else:
                # 快取過期，刪除
                conn.execute("DELETE FROM search_cache WHERE cache_key = ?", (cache_key,))
                conn.commit()
        conn.close()
        return None
    except Exception as e:
        logger.warning(f"快取讀取失敗: {str(e)}")
        return None

def cache_results(query: str, params: Dict, results: List[Dict]):
    """將搜尋結果存入快取"""
    try:
        conn = init_cache_db()
        cache_key = get_cache_key(query, params)
        results_json = json.dumps(results, ensure_ascii=False)
        timestamp = time.time()
        expiry_time = timestamp + (CACHE_EXPIRY_HOURS * 3600)
        
        conn.execute(
            """INSERT OR REPLACE INTO search_cache 
               (cache_key, query, results, timestamp, expiry_time) 
               VALUES (?, ?, ?, ?, ?)""",
            (cache_key, query[:200], results_json, timestamp, expiry_time)
        )
        conn.commit()
        conn.close()
        logger.info(f"結果已快取: {query[:50]}")
    except Exception as e:
        logger.warning(f"快取寫入失敗: {str(e)}")

def clear_cache():
    """清除所有過期快取"""
    try:
        conn = init_cache_db()
        current_time = time.time()
        cursor = conn.execute("DELETE FROM search_cache WHERE expiry_time < ?", (current_time,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        logger.info(f"已清除 {deleted} 條過期快取")
        return deleted
    except Exception as e:
        logger.warning(f"清除快取失敗: {str(e)}")
        return 0

# ==========================================
# AI 輸出格式驗證
# ==========================================
def validate_ai_output_format(raw_text: str, mode: str = "FUSION") -> Dict[str, Any]:
    """
    驗證 AI 輸出是否包含必需的章節和格式
    
    Returns:
        Dict with validation results and missing sections
    """
    if not raw_text:
        return {
            'is_valid': False,
            'missing_sections': [],
            'has_timeline': False,
            'has_report': False,
            'score': 0.0
        }
    
    if mode == "FUSION":
        required_sections = REQUIRED_SECTIONS_FUSION
    elif mode == "DEEP_SCENARIO":
        required_sections = REQUIRED_SECTIONS_SCENARIO
    elif mode == "TEXT_ANALYSIS":
        required_sections = REQUIRED_SECTIONS_TEXT_ANALYSIS
    else:
        required_sections = []
    
    validation = {
        'is_valid': True,
        'missing_sections': [],
        'has_timeline': False,
        'has_report': False,
        'has_ach_table': False,
        'has_logic_table': False,
        'has_framing_table': False,
        'score': 0.0
    }
    
    # 檢查時間軸
    validation['has_timeline'] = bool('[DATA_TIMELINE]' in raw_text or 'DATA_TIMELINE' in raw_text)
    
    # 檢查報告文本
    validation['has_report'] = bool('[REPORT_TEXT]' in raw_text or 'REPORT_TEXT' in raw_text)
    
    # 檢查必需章節
    for section in required_sections:
        # 使用多種可能的形式檢查
        section_variants = [
            section,
            section.replace(' ', ''),
            section.split('(')[0].strip(),
            section.split('（')[0].strip()
        ]
        
        found = any(variant in raw_text for variant in section_variants)
        if not found:
            validation['missing_sections'].append(section)
            validation['is_valid'] = False
    
    # 檢查表格格式
    validation['has_ach_table'] = bool('| 假設' in raw_text and '| 支持證據' in raw_text)
    validation['has_logic_table'] = bool('| 謬誤類型' in raw_text or '邏輯謬誤偵測表' in raw_text)
    validation['has_framing_table'] = bool(
        ('| 媒體陣營' in raw_text or '| 陣營' in raw_text or '| 框架元素' in raw_text)
        and '| 問題定義' in raw_text
    )
    
    # 計算分數（0-100）
    base_score = 50 if validation['has_timeline'] else 0
    base_score += 30 if validation['has_report'] else 0
    if required_sections:
        base_score += 10 * (len(required_sections) - len(validation['missing_sections'])) / len(required_sections)
    base_score += 3 if validation['has_ach_table'] else 0
    base_score += 3 if validation['has_logic_table'] else 0
    base_score += 4 if validation['has_framing_table'] else 0
    
    validation['score'] = min(100.0, base_score)
    
    return validation

# ==========================================
# 證據強度評估增強
# ==========================================
def analyze_language_style(content: str, title: str) -> Dict[str, Any]:
    """
    分析語言風格異常與情感操控（改進項目：語言風格分析模組）
    
    參考最新研究，檢測：
    - 誇張性標題（clickbait）
    - 情感操控性語言
    - 標題與內容不匹配
    - 語法錯誤與格式異常
    
    Args:
        content: 內容文本
        title: 標題
    
    Returns:
        Dict: {
            "clickbait_score": float,  # 標題誘導性分數 0-1
            "emotional_manipulation": float,  # 情感操控強度 0-1
            "title_content_mismatch": float,  # 標題與內容不匹配度 0-1
            "sensationalism_score": float,  # 聳動性分數 0-1
            "grammar_errors": int,  # 語法錯誤數（預留，未來可整合語法檢查）
            "flags": List[str],  # 警示標記列表
            "emotional_indicators": Dict  # 情感指標詳情
        }
    """
    if not title:
        return {
            "clickbait_score": 0.0,
            "emotional_manipulation": 0.0,
            "title_content_mismatch": 0.0,
            "sensationalism_score": 0.0,
            "grammar_errors": 0,
            "flags": [],
            "emotional_indicators": {}
        }
    
    flags = []
    
    # === 1. 檢測 Clickbait 標題模式 ===
    clickbait_patterns = [
        "你不會相信", "驚人真相", "被隱瞞", "震驚", "驚呆了",
        "99%的人都不知道", "最後竟然", "沒想到", "竟然這樣",
        "秘密曝光", "真相大白", "內幕", "曝光", "震撼",
        "嚇一跳", "不敢相信", "原來是", "終於知道"
    ]
    clickbait_matches = sum(1 for pattern in clickbait_patterns if pattern in title)
    clickbait_score = min(1.0, clickbait_matches * CLICKBAIT_MULTIPLIER)
    
    if clickbait_score > CLICKBAIT_THRESHOLD:
        flags.append("⚠️ 標題具有誘導性（Clickbait）")
    
    # === 2. 檢測過度誇張語言 ===
    exaggeration_indicators = [
        "極度", "非常", "超級", "絕對", "完全", "永遠", "從來沒有",
        "史上最", "前所未有", "史無前例", "絕無僅有", "空前絕後",
        "無與倫比", "登峰造極", "徹底", "全面", "徹底崩潰"
    ]
    combined_text = (title + " " + content[:300]).lower()
    exaggeration_count = sum(1 for indicator in exaggeration_indicators if indicator in combined_text)
    sensationalism_score = min(1.0, exaggeration_count * SENSATIONALISM_MULTIPLIER)
    
    if sensationalism_score > SENSATIONALISM_THRESHOLD:
        flags.append("⚠️ 使用過度誇張語言")
    
    # === 3. 檢測標題格式異常 ===
    alpha_chars = [c for c in title if c.isalpha()]
    caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) if alpha_chars else 0
    exclamation_count = title.count('!')
    question_mark_count = title.count('?')
    
    format_anomaly = False
    if caps_ratio > CAPS_RATIO_THRESHOLD:
        format_anomaly = True
        flags.append("⚠️ 標題包含過多大寫字母")
    if exclamation_count > 2:
        format_anomaly = True
        flags.append("⚠️ 標題使用過多感嘆號")
    if question_mark_count > 2:
        format_anomaly = True
    
    # === 4. 標題與內容語義相似度（不匹配度檢測）===
    title_words = set(re.findall(r'\w+', title.lower()))
    content_sample = content[:500].lower()  # 取前500字元
    content_words = set(re.findall(r'\w+', content_sample))
    
    # 過濾停用詞（簡化版）
    stopwords = {'的', '是', '在', '了', '和', '與', '及', '或', '但', '而', '也', '都', '就', '還', '更', '又', '要', '為', '會', '可以', '能', '可', '被', '將', '已', '對', '從', '由', '以', '等', '個', '一', '二', '三', '這', '那', '他', '她', '它', '我們', '你們', '他們', '這', '那'}
    title_keywords = title_words - stopwords
    content_keywords = content_words - stopwords
    
    if title_keywords:
        overlap_ratio = len(title_keywords & content_keywords) / len(title_keywords)
        title_content_mismatch = 1.0 - overlap_ratio
    else:
        title_content_mismatch = TITLE_CONTENT_UNKNOWN_SCORE  # 無法判斷
    
    if title_content_mismatch > TITLE_CONTENT_MISMATCH_THRESHOLD:
        flags.append("⚠️ 標題與內容不匹配（標題黨）")
    
    # === 5. 情感操控檢測 ===
    # 恐懼詞彙
    fear_words = ["恐懼", "害怕", "危險", "威脅", "災難", "毀滅", "崩潰", 
                  "危機", "失控", "恐怖", "驚嚇", "恐慌", "滅亡"]
    # 憤怒詞彙
    anger_words = ["憤怒", "不滿", "抗議", "譴責", "暴怒", "憤慨", "反感",
                   "痛批", "怒斥", "抨擊", "譏諷", "嘲諷"]
    # 興奮/煽動詞彙
    excitement_words = ["興奮", "激動", "震撼", "驚人", "爆炸性", "轟動",
                        "驚爆", "勁爆", "爆裂", "炸裂", "驚天"]
    
    content_lower = content.lower()
    emotional_indicators = {
        "fear": sum(1 for w in fear_words if w in content_lower),
        "anger": sum(1 for w in anger_words if w in content_lower),
        "excitement": sum(1 for w in excitement_words if w in content_lower)
    }
    
    # 計算情感操控強度（多種情感同時出現時更強）
    total_emotional_words = sum(emotional_indicators.values())
    emotional_manipulation = min(1.0, total_emotional_words / EMOTIONAL_WORDS_DIVISOR)
    
    # 如果標題也包含情感詞彙，加權
    title_lower = title.lower()
    title_emotional = any(w in title_lower for w in fear_words + anger_words + excitement_words)
    if title_emotional:
        emotional_manipulation = min(1.0, emotional_manipulation + EMOTIONAL_TITLE_BONUS)
    
    if emotional_manipulation > EMOTIONAL_MANIPULATION_THRESHOLD:
        dominant_emotion = max(emotional_indicators.items(), key=lambda x: x[1])[0]
        emotion_names = {"fear": "恐懼", "anger": "憤怒", "excitement": "興奮"}
        flags.append(f"⚠️ 情感操控性強（主要為{emotion_names.get(dominant_emotion, '情緒化')}語言）")
    
    return {
        "clickbait_score": clickbait_score,
        "emotional_manipulation": emotional_manipulation,
        "title_content_mismatch": title_content_mismatch,
        "sensationalism_score": sensationalism_score,
        "grammar_errors": 0,  # 預留：未來可整合語法檢查 API
        "flags": flags,
        "emotional_indicators": emotional_indicators,
        "format_anomaly": format_anomaly
    }

def assess_website_quality(url: str, content: str) -> Dict[str, Any]:
    """
    評估網頁結構品質（改進項目：topic-agnostic 方法）
    
    檢測：
    - 域名可疑性（拼寫錯誤、模仿知名媒體）
    - 域名模式異常（非標準後綴）
    
    Args:
        url: 來源 URL
        content: 內容文本（預留，未來可用於檢測廣告密度等）
    
    Returns:
        Dict: {
            "domain_suspicious": bool,
            "domain_issues": List[str],
            "quality_score": float,  # 0-1
            "flags": List[str]
        }
    """
    domain = get_domain_name(url)
    flags = []
    issues = []
    
    if not domain:
        return {
            "domain_suspicious": True,
            "domain_issues": ["無法解析域名"],
            "quality_score": 0.3,
            "flags": ["⚠️ 域名無法解析"]
        }
    
    domain_lower = domain.lower()
    
    # === 1. 檢查域名可疑模式 ===
    suspicious_patterns = [
        (r'\.co$', "使用非標準後綴 .co（可能是仿冒）"),
        (r'\.info$', "使用 .info 後綴（可信度較低）"),
        (r'\.biz$', "使用 .biz 後綴（可信度較低）"),
        (r'[0-9]{3,}', "域名包含多個數字（可能是仿冒）"),
        (r'news-.*-news', "域名模式可疑（news-xxx-news）"),
        (r'-news-', "域名包含多個 news 關鍵字"),
    ]
    
    for pattern, description in suspicious_patterns:
        if re.search(pattern, domain_lower):
            issues.append(description)
            flags.append(f"⚠️ {description}")
    
    # === 2. 檢查是否模仿知名媒體 ===
    known_media_domains = {
        "bbc.com": ["bbcc", "bbcnews", "bbc-news"],
        "cnn.com": ["cnnnews", "cnn-news", "cnninfo"],
        "nytimes.com": ["nytimesnews", "ny-times"],
        "udn.com": ["udnnews", "udn-news"],
        "ltn.com.tw": ["ltnnews", "ltn-news"],
        "cna.com.tw": ["cnanews", "cna-news"],
    }
    
    for legit_domain, variations in known_media_domains.items():
        legit_base = legit_domain.replace('.com', '').replace('.tw', '').replace('.', '')
        if legit_base in domain_lower and domain_lower != legit_domain.lower():
            # 檢查是否為變體
            is_variant = any(var in domain_lower for var in variations)
            if not is_variant and legit_domain.lower() not in domain_lower:
                issues.append(f"可能模仿知名媒體：{legit_domain}")
                flags.append(f"⚠️ 域名可能為仿冒：疑似模仿 {legit_domain}")
    
    # === 3. 評估品質分數 ===
    quality_score = 1.0
    if issues:
        # 每個問題扣分，最低分數
        quality_score = max(WEBSITE_QUALITY_MIN_SCORE, 1.0 - len(issues) * WEBSITE_QUALITY_PENALTY_PER_ISSUE)
    
    return {
        "domain_suspicious": len(issues) > 0,
        "domain_issues": issues,
        "quality_score": quality_score,
        "flags": flags
    }

def assess_content_quality(content: str, title: str) -> Dict[str, Any]:
    """
    評估內容品質（內容導向篩選的核心）
    
    依據文章內文特徵評分，不依賴來源黑白名單：
    - 長度、結構、事實密度、引用、attribution、標題內容一致
    權重已上調（2026.02）：使扎實文本更容易獲得「強」「中強」證據等級。
    """
    quality_score = 0.0
    indicators = {}
    
    if not content:
        return {'score': 0.0, 'indicators': indicators}
    
    # 長度評估（扎實報導通常較長；上調給分以反映文本內容價值）
    content_length = len(content)
    if content_length > CONTENT_QUALITY_LONG:
        quality_score += 0.28
        indicators['length'] = '長'
    elif content_length > CONTENT_QUALITY_MEDIUM:
        quality_score += 0.22
        indicators['length'] = '中'
    elif content_length > CONTENT_QUALITY_SHORT:
        quality_score += 0.16
        indicators['length'] = '短'
    else:
        quality_score += 0.06
        indicators['length'] = '極短'
    
    # 完整性評估（結構化資訊：日期、數字、引述）
    has_dates = bool(re.search(r'\d{4}[-年]\d{1,2}[-月]\d{1,2}', content))
    has_numbers = bool(re.search(r'\d+', content))
    has_quotes = bool('"' in content or '"' in content or "'" in content)
    
    if has_dates:
        quality_score += 0.14
        indicators['has_dates'] = True
    if has_numbers:
        quality_score += 0.10
        indicators['has_numbers'] = True
    if has_quotes:
        quality_score += 0.12
        indicators['has_quotes'] = True
    
    # 引用與 attribution（記者、據、指出、來源、專家；上調以肯定有依據的文本）
    citation_patterns = ['來源', '引用', '據', '指出', '表示', 'Source', 'reference', '記者', '報導', '專家', '認為', '分析']
    citation_count = sum(1 for pattern in citation_patterns if pattern in content)
    if citation_count > 3:
        quality_score += 0.22
        indicators['citations'] = '多'
    elif citation_count > 1:
        quality_score += 0.16
        indicators['citations'] = '有'
    elif citation_count > 0:
        quality_score += 0.10
        indicators['citations'] = '少'
    
    # 標題與內容相關性（標題黨扣分已在 style_score）
    if title:
        title_words = set(title.lower().split()[:5])
        content_words = set(content.lower().split()[:50])
        overlap = len(title_words & content_words) / len(title_words) if title_words else 0
        if overlap > CONTENT_OVERLAP_HIGH:
            quality_score += 0.14
            indicators['relevance'] = '高'
        elif overlap > CONTENT_OVERLAP_MEDIUM:
            quality_score += 0.10
            indicators['relevance'] = '中'
        else:
            indicators['relevance'] = '低'
    
    return {
        'score': min(1.0, quality_score),
        'indicators': indicators
    }

def calculate_academic_evidence_level(url: str, source_category: str, content: str, title: str, all_sources: Optional[List[Dict]] = None) -> Tuple[str, float, Dict[str, Any]]:
    """
    學術級證據強度分級系統（方案 2.2，依 GRADE 多維度評分）
    
    參考 GRADE 標準，實作多維度評分：
    - Tier 1: 官方原始文檔、同儕評審論文、權威機構報告
    - Tier 2: 專業媒體深度調查、獨立媒體機構報告、國際權威媒體
    - Tier 3: 一般媒體報導、專家評論、組織聲明
    - Tier 4: 社群媒體、個人部落格、內容農場
    
    綜合公式權重：內容品質 46%、來源 10%、公信力 10%、語言風格 17%、交叉驗證 12%、網站品質 5%；
    扎實文本（content_score≥0.55 且 style_score≥0.55）享有保底加分，使「強」「中強」更易出現。
    
    Args:
        url: 來源 URL
        source_category: 來源類型
        content: 內容文本
        title: 標題
        all_sources: 所有來源列表（用於交叉驗證）
    
    Returns:
        Tuple[str, float, Dict]: (強度等級, 數值分數 0-1, 詳細評分明細)
    """
    details = {
        "source_tier": 3,
        "source_score": 0.0,
        "content_score": 0.0,
        "cross_validation": 0.0,
        "conflict_of_interest": 0.0,
        "temporal_score": 0.0
    }
    
    # === 層次一：證據類型分級 (GRADE 標準) ===
    source_tier = 3
    source_score = 0.0
    
    # Tier 1 (最高證據強度): 官方原始文檔、學術論文、權威機構報告
    if source_category == "OFFICIAL":
        official_patterns = ["gov.tw", "mnd.gov.tw", "mac.gov.tw", "tfc-taiwan.org.tw", "judicial.gov.tw"]
        if any(pattern in url.lower() for pattern in official_patterns):
            source_tier = 1
            source_score = 0.9
            details["source_tier"] = 1
        else:
            source_tier = 2
            source_score = 0.7
            details["source_tier"] = 2
    
    # Tier 2 (高證據強度): 專業媒體、獨立媒體、國際權威媒體
    elif source_category == "INDIE":
        source_tier = 2
        source_score = 0.75
        details["source_tier"] = 2
    elif source_category == "INTL":
        intl_authorities = ["bbc.com", "reuters.com", "apnews.com", "afp.com", "bloomberg.com", "wsj.com", "nytimes.com",
                           "brookings.edu", "csis.org", "cfr.org", "chathamhouse.org", "foreignaffairs.com", "foreignpolicy.com"]
        if any(auth in url.lower() for auth in intl_authorities):
            source_tier = 2
            source_score = 0.75
            details["source_tier"] = 2
        else:
            source_tier = 3
            source_score = 0.55
            details["source_tier"] = 3
    
    # Tier 3 (中等證據強度): 一般媒體
    elif source_category in ["BLUE", "GREEN"]:
        source_tier = 3
        source_score = 0.5
        details["source_tier"] = 3
    elif source_category == "NEUTRAL":
        source_tier = 3
        source_score = 0.55  # 中立商業媒體，略高於藍綠（無明顯立場偏倚）
        details["source_tier"] = 3
    
    # Tier 4 (低證據強度): 內容農場、社群媒體
    elif source_category in ["FARM", "SOCIAL"]:
        source_tier = 4
        source_score = 0.2
        details["source_tier"] = 4
    elif source_category == "CHINA":
        source_tier = 3
        source_score = 0.4
        details["source_tier"] = 3
    elif source_category == "OTHER":
        source_tier = 3
        source_score = 0.45
        details["source_tier"] = 3
    
    details["source_score"] = source_score
    
    # === 層次二：來源公信力評分（方案 2）===
    domain = get_domain_name(url)
    credibility_score, tier = _reputation_manager.calculate_credibility_score(url, domain)
    details["credibility_score"] = credibility_score
    details["credibility_tier"] = tier
    
    # === 層次三：內容品質評估 (CERQual) ===
    quality = assess_content_quality(content, title)
    content_score = quality['score']
    details["content_score"] = content_score
    
    # === 層次四：交叉驗證機制 ===
    cross_validation_score = 0.0
    if all_sources and len(all_sources) > 1:
        similar_title_count = 0
        title_lower = title.lower() if title else ""
        
        for other_source in all_sources:
            if other_source.get('url') == url:
                continue
            other_title = other_source.get('title', '').lower()
            if title_lower and other_title:
                common_words = set(title_lower.split()) & set(other_title.split())
                if len(common_words) >= 3:
                    similar_title_count += 1
        
        if similar_title_count > 0:
            consensus_ratio = min(1.0, similar_title_count / max(1, len(all_sources) - 1))
            if consensus_ratio > CROSS_VALIDATION_HIGH_RATIO:
                cross_validation_score = 0.2
            elif consensus_ratio > CROSS_VALIDATION_MEDIUM_RATIO:
                cross_validation_score = 0.1
    
    details["cross_validation"] = cross_validation_score
    
    # === 層次四：利益衝突檢測 ===
    conflict_score = 0.0
    conflict_patterns = ["贊助", "廣告", "業配", "合作", "投資", "股東"]
    if any(pattern in content for pattern in conflict_patterns):
        conflict_score = -0.15
    details["conflict_of_interest"] = conflict_score
    
    # === 層次五：語言風格分析（改進項目：新增）===
    language_style = analyze_language_style(content, title)
    # 計算風格分數（負向指標，需要轉換為正向分數）
    style_penalty = max(
        language_style['clickbait_score'] * 0.4,
        language_style['emotional_manipulation'] * 0.3,
        language_style['title_content_mismatch'] * 0.3,
        language_style['sensationalism_score'] * 0.2
    )
    style_score = 1.0 - style_penalty  # 轉換為正向分數
    details["language_style"] = language_style
    details["style_score"] = style_score
    
    # === 層次六：網頁品質評估（改進項目：新增）===
    website_quality = assess_website_quality(url, content)
    details["website_quality"] = website_quality
    
    # === 綜合評分（GRADE 多維度；內容權重已上調，使扎實文本易達強/中強）===
    # 方法論：來源類型、公信力、內容品質、語言風格、交叉驗證、網站品質、利益衝突
    final_score = (
        source_score * 0.10 +      # 來源類型（GRADE Tier）
        credibility_score * 0.10 + # 來源公信力
        content_score * 0.46 +     # 內容品質（上調：長度、結構、引用、事實密度為核心）
        style_score * 0.17 +       # 語言風格（非聳動、非標題黨）
        cross_validation_score * 0.12 +
        website_quality['quality_score'] * 0.05
    )
    if conflict_score < 0:
        final_score += conflict_score * 0.06  # 利益衝突扣分
    # 內容品質保底：扎實內文可拉高證據等級，不受限於來源評級
    if content_score >= 0.55 and style_score >= 0.55:
        content_bonus = min(0.18, (content_score - 0.45) * 0.35)
        final_score = max(final_score, 0.48 + content_bonus)  # 至少傾向「中強」起跳
    final_score = max(0.0, min(1.0, final_score))
    
    # === 轉換為等級 ===
    if final_score >= EVIDENCE_LEVEL_A_PLUS:
        level = "A+"
        level_cn = "極強"
    elif final_score >= EVIDENCE_LEVEL_A:
        level = "A"
        level_cn = "強"
    elif final_score >= EVIDENCE_LEVEL_B_PLUS:
        level = "B+"
        level_cn = "中強"
    elif final_score >= EVIDENCE_LEVEL_B:
        level = "B"
        level_cn = "中等"
    elif final_score >= EVIDENCE_LEVEL_C:
        level = "C"
        level_cn = "中弱"
    else:
        level = "D"
        level_cn = "弱"
    
    details["final_score"] = final_score
    details["level"] = level
    details["level_cn"] = level_cn
    
    return level_cn, final_score, details

def calculate_enhanced_evidence_level(url: str, source_category: str, content: str, title: str) -> Tuple[str, float]:
    """
    向後相容：計算增強版證據強度（考慮來源類型和內容品質）
    
    Returns:
        Tuple[str, float]: (強度等級, 數值分數 0-1)
    """
    level_cn, score, _ = calculate_academic_evidence_level(url, source_category, content, title)
    return level_cn, score

def search_cofacts(query: str) -> Tuple[str, List[Dict]]:
    """
    查詢 Cofacts 謠言資料庫
    
    Returns:
        Tuple[str, List[Dict]]: (結果文字, 謠言清單)
    """
    url = "https://cofacts-api.g0v.tw/graphql"
    graphql_query = """query ListArticles($text: String!) { ListArticles(filter: {q: $text}, orderBy: [{_score: DESC}], first: 3) { edges { node { text articleReplies(status: NORMAL) { reply { text type } } } } } }"""
    try:
        response = requests.post(url, json={'query': graphql_query, 'variables': {'text': query}}, timeout=TIMEOUT_COFACTS)
        if response.status_code == 200:
            data = response.json()
            articles = data.get('data', {}).get('ListArticles', {}).get('edges', [])
            result_text = ""
            rumor_list = []
            if articles:
                result_text += "【Cofacts 查核資料庫】\n"
                for i, art in enumerate(articles):
                    node = art.get('node', {})
                    rumor_text = node.get('text', '')
                    rumor_short = rumor_text[:50] if rumor_text else ""
                    replies = node.get('articleReplies', [])
                    if replies:
                        reply = replies[0].get('reply', {})
                        r_type = reply.get('type', 'UNKNOWN')
                        r_text = reply.get('text', '')
                        result_text += f"- 謠言: {rumor_short}... (判定: {r_type})\n"
                        rumor_list.append({
                            'text': rumor_text,
                            'type': r_type,
                            'reply': r_text
                        })
            return result_text, rumor_list
    except Exception as e:
        logger.warning(f"Cofacts 查詢失敗: {str(e)}")
        return "", []
    return "", []

# ==========================================
# 方案一：Google Fact Check Tools API 整合
# ==========================================
def extract_claims_from_sources(sources: List[Dict], api_key: str) -> List[Dict[str, Any]]:
    """
    從來源中提取核心聲明（方案 1.1 - 聲明提取）
    
    Args:
        sources: 來源列表
        api_key: Google API Key (用於 LLM)
    
    Returns:
        List[Dict]: [{"text": "聲明文本", "source_id": 1, "url": "...", "claim_type": "factual"}, ...]
    """
    claims = []
    
    try:
        llm = ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, google_api_key=api_key, temperature=0.3)
        
        # 優化：增加批次大小，減少 API 調用次數（從 5 增加到 8）
        # 注意：如果來源很多，可以進一步優化為只提取前 N 個高品質來源的聲明
        batch_size = 8
        for i in range(0, len(sources), batch_size):
            batch = sources[i:i+batch_size]
            batch_texts = []
            
            for j, source in enumerate(batch):
                title = source.get('title', 'No Title')
                content = source.get('content', '')[:500]  # 只取前 500 字元
                batch_texts.append(f"Source {i+j+1}: {title}\n{content[:300]}")
            
            prompt = f"""
            請從以下新聞來源中提取核心聲明（主張），每個來源提取 1-2 個最重要的聲明。
            只提取事實性聲明，不需要提取評論或意見。
            
            {chr(10).join(batch_texts)}
            
            請以 JSON 格式輸出：
            [
                {{"source_id": 1, "claim": "聲明文本", "claim_type": "factual"}},
                ...
            ]
            """
            
            try:
                resp = _extract_text_from_llm_content(llm.invoke(prompt).content)
                # 嘗試解析 JSON
                json_match = re.search(r'\[.*\]', resp, re.DOTALL)
                if json_match:
                    extracted = json.loads(json_match.group())
                    for item in extracted:
                        source_idx = item.get('source_id', 0) - 1
                        if 0 <= source_idx < len(batch):
                            claims.append({
                                "text": item.get('claim', ''),
                                "source_id": i + source_idx + 1,
                                "url": batch[source_idx].get('url', ''),
                                "claim_type": item.get('claim_type', 'factual')
                            })
            except Exception as e:
                logger.warning(f"批次 {i//batch_size+1} 聲明提取失敗: {str(e)}")
                # 降級：直接使用標題作為聲明
                for j, source in enumerate(batch):
                    claims.append({
                        "text": source.get('title', ''),
                        "source_id": i + j + 1,
                        "url": source.get('url', ''),
                        "claim_type": "factual"
                    })
    
    except Exception as e:
        logger.error(f"聲明提取失敗: {str(e)}")
        # 降級策略：使用標題作為聲明
        for i, source in enumerate(sources):
            claims.append({
                "text": source.get('title', ''),
                "source_id": i + 1,
                "url": source.get('url', ''),
                "claim_type": "factual"
            })
    
    return claims

async def verify_single_claim(session: aiohttp.ClientSession, claim_text: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    驗證單一聲明（異步版本）
    
    Args:
        session: aiohttp session
        claim_text: 聲明文本
        api_key: Google API Key
    
    Returns:
        Dict 或 None: {"textualRating": "VERIFIED_FALSE", ...} 或 None（如果失敗）
    """
    try:
        endpoint = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": claim_text[:100],  # Google API 限制長度
            "languageCode": "zh-TW",
            "key": api_key,
            "pageSize": 3,
            "maxAgeDays": 365
        }
        
        async with session.get(endpoint, params=params, timeout=aiohttp.ClientTimeout(total=TIMEOUT_FACT_CHECK)) as response:
            if response.status == 200:
                data = await response.json()
                claims = data.get('claims', [])
                if claims:
                    # 返回第一個最相關的結果
                    return claims[0]
            return None
    except Exception as e:
        logger.warning(f"Fact Check API 驗證失敗: {claim_text[:50]}... 錯誤: {str(e)}")
        return None

async def verify_claims_async(claims: List[Dict[str, Any]], api_key: str) -> Dict[str, List[Dict]]:
    """
    異步批次驗證聲明（方案 1.2 - API 驗證）
    
    Returns:
        Dict: {
            "verified_claims": [...],
            "false_claims": [...],
            "misleading_claims": [...],
            "unverified_claims": [...]
        }
    """
    results = {
        "verified_claims": [],
        "false_claims": [],
        "misleading_claims": [],
        "unverified_claims": []
    }
    
    if not claims or not api_key:
        return results
    
    # 限制並發數，避免 API 配額問題
    semaphore = asyncio.Semaphore(5)
    
    async def verify_with_semaphore(session, claim):
        async with semaphore:
            return await verify_single_claim(session, claim['text'], api_key)
    
    async def run_verifications():
        async with aiohttp.ClientSession() as session:
            tasks = [verify_with_semaphore(session, claim) for claim in claims]
            api_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, (claim, api_result) in enumerate(zip(claims, api_results)):
                if isinstance(api_result, Exception):
                    logger.warning(f"驗證失敗: {claim['text'][:50]}... 錯誤: {str(api_result)}")
                    results["unverified_claims"].append(claim)
                    continue
                
                if not api_result:
                    results["unverified_claims"].append(claim)
                    continue
                
                # 解析評級
                claim_review = api_result.get('claimReview', [])
                if not claim_review:
                    results["unverified_claims"].append(claim)
                    continue
                
                # 取第一個查核結果
                review = claim_review[0]
                textual_rating = review.get('textualRating', '').upper()
                
                claim_with_rating = {**claim, 'rating': textual_rating, 'review_url': review.get('url', '')}
                
                if 'FALSE' in textual_rating:
                    results["false_claims"].append(claim_with_rating)
                elif 'MISLEADING' in textual_rating or 'PARTLY_FALSE' in textual_rating:
                    results["misleading_claims"].append(claim_with_rating)
                else:
                    results["verified_claims"].append(claim_with_rating)
    
    try:
        # 執行異步驗證
        await run_verifications()
    except Exception as e:
        logger.error(f"異步驗證執行失敗: {str(e)}")
        # 降級：標記為未驗證
        results["unverified_claims"].extend(claims)
    
    return results

def verify_claims(claims: List[Dict[str, Any]], api_key: str) -> Dict[str, List[Dict]]:
    """
    驗證聲明（同步包裝器）
    
    Returns:
        Dict: 驗證結果分類
    """
    if not claims:
        return {
            "verified_claims": [],
            "false_claims": [],
            "misleading_claims": [],
            "unverified_claims": []
        }
    
    try:
        # 創建新的事件循環
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(verify_claims_async(claims, api_key))
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"驗證執行失敗: {str(e)}")
        return {
            "verified_claims": [],
            "false_claims": [],
            "misleading_claims": [],
            "unverified_claims": claims  # 所有聲明標記為未驗證
        }

def apply_fact_check_tags(sources: List[Dict], fact_check_results: Dict[str, List[Dict]]) -> List[Dict]:
    """
    應用事實查核標籤和降權（方案 1.3 - 標註與降權）
    
    Args:
        sources: 來源列表
        fact_check_results: 事實查核結果
    
    Returns:
        List[Dict]: 更新後的來源列表
    """
    # 確保 fact_check_results 是字典類型
    if not isinstance(fact_check_results, dict):
        logger.warning(f"fact_check_results 不是字典類型: {type(fact_check_results).__name__}，跳過事實查核標籤")
        return sources
    
    # 建立 source_id 到查核結果的映射
    false_map = {c['source_id']: c for c in fact_check_results.get('false_claims', [])}
    misleading_map = {c['source_id']: c for c in fact_check_results.get('misleading_claims', [])}
    
    for i, source in enumerate(sources):
        source_id = i + 1
        
        # 檢查是否為已證偽
        if source_id in false_map:
            # 降權：證據強度降 2 級
            current_level = source.get('evidence_level', '中等')
            if current_level == '強':
                source['evidence_level'] = '弱'
            elif current_level == '中等':
                source['evidence_level'] = '弱'
            source['fact_check_status'] = '❌ 已證偽'
            source['fact_check_rating'] = false_map[source_id].get('rating', 'VERIFIED_FALSE')
            source['fact_check_url'] = false_map[source_id].get('review_url', '')
            logger.warning(f"Source {source_id} 已證偽，證據強度降級")
        
        # 檢查是否為誤導性
        elif source_id in misleading_map:
            # 降權：證據強度降 1 級
            current_level = source.get('evidence_level', '中等')
            if current_level == '強':
                source['evidence_level'] = '中等'
            source['fact_check_status'] = '⚠️ 誤導性內容'
            source['fact_check_rating'] = misleading_map[source_id].get('rating', 'MISLEADING')
            source['fact_check_url'] = misleading_map[source_id].get('review_url', '')
            logger.info(f"Source {source_id} 標記為誤導性")
    
    return sources

def generate_fact_check_warning(fact_check_results: Dict[str, List[Dict]]) -> str:
    """
    生成事實查核警告文字（用於 context）
    """
    # 確保 fact_check_results 是字典類型
    if not isinstance(fact_check_results, dict):
        logger.warning(f"generate_fact_check_warning: fact_check_results 不是字典類型: {type(fact_check_results).__name__}")
        return ""

    warning_text = "\n【⚠️ 事實查核警告】\n"
    
    false_claims = fact_check_results.get('false_claims', [])
    misleading_claims = fact_check_results.get('misleading_claims', [])
    if not false_claims and not misleading_claims:
        return ""
    
    if false_claims:
        warning_text += f"❌ 已證偽的聲明（{len(false_claims)} 項）：\n"
        for claim in false_claims[:5]:  # 最多顯示 5 項
            warning_text += f"  - Source {claim['source_id']}: {claim['text'][:80]}... (評級: {claim.get('rating', 'VERIFIED_FALSE')})\n"
    
    if misleading_claims:
        warning_text += f"⚠️ 誤導性內容（{len(misleading_claims)} 項）：\n"
        for claim in misleading_claims[:5]:
            warning_text += f"  - Source {claim['source_id']}: {claim['text'][:80]}... (評級: {claim.get('rating', 'MISLEADING')})\n"
    
    return warning_text + "\n"

def calculate_title_similarity(title1: str, title2: str) -> float:
    """計算兩個標題的相似度（0-1之間）- 優化版使用 TF-IDF 概念"""
    if not title1 or not title2:
        return 0.0
    
    # 快速字元級相似度（用於初步過濾）
    char_similarity = SequenceMatcher(None, title1.lower(), title2.lower()).ratio()
    if char_similarity < 0.5:  # 快速過濾明顯不同的標題
        return char_similarity
    
    # 詞級相似度（更準確）
    words1 = set(title1.lower().split())
    words2 = set(title2.lower().split())
    
    if not words1 or not words2:
        return char_similarity
    
    # Jaccard 相似度
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    word_similarity = intersection / union if union > 0 else 0
    
    # 加權組合（字元相似度 30%，詞相似度 70%）
    return 0.3 * char_similarity + 0.7 * word_similarity

def detect_coordinated_behavior(sources: List[Dict]) -> Dict[str, Any]:
    """
    檢測協調行為（改進項目：網軍協調行為偵測 - 簡化版）
    
    基於現有資料分析：
    1. 內容相似度（已實作在 analyze_volume_weight）
    2. 來源域名集中度
    3. 時間分佈模式（如有時間數據）
    
    Args:
        sources: 來源列表
    
    Returns:
        Dict: {
            "coordination_score": float,  # 0-1，協調性分數
            "flags": List[str],
            "similar_content_groups": List[List[int]],
            "domain_concentration": float,
            "time_clustering": Dict  # 時間聚集模式（如有）
        }
    """
    if not sources:
        return {
            "coordination_score": 0.0,
            "flags": [],
            "similar_content_groups": [],
            "domain_concentration": 0.0,
            "time_clustering": {}
        }
    
    flags = []
    
    # === 1. 內容相似度分析（重用現有函數）===
    volume_analysis = analyze_volume_weight(sources)
    duplicate_groups = volume_analysis.get('duplicate_groups', [])
    duplicate_count = volume_analysis.get('duplicate_count', 0)
    total_count = len(sources)
    
    # === 2. 來源域名集中度 ===
    domain_counts = Counter(get_domain_name(s.get('url', '')) for s in sources)
    domain_concentration = max(domain_counts.values()) / total_count if total_count > 0 else 0
    
    # === 3. 計算協調性分數 ===
    coordination_score = 0.0
    
    # 如果重複內容超過閾值，可能為協調行為
    duplicate_ratio = duplicate_count / total_count if total_count > 0 else 0
    if duplicate_ratio > COORDINATION_DUPLICATE_RATIO_THRESHOLD:
        coordination_score += COORDINATION_DUPLICATE_PENALTY
        flags.append(f"⚠️ 高度重複內容（>{COORDINATION_DUPLICATE_RATIO_THRESHOLD*100:.0f}%），可能存在協調發布")
    
    # 如果單一域名超過閾值，可能為組織性操作
    if domain_concentration > COORDINATION_DOMAIN_CONCENTRATION_THRESHOLD:
        top_domain = domain_counts.most_common(1)[0][0] if domain_counts else ""
        coordination_score += COORDINATION_DOMAIN_PENALTY
        flags.append(f"⚠️ 來源過度集中（{top_domain} 佔 {domain_concentration*100:.1f}%），可能為組織性操作")
    
    # === 4. 時間聚集分析（簡化版：基於日期）===
    date_counts = Counter()
    for source in sources:
        date_str = source.get('published_date') or source.get('final_date', '')
        if date_str and date_str != 'Missing':
            date_counts[date_str[:10]] += 1
    
    if date_counts:
        max_same_date = max(date_counts.values())
        date_concentration = max_same_date / total_count
        
        if date_concentration > COORDINATION_DATE_CONCENTRATION_THRESHOLD:  # 同一天發布超過閾值
            coordination_score += COORDINATION_TIME_PENALTY
            flags.append(f"⚠️ 時間高度集中（{max_same_date} 篇在同一天發布），可能存在同步操作")
    
    coordination_score = min(1.0, coordination_score)
    
    if coordination_score > COORDINATION_HIGH_RISK_SCORE:
        flags.insert(0, "🚨 高風險：檢測到明顯的協調行為特徵")
    
    return {
        "coordination_score": coordination_score,
        "flags": flags,
        "similar_content_groups": duplicate_groups,
        "domain_concentration": domain_concentration,
        "time_clustering": dict(date_counts) if date_counts else {},
        "duplicate_ratio": duplicate_ratio
    }


def _jaccard_similarity_ngram(text_a: str, text_b: str, n: int = 3) -> float:
    """
    計算兩段文字的字元 n-gram Jaccard 相似度（無 sklearn 時的降級方案）。
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    """
    if not text_a.strip() or not text_b.strip():
        return 0.0
    def ngrams(s: str, k: int) -> Set[str]:
        s = re.sub(r'\s+', ' ', s.strip())
        return set(s[i:i + k] for i in range(max(0, len(s) - k + 1)))
    a, b = ngrams(text_a, n), ngrams(text_b, n)
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def _build_similarity_matrix(texts: List[str], similarity_threshold: float) -> Tuple[Any, List[List[int]]]:
    """
    建立正文相似度矩陣並回傳聚類（連通分量）。
    若 SKLEARN_AVAILABLE 則用 TfidfVectorizer + cosine_similarity，否則用 n-gram Jaccard。
    聚類邏輯：相似度 >= threshold 的兩兩合併（Union-Find）。
    """
    n = len(texts)
    if n == 0:
        return None, []
    # 空白或過短補齊，避免 vectorizer 報錯
    texts = [t.strip() or " " for t in texts]

    if SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
                min_df=1,
                token_pattern=r'(?u)\b\w+\b'
            )
            X = vectorizer.fit_transform(texts)
            sim = cosine_similarity(X)
        except Exception as e:
            logger.warning(f"TfidfVectorizer/cosine_similarity 失敗，降級為 n-gram Jaccard: {e}")
            sim = None
    else:
        sim = None

    if sim is None:
        # 降級：n-gram Jaccard 兩兩計算（純 Python 二維 list）
        sim = [[0.0] * n for _ in range(n)]
        for i in range(n):
            sim[i][i] = 1.0
            for j in range(i + 1, n):
                s = _jaccard_similarity_ngram(texts[i], texts[j], n=3)
                sim[i][j] = sim[j][i] = s

    # Union-Find 聚類
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= similarity_threshold:
                union(i, j)

    # 依 root 分組
    clusters_by_root: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        clusters_by_root.setdefault(r, []).append(i)
    clusters = [sorted(indices) for indices in clusters_by_root.values() if len(indices) > 1]
    return sim, clusters


def detect_cross_domain_syndication(
    sources: List[Dict],
    similarity_threshold: float = 0.75
) -> List[Dict]:
    """
    偵測跨網域聯播／複製內容（Syndication Network / PR 推送）。
    
    僅依標題相似度無法辨識「同一正文、多站發布」的協調行為。本函數以正文為準，
    計算成對相似度後聚類，並**只保留橫跨至少 2 個不同網域**的群組，以區分：
    - 單站內多篇重複（如 Yahoo 分頁）→ 不計入
    - 多個獨立網站出現高度雷同正文 → 視為聯播網／協調推送
    
    數學與邏輯：
    - 正文表示：TfidfVectorizer(ngram_range=(1,2)) 或字元 3-gram Jaccard（降級）
    - 相似度：cosine_similarity 或 Jaccard
    - 聚類：相似度 >= similarity_threshold 的節點做 Union-Find 連通分量
    - 篩選：每個群組取 url 的 get_domain_name，僅保留 unique_domain_count >= 2 的群組
    
    Args:
        sources: 來源列表，每項需含 'content'（可選 'title'），'url'
        similarity_threshold: 正文相似度閾值，預設 0.75
    
    Returns:
        List[Dict]: 每個元素為一跨網域群組，格式：
        {
            "source_indices": List[int],
            "domains": List[str],
            "unique_domain_count": int,
            "mean_similarity": float,
        }
    """
    if not sources:
        return []

    # 1. 萃取正文（標題 + 內容前段，避免過長）
    texts: List[str] = []
    for s in sources:
        title = s.get("title") or ""
        content = (s.get("content") or "")[:3000]
        text = f"{title} {content}".strip()
        texts.append(text)

    try:
        sim_matrix, clusters = _build_similarity_matrix(texts, similarity_threshold)
    except Exception as e:
        logger.warning(f"detect_cross_domain_syndication 相似度矩陣失敗: {e}")
        return []

    # 2. 只保留跨多網域的群組
    result: List[Dict] = []
    for indices in clusters:
        domains = [get_domain_name(sources[i].get("url") or "") for i in indices]
        unique_domains = [d for d in domains if d]
        unique_domain_set = set(unique_domains)
        if len(unique_domain_set) < 2:
            continue

        # 群組內平均相似度（上三角）
        if sim_matrix is not None and len(indices) > 1:
            total, cnt = 0.0, 0
            for ii in range(len(indices)):
                for jj in range(ii + 1, len(indices)):
                    i_idx, j_idx = indices[ii], indices[jj]
                    val = sim_matrix[i_idx][j_idx] if isinstance(sim_matrix, list) else sim_matrix[i_idx, j_idx]
                    total += val
                    cnt += 1
            mean_sim = total / cnt if cnt else 0.0
        else:
            mean_sim = float(similarity_threshold)

        result.append({
            "source_indices": indices,
            "domains": list(unique_domain_set),
            "unique_domain_count": len(unique_domain_set),
            "mean_similarity": round(mean_sim, 4),
        })

    return result


def _parse_source_datetime(source: Dict) -> Optional[datetime]:
    """從來源取得可解析的發布時間，支援 YYYY-MM-DD 或帶時間字串。"""
    for key in ("published_date", "final_date", "date"):
        raw = source.get(key)
        if not raw or not isinstance(raw, str):
            continue
        raw = raw.strip()[:19]
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(raw[:len(fmt)], fmt)
            except ValueError:
                continue
        if len(raw) >= 10:
            try:
                return datetime.strptime(raw[:10], "%Y-%m-%d")
            except ValueError:
                pass
    return None


def track_narrative_diffusion(
    syndication_clusters: List[Dict],
    sources: List[Dict]
) -> List[Dict]:
    """
    依跨網域聯播群組計算「敘事擴散速度」，標記高風險協調敘事。
    
    同一敘事在極短時間內出現在多個獨立網域，常與 Astroturfing / CIB 相關。
    定義：velocity = unique_domain_count / time_span_hours，其中 time_span 為群組內
    最早與最晚發布時間差（小時）；若無有效時間則以 time_span = 1 小時避免除零。
    
    高 velocity 表示「多站、短時」擴散，可標記為 High-Risk Coordinated Narrative。
    
    Args:
        syndication_clusters: detect_cross_domain_syndication 的輸出
        sources: 與當時分析對應的來源列表（依 index 對應）
    
    Returns:
        List[Dict]: 每個群組一筆，新增欄位：
        - time_span_hours: float
        - velocity: float (domains per hour)
        - risk_level: "high" | "medium" | "low"
        - is_high_risk_coordinated: bool
    """
    if not syndication_clusters or not sources:
        return []

    # 高風險門檻：每小時 0.5 個以上網域 且 至少 2 網域
    VELOCITY_HIGH = 0.5
    VELOCITY_MEDIUM = 0.2

    result: List[Dict] = []
    for cluster in syndication_clusters:
        indices = cluster.get("source_indices", [])
        unique_domain_count = cluster.get("unique_domain_count", 0)
        out = dict(cluster)

        datetimes: List[datetime] = []
        for i in indices:
            if 0 <= i < len(sources):
                dt = _parse_source_datetime(sources[i])
                if dt is not None:
                    datetimes.append(dt)

        if len(datetimes) < 2:
            time_span_hours = 1.0
        else:
            min_dt = min(datetimes)
            max_dt = max(datetimes)
            time_span_hours = max(0.01, (max_dt - min_dt).total_seconds() / 3600.0)

        velocity = unique_domain_count / time_span_hours
        out["time_span_hours"] = round(time_span_hours, 4)
        out["velocity"] = round(velocity, 4)

        if velocity >= VELOCITY_HIGH and unique_domain_count >= 2:
            risk_level = "high"
            is_high_risk = True
        elif velocity >= VELOCITY_MEDIUM and unique_domain_count >= 2:
            risk_level = "medium"
            is_high_risk = False
        else:
            risk_level = "low"
            is_high_risk = False

        out["risk_level"] = risk_level
        out["is_high_risk_coordinated"] = is_high_risk
        result.append(out)

    return result


def detect_semantic_spin(
    syndication_clusters: List[Dict],
    sources: List[Dict],
    api_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    語義旋轉偵測：對比聯播群組中的「主推敘事」(Source A) 與一篇非群組來源 (Source B)，
    以 LLM 辨識同一事實下的對立框架 (Spin)。若 api_key 缺失或為 None 則回傳 None。
    
    Args:
        syndication_clusters: detect_cross_domain_syndication 的輸出（每項含 source_indices）
        sources: 與當時分析對應的來源列表（依 index 對應）
        api_key: Google Gemini API Key（無則跳過，回傳 None）
    
    Returns:
        Optional[Dict]: 成功時為 {"shared_fact", "spin_a", "spin_b", "spin_score"}；失敗或無群組時為 None
    """
    if not syndication_clusters or not sources or not api_key:
        return None

    # 1. Source A (The Narrative Push): 最大群組的第一個有效來源
    largest = max(syndication_clusters, key=lambda c: len(c.get("source_indices", [])))
    cluster_indices = set(largest.get("source_indices", []))
    if not cluster_indices:
        return None

    idx_a = largest["source_indices"][0]
    if idx_a < 0 or idx_a >= len(sources):
        return None
    source_a = sources[idx_a]
    cat_a = source_a.get("source_category") or "OTHER"

    # 2. Source B (The Counter-Narrative): 不在群組內，優先不同 source_category
    preferred_other = (
        {"GREEN", "INTL", "CHINA", "OFFICIAL", "NEUTRAL", "INDIE"} if cat_a == "BLUE"
        else {"BLUE", "INTL", "CHINA", "OFFICIAL", "NEUTRAL", "INDIE"} if cat_a == "GREEN"
        else {"BLUE", "GREEN", "INTL", "CHINA"}
    )
    idx_b = None
    for i in range(len(sources)):
        if i in cluster_indices:
            continue
        cat_b = (sources[i].get("source_category") or "OTHER")
        if cat_b in preferred_other:
            idx_b = i
            break
    if idx_b is None:
        for i in range(len(sources)):
            if i not in cluster_indices:
                idx_b = i
                break
    if idx_b is None:
        return None

    source_b = sources[idx_b]
    title_a = (source_a.get("title") or "")[:300]
    title_b = (source_b.get("title") or "")[:300]
    content_a = (source_a.get("content") or "")[:500]
    content_b = (source_b.get("content") or "")[:500]

    prompt_instruction = (
        'Compare these two news excerpts. '
        '1. Identify the shared objective fact. '
        '2. Identify the \'Spin\' or \'Framing\' used by each side (e.g., emotional adjectives, omitted context, shifting blame). '
        '3. Provide a Spin Score (0.0 to 1.0, where >0.6 means high manipulation). '
        'Output JSON strictly: {"shared_fact": "...", "spin_a": "...", "spin_b": "...", "spin_score": 0.8}'
    )
    prompt = f"""{prompt_instruction}

**Source A (Narrative Push):**
Title: {title_a}
Excerpt: {content_a}

**Source B (Counter-Narrative):**
Title: {title_b}
Excerpt: {content_b}"""

    try:
        llm = ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, google_api_key=api_key, temperature=0.2)
        raw = _extract_text_from_llm_content(llm.invoke(prompt).content)
        if not raw:
            return None
        obj = {}
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            if start >= 0:
                depth = 0
                for i, c in enumerate(raw[start:], start):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(raw[start : i + 1])
                            except json.JSONDecodeError:
                                pass
                            break
        if isinstance(obj, dict) and "shared_fact" in obj and "spin_score" in obj:
            score = obj.get("spin_score")
            if isinstance(score, (int, float)):
                obj["spin_score"] = float(score)
            return obj
    except Exception as e:
        logger.warning(f"detect_semantic_spin LLM 失敗: {e}")
    return None


def analyze_volume_weight(sources: List[Dict], progress_callback=None) -> Dict[str, Any]:
    """
    聲量權重校正：識別重複論述和獨特觀點（優化版）
    
    Args:
        sources: 來源列表
        progress_callback: 進度回調函數 (current, total) -> None
    
    Returns:
        Dict containing: duplicate_groups, unique_articles
    """
    duplicate_groups = []
    processed_indices = set()
    unique_articles = []
    total = len(sources)
    
    # 預處理：建立標題索引（去除空標題）
    valid_sources = [(i, s.get('title', '')) for i, s in enumerate(sources) if s.get('title', '')]
    
    for idx, (i, title1) in enumerate(valid_sources):
        if i in processed_indices:
            if progress_callback:
                progress_callback(idx + 1, total)
            continue
        
        group = [i]
        # 只比較尚未處理的來源（優化：不重複比較）
        for j, title2 in valid_sources[idx+1:]:
            if j in processed_indices:
                continue
            
            # 快速過濾：先檢查字首相似度
            if title1[:10].lower() != title2[:10].lower():
                similarity = calculate_title_similarity(title1, title2)
                if similarity >= SIMILARITY_THRESHOLD:
                    group.append(j)
                    processed_indices.add(j)
            else:
                # 字首相同，直接計算完整相似度
                similarity = calculate_title_similarity(title1, title2)
                if similarity >= SIMILARITY_THRESHOLD:
                    group.append(j)
                    processed_indices.add(j)
        
        processed_indices.add(i)
        
        if len(group) > 1:
            duplicate_groups.append(group)
        else:
            unique_articles.append(i)
        
        if progress_callback and (idx + 1) % 10 == 0:
            progress_callback(idx + 1, total)
    
    return {
        'duplicate_groups': duplicate_groups,
        'unique_articles': unique_articles,
        'duplicate_count': len(duplicate_groups),
        'unique_count': len(unique_articles)
    }

def execute_hybrid_search(
    query: str,
    api_key_tavily: str,
    search_params: Dict,
    is_strict_mode: bool,
    dynamic_keywords: List,
    selected_regions: List[str],
    english_queries: Optional[List[str]] = None,
    japanese_queries: Optional[List[str]] = None,
    korean_queries: Optional[List[str]] = None,
) -> List[Dict]:
    """
    執行混和搜尋（完整版 - 基於 Tavily 最佳實踐）
    
    基於 Tavily API 官方最佳實踐：
    1. 查詢優化：保持查詢少於 400 字元，拆分複雜查詢
    2. 搜尋深度：通用搜尋使用 basic，保底搜尋使用 advanced
    3. 結果過濾：使用 topic: "news" 和網域過濾
    4. 平衡報導：多視角查詢 + 分眾保底機制
    5. 建議一：當選歐洲/美洲且傳入 english_queries 時，新增英文查詢任務（非中文檢索），結果與現有任務共用 seen_urls 去重。
    6. 亞洲視角：當選亞洲且傳入 japanese_queries / korean_queries 時，新增日文/韓文查詢任務，對日本/韓國網域檢索。
    
    Args:
        dynamic_keywords: 可以是 List[str] 或 List[Dict] (擴展查詢格式)
        english_queries: 英文搜尋關鍵字列表；當 selected_regions 含歐洲/美洲時會用於非中文檢索任務（可選）
        japanese_queries: 日文搜尋關鍵字列表；當 selected_regions 含亞洲時會用於日本媒體保底（可選）
        korean_queries: 韓文搜尋關鍵字列表；當 selected_regions 含亞洲時會用於韓國媒體保底（可選）
    """
    tavily = TavilyClient(api_key=api_key_tavily)
    seen_urls = set()
    tasks = []
    
    # === Tavily API 參數優化策略（基於官方最佳實踐）===
    # 從 search_params 複製，但只保留 Tavily API 支持的參數
    optimized_params = {
        "search_depth": search_params.get("search_depth", "basic"),  # 預設使用 basic
        "max_results": search_params.get("max_results", 10),
        "topic": "general",  # 改回 general：因為 news 模式會排除 YouTube/PTT 等社群媒體，導致某些議題搜尋不到
    }
    
    # 條件性添加 exclude_domains
    if "exclude_domains" in search_params and search_params["exclude_domains"]:
        optimized_params["exclude_domains"] = search_params["exclude_domains"]
    
    # 條件性添加 country（如果選定了台灣）
    if selected_regions and any("台灣" in str(r) for r in selected_regions):
        optimized_params["country"] = "taiwan"  # 優先台灣來源
    
    # 1. 通用熱度搜尋
    general_domains = []
    selected_str = str(selected_regions)
    if "台灣" in selected_str: general_domains.extend(FULL_TAIWAN_WHITELIST)
    if "獨立" in selected_str: general_domains.extend(INDIE_WHITELIST)
    if "亞洲" in selected_str: general_domains.extend(INTL_ASIA_WHITELIST)
    if "歐洲" in selected_str: general_domains.extend(INTL_EUROPE_WHITELIST)
    if "美洲" in selected_str: general_domains.extend(INTL_AMERICAS_WHITELIST)
    
    # === 通用搜尋參數（基於 Tavily 最佳實踐）===
    general_params = optimized_params.copy()
    general_params['max_results'] = 15  # Tavily 建議：不要設定過高，避免低品質結果
    general_params['search_depth'] = 'basic'  # 通用搜尋使用 basic（平衡速度與品質）
    
    # 改進：通用搜尋完全不使用 include_domains，讓搜尋範圍最大化
    # 域名過濾只在保底搜尋中使用（確保立場平衡）
    if is_strict_mode and general_domains:
        logger.info(f"通用搜尋：檢測到嚴格模式，但不使用域名過濾（共 {len(set(general_domains))} 個候選域名可用於保底搜尋）")
        logger.info(f"域名過濾僅用於保底搜尋（藍/綠/官方），確保立場平衡的同時，通用搜尋仍能獲取廣泛結果")
    # 不設置 general_params['include_domains']，讓通用搜尋範圍最大化
    
    # === 查詢擴展機制（基於 Tavily 最佳實踐：拆分複雜查詢）===
    # Tavily 建議：將複雜查詢拆分為多個子查詢，每個查詢少於 400 字元
    
    def validate_query_length(q: str, max_length: int = 400) -> str:
        """確保查詢長度符合 Tavily 建議（少於 400 字元）"""
        if len(q) <= max_length:
            return q
        # 如果超過，截斷並添加省略號
        logger.warning(f"查詢過長 ({len(q)} 字元)，截斷至 {max_length} 字元: {q[:50]}...")
        return q[:max_length-3] + "..."
    
    # 處理擴展查詢列表
    if isinstance(dynamic_keywords, list) and len(dynamic_keywords) > 0:
        if all(isinstance(k, str) for k in dynamic_keywords):
            # 字串列表，轉換為 Dict 格式並驗證長度
            expanded_queries = [
                {"query": validate_query_length(k), "type": f"查詢{i+1}", "priority": 1}
                for i, k in enumerate(dynamic_keywords)
            ]
        else:
            # 已經是 Dict 列表，驗證查詢長度
            expanded_queries = []
            for i, q in enumerate(dynamic_keywords):
                if isinstance(q, dict) and 'query' in q:
                    q_copy = q.copy()
                    q_copy['query'] = validate_query_length(q['query'])
                    expanded_queries.append(q_copy)
                elif isinstance(q, str):
                    # 如果是字串，轉換為 dict 格式
                    expanded_queries.append({
                        "query": validate_query_length(q),
                        "type": f"查詢{i+1}",
                        "priority": 1
                    })
                else:
                    # 其他類型，嘗試轉換為 dict
                    logger.warning(f"動態關鍵字項目既不是 dict 也不是 str，類型: {type(q).__name__}，跳過")
    else:
        # 預設查詢（三軌查詢法：事實軌、觀點軌、深度軌）
        base_query = validate_query_length(query)
        expanded_queries = [
            {"query": base_query, "type": "主查詢", "priority": 1},
            {"query": validate_query_length(f"{query} 新聞 事件"), "type": "事實軌", "priority": 1},
            {"query": validate_query_length(f"{query} 爭議 評論"), "type": "觀點軌", "priority": 1},
            {"query": validate_query_length(f"{query} 懶人包 分析"), "type": "深度軌", "priority": 1}
        ]
    
    # 使用所有高優先級查詢（優化：減少請求數量以避免超過 API 負荷）
    priority_queries = [q for q in expanded_queries if q.get("priority", 3) <= 2]
    logger.info(f"處理擴展查詢: 總數={len(expanded_queries)}, 優先級查詢={len(priority_queries)}")
    
    # 大幅減少到最多 5 個通用查詢，避免 429 錯誤
    for q in priority_queries[:5]:
        # 通用查詢不限制 topic，以便包含新聞和社群內容
        tasks.append({"name": f"General_{q['type']}", "query": q["query"], "params": general_params})
    
    logger.info(f"已建立 {len(tasks)} 個通用搜尋任務（已優化以避免 429 錯誤）")
    
    # === 分眾保底搜尋（平衡報導策略：確保立場多樣化）===
    # Tavily 建議：使用 include_domains 限制到特定網域，但保持列表簡短
    if "台灣" in selected_str:
        guard_max = 8  # Tavily 建議：不要設定過高，避免低品質結果
        
        # 藍營保底搜尋（使用 advanced 深度以確保品質）
        blue_params = optimized_params.copy()
        blue_params['max_results'] = guard_max
        blue_params['search_depth'] = 'advanced'  # 保底搜尋使用 advanced（最高相關性）
        if BLUE_WHITELIST and len(BLUE_WHITELIST) > 0:
            # Tavily 建議：保持網域列表簡短（< 50 個）
            blue_domains = BLUE_WHITELIST[:50] if len(BLUE_WHITELIST) > 50 else BLUE_WHITELIST
            blue_params['include_domains'] = blue_domains
            logger.debug(f"藍營保底搜尋: {len(blue_domains)} 個網域（已限制在 50 個以內）")
        tasks.append({"name": "Blue_Guard", "query": validate_query_length(query), "params": blue_params})
        
        # 綠營保底搜尋
        green_params = optimized_params.copy()
        green_params['max_results'] = guard_max
        green_params['search_depth'] = 'advanced'
        if GREEN_WHITELIST and len(GREEN_WHITELIST) > 0:
            green_domains = GREEN_WHITELIST[:50] if len(GREEN_WHITELIST) > 50 else GREEN_WHITELIST
            green_params['include_domains'] = green_domains
            logger.debug(f"綠營保底搜尋: {len(green_domains)} 個網域（已限制在 50 個以內）")
        tasks.append({"name": "Green_Guard", "query": validate_query_length(query), "params": green_params})
        
        # 官方保底搜尋
        official_params = optimized_params.copy()
        official_params['max_results'] = guard_max
        official_params['search_depth'] = 'advanced'
        if OFFICIAL_WHITELIST and len(OFFICIAL_WHITELIST) > 0:
            official_domains = OFFICIAL_WHITELIST[:50] if len(OFFICIAL_WHITELIST) > 50 else OFFICIAL_WHITELIST
            official_params['include_domains'] = official_domains
            logger.debug(f"官方保底搜尋: {len(official_domains)} 個網域（已限制在 50 個以內）")
        tasks.append({"name": "Official_Guard", "query": validate_query_length(f"{query} 聲明 新聞稿"), "params": official_params})

        logger.info(f"已建立 3 個保底搜尋任務（藍/綠/官方），總任務數: {len(tasks)}")

    # === 國際/亞洲保底搜尋（確保日本、韓國等區域議題有在地媒體報導）===
    # 觸發條件：1) 使用者選擇「亞洲」 或 2) 查詢包含日本/韓國等關鍵字
    _query_lower = query.lower().strip()
    _japan_keywords = ["日本", "自民黨", "岸田", "東京", "大阪", "nhk", "ldp", "japan", "日經", "朝日", "讀賣"]
    _korea_keywords = ["韓國", "南韓", "北韓", "首爾", "korea", "kim jong", "尹錫悅"]
    _needs_asia_guard = "亞洲" in selected_str
    _needs_japan_guard = any(kw in _query_lower for kw in _japan_keywords)
    _needs_korea_guard = any(kw in _query_lower for kw in _korea_keywords)

    if _needs_asia_guard and INTL_ASIA_WHITELIST:
        asia_params = optimized_params.copy()
        asia_params['max_results'] = 8
        asia_params['search_depth'] = 'advanced'
        asia_domains = INTL_ASIA_WHITELIST[:55]  # 含外電、智庫、亞洲媒體
        asia_params['include_domains'] = asia_domains
        tasks.append({"name": "Intl_Asia_Guard", "query": validate_query_length(query), "params": asia_params})
        logger.info(f"已建立亞洲國際保底搜尋（{len(asia_domains)} 個網域），總任務數: {len(tasks)}")

    if _needs_japan_guard and INTL_JAPAN_WHITELIST:
        japan_params = optimized_params.copy()
        japan_params['max_results'] = 8
        japan_params['search_depth'] = 'advanced'
        # 與藍/綠/官方一致：擴增白名單時仍限制網域數，避免單次請求過大並符合 Tavily 建議
        japan_domains = INTL_JAPAN_WHITELIST[:50] if len(INTL_JAPAN_WHITELIST) > 50 else INTL_JAPAN_WHITELIST
        japan_params['include_domains'] = japan_domains
        tasks.append({"name": "Japan_Guard", "query": validate_query_length(query), "params": japan_params})
        logger.info(f"已建立日本媒體保底搜尋（查詢含日本關鍵字），總任務數: {len(tasks)}")

    if _needs_korea_guard:
        korea_domains = [d for d in INTL_ASIA_WHITELIST if "korea" in d.lower() or "yna" in d.lower()]
        if korea_domains:
            korea_params = optimized_params.copy()
            korea_params['max_results'] = 6
            korea_params['search_depth'] = 'advanced'
            korea_params['include_domains'] = korea_domains[:25]
            tasks.append({"name": "Korea_Guard", "query": validate_query_length(query), "params": korea_params})
            logger.info(f"已建立韓國媒體保底搜尋，總任務數: {len(tasks)}")

    # === 亞洲視角：日文/韓文關鍵字檢索（日本/韓國在地媒體，與既有中文保底並行）===
    _needs_asia_lang = "亞洲" in selected_str
    if _needs_asia_lang and japanese_queries and INTL_JAPAN_WHITELIST:
        ja_params = optimized_params.copy()
        ja_params.pop("country", None)
        ja_params["max_results"] = 8
        ja_params["search_depth"] = "advanced"
        ja_params["include_domains"] = INTL_JAPAN_WHITELIST[:15]
        for i, jq in enumerate(japanese_queries[:2]):
            if jq and isinstance(jq, str) and jq.strip():
                tasks.append({
                    "name": f"Asia_Japanese_Guard_{i+1}",
                    "query": validate_query_length(jq.strip()),
                    "params": ja_params,
                })
        logger.info(f"已建立 {min(2, len(japanese_queries))} 個日文檢索任務（日本媒體），總任務數: {len(tasks)}")
    _korea_domains_for_asia = [d for d in INTL_ASIA_WHITELIST if "korea" in d.lower() or "yna" in d.lower()]
    if _needs_asia_lang and korean_queries and _korea_domains_for_asia:
        ko_params = optimized_params.copy()
        ko_params.pop("country", None)
        ko_params["max_results"] = 8
        ko_params["search_depth"] = "advanced"
        ko_params["include_domains"] = _korea_domains_for_asia[:15]
        for i, kq in enumerate(korean_queries[:2]):
            if kq and isinstance(kq, str) and kq.strip():
                tasks.append({
                    "name": f"Asia_Korean_Guard_{i+1}",
                    "query": validate_query_length(kq.strip()),
                    "params": ko_params,
                })
        logger.info(f"已建立 {min(2, len(korean_queries))} 個韓文檢索任務（韓國媒體），總任務數: {len(tasks)}")

    # === 獨立/自媒體保底（勾選「獨立/自媒體」時，對 INDIE_WHITELIST 執行專用搜尋）===
    if "獨立" in selected_str and INDIE_WHITELIST:
        indie_domains = INDIE_WHITELIST[:20]  # 精簡以符合 Tavily 建議
        indie_params = optimized_params.copy()
        indie_params.pop("country", None)
        indie_params["max_results"] = 8
        indie_params["search_depth"] = "advanced"
        indie_params["include_domains"] = indie_domains
        tasks.append({"name": "Indie_Guard", "query": validate_query_length(query), "params": indie_params})
        logger.info(f"已建立獨立/自媒體保底搜尋（{len(indie_domains)} 個網域），總任務數: {len(tasks)}")

    # === 建議二：歐洲/美洲分區保底（含 INTL_WEST_WHITELIST 調查／國際權威，網域精簡 15）===
    # 每個選定視角都有對應保底；併入西方調查媒體白名單，避免只靠通用搜尋
    if "歐洲" in selected_str and (INTL_EUROPE_WHITELIST or INTL_WEST_WHITELIST):
        _europe_pool = list(INTL_EUROPE_WHITELIST[:10]) if INTL_EUROPE_WHITELIST else []
        _europe_pool.extend(INTL_WEST_WHITELIST[:8] if INTL_WEST_WHITELIST else [])
        europe_domains = list(dict.fromkeys(_europe_pool))[:15]
        if europe_domains:
            europe_params = optimized_params.copy()
            europe_params.pop("country", None)
            europe_params["max_results"] = 8
            europe_params["search_depth"] = "advanced"
            europe_params["include_domains"] = europe_domains
            tasks.append({"name": "Europe_Guard", "query": validate_query_length(query), "params": europe_params})
            logger.info(f"已建立歐洲區域保底搜尋（含西方調查，{len(europe_domains)} 個網域），總任務數: {len(tasks)}")
    if "美洲" in selected_str and (INTL_AMERICAS_WHITELIST or INTL_WEST_WHITELIST):
        _americas_pool = list(INTL_AMERICAS_WHITELIST[:10]) if INTL_AMERICAS_WHITELIST else []
        _americas_pool.extend(INTL_WEST_WHITELIST[:8] if INTL_WEST_WHITELIST else [])
        americas_domains = list(dict.fromkeys(_americas_pool))[:15]
        if americas_domains:
            americas_params = optimized_params.copy()
            americas_params.pop("country", None)
            americas_params["max_results"] = 8
            americas_params["search_depth"] = "advanced"
            americas_params["include_domains"] = americas_domains
            tasks.append({"name": "Americas_Guard", "query": validate_query_length(query), "params": americas_params})
            logger.info(f"已建立美洲區域保底搜尋（含西方調查，{len(americas_domains)} 個網域），總任務數: {len(tasks)}")

    # === 建議一：歐洲/美洲非中文檢索（英文關鍵字 + 國際網域含 INTL_WEST，結果依 URL 去重）===
    # 衝突避免：英文任務使用獨立參數（不設 country，不與台灣保底混用），且僅在選歐洲/美洲時加入
    _needs_english_guard = "歐洲" in selected_str or "美洲" in selected_str
    if _needs_english_guard and english_queries:
        # 合併歐洲+美洲+西方調查網域，精簡至 20 個（符合 Tavily 建議）
        _intl_domains = []
        if "歐洲" in selected_str and INTL_EUROPE_WHITELIST:
            _intl_domains.extend(INTL_EUROPE_WHITELIST[:8])
        if "美洲" in selected_str and INTL_AMERICAS_WHITELIST:
            _intl_domains.extend(INTL_AMERICAS_WHITELIST[:8])
        if INTL_WEST_WHITELIST:
            _intl_domains.extend(INTL_WEST_WHITELIST[:10])
        _intl_domains = list(dict.fromkeys(_intl_domains))[:20]
        if _intl_domains:
            # 英文任務不設 country，避免與台灣優先衝突
            en_params = optimized_params.copy()
            en_params.pop("country", None)
            en_params["max_results"] = 8
            en_params["search_depth"] = "advanced"
            en_params["include_domains"] = _intl_domains
            for i, eq in enumerate(english_queries[:3]):
                if eq and isinstance(eq, str) and eq.strip():
                    tasks.append({
                        "name": f"English_Guard_{i+1}",
                        "query": validate_query_length(eq.strip()),
                        "params": en_params,
                    })
            logger.info(f"已建立 {min(3, len(english_queries))} 個英文檢索任務（歐洲/美洲/西方調查網域），總任務數: {len(tasks)}")

    def fetch(task, retry_count=0):
        try:
            # 添加延遲以避免超過 API 速率限制
            if SEARCH_REQUEST_DELAY > 0:
                time.sleep(SEARCH_REQUEST_DELAY)
            
            # 清理參數：只保留 Tavily API 支持的參數
            clean_params = {}
            tavily_supported_params = [
                'search_depth', 'max_results', 'include_domains', 'exclude_domains',
                'include_answer', 'topic', 'days', 'include_raw_content', 'include_images', 'include_image_descriptions'
            ]
            
            for key, value in task['params'].items():
                if key in tavily_supported_params and value is not None:
                    # 特別處理 include_domains 和 exclude_domains，確保是列表
                    if key in ['include_domains', 'exclude_domains']:
                        if isinstance(value, list) and len(value) > 0:
                            clean_params[key] = value
                        elif isinstance(value, (str, tuple)) and len(value) > 0:
                            clean_params[key] = list(value) if isinstance(value, tuple) else [value]
                    else:
                        clean_params[key] = value
            
            # 記錄清理後的參數（用於調試）
            param_summary = {}
            for k, v in clean_params.items():
                if k in ['include_domains', 'exclude_domains']:
                    param_summary[k] = f"列表({len(v)}個網域)" if isinstance(v, list) else str(v)[:50]
                else:
                    param_summary[k] = str(v)[:50]
            
            logger.info(f"執行搜尋任務: {task['name']}, 查詢: {task['query'][:50]}, 參數: {param_summary}, 重試次數: {retry_count}")
            
            # 執行搜尋
            search_response = tavily.search(query=task['query'], **clean_params)
            
            # 確保 search_response 是字典類型
            if not isinstance(search_response, dict):
                logger.error(f"搜尋任務 {task['name']} 返回非字典類型: {type(search_response).__name__}，值: {str(search_response)[:500]}")
                return []
            
            # 檢查是否有錯誤
            if 'error' in search_response:
                logger.error(f"搜尋任務 {task['name']} API 返回錯誤: {search_response.get('error', 'Unknown error')}")
                return []
            
            results = search_response.get('results', [])
            logger.info(f"搜尋任務 {task['name']} 完成: 找到 {len(results)} 筆結果")
            
            # 後處理：使用分數過濾（基於 Tavily 最佳實踐）
            # Tavily 建議：使用 score 來過濾和排序結果
            if results:
                # 按分數排序（分數越高，相關性越高）
                results.sort(key=lambda x: x.get('score', 0), reverse=True)
                # 記錄分數範圍
                scores = [r.get('score', 0) for r in results]
                if scores:
                    logger.debug(f"搜尋任務 {task['name']} 分數範圍: {min(scores):.3f} - {max(scores):.3f}")
            
            # 如果結果為空，記錄詳細資訊用於調試
            if len(results) == 0:
                logger.warning(f"搜尋任務 {task['name']} 返回 0 筆結果")
                logger.warning(f"  - 查詢: {task['query'][:100]}")
                logger.warning(f"  - 參數: {param_summary}")
                logger.warning(f"  - API 回應 keys: {list(search_response.keys())}")
                if 'query' in search_response:
                    logger.warning(f"  - API 回應 query: {search_response.get('query', 'N/A')}")
            
            return results
        except Exception as e:
            error_str = str(e)
            # 檢查是否為 429 錯誤（速率限制）
            if "429" in error_str or "TooManyRequests" in error_str or "rate limit" in error_str.lower():
                if retry_count < MAX_SEARCH_RETRIES:
                    wait_time = SEARCH_RETRY_DELAY * (retry_count + 1)  # 指數退避
                    logger.warning(f"搜尋任務 {task['name']} 遇到 429 錯誤，等待 {wait_time} 秒後重試 ({retry_count + 1}/{MAX_SEARCH_RETRIES})")
                    time.sleep(wait_time)
                    return fetch(task, retry_count + 1)  # 遞迴重試
                else:
                    logger.error(f"搜尋任務 {task['name']} 重試 {MAX_SEARCH_RETRIES} 次後仍失敗（429 錯誤）")
            else:
                logger.warning(f"搜尋任務失敗: {task['name']}, 查詢: {task['query'][:50]}, 錯誤: {error_str[:200]}")
            return []

    # 檢查是否有任務
    if not tasks:
        logger.warning(f"沒有搜尋任務，查詢: {query[:50]}")
        return []
    
    logger.info(f"開始執行 {len(tasks)} 個搜尋任務")
    
    # 使用較少的並發數以避免超過 API 速率限制
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_SEARCH_WORKERS, len(tasks))) as executor:
        futures = {executor.submit(fetch, t): t['name'] for t in tasks}
        results_map = {}
        success_count = 0
        error_details = {}  # 記錄錯誤詳情
        
        for future in concurrent.futures.as_completed(futures):
            t_name = futures[future]
            try:
                task_results = future.result(timeout=30)  # 添加超時
                results_map[t_name] = task_results
                if len(task_results) > 0:
                    success_count += 1
                    logger.info(f"任務 {t_name} 成功: {len(task_results)} 筆結果")
                else:
                    logger.warning(f"任務 {t_name} 返回 0 筆結果")
                    error_details[t_name] = "返回 0 筆結果"
            except concurrent.futures.TimeoutError:
                logger.error(f"任務 {t_name} 執行超時（30秒）")
                results_map[t_name] = []
                error_details[t_name] = "執行超時"
            except Exception as e:
                error_str = str(e)
                logger.error(f"任務 {t_name} 執行失敗: {error_str[:300]}")
                results_map[t_name] = []
                error_details[t_name] = error_str[:200]
        
        logger.info(f"搜尋完成: {success_count}/{len(tasks)} 個任務成功取得結果")
        if error_details:
            logger.warning(f"失敗任務詳情: {error_details}")
            
    final_list = []
    
    # A. 優先加入保底
    guards = ["Blue_Guard", "Green_Guard", "Official_Guard"]
    guard_count = 0
    for guard_name in guards:
        if guard_name in results_map:
            guard_results = results_map[guard_name]
            logger.info(f"處理保底搜尋 {guard_name}: {len(guard_results)} 筆結果")
            for item in guard_results:
                # 確保 item 是字典類型
                if not isinstance(item, dict):
                    logger.warning(f"保底搜尋 {guard_name} 的結果項不是字典類型: {type(item).__name__}，跳過")
                    continue
                if item.get('url') and item['url'] not in seen_urls:
                    seen_urls.add(item['url'])
                    final_list.append(item)
                    guard_count += 1
    
    logger.info(f"保底搜尋共加入 {guard_count} 筆結果")
    
    # 🔍 DEBUG: 檢查保底搜尋的原始資料
    for guard_name in guards:
        if guard_name in results_map:
            raw_count = len(results_map[guard_name])
            if raw_count > 0 and guard_count == 0:
                logger.warning(f"⚠️ {guard_name} 有 {raw_count} 筆原始結果，但 0 筆被加入 final_list！")
                # 取樣檢查第一筆資料
                sample = results_map[guard_name][0]
                logger.warning(f"   範例資料類型: {type(sample).__name__}")
                if isinstance(sample, dict):
                    logger.warning(f"   範例資料 keys: {list(sample.keys())[:10]}")
                    logger.warning(f"   有 'url' 欄位: {'url' in sample}")
                    if 'url' in sample:
                        logger.warning(f"   URL 值: {sample.get('url', 'N/A')[:50]}")
    
    # B. 再加入通用搜尋結果（重視正確度，使用所有查詢）
    # 收集所有 General_ 開頭的任務結果
    general_keys = [k for k in results_map.keys() if k.startswith("General_")]
    if general_keys:
        max_len = max([len(results_map.get(k, [])) for k in general_keys])
        logger.info(f"處理通用搜尋: {len(general_keys)} 個任務，最大結果數: {max_len}")
        
        general_count = 0
        for i in range(max_len):
            for key in general_keys:
                if key in results_map and i < len(results_map[key]):
                    item = results_map[key][i]
                    # 確保 item 是字典類型
                    if not isinstance(item, dict):
                        logger.warning(f"通用搜尋 {key} 的結果項不是字典類型: {type(item).__name__}，跳過")
                        continue
                    if item.get('url') and item['url'] not in seen_urls:
                        seen_urls.add(item['url'])
                        final_list.append(item)
                        general_count += 1
        
        logger.info(f"通用搜尋共加入 {general_count} 筆結果")
        
        # 🔍 DEBUG: 檢查通用搜尋的原始資料
        if max_len > 0 and general_count == 0:
            logger.warning(f"⚠️ 通用搜尋有 {max_len} 筆原始結果（跨 {len(general_keys)} 個任務），但 0 筆被加入 final_list！")
            # 取樣檢查第一個任務的第一筆資料
            if general_keys and results_map.get(general_keys[0]):
                sample = results_map[general_keys[0]][0]
                logger.warning(f"   範例資料類型: {type(sample).__name__}")
                if isinstance(sample, dict):
                    logger.warning(f"   範例資料 keys: {list(sample.keys())[:10]}")
                    logger.warning(f"   有 'url' 欄位: {'url' in sample}")
                    if 'url' in sample:
                        logger.warning(f"   URL 值: {sample.get('url', 'N/A')[:50]}")
    else:
        logger.warning("沒有通用搜尋任務結果")

    # C. 國際保底（亞洲/日本/韓國）— 與通用搜尋共用 seen_urls 去重
    other_guards = ["Intl_Asia_Guard", "Japan_Guard", "Korea_Guard"]
    other_guard_count = 0
    for guard_name in other_guards:
        if guard_name in results_map:
            for item in results_map[guard_name]:
                if not isinstance(item, dict):
                    continue
                if item.get("url") and item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    final_list.append(item)
                    other_guard_count += 1
    if other_guard_count:
        logger.info(f"國際保底（亞洲/日本/韓國）共加入 {other_guard_count} 筆結果")

    # C1a. 獨立/自媒體保底結果
    if "Indie_Guard" in results_map:
        indie_count = 0
        for item in results_map["Indie_Guard"]:
            if not isinstance(item, dict):
                continue
            if item.get("url") and item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                final_list.append(item)
                indie_count += 1
        if indie_count:
            logger.info(f"獨立/自媒體保底共加入 {indie_count} 筆結果")

    # C1b. 亞洲日文/韓文檢索結果（日本/韓國在地媒體）— 共用 seen_urls 去重
    asia_lang_keys = [k for k in results_map.keys() if k.startswith("Asia_Japanese_Guard_") or k.startswith("Asia_Korean_Guard_")]
    asia_lang_count = 0
    for key in sorted(asia_lang_keys):
        for item in results_map[key]:
            if not isinstance(item, dict):
                continue
            if item.get("url") and item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                final_list.append(item)
                asia_lang_count += 1
    if asia_lang_count:
        logger.info(f"亞洲日文/韓文檢索共加入 {asia_lang_count} 筆結果")

    # C2. 建議二：區域保底（歐洲/美洲）— 分區子查詢結果，共用 seen_urls 去重
    region_guards = ["Europe_Guard", "Americas_Guard"]
    region_guard_count = 0
    for guard_name in region_guards:
        if guard_name in results_map:
            for item in results_map[guard_name]:
                if not isinstance(item, dict):
                    continue
                if item.get("url") and item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    final_list.append(item)
                    region_guard_count += 1
    if region_guard_count:
        logger.info(f"區域保底（歐洲/美洲）共加入 {region_guard_count} 筆結果")

    # D. 建議一：英文檢索結果（歐洲/美洲非中文）— 與前述任務共用 seen_urls，避免重複
    english_guard_keys = [k for k in results_map.keys() if k.startswith("English_Guard_")]
    english_guard_count = 0
    for key in sorted(english_guard_keys):
        for item in results_map[key]:
            if not isinstance(item, dict):
                continue
            if item.get("url") and item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                final_list.append(item)
                english_guard_count += 1
    if english_guard_count:
        logger.info(f"英文檢索（歐洲/美洲）共加入 {english_guard_count} 筆結果")
    
    logger.info(f"搜尋總結果: {len(final_list)} 筆（去重後）")
    
    if len(final_list) == 0:
        # 詳細記錄所有任務的結果
        detailed_results = []
        for task_name, task_results in results_map.items():
            detailed_results.append(f"{task_name}: {len(task_results)} 筆")
        
        logger.error(f"⚠️ 搜尋結果為空！查詢: {query[:50]}, 任務數: {len(tasks)}, 詳細結果: {', '.join(detailed_results)}")
        st.write(f"⚠️ 原始搜尋 (news 模式及網域限制) 為空，嘗試自動降級至全網通用模式...")
        
        # 降級策略：嘗試多個簡單的搜尋（逐步放寬條件）
        logger.info("嘗試降級策略：執行簡單搜尋（無過濾條件）")
        
        # 策略1：完全無過濾的搜尋（使用原始查詢）
        try:
            simple_params = {
                'query': query,
                'max_results': 20,  # 增加結果數量
                'search_depth': 'basic'  # 使用 basic 以加快速度
            }
            logger.info(f"降級搜尋（策略1）：查詢='{query[:50]}', 參數={simple_params}")
            simple_response = tavily.search(**simple_params)
            
            if isinstance(simple_response, dict):
                if 'error' in simple_response:
                    logger.error(f"降級搜尋 API 錯誤: {simple_response.get('error')}")
                else:
                    simple_results = simple_response.get('results', [])
                    logger.info(f"降級搜尋（策略1）找到 {len(simple_results)} 筆結果")
                    
                    if len(simple_results) > 0:
                        # 加入降級搜尋的結果
                        for item in simple_results:
                            if isinstance(item, dict) and item.get('url') and item['url'] not in seen_urls:
                                seen_urls.add(item['url'])
                                final_list.append(item)
                        logger.info(f"降級搜尋後總結果: {len(final_list)} 筆")
                    else:
                        logger.warning(f"降級搜尋（策略1）也返回 0 筆結果 - 查詢可能太特定")
                        
                        # 策略2：嘗試簡化查詢（改進版：處理中文無空格情況）
                        try:
                            # 1. 嘗試依標點符號或關鍵連詞拆分
                            separators = [' ', ' 與 ', ' 和 ', ' 及 ', '、', '，', '。']
                            simplified_query = query
                            for sep in separators:
                                if sep in query:
                                    parts = [p.strip() for p in query.split(sep) if p.strip()]
                                    if len(parts) >= 1:
                                        simplified_query = parts[0]
                                        break
                            
                            # 2. 如果還是太長且無分割，取前 6 個字（通常是主詞）
                            if len(simplified_query) > 10 and simplified_query == query:
                                simplified_query = query[:8]
                                
                            logger.info(f"降級搜尋（策略2）：簡化查詢 '{simplified_query}'")
                            simple2_params = {
                                'query': simplified_query,
                                'max_results': 15,
                                'search_depth': 'basic',
                                'topic': 'general'  # 降級時使用 general 以跳過新聞時效限制
                            }
                            simple2_response = tavily.search(**simple2_params)
                            if isinstance(simple2_response, dict) and 'results' in simple2_response:
                                simple2_results = simple2_response.get('results', [])
                                logger.info(f"降級搜尋（策略2）找到 {len(simple2_results)} 筆結果")
                                if len(simple2_results) > 0:
                                    for item in simple2_results:
                                        if isinstance(item, dict) and item.get('url') and item['url'] not in seen_urls:
                                            seen_urls.add(item['url'])
                                            final_list.append(item)
                                    logger.info(f"降級搜尋（策略2）後總結果: {len(final_list)} 筆")
                        except Exception as e2:
                            logger.warning(f"降級搜尋（策略2）失敗: {str(e2)[:200]}")
                        
                        # 策略3：嘗試通用測試查詢
                        try:
                            test_query = "台灣新聞" if "台灣" in str(selected_regions) else "news"
                            logger.info(f"降級搜尋（策略3）：使用測試查詢 '{test_query}' 驗證 API")
                            test_response = tavily.search(query=test_query, max_results=5, search_depth='basic')
                            if isinstance(test_response, dict) and 'results' in test_response:
                                test_results = test_response.get('results', [])
                                logger.info(f"測試查詢 '{test_query}' 找到 {len(test_results)} 筆結果")
                                if len(test_results) == 0:
                                    logger.error("⚠️ 測試查詢也返回 0 筆結果，可能是 API 服務異常")
                                else:
                                    logger.info("✅ API 正常，問題可能是查詢關鍵字太特定或時間範圍內無相關內容")
                        except Exception as e3:
                            logger.error(f"測試查詢失敗: {str(e3)[:200]}")
            else:
                logger.error(f"降級搜尋返回非字典類型: {type(simple_response).__name__}, 值: {str(simple_response)[:200]}")
        except Exception as e:
            error_str = str(e)
            logger.error(f"降級搜尋失敗: {error_str[:300]}")
            # 記錄更詳細的錯誤資訊
            if "401" in error_str or "Unauthorized" in error_str or "Invalid API key" in error_str:
                logger.error("❌ API Key 認證失敗 - 請檢查 API Key 是否正確")
            elif "429" in error_str or "rate limit" in error_str.lower():
                logger.error("❌ API 配額用完或超過速率限制")
            elif "400" in error_str or "Bad Request" in error_str:
                logger.error("❌ 查詢參數格式錯誤")
            else:
                logger.error(f"❌ 未知錯誤: {error_str[:200]}")
                
    return final_list

def summarize_content(content: str, max_length: int = 800) -> str:
    """
    快速摘要內容（使用簡單的句子提取）
    
    Args:
        content: 原始內容
        max_length: 目標最大長度
    
    Returns:
        摘要後的內容
    """
    if len(content) <= max_length:
        return content
    
    # 簡單摘要：保留開頭和結尾的重要句子
    sentences = content.split('。')
    if len(sentences) <= 3:
        return content[:max_length]
    
    # 保留前 40% 和後 30% 的句子
    start_count = max(1, int(len(sentences) * 0.4))
    end_count = max(1, int(len(sentences) * 0.3))
    
    summary = '。'.join(sentences[:start_count]) + '。'
    if end_count > 0:
        summary += '...' + '。'.join(sentences[-end_count:])
    
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    
    return summary


def determine_issue_category(query: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    在搜尋前將查詢分類為議題類型，供後續立場平衡與 gap-fill 動態調整。
    
    理論基礎：台灣國內議題適用藍/綠/官方平衡；國際議題應檢視地理/地緣多元性（INTL、CHINA 等）；
    兩岸議題需同時涵蓋台灣陣營 + 中國視角 + 國際視角。
    
    Args:
        query: 使用者查詢字串
        api_key: Google Gemini API Key（可選；無則僅用 regex  fallback）
    
    Returns:
        Dict: {"issue_type": "TAIWAN_DOMESTIC"|"CROSS_STRAIT"|"INTERNATIONAL", "key_actors": ["US", "China", ...]}
    """
    out: Dict[str, Any] = {"issue_type": "TAIWAN_DOMESTIC", "key_actors": []}
    q = (query or "").strip()
    if not q:
        return out

    # --- LLM 分類（僅在有 api_key 時呼叫）---
    if api_key:
        try:
            system_prompt = "Classify the query into one of three issue_types: 'TAIWAN_DOMESTIC', 'CROSS_STRAIT', or 'INTERNATIONAL'. Also extract key_actors. Output ONLY valid JSON: {\"issue_type\": \"...\", \"key_actors\": [\"...\"]}."
            prompt = f"{system_prompt}\n\nQuery: {q[:300]}"
            llm = ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, google_api_key=api_key, temperature=0.0)
            raw = _extract_text_from_llm_content(llm.invoke(prompt).content)
            if raw:
                obj: Dict[str, Any] = {}
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    start = raw.find("{")
                    if start >= 0:
                        depth = 0
                        for i, c in enumerate(raw[start:], start):
                            if c == "{":
                                depth += 1
                            elif c == "}":
                                depth -= 1
                                if depth == 0:
                                    try:
                                        obj = json.loads(raw[start : i + 1])
                                    except json.JSONDecodeError:
                                        pass
                                    break
                if isinstance(obj, dict):
                    it = (obj.get("issue_type") or "").strip().upper()
                    if it in ("TAIWAN_DOMESTIC", "CROSS_STRAIT", "INTERNATIONAL"):
                        out["issue_type"] = it
                    actors = obj.get("key_actors")
                    if isinstance(actors, list):
                        out["key_actors"] = [str(a).strip() for a in actors if a]
        except Exception as e:
            logger.debug(f"determine_issue_category LLM 失敗，改用 regex: {e}")

    # --- Regex/Keyword fallback（無 API 或 LLM 失敗時）---
    if out["issue_type"] == "TAIWAN_DOMESTIC":
        q_lower = q.lower()
        # INTERNATIONAL: us election, ukraine, russia, nato, 烏克蘭, 俄羅斯, 北約, 國際經濟, 中東, israel, gaza
        intl_keywords = [
            "us election", "ukraine", "russia", "nato", "烏克蘭", "俄羅斯", "北約",
            "國際經濟", "中東", "israel", "gaza"
        ]
        for kw in intl_keywords:
            if kw in q_lower:
                out["issue_type"] = "INTERNATIONAL"
                break
        # CROSS_STRAIT: 兩岸, 台海, 美中, ecfa, 共機, 武統, 北京, 中國觀點
        if out["issue_type"] == "TAIWAN_DOMESTIC":
            cross_strait_keywords = ["兩岸", "台海", "美中", "ecfa", "共機", "武統", "北京", "中國觀點"]
            if any(kw in q_lower for kw in cross_strait_keywords):
                out["issue_type"] = "CROSS_STRAIT"
            else:
                # TAIWAN_DOMESTIC (default): 選舉, 立委, 藍綠, 民進黨, 國民黨, 柯文哲, 侯友宜, 賴清德
                domestic_keywords = ["選舉", "立委", "藍綠", "民進黨", "國民黨", "柯文哲", "侯友宜", "賴清德"]
                if any(kw in q_lower for kw in domestic_keywords):
                    out["issue_type"] = "TAIWAN_DOMESTIC"

    return out


def analyze_stance_balance(sources: List[Dict], issue_type: str = "TAIWAN_DOMESTIC") -> Dict[str, Any]:
    """
    系統化立場分析框架（方案 2.1 + Phase 3 動態議題類型）。
    
    依 issue_type 動態決定平衡標準：
    - TAIWAN_DOMESTIC：沿用藍/綠/官方門檻與缺失檢測。
    - INTERNATIONAL：檢視地理/地緣多元性（INTL、CHINA、NEUTRAL）；若僅有台灣媒體則缺失 INTL_PERSPECTIVE。
    - CROSS_STRAIT：要求藍/綠/官方 + 中國視角 + 國際視角之混合。
    
    Returns:
        Dict: 包含立場分佈、平衡度評估、建議補充來源（missing_stances 依議題類型語意化）
    """
    stance_analysis: Dict[str, Any] = {
        "camp_distribution": {},
        "balance_score": 0.0,
        "missing_stances": [],
        "recommendations": []
    }
    if not sources:
        return stance_analysis

    camp_counts = Counter()
    for source in sources:
        category = source.get("source_category", "OTHER")
        camp_counts[category] += 1
    stance_analysis["camp_distribution"] = dict(camp_counts)

    it = (issue_type or "TAIWAN_DOMESTIC").strip().upper()
    if it not in ("TAIWAN_DOMESTIC", "CROSS_STRAIT", "INTERNATIONAL"):
        it = "TAIWAN_DOMESTIC"

    taiwan_camps = ["BLUE", "GREEN", "OFFICIAL"]
    taiwan_counts = {c: camp_counts.get(c, 0) for c in taiwan_camps}
    intl_count = camp_counts.get("INTL", 0)
    china_count = camp_counts.get("CHINA", 0)
    taiwan_total = sum(taiwan_counts.values())

    if it == "TAIWAN_DOMESTIC":
        # 原有邏輯：藍綠官方平衡
        if taiwan_total > 0:
            blue_count = taiwan_counts.get("BLUE", 0)
            green_count = taiwan_counts.get("GREEN", 0)
            official_count = taiwan_counts.get("OFFICIAL", 0)
            total_political = blue_count + green_count
            if total_political > 0:
                blue_ratio = blue_count / total_political
                green_ratio = green_count / total_political
                balance_ratio = min(blue_ratio, green_ratio) / max(blue_ratio, green_ratio) if max(blue_ratio, green_ratio) > 0 else 0
                stance_analysis["balance_score"] = balance_ratio * 0.6 + (1 if official_count > 0 else 0) * 0.4
            else:
                stance_analysis["balance_score"] = 0.4 if official_count > 0 else 0.2
            min_threshold = max(2, int(taiwan_total * 0.15))
            if blue_count < min_threshold and green_count >= blue_count * 2:
                stance_analysis["missing_stances"].append("BLUE")
                stance_analysis["recommendations"].append({"type": "BLUE", "reason": f"藍營觀點不足（僅 {blue_count} 篇，綠營 {green_count} 篇）", "priority": "高"})
            if green_count < min_threshold and blue_count >= green_count * 2:
                stance_analysis["missing_stances"].append("GREEN")
                stance_analysis["recommendations"].append({"type": "GREEN", "reason": f"綠營觀點不足（僅 {green_count} 篇，藍營 {blue_count} 篇）", "priority": "高"})
            if official_count == 0 and len(sources) > 5:
                stance_analysis["missing_stances"].append("OFFICIAL")
                stance_analysis["recommendations"].append({"type": "OFFICIAL", "reason": "缺少官方/中立觀點", "priority": "中"})
        else:
            stance_analysis["balance_score"] = 0.5

    elif it == "INTERNATIONAL":
        total = len(sources)
        tw_media_count = taiwan_counts.get("BLUE", 0) + taiwan_counts.get("GREEN", 0) + taiwan_counts.get("OFFICIAL", 0) + camp_counts.get("NEUTRAL", 0)
        tw_media_ratio = tw_media_count / total if total > 0 else 0.0
        if total > 0:
            if tw_media_ratio >= 0.8 and intl_count == 0:
                stance_analysis["missing_stances"].append("INTL_PERSPECTIVE")
                stance_analysis["recommendations"].append({"type": "INTL_PERSPECTIVE", "reason": "國際議題但多為台灣媒體，缺少國際視角", "priority": "高"})
            # balance_score 依 INTL 來源是否存在
            stance_analysis["balance_score"] = 0.7 if intl_count > 0 else 0.3
        else:
            stance_analysis["balance_score"] = 0.5

    else:
        # CROSS_STRAIT：藍/綠/官方 + CHINA + INTL
        score = 0.0
        if taiwan_total > 0:
            blue_count = taiwan_counts.get("BLUE", 0)
            green_count = taiwan_counts.get("GREEN", 0)
            official_count = taiwan_counts.get("OFFICIAL", 0)
            total_political = blue_count + green_count
            if total_political > 0:
                balance_ratio = min(blue_count, green_count) / max(blue_count, green_count) if max(blue_count, green_count) > 0 else 0
                score += balance_ratio * 0.3
            if official_count > 0:
                score += 0.2
        if china_count > 0:
            score += 0.25
        else:
            stance_analysis["missing_stances"].append("CHINA")
            stance_analysis["recommendations"].append({"type": "CHINA", "reason": "兩岸議題缺少中國/北京視角", "priority": "高"})
        if intl_count > 0:
            score += 0.25
        else:
            stance_analysis["missing_stances"].append("INTL_PERSPECTIVE")
            stance_analysis["recommendations"].append({"type": "INTL_PERSPECTIVE", "reason": "兩岸議題缺少國際第三方視角", "priority": "中"})
        # 藍綠缺口（兩岸議題仍建議台灣內部平衡）
        min_threshold = max(2, int(taiwan_total * 0.15))
        blue_count = taiwan_counts.get("BLUE", 0)
        green_count = taiwan_counts.get("GREEN", 0)
        if blue_count < min_threshold and green_count >= blue_count * 2:
            stance_analysis["missing_stances"].append("BLUE")
            stance_analysis["recommendations"].append({"type": "BLUE", "reason": f"藍營觀點不足（僅 {blue_count} 篇）", "priority": "中"})
        if green_count < min_threshold and blue_count >= green_count * 2:
            stance_analysis["missing_stances"].append("GREEN")
            stance_analysis["recommendations"].append({"type": "GREEN", "reason": f"綠營觀點不足（僅 {green_count} 篇）", "priority": "中"})
        stance_analysis["balance_score"] = min(1.0, score + 0.2)

    return stance_analysis

def process_source_item(res: Dict, index: int) -> Tuple[str, Dict]:
    """
    並行處理單一來源項目（已整合公信力評分）
    
    Returns:
        Tuple[str, Dict]: (context 文字行, 處理後的結果字典)
    """
    title = res.get('title', 'No Title')
    url = res.get('url', '#')
    
    pub_date = res.get('published_date')
    if not pub_date:
        url_date = extract_date_from_url(url)
        pub_date = url_date if url_date else "Missing"
    else:
        pub_date = pub_date[:10]
    
    res['final_date'] = pub_date
    content = res.get('content', '')
    
    # 重視正確度：使用完整內容長度
    content = content[:MAX_CONTENT_LENGTH]
    # 如果內容太長，進行摘要
    if len(res.get('content', '')) > SUMMARY_THRESHOLD:
        content = summarize_content(content, MAX_CONTENT_LENGTH)
    
    # 學術級證據強度分級（需要所有來源用於交叉驗證，但在這裡只能使用現有來源）
    source_category = classify_source(url)
    domain = get_domain_name(url)
    
    # 注意：在 process_source_item 階段，all_sources 可能還未完全構建
    # 完整的交叉驗證會在後續階段進行
    evidence_level, evidence_score, evidence_details = calculate_academic_evidence_level(
        url, source_category, content, title, all_sources=None
    )
    
    # 來源公信力評分（方案 2）
    credibility_score, tier = _reputation_manager.calculate_credibility_score(url, domain)
    weight_coefficient = _reputation_manager.get_weight_coefficient(source_category, domain)
    
    res['evidence_level'] = evidence_level
    res['evidence_score'] = evidence_score
    res['evidence_details'] = evidence_details
    res['source_category'] = source_category
    res['credibility_score'] = credibility_score
    res['credibility_tier'] = tier
    res['weight_coefficient'] = weight_coefficient
    
    # RAG 權重應用（方案 2.3）：根據公信力調整內容長度
    if credibility_score >= 0.8:
        # 高公信力：完整展示
        prefix = "[高可信度來源] "
        adjusted_content_length = MAX_CONTENT_LENGTH
    elif credibility_score >= 0.6:
        # 中等：正常展示
        prefix = ""
        adjusted_content_length = MAX_CONTENT_LENGTH
    else:
        # 低公信力：縮短並標註
        prefix = "[⚠️ 低可信度，請謹慎參考] "
        adjusted_content_length = MAX_CONTENT_LENGTH // 2
    
    # 確保內容不超過調整後的長度
    if len(content) > adjusted_content_length:
        content = content[:adjusted_content_length] + "..."
    
    # 優化 context 格式（包含公信力標註與語言風格警示）
    evidence_label = evidence_details.get('level', 'B')
    
    # 添加語言風格警示標記（改進項目：新增）
    language_flags = evidence_details.get('language_style', {}).get('flags', [])
    style_warning = ""
    if language_flags:
        # 只顯示最重要的警示（最多2個）
        critical_flags = [f for f in language_flags if '⚠️' in f][:2]
        if critical_flags:
            style_warning = " " + " ".join(critical_flags)
    
    context_line = f"Source {index+1}: [Date: {pub_date}] [Evidence: {evidence_level} ({evidence_label}, {evidence_score:.2f})] [Credibility: {credibility_score:.2f}] {prefix}[Title: {title}]{style_warning} {content} (URL: {url})\n"
    
    return context_line, res

def get_search_context(
    query: str,
    api_key_tavily: str,
    days_back: int,
    selected_regions: List[str],
    max_results: int,
    dynamic_keywords: List[str],
    use_cache: bool = True,
    google_api_key: str = None,
    enable_english_for_regions: bool = True,
    enable_google_fact_check: bool = False,
):
    """
    獲取搜尋上下文（完整版 - 整合事實查核、公信力評分、平衡檢索）
    
    Args:
        google_api_key: 用於事實查核的 Google API Key
        enable_english_for_regions: 建議三；當勾選歐洲/美洲時是否自動加入英文關鍵字檢索（預設 True）
        enable_google_fact_check: 是否啟用 Google Fact Check Tools 查核管線（預設 False 以節省配額）
    
    Returns:
        Tuple: (context_text, results, query, is_strict_mode, stance_analysis, fact_check_results, consensus_analysis)
    """
    try:
        # === Phase 3：議題類型判定（用於後續立場平衡與 gap-fill 動態調整）===
        issue_category = determine_issue_category(query, google_api_key)
        current_issue_type = issue_category.get("issue_type", "TAIWAN_DOMESTIC")

        active_blacklist = NOISE_BLACKLIST

        # 只包含 Tavily API 支持的參數
        search_params = {
            "search_depth": "advanced",  # 重視正確度，始終使用 advanced
            "max_results": max_results,
            "days": days_back,           # 正確傳遞時間範圍參數
        }
        
        # 條件性添加 exclude_domains（如果黑名單不為空）
        if active_blacklist and len(active_blacklist) > 0:
            search_params["exclude_domains"] = active_blacklist
        
        # 注意：Tavily API 不支持 "topic", "days", "selected_regions" 參數
        # 這些參數會在 execute_hybrid_search 中通過 include_domains 來實現區域過濾

        # 嘗試從快取獲取
        cached_results = None
        if use_cache:
            cached_results = get_cached_results(query, search_params)

        is_strict_mode = bool(selected_regions)
        
        if cached_results:
            results = cached_results
            logger.info(f"使用快取結果: {len(results)} 篇")
        else:
            # === 多維度平衡檢索（方案 3，優化：使用快取）===
            # 生成平衡查詢（使用快取減少 API 調用）
            if google_api_key:
                balanced_queries = generate_balanced_queries(query, google_api_key, use_cache=use_cache)
            else:
                # 如果沒有 API Key，使用降級策略
                balanced_queries = {
                    "pro_arguments": [f"{query} 支持 優點", f"{query} 贊成 好處"],
                    "con_arguments": [f"{query} 反對 缺點", f"{query} 批評 風險"],
                    "neutral_analysis": [f"{query} 研究", f"{query} 數據分析", f"{query} 學術"],
                    "factual_timeline": [f"{query} 時間軸", f"{query} 發展歷程"]
                }
            
            # 確保 balanced_queries 是字典類型（加強檢查）
            if not isinstance(balanced_queries, dict):
                logger.warning(f"balanced_queries 不是字典類型: {type(balanced_queries).__name__}，值: {str(balanced_queries)[:100]}，使用降級策略")
                balanced_queries = {
                    "pro_arguments": [f"{query} 支持 優點", f"{query} 贊成 好處"],
                    "con_arguments": [f"{query} 反對 缺點", f"{query} 批評 風險"],
                    "neutral_analysis": [f"{query} 研究", f"{query} 數據分析", f"{query} 學術"],
                    "factual_timeline": [f"{query} 時間軸", f"{query} 發展歷程"]
                }
            
            # 再次確認（防禦性編程）
            if not isinstance(balanced_queries, dict):
                logger.error(f"balanced_queries 仍然不是字典類型，強制使用降級策略")
                balanced_queries = {
                    "pro_arguments": [f"{query} 支持 優點", f"{query} 贊成 好處"],
                    "con_arguments": [f"{query} 反對 缺點", f"{query} 批評 風險"],
                    "neutral_analysis": [f"{query} 研究", f"{query} 數據分析", f"{query} 學術"],
                    "factual_timeline": [f"{query} 時間軸", f"{query} 發展歷程"]
                }
            
            # 合併到擴展查詢列表（優化：大幅減少查詢數量以避免 429 錯誤）
            balanced_expanded = []
            pro_args = balanced_queries.get("pro_arguments", []) if isinstance(balanced_queries, dict) else []
            # 大幅減少到最多 1 個查詢（從 2 減少到 1），避免 429 錯誤
            for q in pro_args[:1]:
                balanced_expanded.append({"query": q, "type": "正方觀點", "priority": 1, "perspective": "pro"})
            con_args = balanced_queries.get("con_arguments", []) if isinstance(balanced_queries, dict) else []
            for q in con_args[:1]:
                balanced_expanded.append({"query": q, "type": "反方觀點", "priority": 1, "perspective": "con"})
            neutral_args = balanced_queries.get("neutral_analysis", []) if isinstance(balanced_queries, dict) else []
            for q in neutral_args[:1]:
                balanced_expanded.append({"query": q, "type": "中立分析", "priority": 1, "perspective": "neutral"})
            
            # 合併原有查詢和平衡查詢
            all_queries = dynamic_keywords + balanced_expanded
            logger.info(f"開始執行混和搜尋: 查詢={query[:50]}, 擴展查詢數={len(all_queries)}, strict_mode={is_strict_mode}, selected_regions={selected_regions}")

            # 建議一＋建議三：選歐洲/美洲時可自動產出英文關鍵字並進行非中文檢索（可由 UI 核取方塊關閉）
            english_queries: List[str] = []
            if enable_english_for_regions and google_api_key and selected_regions:
                _sel_str = str(selected_regions)
                if "歐洲" in _sel_str or "美洲" in _sel_str:
                    english_queries = translate_queries_to_english(
                        query, all_queries, google_api_key, max_english_queries=3
                    )
                    if english_queries:
                        logger.info(f"已產出 {len(english_queries)} 條英文查詢，將用於歐洲/美洲非中文檢索")
            # 亞洲視角：產出日文與韓文關鍵字，用於日本/韓國在地媒體檢索
            japanese_queries: List[str] = []
            korean_queries: List[str] = []
            if google_api_key and selected_regions and "亞洲" in str(selected_regions):
                ja_ko = translate_queries_to_japanese_korean(
                    query, all_queries, google_api_key, max_per_language=2
                )
                japanese_queries = ja_ko.get("ja") or []
                korean_queries = ja_ko.get("ko") or []
                if japanese_queries or korean_queries:
                    logger.info(f"已產出日文 {len(japanese_queries)} 條、韓文 {len(korean_queries)} 條查詢，將用於亞洲日韓檢索")
            # 建議三：寫入上次英文/日文/韓文查詢供策略表顯示（僅在非快取路徑更新）
            try:
                st.session_state["last_english_queries"] = english_queries
                st.session_state["last_japanese_queries"] = japanese_queries
                st.session_state["last_korean_queries"] = korean_queries
            except Exception:
                pass
            
            # 如果 strict_mode 且選定了區域，先嘗試無網域過濾的測試搜尋
            if is_strict_mode and selected_regions:
                logger.info("檢測到嚴格模式，先執行無網域過濾的測試搜尋...")
                try:
                    test_tavily = TavilyClient(api_key=api_key_tavily)
                    test_params = {
                        'query': query,
                        'max_results': 5,
                        'search_depth': 'basic'
                    }
                    test_response = test_tavily.search(**test_params)
                    if isinstance(test_response, dict):
                        test_results = test_response.get('results', [])
                        logger.info(f"無網域過濾測試搜尋: 找到 {len(test_results)} 筆結果")
                        if len(test_results) == 0:
                            logger.warning(f"⚠️ 即使無網域過濾，查詢 '{query[:50]}' 也返回 0 筆結果")
                            logger.warning(f"   這可能表示：1) 查詢關鍵字太特定 2) 時間範圍內無相關內容 3) API 服務問題")
                except Exception as e:
                    logger.warning(f"測試搜尋失敗: {str(e)[:200]}")
            
            results = execute_hybrid_search(
                query, api_key_tavily, search_params, is_strict_mode, all_queries, selected_regions,
                english_queries=english_queries,
                japanese_queries=japanese_queries,
                korean_queries=korean_queries,
            )
            
            logger.info(f"混和搜尋完成: 取得 {len(results)} 筆結果")
            
            if len(results) == 0:
                logger.error(f"⚠️ 混和搜尋返回 0 筆結果！")
                logger.error(f"   查詢: {query[:50]}")
                logger.error(f"   擴展查詢: {[q.get('query', q) if isinstance(q, dict) else q for q in all_queries[:5]]}")
                logger.error(f"   嚴格模式: {is_strict_mode}, 選定區域: {selected_regions}")
                logger.error(f"   API Key 前綴: {api_key_tavily[:10] if api_key_tavily else 'None'}...")
            
            # 存入快取
            if use_cache and results:
                cache_results(query, search_params, results)
        
        results.sort(key=lambda x: x.get('published_date') or "", reverse=True)
        results = results[:max_results]
        
        # 標註來源立場（基於查詢類型）
        perspective_map = {}  # source_id -> perspective
        for i, result in enumerate(results):
            # 根據來源標題和內容推斷立場（簡單實現）
            # 可以進一步使用 LLM 進行更精確的立場檢測
            perspective_map[i+1] = "neutral"  # 預設中立
        
        # 並行處理所有來源（應用公信力評分）
        context_lines = []
        if len(results) > 10:  # 大量來源時使用並行處理
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(results))) as executor:
                futures = {executor.submit(process_source_item, res, i): i for i, res in enumerate(results)}
                processed_results = [None] * len(results)
                
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    try:
                        context_line, processed_res = future.result()
                        context_lines.append(context_line)
                        processed_results[idx] = processed_res
                    except Exception as e:
                        logger.warning(f"處理來源 {idx} 失敗: {str(e)}")
                        processed_results[idx] = results[idx]
                        context_lines.append(f"Source {idx+1}: [Date: Missing] [Title: {results[idx].get('title', 'No Title')}] (URL: {results[idx].get('url', '#')})\n")
                
                results = [r for r in processed_results if r is not None]
        else:
            # 少量來源時使用串行處理
            for i, res in enumerate(results):
                context_line, processed_res = process_source_item(res, i)
                context_lines.append(context_line)
                results[i] = processed_res
        
        context_text = "".join(context_lines)
        
        # === 事實查核驗證（方案 1，可選功能）===
        # 真正的 claim-based Google Fact Check 管線會在 gap-fill 完成後執行，避免漏掉補足來源。
        fact_check_results = None
        
        # === 立場平衡分析（方案 2.1 + Phase 3 依議題類型動態評估）===
        stance_analysis = analyze_stance_balance(results, issue_type=current_issue_type)
        
        # === Phase 2: 主動補足機制（迭代式平衡搜尋）===
        MAX_GAP_FILL_ITERATIONS = 2  # 最多補充 2 次，控制 API 成本
        gap_fill_iteration = 0
        
        while gap_fill_iteration < MAX_GAP_FILL_ITERATIONS:
            # 檢查是否有缺口需要補足
            missing_stances = stance_analysis.get("missing_stances", [])
            balance_score = stance_analysis.get("balance_score", 1.0)
            
            # 如果平衡度已達標或無缺口，停止迭代
            if balance_score >= 0.7 or not missing_stances:
                if gap_fill_iteration > 0:
                    logger.info(f"經過 {gap_fill_iteration} 次補充搜尋後達到平衡（平衡度: {balance_score:.2f}）")
                break
            
            gap_fill_iteration += 1
            logger.info(f"檢測到立場缺口: {missing_stances}，執行第 {gap_fill_iteration} 次補充搜尋（平衡度: {balance_score:.2f}）")
            
            # 生成針對缺失立場的補充關鍵字（Phase 3：依議題類型與 missing_stances 動態生成）
            gap_fill_keywords = []
            for stance in missing_stances:
                if stance == "BLUE" and google_api_key:
                    gap_fill_keywords.extend([
                        f"{query} 國民黨 觀點",
                        f"{query} 保守派 看法",
                        f"{query} 藍營 立場"
                    ])
                elif stance == "GREEN" and google_api_key:
                    gap_fill_keywords.extend([
                        f"{query} 民進黨 觀點",
                        f"{query} 進步派 看法",
                        f"{query} 綠營 立場"
                    ])
                elif stance == "OFFICIAL":
                    gap_fill_keywords.extend([
                        f"{query} 政府 官方 聲明",
                        f"{query} 官方 新聞稿"
                    ])
                elif stance == "INTL_PERSPECTIVE":
                    gap_fill_keywords.extend([
                        f"{query} global view",
                        f"{query} international news",
                        f"{query} 國際 觀點",
                        f"{query} 外電"
                    ])
                elif stance == "CHINA":
                    gap_fill_keywords.extend([
                        f"{query} 中國 觀點",
                        f"{query} 北京 立場",
                        f"{query} 大陸 看法"
                    ])
            
            if not gap_fill_keywords:
                gap_fill_keywords = [f"{query} {stance}" for stance in missing_stances]
            
            logger.info(f"生成補充關鍵字: {gap_fill_keywords[:3]}...")
            
            # 執行補充搜尋
            gap_fill_results = []
            for kw in gap_fill_keywords[:3]:  # 限制最多 3 個補充查詢
                try:
                    tavily_client = TavilyClient(api_key=api_key_tavily)
                    resp = tavily_client.search(
                        query=kw,
                        max_results=5,
                        search_depth='basic',
                        topic='general'
                    )
                    if isinstance(resp, dict) and 'results' in resp:
                        gap_fill_results.extend(resp.get('results', []))
                except Exception as e:
                    logger.warning(f"補充搜尋失敗: {str(e)[:100]}")
            
            # 去重並整合補充結果
            existing_urls = {r.get('url') for r in results}
            new_results = [r for r in gap_fill_results if r.get('url') and r.get('url') not in existing_urls]
            
            if new_results:
                logger.info(f"補充搜尋獲得 {len(new_results)} 筆新結果")
                results.extend(new_results)
                stance_analysis = analyze_stance_balance(results, issue_type=current_issue_type)
            else:
                logger.warning(f"補充搜尋未獲得新結果，停止迭代")
                break
        
        # === 關鍵修復：如果經過補充搜尋，重新生成 context_text ===
        # 修復原因：gap-filling 迭代會新增結果到 results，但 context_text 在迭代前已生成
        # 導致 context_text 和 results 不同步，造成分析錯誤和重複資料
        if gap_fill_iteration > 0:
            logger.info(f"經過 {gap_fill_iteration} 次補充搜尋，重新生成完整 context_text")
            # 重新處理所有來源（包含補充的新來源）
            context_lines = []
            if len(results) > 10:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(results))) as executor:
                    futures = {executor.submit(process_source_item, res, i): i for i, res in enumerate(results)}
                    processed_results = [None] * len(results)
                    
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        try:
                            context_line, processed_res = future.result()
                            context_lines.append(context_line)
                            processed_results[idx] = processed_res
                        except Exception as e:
                            logger.warning(f"處理來源 {idx} 失敗: {str(e)}")
                            processed_results[idx] = results[idx]
                            context_lines.append(f"Source {idx+1}: [Date: Missing] [Title: {results[idx].get('title', 'No Title')}] (URL: {results[idx].get('url', '#')})\n")
                    
                    results = [r for r in processed_results if r is not None]
            else:
                for i, res in enumerate(results):
                    context_line, processed_res = process_source_item(res, i)
                    context_lines.append(context_line)
                    results[i] = processed_res
            
            # 重新生成 context_text
            context_text = "".join(context_lines)
            logger.info(f"已重新生成 context_text，包含 {len(results)} 筆完整來源")

        if enable_google_fact_check and google_api_key and results:
            try:
                logger.info("啟用 Google Fact Check：開始抽取聲明並查核")
                claims = extract_claims_from_sources(results[:20], google_api_key)
                fact_check_results = verify_claims(claims, google_api_key)
                results = apply_fact_check_tags(results, fact_check_results)
                fact_check_warning = generate_fact_check_warning(fact_check_results)
                if fact_check_warning:
                    context_text += f"\n{fact_check_warning}\n"
                logger.info(
                    "Google Fact Check 完成：已證偽 %s，誤導 %s，未驗證 %s",
                    len(fact_check_results.get("false_claims", [])),
                    len(fact_check_results.get("misleading_claims", [])),
                    len(fact_check_results.get("unverified_claims", [])),
                )
            except Exception as e:
                logger.warning(f"Google Fact Check 管線失敗，略過查核結果: {str(e)[:200]}")
                fact_check_results = {
                    "verified_claims": [],
                    "false_claims": [],
                    "misleading_claims": [],
                    "unverified_claims": [],
                    "error": str(e)[:300],
                }
        
        # === 共識分析（方案 3.3 - LLM 增強版）===
        # 分類來源為不同立場
        perspective_sources = {
            "pro_sources": [],
            "con_sources": [],
            "neutral_sources": [],
            "factual_sources": results  # 所有來源都可作為事實來源
        }
        # 傳遞 api_key 和 query 以啟用 LLM 分析
        consensus_analysis = analyze_consensus(perspective_sources, api_key=google_api_key, query=query)
        
        # === Phase 2：跨網域聯播與敘事擴散偵測（CIB / 洗稿）===
        manipulation_signals_text = ""
        try:
            syndication_clusters = detect_cross_domain_syndication(results)
            diffusion_metrics = track_narrative_diffusion(syndication_clusters, results)
            if diffusion_metrics:
                parts = []
                for i, m in enumerate(diffusion_metrics, 1):
                    n_articles = len(m.get("source_indices", []))
                    n_domains = m.get("unique_domain_count", 0)
                    velocity = m.get("velocity", 0)
                    risk = m.get("risk_level", "low")
                    risk_label = "高" if risk == "high" else "中" if risk == "medium" else "低"
                    flag = "🚨" if m.get("is_high_risk_coordinated") else "⚠️"
                    parts.append(
                        f"{flag} 群組 {i}：{n_articles} 篇高度相似文章，橫跨 {n_domains} 個不同網域，"
                        f"擴散速度 {velocity:.2f} 網域/小時。風險等級：{risk_label}。"
                    )
                manipulation_signals_text = "【MANIPULATION_SIGNALS】\n" + "偵測到跨網域洗稿網路：\n" + "\n".join(parts)
            else:
                manipulation_signals_text = "【MANIPULATION_SIGNALS】\n本輪未偵測到跨網域聯播群組（無高風險協調敘事信號）。"
            # Phase 4: 語義旋轉偵測（失敗不影響主流程）
            try:
                spin_analysis = detect_semantic_spin(syndication_clusters, results, google_api_key)
                if spin_analysis and float(spin_analysis.get("spin_score", 0)) > 0.6:
                    spin_score = spin_analysis.get("spin_score", 0)
                    shared_fact = spin_analysis.get("shared_fact", "")
                    spin_a = spin_analysis.get("spin_a", "")
                    spin_b = spin_analysis.get("spin_b", "")
                    spin_text = (
                        f"\n\n【語義旋轉偵測 (Semantic Spin)】\n"
                        f"⚠️ 偵測到高對立敘事框架 (Spin Score: {spin_score})。\n"
                        f"- 共享事實：{shared_fact}\n"
                        f"- 敘事 A 框架：{spin_a}\n"
                        f"- 敘事 B 框架：{spin_b}"
                    )
                    manipulation_signals_text += spin_text
            except Exception as e:
                logger.warning(f"語義旋轉偵測跳過: {str(e)}")
            try:
                coordination_analysis = detect_coordinated_behavior(results)
                coordination_score = coordination_analysis.get("coordination_score", 0.0)
                coordination_flags = coordination_analysis.get("flags", [])
                if coordination_score > 0 or coordination_flags:
                    flag_text = "\n".join(f"- {flag}" for flag in coordination_flags) if coordination_flags else "- 未達警示門檻"
                    manipulation_signals_text += (
                        f"\n\n【傳統協調行為指標】\n"
                        f"協調性分數：{coordination_score:.2f}。\n"
                        f"重複論述比例：{coordination_analysis.get('duplicate_ratio', 0):.2f}。\n"
                        f"來源集中度：{coordination_analysis.get('domain_concentration', 0):.2f}。\n"
                        f"{flag_text}"
                    )
            except Exception as e:
                logger.warning(f"傳統協調行為偵測跳過: {str(e)}")
        except Exception as e:
            logger.warning(f"跨網域聯播/擴散偵測失敗，不注入操作信號: {e}")
            manipulation_signals_text = "【MANIPULATION_SIGNALS】\n（本輪操作信號因技術原因未產生，請依既有來源分析。）"
            
        return context_text, results, query, is_strict_mode, stance_analysis, fact_check_results, consensus_analysis, manipulation_signals_text
        
    except Exception as e:
        logger.error(f"搜尋上下文獲取失敗: {str(e)}")
        return f"Error: {str(e)}", [], "Error", False, None, None, None, ""

def validate_google_api_key(google_key: str) -> Tuple[bool, str]:
    """僅驗證 Google Gemini API Key。"""
    if not (google_key or "").strip():
        return False, "未提供 Gemini API Key"
    try:
        os.environ["GOOGLE_API_KEY"] = (google_key or "").strip()
        llm = ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, google_api_key=(google_key or "").strip(), temperature=0.0)
        test_response = llm.invoke("test")
        if not test_response or not test_response.content:
            return False, "Gemini API Key 無效：無法取得回應"
    except Exception as e:
        logger.error(f"Gemini API 驗證失敗: {str(e)}")
        return False, f"Gemini API Key 無效：{str(e)[:100]}"
    return True, "✅ Gemini API Key 驗證通過"


def validate_tavily_api_key(tavily_key: str) -> Tuple[bool, str]:
    """驗證 Tavily API Key（需網路）。"""
    if not (tavily_key or "").strip():
        return False, "未提供 Tavily API Key"
    try:
        tavily = TavilyClient(api_key=(tavily_key or "").strip())
        test_results = tavily.search(query="台灣新聞", max_results=1, search_depth="basic")
        if not test_results:
            return False, "Tavily API Key 無效：API 返回空結果"
        results = test_results.get('results', [])
        if len(results) == 0:
            logger.warning("Tavily API 測試搜尋返回 0 筆結果，可能是配額問題")
            return False, "Tavily API 測試搜尋無結果（可能是配額用完或服務異常）"
    except Exception as e:
        error_str = str(e)
        logger.error(f"Tavily API 驗證失敗: {error_str}")
        if "401" in error_str or "Unauthorized" in error_str or "Invalid API key" in error_str:
            return False, "Tavily API Key 無效：認證失敗"
        elif "429" in error_str or "rate limit" in error_str.lower():
            return False, "Tavily API 配額已用完或超過速率限制"
        elif "500" in error_str or "Internal Server Error" in error_str:
            return False, "Tavily API 服務暫時不可用（伺服器錯誤）"
        else:
            return False, f"Tavily API 驗證失敗：{error_str[:100]}"
    return True, "✅ Tavily API Key 驗證通過"


def validate_api_keys(google_key: str, tavily_key: str, require_tavily: bool = True) -> Tuple[bool, str]:
    """
    驗證 API Key。require_tavily=False 時僅驗證 Gemini（適用新聞文本分析／方法論／本次修改頁）。
    """
    ok, msg = validate_google_api_key(google_key)
    if not ok:
        return False, msg
    if not require_tavily:
        return True, "✅ Gemini API Key 驗證通過（未檢查 Tavily）"
    ok2, msg2 = validate_tavily_api_key(tavily_key)
    if not ok2:
        return False, msg2
    return True, "✅ Gemini 與 Tavily API Key 驗證通過"


def call_gemini(system_prompt: str, user_text: str, model_name: str, api_key: str) -> str:
    """
    呼叫 Google Gemini。模型不可用或配額不足時，僅在 **Gemini 型號清單** 內嘗試降級（不使用其他供應商）。
    """
    os.environ["GOOGLE_API_KEY"] = api_key

    try:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        chain = prompt | llm
        response = chain.invoke({"input": user_text})
        return _extract_text_from_llm_content(response.content)
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__

        if "NOT_FOUND" in error_msg or ("404" in error_msg and "not found" in error_msg.lower()):
            logger.warning(f"模型 {model_name} 不存在或不可用，嘗試降級到可用 Gemini 型號")
            fallback_models = [m for m in GEMINI_FALLBACK_MODELS if m != model_name]
            for fallback_model in fallback_models:
                try:
                    llm = ChatGoogleGenerativeAI(model=fallback_model, temperature=0.0)
                    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
                    chain = prompt | llm
                    response = chain.invoke({"input": user_text})
                    return _extract_text_from_llm_content(response.content)
                except Exception:
                    logger.warning(f"降級到 {fallback_model} 失敗，嘗試下一個")
                    continue
            err = (
                f"❌ 模型 {model_name} 不存在或不可用\n\n"
                f"**錯誤類型**：{error_type}\n**訊息**：{error_msg[:200]}\n\n"
                f"請改用側欄中的其他 Gemini 3.x 型號，或至 AI Studio 確認可用列表。"
            )
            raise ChatGoogleGenerativeAIError(err) from e

        if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg or "quota" in error_msg.lower():
            fallback_models = [m for m in GEMINI_MODEL_OPTIONS if m != model_name]
            if fallback_models and ("pro" in model_name.lower() or "flash" in model_name.lower()):
                for fallback_model in fallback_models:
                    try:
                        llm = ChatGoogleGenerativeAI(model=fallback_model, temperature=0.0)
                        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
                        chain = prompt | llm
                        response = chain.invoke({"input": user_text})
                        return _extract_text_from_llm_content(response.content)
                    except Exception:
                        logger.warning(f"配額不足，降級到 {fallback_model} 失敗")
                        continue
            detail = (
                f"❌ Gemini API 配額已耗盡或暫時受限\n\n"
                f"- 嘗試模型：{model_name}\n"
                f"- 已嘗試其他 Gemini 備援：{', '.join(fallback_models) if fallback_models else '（無）'}\n\n"
                f"請見 https://ai.dev/rate-limit ；或改用 gemini-3.1-flash-preview / gemini-3-flash-preview。\n"
                f"**原始錯誤**：{error_msg[:200]}"
            )
            raise ChatGoogleGenerativeAIError(detail) from e
        raise


def _call_llm_for_feed(
    system_prompt: str,
    user_text: str,
    model_name: str,
    api_key: Optional[str] = None,
) -> str:
    """全球情報摘要：僅使用 Gemini。"""
    if not api_key or not (api_key or "").strip():
        raise ValueError("未提供 Gemini API Key")
    return call_gemini(system_prompt, user_text, model_name or DEFAULT_GEMINI_MODEL, api_key)


def optimize_context_for_ai(context_text: str, max_tokens: int = 20000) -> str:
    """
    優化 context 以減少 token 使用
    
    Args:
        context_text: 原始 context
        max_tokens: 目標最大 token 數（約 4 字元 = 1 token）
    
    Returns:
        優化後的 context
    """
    max_chars = max_tokens * 4  # 粗略估計
    
    if len(context_text) <= max_chars:
        return context_text
    
    # 優先保留前面和後面的 Source（通常是重要的）
    lines = context_text.split('\n')
    source_lines = [line for line in lines if line.startswith('Source')]
    
    if len(source_lines) <= 1:
        return context_text[:max_chars]
    
    # 保留前 60% 和後 20% 的來源
    keep_start = int(len(source_lines) * 0.6)
    keep_end = int(len(source_lines) * 0.2)
    
    kept_lines = source_lines[:keep_start] + source_lines[-keep_end:]
    non_source_lines = [line for line in lines if not line.startswith('Source')]
    
    optimized = '\n'.join(non_source_lines + kept_lines)
    
    if len(optimized) > max_chars:
        optimized = optimized[:max_chars] + "\n...（內容已截斷以優化處理速度）"
    
    return optimized

def run_strategic_analysis(query: str, context_text: str, model_name: str, api_key: str, mode: str="FUSION", fast_mode: bool = False, manipulation_signals: Optional[str] = None, analysis_depth: str = "標準") -> str:
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    depth_value = (analysis_depth or "標準").strip()
    if depth_value.startswith("快速") or fast_mode:
        context_text = optimize_context_for_ai(context_text, max_tokens=9000)
        depth_instruction = """
        【分析詳盡度】：快速
        - 優先產出可快速閱讀的結論，每節以 2-4 點為主。
        - 表格列數控制在必要範圍，避免過長段落。
        - 必須保留關鍵證據、資訊不足與查證建議，不得為了簡短而臆測。
        """
    elif depth_value.startswith("深度"):
        depth_instruction = """
        【分析詳盡度】：深度
        - 允許較長篇幅，完整展開 ACH、Entman、謬誤、偏見、Cui Bono 與資訊操作分析。
        - 可使用較多表格列與結構化條列，但仍需避免重複與空泛敘述。
        - 每個重要判斷都應標註 Source ID 或說明資訊缺口。
        """
    else:
        context_text = optimize_context_for_ai(context_text, max_tokens=16000)
        depth_instruction = """
        【分析詳盡度】：標準
        - 在完整性與可讀性之間取得平衡。
        - 每節保留必要證據與推論，但避免過度展開。
        - 優先提供可行的查證建議與清楚的風險判斷。
        """
    
    tone_instruction = """
    【⚠️ 語氣風格指令】：
    1. **極度審慎**：嚴禁臆測。若證據不足，請明確說明：
       - 哪些部分資訊不足
       - 需要哪些類型的資料才能進行完整分析
       - 基於現有資料可以得出哪些有限但可靠的結論
       - 避免僅簡單標示「目前資訊不足」，應提供具體的資訊缺口分析
    2. **去軍事化／避免戰爭風**：全文（含標題、小標、表格內文）嚴禁軍事隱喻與戰爭框架。除非議題確為武裝衝突，否則勿使用「開戰」「對決」「攻防」「烽火」「戰線」「戰場」「征戰」等用語；改以「政策討論」「立場差異」「協商」「爭議」「角力」等中性表述。Section 8/9 的標題與內文也須避免過度軍事化修辭。
    3. **平衡報導風格**：標題與內文請採中性、多元觀點，避免單一立場或煽動用語；可並陳不同看法，勿過度戲劇化。
    4. **中性專業**：使用社會科學術語。
    """

    if mode == "FUSION":
        system_prompt = f"""
        你是一位極度嚴謹的情報分析師與社會科學家。
        
        【⚠️ 時間錨點】：今天是 {today_str}。
        {tone_instruction}
        {depth_instruction}
        
        【⚠️ 深度分析指令 (CRITICAL)】：
        1. **拒絕淺層摘要**：你的任務不是總結新聞，而是「解構」新聞。請挖掘文本中未言明的假設、結構性的偏見以及操縱手法。
        2. **詳盡性優先**：表格與各節內容應**完整展開**，包含完整論證邏輯與細節；勿僅以一句話帶過。每一點分析都必須有具體的來源 (Source ID) 支持。讀者偏好**詳細、可深入檢視**的分析。
        3. **多維度視角**：分析時必須同時考慮政治、經濟、社會心理與地緣戰略維度。
        
        【⚠️ 詳細分析原則】：
        表格內的每一格可填寫**多句或分點說明**，提供具體脈絡、數據或引用，不應過度簡化。請挖掘事件背後的深層邏輯。各節（含 ACH、爭議點、框架分析、共識、深層偏見、Cui Bono 等）請**充分展開論證**，必要時可拆成多列或多段呈現。
        
        【⚠️ 輸出長度與詳盡度】：
        請產出**詳盡、完整**的分析報告。各節請充分展開，表格欄位可多句或分點；若支持/反對證據有多筆，請分別列出或合併成完整段落，勿僅寫一句摘要。Section 5、6 的結構化條列請**詳細論證並引用原文**，勿精簡成要點式一句話。
        
        【⚠️ 深度分析區塊禁止表格 (CRITICAL)】：
        針對「深層偏見與認知盲區解構」(Section 5) 與「Cui Bono 利益分析」(Section 6)，**禁止使用表格**。請改用**結構化條列 (structured bullet points)**，以便展開詳細論證與**文本取證 (Textual Forensics)**——即引用原文具體字句、用詞以支持你的論斷。
        
        【分析方法論】：
        1. **ACH 競爭假設分析**：提出多個解釋假設，並進行詳細的證據權重評估。
        2. **邏輯謬誤偵測**：識別文本中的推論缺陷，包含但不限於：
           - 滑坡謬誤、假兩難悖論（非黑即白）、預設謬誤／乞題、稻草人、訴諸人身、訴諸權威、訴諸情感、因果謬誤、以偏概全、訴諸無知、循環論證、紅鯡魚（轉移焦點）。
        3. **Entman 框架分析**：解構不同陣營如何透過「選擇」與「凸顯」來建構現實。
        4. **深層偏見解構**：分析結構性遺漏（Missing Voices）、知識論模態（Epistemic Modality）與隱含前提（Enthymeme）。
        5. **資訊操作偵測**：識別協同性行為與語義旋轉。
        
        【⚠️ 資訊不足處理原則】：
        - 如果 Context 中來源數量少於 3 篇，請在報告開頭明確標註「⚠️ 資訊不足警告」
        - 列出具體的資訊缺口，基於現有資料提供有限但可靠的結論，避免過度推測
        
        【輸出格式 (嚴格遵守)】：
        ### [DATA_TIMELINE]
        (格式：YYYY-MM-DD|媒體|標題|Source_ID)
        *請注意：只能列出 Context 中實際存在的 Source，嚴禁捏造 Source ID。若無 Source ID 則不列出。*
        
        ### [REPORT_TEXT]
        (Markdown 報告 - 繁體中文)
        
        **⚠️ 標題與內文風格**：報告開頭請以一句 **# 簡短標題** 總括本議題；標題須**中性、平衡報導風格**，勿使用戰爭隱喻或聳動用語（如「開戰」「對決」「烽火」）。全文各節小標與表格內文字亦須符合平衡報導與去軍事化表述。
        
        **📚 NBLM 核心導讀指南 (Document Study Guide)**（請緊接標題後產出，嚴格依據 Context 文本內容提取，幫助讀者快速掌握陌生議題）
        * **1. 📖 關鍵術語辭典 (Glossary)**
          (提取 3-5 個文本中出現的專有名詞、法案或行話，並依據文本給出精確定義；每項須標註 Source ID)
          - **[術語 A]**：(定義說明) [Source X]
          - **[術語 B]**：(定義說明) [Source X]
        * **2. 👤 關鍵實體與角色 (Key Entities)**
          (列出核心參與的人物、國家或機構，並用一句話總結他們在此事件中的行為或立場；每項須標註 Source ID)
          - **[實體 A]**：(角色與行為總結) [Source X]
          - **[實體 B]**：(角色與行為總結) [Source X]
        * **3. ❓ 核心問答集 (FAQ)**
          (預判讀者對於此議題最想了解的 3 個核心問題，並給出**充分展開**的解答；解答須僅依據 Context 來源，可多句論證，並標註 Source ID)
          - **Q1: [問題]**
            **A:** [解答] [Source X]
          - **Q2: [問題]**
            **A:** [解答] [Source X]
          - **Q3: [問題]**
            **A:** [解答] [Source X]
        
        0. **🎯 ACH 競爭假設分析 (Analysis of Competing Hypotheses)**
           | 假設 | 支持證據 (請詳述，可多句或分點) | 反對證據 (請詳述，可多句或分點) | 可信度評估 | 關鍵變數 |
           |:---|:---|:---|:---|:---|
           (請提供至少 3 個假設，內容需詳盡，包含具體的人事時地物；支持/反對證據欄請充分展開，勿僅一句話)
           
        1. **📊 整體現況與脈絡 (Situational Context)**
           | 日期 | 事件摘要 (可多句) | 戰略影響與後續效應 (請深入推演，可多句或分點) | 證據強度 |
           |:---|:---|:---|:---|
           
        2. **🔍 爭議點與事實查核 (Fact-Check & Logic Scan)**
           **邏輯謬誤偵測**（請掃描並標註以下類型，若無則留空）：
           - 滑坡謬誤 (Slippery Slope)、假兩難悖論 (False Dilemma)、預設謬誤／乞題 (Begging the Question)、稻草人 (Straw Man)、訴諸人身 (Ad Hominem)、訴諸權威 (Appeal to Authority)、訴諸情感 (Appeal to Emotion)、因果謬誤 (Causal Fallacy)、以偏概全 (Hasty Generalization)、訴諸無知 (Appeal to Ignorance)、循環論證 (Circular Reasoning)、紅鯡魚 (Red Herring)
           | 謬誤類型 | 來源 (Source ID) | 原文片段 | 深度分析 (該謬誤如何影響受眾認知? 請充分展開) |
           |:---|:---|:---|:---|
           
           **事實查核結果** (若有)：
           (列出已證偽或誤導性內容，可多句說明)
           
        3. **⚖️ 媒體框架光譜分析 (Entman Framing Analysis)**
           | 陣營 | 問題定義 (如何界定核心矛盾? 可多句) | 歸因分析 (責任歸咎於誰? 可多句) | 道德評價 (正當性基礎為何? 可多句) | 關鍵修辭/隱喻 |
           |:---|:---|:---|:---|:---|
           
        4. **🤝 共識與分歧深度解析 (Consensus & Divergence)**
           * **核心共識**：(各方均無異議的事實基礎；請充分展開，可多點列舉)
           * **關鍵分歧**：(價值觀或利益的根本衝突點，請分析「為什麼」會有此分歧；可多段論證)
           
        5. **👁️ 深層偏見與認知盲區解構 (Deep Bias & Cognitive Blindspots Deconstruction)**
           (⚠️ 本節**禁止使用表格**，請以**結構化條列**呈現，並進行**文本取證**：引用原文具體字句、用詞以支持論斷。**請充分展開**每一子項，勿僅一句話帶過。)
           * **陣營 A**：[名稱，可標註 Source ID]
             * **結構性遺漏**：哪些聲音被系統性排除？哪些視角未被呈現？請詳細論證並引用原文。
             * **隱含前提**：論述背後未言明但必須成立的價值觀或假設為何？請引用原文支持並展開說明。
             * **狗哨/暗語**：針對特定群體的政治動員詞彙為何？**必須引用原文具體用語**（Textual Forensics），並說明脈絡。
             * **知識論風險**：是否將推測或立場包裝成事實？請指出具體表述並展開分析。
           (對每個主要陣營/來源重複上述結構，務必附上 Source ID 與原文引用；各子項請詳盡撰寫。)
           
        6. **🧠 Cui Bono 與利益分析 (Cui Bono & Interest Analysis)**
           (分析**利益如何驅動**前述 Section 5 所識別的框架與偏見；本節**禁止使用表格**，請以結構化條列呈現。**請充分展開**各利益相關者的動機與機制，勿僅一句話。)
           * **利益相關者 A**：[誰？]
             * **核心動機**：權力 / 利益 / 選票 / 地緣戰略 / 其他（請標註並展開說明）
             * **機制**：透過推動 [框架 X] 以達成 [目標 Y]；與 Section 5 的哪些偏見或遺漏相呼應？請詳細論證。
           (對每個關鍵利益相關者重複，說明其動機如何體現於報導框架與遺漏；可多段論證。)
           
        7. **🛡️ 敘事操縱與資訊操作風險 (Narrative Manipulation Analysis)**
           - **協同行為特徵**：(分析 [MANIPULATION_SIGNALS] 中的擴散模式)
           - **語義旋轉 (Semantic Spin)**：(分析同一事實如何被不同框架扭曲)
           - **風險評估**：(高/中/低，並提供詳細理由)

        8. 🕵️ 影響力網絡與預警指標 (Influence Network & Early Warning)
           * **代理人與網絡分析 (Proxy Network)**：
             請解構主要行動者背後的深層支持網絡（誰在出錢？誰在出論述？）。
             - **陣營 A 網絡**：(例如：智庫 -> 媒體代理人 -> 企業金主)
             - **陣營 B 滲透路徑**：(例如：外部勢力 -> 友好協會 -> 地方議員)
           * **早期預警指標 (Early Warning Watchlist)**：
             請列出具體、可觀測的「微弱訊號 (Weak Signals)」儀表板：
             | 指標類型 | 具體訊號 (What to watch) | 代表意義 | 監測頻率 |
             |:---|:---|:---|:---|
             | (軍事/經濟/政治/社會) | ... | ... | ... |
           （本節用語請保持中性學術表述，勿過度軍事化修辭。）

        9. ⚔️ 混合戰威脅建模 (Hybrid Warfare: Cognitive & Lawfare)
           * **認知戰戰術解構 (Cognitive Warfare - DISARM Framework)**：
             - **攻擊目標**：(例如：年輕選民、特定產業從業者)
             - **核心敘事**：(例如：修憲=徵兵、恐懼訴求、分化族群)
             - **傳播載體**：(例如：TikTok 短影音、匿名帳號集群)
             - **預期效果**：...
           * **國際法理戰推演 (Lawfare Gaming)**：
             - **甲方論述**：(引用之國際法或條約，如聯合國憲章第51條)
             - **乙方反制**：(引用之歷史權利、波茨坦公告或主權宣示)
           （本節雖涉及威脅建模，內文仍請以中性學術用語呈現，避免額外戰爭隱喻如「戰場」「攻防」等。）
        """
        
    elif mode == "DEEP_SCENARIO":
        system_prompt = f"""
        你是一位專精於未來學 (Futures Studies) 的戰略顧問。
        
        【⚠️ 時間錨點】：今天是 {today_str}。
        {tone_instruction}
        {depth_instruction}
        
        【分析任務】：
        1. **早期預警指標**：列出監測訊號。
        2. **驗屍分析**：反推失敗變數。

        【輸出格式】：
        ### [DATA_TIMELINE]
        (留空)
        
        ### [REPORT_TEXT]
        (Markdown 報告 - 繁體中文)
        1. **🎯 CLA 深度解構 (Causal Layered Analysis)**
           - Litany / System / Worldview / Myth
        2. **🔮 未來趨勢路徑模擬 (Scenario Planning)**
           - **基準路徑 (Baseline)** + 🚩 預警指標
           - **轉折路徑 (Alternative)** + 🚩 預警指標
           - **極端路徑 (Wild Card)** + 🚩 預警指標
        3. **💀 驗屍分析 (Pre-mortem Analysis)**
        4. **💡 綜合發展與因應建議**
        """
    elif mode == "TEXT_ANALYSIS":
        system_prompt = f"""
        你是一位極度嚴謹的新聞文本分析師、媒體識讀教師與社會科學研究者。

        【⚠️ 時間錨點】：今天是 {today_str}。
        {tone_instruction}
        {depth_instruction}

        【分析任務】：
        使用使用者貼上的單篇新聞文本進行文本取證式分析。你的任務不是重寫新聞摘要，而是解構這篇新聞如何建構現實、如何使用語言、哪些證據可靠、哪些推論需要查證。

        【方法論要求】：
        1. **GRADE / CERQual 證據強度**：根據 Context 中提供的來源分類、內容品質、語言風格、引用密度與資訊完整性評估可信度。
        2. **Entman 框架分析**：分析問題定義、歸因、道德評價、解決方案或行動暗示。
        3. **邏輯謬誤偵測**：掃描滑坡謬誤、假兩難、預設謬誤、稻草人、訴諸人身、訴諸權威、訴諸情感、因果謬誤、以偏概全、訴諸無知、循環論證、紅鯡魚。
        4. **深層偏見解構**：分析結構性遺漏、Missing Voices、隱含前提、知識論模態與狗哨/暗語。
        5. **Cui Bono 利益分析**：辨識這種敘事可能使哪些行動者受益，並說明受益機制。
        6. **水平閱讀建議**：指出若要查證，應補哪些外部資料或不同立場來源。

        【引用規則】：
        - 只能引用 Context 中的內容。
        - 單篇新聞一律標註為 [Source 1]。
        - 分析必須引用原文片段或具體用詞，避免空泛評論。
        - 若證據不足，請明確標註「資訊不足」，不要臆測。

        【輸出格式 (嚴格遵守)】：
        ### [DATA_TIMELINE]
        (單篇文本分析通常留空；若文本中有明確日期事件，可用 YYYY-MM-DD|來源|事件|Source 1 列出)

        ### [REPORT_TEXT]
        (Markdown 報告 - 繁體中文)

        # [中性、平衡的新聞文本分析標題]

        ## 1. 新聞核心主張摘要
        - **主張 A**：[用 1-2 句說明，標註 Source 1]
        - **主張 B**：[如有]
        - **可確認事實 / 解釋性說法 / 推測性說法**：請分開說明。

        ## 2. 文本可信度與證據強度
        | 評估面向 | 觀察 | 風險或可信度影響 |
        |:---|:---|:---|
        | 內容完整性 | ... | ... |
        | 引用與歸因 | ... | ... |
        | 來源與利益關係 | ... | ... |
        | 證據強度 | ... | ... |

        ## 3. 語言與情緒操控檢測
        - **情緒詞與修辭**：引用具體字詞並分析其效果。
        - **標題與內文落差**：若有，說明落差如何影響理解。
        - **知識論模態**：辨識「可能、恐、據稱、專家認為」等不確定性標記。

        ## 4. Entman 框架分析
        | 框架元素 | 本文如何呈現 | 原文依據 |
        |:---|:---|:---|
        | 問題定義 | ... | ... |
        | 歸因分析 | ... | ... |
        | 道德評價 | ... | ... |
        | 解決方案 / 行動暗示 | ... | ... |

        ## 5. 邏輯謬誤與事實查核風險
        | 類型 | 原文片段 | 分析 | 查證需求 |
        |:---|:---|:---|:---|
        | ... | ... | ... | ... |

        ## 6. 深層偏見與認知盲區
        - **結構性遺漏**：哪些人、資料或反方觀點缺席？
        - **隱含前提**：文本要成立，讀者必須先接受哪些假設？
        - **狗哨/暗語與身份動員**：若有，引用原文並說明。
        - **知識論風險**：哪些推測被包裝成事實？

        ## 7. Cui Bono 與利益分析
        - **可能受益者 A**：其利益、動機與文本框架之間的關聯。
        - **可能受益者 B**：如有。

        ## 8. 資訊不足與橫向查證建議
        - **目前不足**：列出無法只靠本文確認的關鍵資訊。
        - **建議查證來源**：官方文件、原始數據、當事方說法、獨立媒體、事實查核機構等。
        - **讀者判讀提醒**：用 2-3 點給出媒體識讀建議。
        """
    else:
        system_prompt = f"請針對 {query} 進行分析。"

    # 將操作信號注入 FUSION 模式提示（替換 [MANIPULATION_SIGNALS]）
    if mode == "FUSION" and "[MANIPULATION_SIGNALS]" in system_prompt:
        replacement = (manipulation_signals or "").strip() or "（本輪未提供操作信號資料。）"
        system_prompt = system_prompt.replace("[MANIPULATION_SIGNALS]", replacement)

    return call_gemini(system_prompt, context_text, model_name, api_key)


# --- LLM JSON Parsing Helper (Used by News Feed & General Output) ---

def _extract_json_from_llm_raw(raw: str) -> Optional[Dict[str, Any]]:
    """
    從 LLM 回傳文字中萃取出單一 JSON 物件。容忍 markdown 程式碼區塊、前後說明文字。
    """
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    # 移除 markdown 程式碼區塊
    if "```" in raw:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if match:
            raw = match.group(1).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    # 找到第一個 { 與最後一個 }，避免前後多餘文字
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]
    # 移除可能導致 JSON 解析失敗的尾隨逗號（僅限 ,] 與 ,}）
    raw = re.sub(r",\s*]", "]", raw)
    raw = re.sub(r",\s*}", "}", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _extract_json_list_from_llm_raw(raw: str) -> Optional[List[Dict[str, Any]]]:
    """
    從 LLM 回傳文字中萃取出 JSON 陣列。容忍 markdown 程式碼區塊、前後說明文字。
    支援部分模型回傳之思考區塊標記：會先移除 <think> 等區塊再解析 JSON。
    用於全球情報摘要等回傳 list 的場景。
    """
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip()
    # 移除 thinking 模型輸出的 <think>...</think> 區塊，避免 JSON 被當成前綴而解析失敗
    if "<think>" in text:
        text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    # 陣列：第一個 [ 到最後一個 ]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    text = re.sub(r",\s*]", "]", text)
    text = re.sub(r",\s*}", "}", text)
    try:
        out = json.loads(text)
        if isinstance(out, list):
            return out
        if isinstance(out, dict) and ("items" in out or "stories" in out):
            key = "items" if "items" in out else "stories"
            return out[key] if isinstance(out[key], list) else None
        return None
    except json.JSONDecodeError:
        return None


# ==========================================
# 全球情報摘要 (Global Intelligence Feed)：五大洲 × 政治/經濟/科技
# ==========================================

# 全球情報「僅從白名單來源」時的網域：台灣藍綠官方＋國際（通訊社／亞洲／歐美），最多 50 個以符合 Tavily 建議
def _feed_whitelist_domains() -> List[str]:
    merged = list(dict.fromkeys(
        list(BLUE_WHITELIST) + list(GREEN_WHITELIST) + list(OFFICIAL_WHITELIST) +
        list(INTL_WIRES_WHITELIST) + list(INTL_ASIA_WHITELIST)[:15] + list(INTL_EUROPE_WHITELIST)[:8] + list(INTL_AMERICAS_WHITELIST)[:8]
    ))
    return merged[:50]


# 每洲一組查詢（合併政治/經濟/科技），大幅減少 API 次數：5 次 Tavily + 5 次 Gemini
FEED_BY_CONTINENT = [
    ("亞洲", "Asia top important politics economy technology news today headline"),
    ("歐洲", "Europe top important politics economy technology EU news today"),
    ("美洲", "Americas USA top important politics economy technology news today"),
    ("非洲", "Africa top important politics economy technology news today"),
    ("大洋洲", "Oceania Australia top important politics economy technology news today"),
]

CONTINENT_ORDER = ["亞洲", "歐洲", "美洲", "非洲", "大洋洲"]
TOPIC_ORDER = ["政治", "經濟", "科技"]
CONTINENT_EMOJI = {"亞洲": "🌏", "歐洲": "🌍", "美洲": "🌎", "非洲": "🌍", "大洋洲": "🌏", "其他": "🌐"}


def _normalize_feed_field(value: Any, *, list_join: str = " ") -> str:
    """
    將 LLM／Tavily 可能回傳的 str、list、數字、巢狀 dict 轉成單一字串。
    避免 Gemini 將 summary 等欄位輸出成 JSON 陣列時，對 list 呼叫 .strip() 造成錯誤。
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value).strip()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: List[str] = []
        for x in value:
            s = _normalize_feed_field(x, list_join=" ")
            if s:
                parts.append(s)
        return list_join.join(parts)
    if isinstance(value, dict):
        for k in ("text", "content", "title", "summary", "url", "value"):
            if k in value and value[k] is not None:
                return _normalize_feed_field(value[k], list_join=list_join)
        return ""
    return str(value).strip()


def _dedupe_tavily_raw_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """依 URL 去重（正規化尾隨斜線），減少送入 LLM 的重複稿件。"""
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in items:
        u = _normalize_feed_field(r.get("url")).lower().rstrip("/")
        key = u if u.startswith("http") else f"t:{_normalize_feed_field(r.get('title'))[:120]}"
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


# --- 全球情報 Gemini 輸入量控制（節省 input token；中文大致 2～4 字 ≈ 1 token）---
FEED_LLM_INPUT_CHAR_CAP = 12000  # 單洲 user 區總上限（原約 18000）
FEED_LLM_SNIPPET_CHARS = 300  # 每則內文摘取長度（原 500）
FEED_LLM_MAX_ARTICLES = 12  # 每洲送入摘要的篇數上限（原 15）
# 勾選「精簡輸入」時
FEED_LLM_COMPACT_CHAR_CAP = 8000
FEED_LLM_COMPACT_SNIPPET = 200
FEED_LLM_COMPACT_ARTICLES = 9
# RSS 單次摘要（若日後啟用）
FEED_RSS_INPUT_CHAR_CAP = 18000
FEED_RSS_MAX_LINES = 40
FEED_RSS_SNIPPET_CHARS = 220


def _format_tavily_items_for_feed_llm(
    raw_items: List[Dict[str, Any]],
    *,
    max_articles: int,
    snippet_chars: int,
    total_char_cap: int,
) -> str:
    """
    將 Tavily 結果壓成送進 Gemini 的純文字；在總字數上限內盡可能多帶幾則（由前到後）。
    使用短欄位前綴 T/U/C 並截斷內文，減少冗餘標籤長度。
    """
    blocks: List[str] = []
    for i, r in enumerate(raw_items[:max_articles], 1):
        title = _normalize_feed_field(r.get("title"))
        url = _normalize_feed_field(r.get("url"))
        content = _normalize_feed_field(r.get("content") or r.get("snippet") or "", list_join=" ")
        if len(content) > snippet_chars:
            snippet = content[:snippet_chars].rstrip() + "…"
        else:
            snippet = content
        block = f"[{i}] T:{title}\nU:{url}\nC:{snippet}"
        candidate = "\n\n".join(blocks + [block])
        if len(candidate) > total_char_cap:
            break
        blocks.append(block)
    return "\n\n".join(blocks)


# RSS 來源（免 Tavily）：涵蓋全球＋各區域；前段為主要來源，後段為備援（提高 0 次 API 模式成功率）
RSS_FEED_URLS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
    "https://feeds.bbci.co.uk/news/world/europe/rss.xml",
    "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
    "https://feeds.bbci.co.uk/news/world/africa/rss.xml",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.reuters.com/reuters/technologyNews",
    # 備援來源（格式較單純或較不易被擋）
    "https://feeds.npr.org/1001/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
    # 平衡報導版面用：補充不同立場來源（RSS 若失效會自動略過）
    "https://moxie.foxnews.com/google-publisher/world.xml",
    "https://nypost.com/news/feed/",
    "https://www.washingtontimes.com/rss/headlines/news/world/",
    "https://thehill.com/feed/",
]
RSS_ITEMS_PER_FEED = 8

SOURCE_BIAS_LABELS = {
    "left": "Left",
    "lean_left": "Lean Left",
    "center": "Center",
    "lean_right": "Lean Right",
    "right": "Right",
    "unknown": "Unknown",
}

SOURCE_BIAS_GROUP = {
    "left": "Left",
    "lean_left": "Left",
    "center": "Center",
    "lean_right": "Right",
    "right": "Right",
    "unknown": "Other",
}

SOURCE_BIAS_BY_DOMAIN = {
    "theguardian.com": "left",
    "nytimes.com": "lean_left",
    "npr.org": "lean_left",
    "aljazeera.com": "lean_left",
    "bbc.com": "center",
    "bbc.co.uk": "center",
    "reuters.com": "center",
    "thehill.com": "center",
    "foxnews.com": "right",
    "nypost.com": "right",
    "washingtontimes.com": "lean_right",
}

def _get_feed_source_badges(url_or_domain: str) -> List[str]:
    """
    依來源 URL／網域對照白名單，回傳用於情報卡顯示的標籤列表。
    用於 Intelligence Card 的 Meta Tags（泛藍／泛綠／官方／國際等）。
    """
    if not url_or_domain or not isinstance(url_or_domain, str):
        return []
    domain = get_domain_name(url_or_domain) if url_or_domain.startswith(("http://", "https://")) else url_or_domain.strip().lower()
    if not domain:
        return []
    badges = []
    if any(domain in d or d in domain for d in BLUE_WHITELIST):
        badges.append("🔵 泛藍觀點")
    if any(domain in d or d in domain for d in GREEN_WHITELIST):
        badges.append("🟢 泛綠觀點")
    if any(domain in d or d in domain for d in OFFICIAL_WHITELIST):
        badges.append("🏛️ 官方")
    if any(domain in d or d in domain for d in NEUTRAL_WHITELIST):
        badges.append("⚪ 中立")
    if any(domain in d or d in domain for d in INDIE_WHITELIST):
        badges.append("📰 獨立")
    if any(domain in d or d in domain for d in INTL_WHITELIST):
        badges.append("🇺🇸 國際視角")
    if not badges:
        badges.append("📌 其他來源")
    return badges


def _feed_source_domain(item: Dict[str, Any]) -> str:
    """從情報卡資料取出可比對的來源網域。"""
    ref = _normalize_feed_field(item.get("url")) or _normalize_feed_field(item.get("source"))
    if not ref:
        return ""
    try:
        domain = get_domain_name(ref) if ref.startswith(("http://", "https://")) else ref
    except Exception:
        domain = ref
    return (domain or "").lower().replace("www.", "")


def _feed_source_bias(item: Dict[str, Any]) -> Tuple[str, str]:
    """
    回傳 (bias_key, source_domain)。這不是精密評級，只供 AllSides-inspired 版面做初步分欄。
    未列入的來源會顯示為 Unknown，不硬分到左右。
    """
    domain = _feed_source_domain(item)
    for known, bias in SOURCE_BIAS_BY_DOMAIN.items():
        if domain == known or domain.endswith("." + known) or known in domain:
            return bias, domain
    return "unknown", domain


def _short_feed_text(text: str, limit: int = 120) -> str:
    t = re.sub(r"\s+", " ", _normalize_feed_field(text)).strip()
    return (t[:limit].rstrip() + "…") if len(t) > limit else t


ALLSIDES_BALANCED_NEWS_URL = "https://www.allsides.com/unbiased-balanced-news"


def _clean_allsides_text(text: str) -> str:
    """清理 AllSides HTML 文字中的多餘空白與常見編碼雜訊。"""
    t = unescape(text or "")
    t = t.replace("�X", "—").replace("��", "'")
    return re.sub(r"\s+", " ", t).strip()


def _allsides_abs_url(href: str) -> str:
    return urljoin(ALLSIDES_BALANCED_NEWS_URL, href or "")


@st.cache_data(ttl=1800)
def fetch_allsides_headline_roundups(max_roundups: int = 12) -> List[Dict[str, Any]]:
    """
    直接擷取 AllSides Balanced News 的 Headline Roundups。
    僅讀取公開頁面，不登入、不呼叫 Tavily/Gemini。
    """
    try:
        from bs4 import BeautifulSoup
    except Exception as e:
        logger.error("fetch_allsides_headline_roundups: 缺少 beautifulsoup4: %s", str(e))
        return []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        resp = requests.get(ALLSIDES_BALANCED_NEWS_URL, timeout=25, headers=headers)
        resp.raise_for_status()
    except Exception as e:
        logger.error("fetch_allsides_headline_roundups: 取得頁面失敗: %s", str(e))
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    cards = soup.select(".headline-roundup")
    out: List[Dict[str, Any]] = []

    for card in cards[:max_roundups]:
        content = card.select_one(".headline-roundup-content") or card
        heading = content.find(["h2", "h3"]) or card.find(["h2", "h3"])
        if not heading:
            continue
        title = _clean_allsides_text(heading.get_text(" ", strip=True))
        story_link = heading.find("a") or card.find("a", href=re.compile(r"^/story/"))
        story_url = _allsides_abs_url(story_link.get("href")) if story_link else ALLSIDES_BALANCED_NEWS_URL

        paragraphs: List[str] = []
        for p in content.find_all("p"):
            text = _clean_allsides_text(p.get_text(" ", strip=True))
            if not text:
                continue
            if text.lower().startswith(("written by", "learn more", "support our mission", "suggest an improvement")):
                continue
            paragraphs.append(text)
            if len(paragraphs) >= 3:
                break
        summary = "\n\n".join(paragraphs)

        perspectives: List[Dict[str, str]] = []
        trio = card.select_one(".news-trio")
        if trio:
            for item in trio.select(".news-item"):
                cls = set(item.get("class") or [])
                if "left" in cls:
                    bias = "Left"
                elif "right" in cls:
                    bias = "Right"
                elif "center" in cls:
                    bias = "Center"
                else:
                    label = _clean_allsides_text(item.get_text(" ", strip=True))
                    bias = "Left" if "From the Left" in label else "Right" if "From the Right" in label else "Center" if "From the Center" in label else "Other"
                links = item.find_all("a")
                news_link = links[0] if links else None
                source_link = links[1] if len(links) > 1 else None
                perspectives.append({
                    "bias": bias,
                    "title": _clean_allsides_text(news_link.get_text(" ", strip=True)) if news_link else "",
                    "url": _allsides_abs_url(news_link.get("href")) if news_link else "",
                    "source": _clean_allsides_text(source_link.get_text(" ", strip=True)) if source_link else "",
                    "source_url": _allsides_abs_url(source_link.get("href")) if source_link else "",
                })
        else:
            text = _clean_allsides_text(card.get_text(" ", strip=True))
            match = re.search(r"See how\s+(.+?)\s+cover this story", text, flags=re.I)
            if match:
                names = re.split(r"\s*,\s*|\s+and\s+", match.group(1))
                for idx, name in enumerate([n.strip(" ,") for n in names if n.strip(" ,")][:3]):
                    perspectives.append({
                        "bias": ["Left", "Center", "Right"][idx] if idx < 3 else "Other",
                        "title": "",
                        "url": "",
                        "source": name,
                        "source_url": "",
                    })

        out.append({
            "title": title,
            "summary": summary,
            "story_url": story_url,
            "perspectives": perspectives,
            "source": "AllSides",
            "url": story_url,
            "analysis_keywords": title,
        })

    return out


def _render_allsides_roundup_card(roundup: Dict[str, Any], index: int) -> None:
    """呈現單則 AllSides Headline Roundup。"""
    title = _normalize_feed_field(roundup.get("title")) or "（無標題）"
    summary = _normalize_feed_field(roundup.get("summary"), list_join="\n")
    story_url = _normalize_feed_field(roundup.get("story_url") or roundup.get("url"))
    perspectives = roundup.get("perspectives") if isinstance(roundup.get("perspectives"), list) else []
    news_id = hashlib.md5(f"allsides_{title}_{index}".encode("utf-8")).hexdigest()[:12]

    with st.container(border=True):
        st.caption("HEADLINE ROUNDUP · AllSides")
        if story_url.startswith(("http://", "https://")):
            st.markdown(f"### [{title}]({story_url})")
        else:
            st.markdown(f"### {title}")
        if summary:
            st.write(summary[:900] + ("…" if len(summary) > 900 else ""))

        cols = st.columns(3)
        column_defs = [("Left", "From the Left", "🟦"), ("Center", "From the Center", "⬜"), ("Right", "From the Right", "🟥")]
        for col, (bias, label, icon) in zip(cols, column_defs):
            with col:
                st.markdown(f"**{icon} {label}**")
                match = next((p for p in perspectives if _normalize_feed_field(p.get("bias")) == bias), None)
                if not match:
                    st.caption("AllSides 此則未提供")
                    continue
                p_title = _normalize_feed_field(match.get("title"))
                p_url = _normalize_feed_field(match.get("url"))
                p_source = _normalize_feed_field(match.get("source"))
                p_source_url = _normalize_feed_field(match.get("source_url"))
                if p_title and p_url.startswith(("http://", "https://")):
                    st.markdown(f"[{p_title}]({p_url})")
                elif p_title:
                    st.write(p_title)
                elif p_source:
                    st.write("此來源觀點")
                if p_source:
                    if p_source_url.startswith(("http://", "https://")):
                        st.caption(f"[{p_source}]({p_source_url})")
                    else:
                        st.caption(p_source)

        if st.button("🔍 用此議題做多元分析", key=f"allsides_deep_{news_id}", type="primary"):
            st.session_state["query"] = title
            st.session_state["current_page"] = "🚀 多元議題分析 (Deep Analysis)"
            st.session_state["keyword_plan"] = None
            st.session_state["result"] = None
            st.session_state["scenario_result"] = None
            st.session_state["sources"] = None
            st.session_state["manipulation_signals"] = None
            st.session_state["cofacts_rumors"] = []
            st.session_state["volume_analysis"] = None
            st.session_state["stance_analysis"] = None
            st.rerun()


def _is_single_camp_source(url_or_domain: str) -> bool:
    """若來源僅屬單一立場（僅藍或僅綠），回傳 True，用於同溫層警示。"""
    if not url_or_domain:
        return False
    domain = get_domain_name(url_or_domain) if url_or_domain.startswith(("http://", "https://")) else url_or_domain.strip().lower()
    if not domain:
        return False
    in_blue = any(domain in d or d in domain for d in BLUE_WHITELIST)
    in_green = any(domain in d or d in domain for d in GREEN_WHITELIST)
    return (in_blue and not in_green) or (in_green and not in_blue)


def _summarize_feed_with_llm(
    raw_news_text: str,
    api_key: str,
    continent: Optional[str] = None,
    topic: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    使用 Gemini 將原始搜尋結果整理為結構化 JSON 列表（繁體中文標題與摘要）。
    理論基礎：資訊摘要與多源整合，供全球情報儀表板使用。

    Args:
        raw_news_text: 來自 Tavily 的原始新聞片段彙總文字
        api_key: Google Gemini API Key
        continent: 可選，洲名（如亞洲、歐洲）供 prompt 脈絡
        topic: 可選，類型（政治/經濟/科技）供 prompt 脈絡

    Returns:
        含 title、summary、source、analysis_keywords（及可選 emoji）的字典列表；失敗時返回空列表。
    """
    if not api_key or not (raw_news_text or "").strip():
        logger.warning("_summarize_feed_with_llm: 缺少 api_key 或 raw_news_text 為空")
        return []
    context = ""
    if continent or topic:
        parts = [p for p in [continent, topic] if p]
        context = f"（本批為「{' '.join(parts)}」類要聞）"
    system_prompt = """你是資深情報編輯（Senior Intelligence Editor）。請將提供的原始新聞片段整理成「今日重要要聞」清單。
選稿原則：優先選擇**具政策／市場／國際影響力**的頭條與重大事件，排除次要、八卦或純地方小事。
每則請產出：
- emoji：一個代表該則主題的 emoji（如 🇺🇸 🇹🇼 📉 🤖）
- title：吸引人的繁體中文標題
- summary：2～3 句繁體中文摘要
- source：來源網域或媒體名（從原文擷取或推斷）
- url：該則新聞的完整連結（從該則「U:」或原文「URL:」後面的網址**原樣複製**，若無則空字串 ""）
- analysis_keywords：若要對該議題做「深度分析」時，最適合的搜尋關鍵字（繁體中文，簡短精準）

請「只」輸出一個 JSON 陣列，每則一筆物件，不要其他說明。鍵名必須為：emoji, title, summary, source, url, analysis_keywords。
**重要**：每個欄位值必須為單一字串（string），不可使用陣列；若需多句摘要請合併為一段文字。"""
    user_prompt = f"""以下為「今日最新」搜尋結果{context}。請從中精選**最重要**的 5～8 則（頭條、重大政策、市場或國際要聞），去除重複與次要內容，輸出上述格式的 JSON 陣列。每則的 url 請從對應則「U:」或「URL:」後方完整複製。\n\n{raw_news_text[:FEED_LLM_INPUT_CHAR_CAP]}"""
    try:
        raw = call_gemini(system_prompt, user_prompt, DEFAULT_GEMINI_MODEL, api_key)
        if not raw:
            return []
        items = _extract_json_list_from_llm_raw(raw)
        if not items or not isinstance(items, list):
            return []
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            out.append({
                "emoji": _normalize_feed_field(it.get("emoji"), list_join="") or "📌",
                "title": _normalize_feed_field(it.get("title")) or "（無標題）",
                "summary": _normalize_feed_field(it.get("summary"), list_join="\n"),
                "source": _normalize_feed_field(it.get("source")),
                "url": _normalize_feed_field(it.get("url")),
                "analysis_keywords": _normalize_feed_field(
                    it.get("analysis_keywords") or it.get("keywords") or it.get("title")
                ),
            })
        return out
    except Exception as e:
        logger.error("_summarize_feed_with_llm: 摘要失敗: %s", str(e))
        return []


def _summarize_continent_feed_with_llm(
    raw_news_text: str,
    api_key: str,
    continent: str,
    llm_model: str = DEFAULT_GEMINI_MODEL,
    max_input_chars: int = FEED_LLM_INPUT_CHAR_CAP,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    單一洲別合併摘要：一次 LLM 產出該洲「政治／經濟／科技」三類要聞，每則帶 topic 欄位。
    僅使用 Google Gemini。
    """
    if not api_key or not (raw_news_text or "").strip():
        return [], "未提供 API Key 或無內容"
    system_prompt = """你是資深情報編輯。請將提供的原始新聞**綜整**成「今日重要要聞」：可合併相似主題，並為每則標註類型與**戰略視角**（Why does this matter?）。

**風格要求（重要）**：
- **平衡報導**：標題與摘要請採中性、多元觀點，避免單一立場或煽動用語；可並陳不同看法或「A 方主張…B 方則…」。
- **避免戰爭／軍事框架**：除非事件本身為武裝衝突，否則勿以「開戰」「對決」「攻防」「烽火」等戰爭隱喻描述；改以「政策討論」「立場差異」「協商」「爭議」等中性表述。標題勿過度戲劇化。

每則請產出：
- topic：必須為「政治」「經濟」「科技」其中之一
- emoji：代表該則主題的 emoji
- title：吸引人、中性平衡的繁體中文標題（勿戰爭風）
- summary：2～3 句繁體中文摘要（平衡報導風格）
- strategic_angle：一句話說明「戰略視角／為何重要」（中性表述）
- source：來源網域或媒體名
- url：從原文「URL:」後方**原樣複製**的完整連結，若無則 ""
- analysis_keywords：深度分析用搜尋關鍵字（繁體中文）

請「只」輸出一個 JSON 陣列。鍵名：topic, emoji, title, summary, strategic_angle, source, url, analysis_keywords。政治、經濟、科技三類盡量各 2～3 則。
**重要**：每個欄位值必須為單一字串（string），不可使用陣列；多句摘要請合併為一段文字。"""
    user_prompt = f"""以下為「{continent}」今日搜尋結果（每則為 T:標題、U:網址、C:Tavily 摘錄）。請綜整相似報導、精選 6～9 則，為每則標註 topic 與 strategic_angle，輸出上述 JSON 陣列；url 請從該則「U:」後方**原樣複製**。\n\n{raw_news_text[:max_input_chars]}"""
    try:
        raw = _call_llm_for_feed(system_prompt, user_prompt, llm_model, api_key)
        if not raw:
            return [], "Gemini 回傳為空"
        items = _extract_json_list_from_llm_raw(raw)
        if not items or not isinstance(items, list):
            return [], "Gemini 回傳無法解析為 JSON 陣列（可能為截斷或非 JSON）"
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            topic = _normalize_feed_field(it.get("topic"))
            if topic not in TOPIC_ORDER:
                topic = "政治"
            out.append({
                "topic": topic,
                "emoji": _normalize_feed_field(it.get("emoji"), list_join="") or "📌",
                "title": _normalize_feed_field(it.get("title")) or "（無標題）",
                "summary": _normalize_feed_field(it.get("summary"), list_join="\n"),
                "strategic_angle": _normalize_feed_field(it.get("strategic_angle")),
                "source": _normalize_feed_field(it.get("source")),
                "url": _normalize_feed_field(it.get("url")),
                "analysis_keywords": _normalize_feed_field(
                    it.get("analysis_keywords") or it.get("keywords") or it.get("title")
                ),
            })
        return out, None
    except Exception as e:
        logger.error("_summarize_continent_feed_with_llm: 摘要失敗: %s", str(e))
        return [], str(e)


def _infer_topic_from_text(title: str, content: str) -> str:
    """依標題與內容關鍵字推斷 政治/經濟/科技，供原文 fallback 使用。"""
    text = (title + " " + content).lower()
    economy_keywords = ("economy", "economic", "market", "stock", "gdp", "inflation", "trade", "經濟", "股市", "市場", "通膨", "貿易", "財經")
    tech_keywords = ("tech", "technology", "ai", "software", "digital", "robot", "科技", "人工智慧", "半導體", "軟體", "數位")
    if any(k in text for k in economy_keywords):
        return "經濟"
    if any(k in text for k in tech_keywords):
        return "科技"
    return "政治"


def _tavily_raw_to_feed_items(raw_items: List[Dict[str, Any]], continent: str) -> List[Dict[str, Any]]:
    """
    LLM 額度不足或摘要失敗時，將 Tavily 回傳的原文轉成與摘要相同格式的卡片資料。
    依標題/內容關鍵字分配 topic（政治／經濟／科技），避免全部顯示為政治。
    """
    out: List[Dict[str, Any]] = []
    for r in raw_items[:20]:
        title = _normalize_feed_field(r.get("title")) or "（無標題）"
        url = _normalize_feed_field(r.get("url"))
        content_raw = r.get("content") or r.get("snippet") or ""
        content = _normalize_feed_field(content_raw, list_join=" ")[:300]
        domain = get_domain_name(url) if url else ""
        topic = _infer_topic_from_text(title, content)
        out.append({
            "continent": continent,
            "topic": topic,
            "emoji": "📌",
            "title": title,
            "summary": content or "",
            "strategic_angle": "Tavily 原文（LLM 額度不足或未摘要時顯示）",
            "source": domain,
            "url": url,
            "analysis_keywords": title,
        })
    return out


@st.cache_data(ttl=3600)
def fetch_intelligence_feed_tavily(
    tavily_key: str,
    gemini_api_key: Optional[str],
    gemini_model: str,
    use_whitelist: bool = False,
    save_tokens: bool = False,
) -> List[Dict[str, Any]]:
    """
    彙整五大洲「每日最新重要」頭條（每洲 1 次 Tavily + 1 次 Gemini 摘要）。
    額度不足或未提供 Key 時改顯示 Tavily 原文。快取 1 小時。
    use_whitelist=True 時，僅從白名單網域（台灣藍綠官方＋國際媒體）取得結果。
    save_tokens=True 時縮短每則摘錄與總字數，降低 Gemini input token。
    """
    if not tavily_key:
        return []
    all_items: List[Dict[str, Any]] = []
    use_llm = bool(gemini_api_key and (gemini_api_key or "").strip())
    in_cap = FEED_LLM_COMPACT_CHAR_CAP if save_tokens else FEED_LLM_INPUT_CHAR_CAP
    snip_n = FEED_LLM_COMPACT_SNIPPET if save_tokens else FEED_LLM_SNIPPET_CHARS
    n_for_llm = FEED_LLM_COMPACT_ARTICLES if save_tokens else FEED_LLM_MAX_ARTICLES
    if use_llm:
        st.session_state.pop("feed_llm_last_error", None)
    include_domains: Optional[List[str]] = None
    if use_whitelist:
        include_domains = _feed_whitelist_domains()
        logger.info("fetch_intelligence_feed_tavily: 使用白名單網域 %d 個", len(include_domains))
    try:
        tavily = TavilyClient(api_key=tavily_key)
        for continent, query in FEED_BY_CONTINENT:
            raw_items: List[Dict[str, Any]] = []
            for time_range in ("day", "week"):
                try:
                    search_kw: Dict[str, Any] = {
                        "query": query,
                        "max_results": 12,
                        "search_depth": "basic",
                        "topic": "news",
                        "time_range": time_range,
                    }
                    if include_domains:
                        search_kw["include_domains"] = include_domains
                    resp = tavily.search(**search_kw)
                    results = resp.get("results", []) if isinstance(resp, dict) else getattr(resp, "results", None) or []
                    for r in results:
                        if isinstance(r, dict):
                            raw_items.append(r)
                        else:
                            raw_items.append({
                                "title": getattr(r, "title", ""),
                                "url": getattr(r, "url", ""),
                                "content": getattr(r, "content", "") or getattr(r, "snippet", ""),
                            })
                    if raw_items:
                        break
                except Exception as e:
                    logger.warning("fetch_intelligence_feed_tavily: Tavily %s (time_range=%s) 失敗: %s", continent, time_range, str(e))
                    continue
            if not raw_items:
                continue
            raw_items = _dedupe_tavily_raw_items(raw_items)
            # 結果過少時補一次 advanced（較耗額度但可提高 recall）
            if len(raw_items) < 4:
                try:
                    adv_kw: Dict[str, Any] = {
                        "query": query,
                        "max_results": 15,
                        "search_depth": "advanced",
                        "topic": "news",
                        "time_range": "week",
                    }
                    if include_domains:
                        adv_kw["include_domains"] = include_domains
                    resp_adv = tavily.search(**adv_kw)
                    results_adv = resp_adv.get("results", []) if isinstance(resp_adv, dict) else getattr(resp_adv, "results", None) or []
                    for r in results_adv:
                        if isinstance(r, dict):
                            raw_items.append(r)
                        else:
                            raw_items.append({
                                "title": getattr(r, "title", ""),
                                "url": getattr(r, "url", ""),
                                "content": getattr(r, "content", "") or getattr(r, "snippet", ""),
                            })
                    raw_items = _dedupe_tavily_raw_items(raw_items)
                except Exception as e:
                    logger.info("fetch_intelligence_feed_tavily: advanced 補搜略過 洲=%s err=%s", continent, str(e))
            raw_items = raw_items[:18]
            raw_news_text = _format_tavily_items_for_feed_llm(
                raw_items,
                max_articles=n_for_llm,
                snippet_chars=snip_n,
                total_char_cap=in_cap,
            )
            if use_llm:
                logger.info("fetch_intelligence_feed_tavily: 使用 Gemini 摘要 模型=%s 洲=%s save_tokens=%s", gemini_model, continent, save_tokens)
                summarized, llm_err = _summarize_continent_feed_with_llm(
                    raw_news_text, gemini_api_key, continent,
                    llm_model=gemini_model,
                    max_input_chars=in_cap,
                )
                if not summarized:
                    logger.warning("fetch_intelligence_feed_tavily: LLM 摘要為空，改顯示 Tavily 原文 洲=%s err=%s", continent, llm_err or "")
                    summarized = _tavily_raw_to_feed_items(raw_items, continent)
                    if llm_err:
                        st.session_state["feed_llm_last_error"] = llm_err
                else:
                    st.session_state.pop("feed_llm_last_error", None)
            else:
                logger.info("fetch_intelligence_feed_tavily: 未提供 LLM Key，直接顯示 Tavily 原文 洲=%s", continent)
                summarized = _tavily_raw_to_feed_items(raw_items, continent)
            for it in summarized:
                it["continent"] = continent
                if not _normalize_feed_field(it.get("url")) and raw_items:
                    llm_title = _normalize_feed_field(it.get("title"))
                    for raw in raw_items:
                        raw_title = _normalize_feed_field(raw.get("title"))
                        if raw_title and llm_title and (raw_title[:30] in llm_title or llm_title[:30] in raw_title or SequenceMatcher(None, llm_title[:50], raw_title[:50]).ratio() > 0.5):
                            it["url"] = _normalize_feed_field(raw.get("url"))
                            break
                all_items.append(it)
    except Exception as e:
        logger.error("fetch_intelligence_feed_tavily: 失敗: %s", str(e))
        return []
    return all_items


def _rss_element_plain_text(elem: Any) -> str:
    """
    萃取 RSS/Atom 節點可見文字（含子節點、CDATA）。
    NYTimes 等來源的 title 常以 HTML 巢狀或 type='html'，單讀 .text 會為空。
    """
    if elem is None:
        return ""
    parts = "".join(elem.itertext())
    if not parts.strip() and getattr(elem, "text", None):
        parts = str(elem.text)
    if not parts.strip():
        for attr in ("content", "value"):
            a = elem.get(attr) if hasattr(elem, "get") else None
            if a and str(a).strip():
                parts = str(a)
                break
    raw = (parts or "").strip()
    if "<" in raw:
        raw = re.sub(r"<[^>]+>", " ", raw)
    raw = unescape(re.sub(r"\s+", " ", raw)).strip()
    return raw


def _rss_item_title_from_entry(item: Any, _local_name_fn: Any) -> str:
    """由單一 item/entry 擷取標題：掃描本節點下所有名為 title 的節點（含 media: 命名空間）。"""
    title_els: List[Any] = []
    for child in item.iter():
        if _local_name_fn(child.tag) == "title":
            title_els.append(child)
    for tel in title_els:
        t = _rss_element_plain_text(tel)
        if t and t.lower() not in ("(no title)", "（無標題）"):
            return t
    return ""


def _title_fallback_from_url(url: str) -> str:
    """從路徑最後一段還原可讀標題（例如 nytimes.com/.../article-slug）。"""
    if not url.startswith(("http://", "https://")):
        return ""
    try:
        seg = urlparse(url).path.strip("/").split("/")[-1]
        if not seg or len(seg) < 4 or seg.isdigit():
            return ""
        seg = unquote(seg)
        if not re.search(r"[a-zA-Z\u4e00-\u9fff]", seg):
            return ""
        return re.sub(r"[-_]+", " ", seg).strip()[:200]
    except Exception:
        return ""


def _canonical_http_url(url: str) -> str:
    """正規化 RSS 連結（protocol-relative `//`、空白）。"""
    u = (url or "").strip()
    if not u:
        return ""
    if u.startswith("//"):
        return "https:" + u
    return u


def _continent_hint_from_feed_url(feed_url: str) -> str:
    """從 RSS feed URL 本身推斷洲別，避免只靠文章 URL 造成分類偏斜。"""
    u = (feed_url or "").lower()
    if "/asia" in u or "asia-pacific" in u:
        return "亞洲"
    if "/europe" in u:
        return "歐洲"
    if "/africa" in u:
        return "非洲"
    if "us_and_canada" in u or "americas" in u:
        return "美洲"
    if "australia" in u or "oceania" in u or "pacific" in u:
        return "大洋洲"
    return ""


def _continent_hint_from_text(title: str, content: str, url: str = "") -> str:
    """從標題、摘要、URL 的地名關鍵字粗分洲別。"""
    text = f"{title} {content} {url}".lower()
    keyword_groups = [
        ("大洋洲", ("australia", "new zealand", "oceania", "pacific islands", "fiji", "png", "papua")),
        ("非洲", ("africa", "nigeria", "kenya", "ghana", "mali", "sudan", "ethiopia", "south africa", "egypt", "congo", "somalia")),
        ("亞洲", ("asia", "china", "japan", "korea", "taiwan", "india", "pakistan", "iran", "israel", "gaza", "palestinian", "thailand", "philippines", "indonesia", "vietnam", "singapore", "malaysia", "afghanistan")),
        ("歐洲", ("europe", "ukraine", "russia", "moscow", "eu ", "e.u.", "france", "germany", "britain", "uk ", "italy", "spain", "poland", "austria", "belgium")),
        ("美洲", ("united states", " u.s.", " us ", "america", "canada", "mexico", "brazil", "argentina", "venezuela", "colombia", "caribbean")),
    ]
    for continent, keywords in keyword_groups:
        if any(k in text for k in keywords):
            return continent
    return ""


def _fetch_rss_raw() -> List[Dict[str, Any]]:
    """從 RSS_FEED_URLS 抓取標題／連結／摘要，不依賴 Tavily。使用 requests + xml 解析。"""
    import xml.etree.ElementTree as ET

    entries: List[Dict[str, Any]] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml, application/atom+xml, text/xml;q=0.9, */*;q=0.8",
    }

    def _local_name(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    def _find_any(elem: Optional[ET.Element], *names: str) -> Optional[ET.Element]:
        if elem is None:
            return None
        name_set = set(names)
        for name in names:
            found = elem.find(name) or elem.find(f"{{http://www.w3.org/2005/Atom}}{name}")
            if found is not None:
                return found
        for child in elem:
            if _local_name(child.tag) in name_set:
                return child
        return None

    def _text_or_attr(el: Optional[ET.Element], attr: str = "href") -> str:
        if el is None:
            return ""
        if attr and el.get(attr):
            return (el.get(attr) or "").strip()
        return (el.text or "").strip()

    for url in RSS_FEED_URLS:
        source_count = 0
        feed_continent_hint = _continent_hint_from_feed_url(url)
        try:
            r = requests.get(url, timeout=22, headers=headers)
            r.raise_for_status()
            r.encoding = r.encoding or "utf-8"
            text = r.text
        except Exception as e:
            logger.warning("_fetch_rss_raw: %s 失敗: %s", url, str(e))
            continue
        try:
            root = ET.fromstring(text)
            # 依「本地名」蒐集 item/entry，不受 default namespace 影響（如 RSS 1.0、部分 Atom）
            items = [e for e in root.iter() if _local_name(e.tag) in ("item", "entry")]
            if not items:
                items = list(root.iter("item")) or list(root.iter("{http://www.w3.org/2005/Atom}entry"))
            for item in items:
                title_el = _find_any(item, "title") or item.find("title") or item.find("{http://www.w3.org/2005/Atom}title")
                link_el = _find_any(item, "link") or item.find("link") or item.find("{http://www.w3.org/2005/Atom}link")
                link = _text_or_attr(link_el, "href") if link_el is not None else ""
                if not link and link_el is not None:
                    link = (link_el.text or "").strip()
                if not link:
                    for lk in item.iter():
                        if _local_name(lk.tag) != "link":
                            continue
                        h = (lk.get("href") or "").strip()
                        if h.startswith("http"):
                            link = h
                            break
                if not link:
                    guid_el = _find_any(item, "guid")
                    if guid_el is not None:
                        gt = _rss_element_plain_text(guid_el) or ((guid_el.text or "").strip())
                        if gt.startswith("http"):
                            link = gt
                if not link:
                    id_el = item.find("{http://www.w3.org/2005/Atom}id")
                    if id_el is not None:
                        it = (id_el.text or "").strip()
                        if it.startswith("http"):
                            link = it
                link = _canonical_http_url(link)
                desc_el = (
                    _find_any(item, "description", "summary", "content")
                    or item.find("description")
                    or item.find("{http://www.w3.org/2005/Atom}summary")
                    or item.find("{http://www.w3.org/2005/Atom}content")
                    or item.find("content")
                )
                if desc_el is not None and desc_el.text:
                    desc = (desc_el.text or "").strip()[:400]
                else:
                    desc_plain = _rss_element_plain_text(desc_el) if desc_el is not None else ""
                    desc = (desc_plain or (ET.tostring(desc_el, encoding="unicode", method="text") if desc_el is not None else ""))[:400]
                title = _rss_item_title_from_entry(item, _local_name)
                if not title and title_el is not None:
                    title = _rss_element_plain_text(title_el)
                if not title:
                    title = _title_fallback_from_url(link)
                if title or link:
                    entries.append({
                        "title": title or "（無標題）",
                        "url": link,
                        "content": desc,
                        "feed_url": url,
                        "continent_hint": feed_continent_hint,
                    })
                    source_count += 1
                    if source_count >= RSS_ITEMS_PER_FEED:
                        break
        except ET.ParseError as e:
            logger.warning("_fetch_rss_raw: 解析 %s 失敗: %s", url, str(e))
            continue
    return entries[:80]


def _continent_hint_from_url(url: str) -> str:
    """依 URL 粗分五大洲（RSS 匯集用啟發式）。"""
    u = (url or "").lower()
    if "asia" in u or "asia-pacific" in u or "china" in u or "japan" in u or "korea" in u or "taiwan" in u:
        return "亞洲"
    if "europe" in u or "eu." in u:
        return "歐洲"
    if "africa" in u:
        return "非洲"
    if "australia" in u or "pacific" in u or "nz" in u:
        return "大洋洲"
    if "/us" in u or "us_and_canada" in u or "americas" in u or "united-states" in u:
        return "美洲"
    return "其他"


def _rss_raw_to_feed_items(raw: List[Dict[str, Any]], max_items: int = 48) -> List[Dict[str, Any]]:
    """
    將 RSS 原始條目轉為全球情報卡片格式（不依賴 Gemini／Tavily）。
    依 URL 去重，並以關鍵字啟發式推斷政治／經濟／科技。
    """
    if not raw:
        return []
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in raw:
        url = _canonical_http_url(_normalize_feed_field(r.get("url")))
        title = _normalize_feed_field(r.get("title")) or ""
        if not title or title in ("（無標題）", "(No title)"):
            title = _title_fallback_from_url(url) or "（無標題）"
        if not url.startswith(("http://", "https://")) and title == "（無標題）":
            continue
        key = url.lower().rstrip("/") if url.startswith(("http://", "https://")) else f"t:{title[:160].lower()}"
        if key in seen:
            continue
        seen.add(key)
        content = _normalize_feed_field(r.get("content") or "", list_join=" ")[:400]
        topic = _infer_topic_from_text(title, content)
        continent = (
            _normalize_feed_field(r.get("continent_hint"))
            or _continent_hint_from_text(title, content, url)
            or _continent_hint_from_url(url)
        )
        domain = ""
        try:
            domain = urlparse(url).netloc or ""
        except Exception:
            pass
        bias_key, _ = _feed_source_bias({"url": url, "source": domain})
        out.append({
            "continent": continent,
            "topic": topic,
            "emoji": "📌",
            "title": title,
            "summary": content or "",
            "strategic_angle": "RSS 摘要（未經 Gemini 綜整）",
            "source": domain,
            "url": url,
            "analysis_keywords": title,
            "bias": bias_key,
        })
        if len(out) >= max_items:
            break
    return out


def _rss_fallback_from_raw(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """LLM 失敗時的備援列表（與 `_rss_raw_to_feed_items` 相同邏輯，略多則數）。"""
    return _rss_raw_to_feed_items(raw, max_items=50)


@st.cache_data(ttl=1800)
def fetch_intelligence_feed_rss_display() -> List[Dict[str, Any]]:
    """
    全球情報預設資料源：抓取 RSS 訂閱並轉為卡片列表，**不呼叫** Gemini／Tavily。
    """
    raw = _fetch_rss_raw()
    out = _rss_raw_to_feed_items(raw)
    if raw and not out:
        logger.warning("fetch_intelligence_feed_rss_display: 原始 RSS %d 筆但轉卡片 0 筆", len(raw))
    return out


@st.cache_data(ttl=1800)
def fetch_intelligence_feed_rss(
    llm_api_key: Optional[str],
    llm_model: str,
) -> List[Dict[str, Any]]:
    """
    以 RSS 頭條為來源，用 1 次 Gemini 摘要並分類為五大洲 × 政治/經濟/科技（約 1 次 API，免 Tavily）。
    """
    if not llm_api_key or not (llm_api_key or "").strip():
        return []
    raw = _fetch_rss_raw()
    if not raw:
        return []
    lines = []
    for i, r in enumerate(raw[:FEED_RSS_MAX_LINES], 1):
        t = _normalize_feed_field(r.get("title"))
        u = _normalize_feed_field(r.get("url"))
        c = _normalize_feed_field(r.get("content") or "", list_join=" ")
        c = (c[:FEED_RSS_SNIPPET_CHARS] + "…") if len(c) > FEED_RSS_SNIPPET_CHARS else c
        lines.append(f"[{i}] T:{t}\nU:{u}\nC:{c}")
    raw_text = "\n\n".join(lines)
    system_prompt = """你是資深情報編輯。請將提供的 RSS 新聞**綜整**成「今日重要要聞」：可將相似主題合併為一則綜整稿，並標註**洲別**與**類型**。
**綜整原則**：群組化相似報導、突出戰略意義；每則需有「戰略視角」（Why does this matter?）。
**洲別判斷規則**：美洲（美加拉美）、歐洲（歐盟英俄烏等）、亞洲（中日韓東南亞台灣）、非洲、大洋洲（澳紐）。
**硬性要求**：五大洲每洲至少 2 則，總計約 15～25 則。
每則請產出：
- continent：亞洲／歐洲／美洲／非洲／大洋洲
- topic：政治／經濟／科技
- emoji：代表主題的 emoji
- title：吸引人的繁體中文標題
- summary：2～3 句繁體中文摘要（綜整多源時可合併敘述）
- strategic_angle：一句話說明「戰略視角／為何重要」（Why does this matter?）
- source：來源網域或媒體名（可從 URL 推斷）
- url：從原文「URL:」後方**原樣複製**的完整連結
- analysis_keywords：深度分析用搜尋關鍵字（繁體中文）

請「只」輸出一個 JSON 陣列。鍵名：continent, topic, emoji, title, summary, strategic_angle, source, url, analysis_keywords。
**重要**：每個欄位值必須為單一字串（string），不可使用陣列。**url 請從輸入中每則「U:」後方原樣複製。**"""
    user_prompt = f"""以下為國際 RSS 頭條（每則 T/U/C）。請**綜整**相似報導、精選 15～25 則，五大洲每洲至少 2 則，並為每則填寫 strategic_angle（為何重要），輸出上述 JSON 陣列。\n\n{raw_text[:FEED_RSS_INPUT_CHAR_CAP]}"""
    try:
        raw_out = _call_llm_for_feed(system_prompt, user_prompt, llm_model, llm_api_key)
        if not raw_out:
            return _rss_fallback_from_raw(raw)
        items = _extract_json_list_from_llm_raw(raw_out)
        if not items or not isinstance(items, list):
            return _rss_fallback_from_raw(raw)
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            cont = _normalize_feed_field(it.get("continent"))
            if cont not in CONTINENT_ORDER:
                cont = "其他"
            topic = _normalize_feed_field(it.get("topic"))
            if topic not in TOPIC_ORDER:
                topic = "政治"
            out.append({
                "continent": cont,
                "topic": topic,
                "emoji": _normalize_feed_field(it.get("emoji"), list_join="") or "📌",
                "title": _normalize_feed_field(it.get("title")) or "（無標題）",
                "summary": _normalize_feed_field(it.get("summary"), list_join="\n"),
                "strategic_angle": _normalize_feed_field(it.get("strategic_angle")),
                "source": _normalize_feed_field(it.get("source")),
                "url": _normalize_feed_field(it.get("url")),
                "analysis_keywords": _normalize_feed_field(
                    it.get("analysis_keywords") or it.get("keywords") or it.get("title")
                ),
            })
        return out
    except Exception as e:
        logger.error("fetch_intelligence_feed_rss: 摘要失敗: %s", str(e))
        return _rss_fallback_from_raw(raw)


def _render_balanced_feed_overview(feed: List[Dict[str, Any]]) -> None:
    """
    AllSides-inspired 平衡報導總覽：依 Left / Center / Right 分欄，讓使用者快速看到來源光譜。
    這不是 AllSides 官方資料，只是以本地 RSS 來源的已知傾向做初步分欄。
    """
    if not feed:
        return
    grouped: Dict[str, List[Dict[str, Any]]] = {"Left": [], "Center": [], "Right": [], "Other": []}
    for item in feed:
        bias_key = _normalize_feed_field(item.get("bias"))
        if not bias_key:
            bias_key, _ = _feed_source_bias(item)
        group = SOURCE_BIAS_GROUP.get(bias_key, "Other")
        grouped.setdefault(group, []).append(item)

    with st.expander("⚖️ 平衡報導總覽（Left / Center / Right）", expanded=True):
        st.caption("參考 AllSides 的平衡報導概念：同一批 RSS 來源按媒體傾向分欄，方便掃讀不同來源。未評級來源列為 Other。")
        bias_cols = st.columns(4)
        columns = [
            ("Left", "左／偏左來源", "🟦"),
            ("Center", "中間來源", "⬜"),
            ("Right", "右／偏右來源", "🟥"),
            ("Other", "未分類", "📌"),
        ]
        for col, (group, label, icon) in zip(bias_cols, columns):
            with col:
                st.markdown(f"#### {icon} {label}")
                items = grouped.get(group, [])[:6]
                if not items:
                    st.caption("目前無來源")
                    continue
                for item in items:
                    title = _normalize_feed_field(item.get("title")) or "（無標題）"
                    summary = _short_feed_text(_normalize_feed_field(item.get("summary")), 110)
                    url = _normalize_feed_field(item.get("url"))
                    source = _normalize_feed_field(item.get("source")) or _feed_source_domain(item)
                    bias_key = _normalize_feed_field(item.get("bias")) or _feed_source_bias(item)[0]
                    bias_label = SOURCE_BIAS_LABELS.get(bias_key, "Unknown")
                    with st.container(border=True):
                        if url.startswith(("http://", "https://")):
                            st.markdown(f"**[{title}]({url})**")
                        else:
                            st.markdown(f"**{title}**")
                        if summary:
                            st.caption(summary)
                        st.caption(f"{source or '未知來源'} · {bias_label}")


def render_news_feed_page(
    google_key: str,
    tavily_key: str,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> None:
    """
    渲染「全球情報」：直接匯整 AllSides Headline Roundups。
    不再使用原本 RSS / Tavily 五大洲匯集作為主流程。
    """
    st.markdown("## 📰 全球情報 (News Feed)")
    st.caption(
        "直接匯整 [AllSides Balanced News](https://www.allsides.com/unbiased-balanced-news) 的 **Headline Roundups**，以 Left / Center / Right 方式呈現同一議題的不同來源。"
    )

    current_source_key = "allsides_headline_roundups_v1"
    if "intelligence_feed_source" not in st.session_state:
        st.session_state["intelligence_feed_source"] = None
    if st.session_state.get("intelligence_feed_source") != current_source_key:
        st.session_state["intelligence_feed_data"] = None
        st.session_state["intelligence_feed_source"] = current_source_key
        st.session_state.pop("feed_diag_msg", None)

    if st.session_state.get("feed_do_fetch"):
        st.session_state["feed_do_fetch"] = False
        try:
            with st.spinner("正在載入 AllSides Headline Roundups…"):
                fetch_allsides_headline_roundups.clear()
                feed = fetch_allsides_headline_roundups()
            st.session_state["intelligence_feed_data"] = feed if isinstance(feed, list) else []
            st.session_state.pop("feed_diag_msg", None)
        except Exception as e:
            logger.exception("載入 AllSides Headline Roundups 失敗")
            st.session_state["intelligence_feed_data"] = []
            st.error(f"載入失敗：{str(e)[:200]}。")
        st.rerun()

    col_load, col_src = st.columns([1, 3])
    with col_load:
        st.button(
            "⚖️ 載入 AllSides Roundups",
            type="primary",
            key="feed_fetch_btn",
            on_click=lambda: st.session_state.update({"feed_do_fetch": True}),
            help="直接讀取 AllSides Balanced News 公開頁面的 Headline Roundups。",
        )
    with col_src:
        st.caption("資料來源：AllSides Headline Roundups。AllSides 主張「unbiased news doesn't exist」，所以用 Left / Center / Right 對照呈現。")

    feed = st.session_state.get("intelligence_feed_data")
    if feed is None:
        st.info("請點擊 **「⚖️ 載入 AllSides Roundups」**。")
        return
    if not feed:
        st.warning("**取得完成，但沒有抓到 AllSides Headline Roundups。**")
        with st.expander("🔍 可能原因與建議", expanded=True):
            st.markdown("""
- **AllSides**：頁面 HTML 結構改版、暫時無法連線，或被網路／防火牆阻擋。
- 可直接開啟 [AllSides Balanced News](https://www.allsides.com/unbiased-balanced-news) 確認本機是否能連線。
            """)
        if st.button("🔧 診斷 AllSides", key="feed_diag_btn"):
            fetch_allsides_headline_roundups.clear()
            with st.spinner("正在測試 AllSides…"):
                diag = fetch_allsides_headline_roundups()
            st.session_state["feed_diag_msg"] = (
                f"診斷：AllSides 目前可解析 **{len(diag)}** 則 Headline Roundups。"
            )
        _dm = st.session_state.get("feed_diag_msg")
        if _dm:
            st.info(_dm)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 再試一次", key="feed_retry_empty"):
                st.session_state["feed_do_fetch"] = True
                st.rerun()
        with col_b:
            if st.button("清除診斷訊息", key="feed_clear_diag"):
                st.session_state.pop("feed_diag_msg", None)
                st.rerun()
        return

    col_info, col_refresh = st.columns([3, 1])
    with col_info:
        st.success(f"已載入 {len(feed)} 則 AllSides Headline Roundups。")
        st.caption("每則包含 AllSides 摘要與 Left / Center / Right 來源欄位（依 AllSides 頁面提供內容）。")
    with col_refresh:
        if st.button("🔄 重新整理", key="feed_refresh_btn"):
            try:
                fetch_allsides_headline_roundups.clear()
                with st.spinner("重新取得 AllSides…"):
                    feed_new = fetch_allsides_headline_roundups()
                st.session_state["intelligence_feed_data"] = feed_new if feed_new else []
            except Exception as e:
                st.error(f"重新整理失敗：{str(e)[:200]}")
            st.rerun()

    for idx, roundup in enumerate(feed):
        _render_allsides_roundup_card(roundup, idx)


# 政治／經濟／科技新聞卡片外框色（方便閱讀區分）
_TOPIC_BORDER_COLORS = {"政治": "#2196F3", "經濟": "#4CAF50", "科技": "#FF9800"}


def _render_feed_card(item: Dict[str, Any], index: int, total: int) -> None:
    """
    Intelligence Card：高資訊密度、可操作按鈕（Ground News / Particle 風格）。
    含 Header、Meta 標籤、摘要、同溫層警示、深度戰略分析按鈕。
    依 topic（政治／經濟／科技）以不同顏色外框標示。
    """
    topic = _normalize_feed_field(item.get("topic")) or "政治"
    if topic not in TOPIC_ORDER:
        topic = "政治"
    border_color = _TOPIC_BORDER_COLORS.get(topic, "#9E9E9E")
    emoji = _normalize_feed_field(item.get("emoji"), list_join="") or "📌"
    title = _normalize_feed_field(item.get("title")) or "（無標題）"
    summary = _normalize_feed_field(item.get("summary"), list_join="\n")
    source = _normalize_feed_field(item.get("source"))
    url = _normalize_feed_field(item.get("url"))
    keywords = _normalize_feed_field(
        item.get("analysis_keywords") or item.get("keywords") or item.get("title")
    ) or title
    strategic_angle = _normalize_feed_field(item.get("strategic_angle"))
    # 穩定唯一 key（避免重複 key 導致按鈕失效）
    news_id = hashlib.md5(f"{title}_{url}_{index}".encode("utf-8")).hexdigest()[:12]
    # 主題色條：卡片上方色帶標示政治／經濟／科技，方便閱讀區分
    st.markdown(
        f'<div style="background: {border_color}; color: white; padding: 4px 10px; font-size: 0.75rem; border-radius: 6px 6px 0 0; margin-bottom: -1px;">{topic}</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        # Header: Emoji + Title
        st.markdown(f"### {emoji} {title}")
        # Meta Tags: 來源類別標籤（泛藍／泛綠／國際等）
        ref_url = url or source
        badges = _get_feed_source_badges(ref_url)
        if badges:
            st.caption(" | ".join(badges))
        # 來源連結
        if source or url:
            if url and url.startswith(("http://", "https://")):
                st.caption(f"來源：[{source or url}]({url})")
            elif source:
                st.caption(f"來源：{source}")
        # Summary（約 3 行內；移除行首 # 避免被渲染成標題，統一為內文字級）
        if summary:
            summary_short = "\n".join(summary.split("\n")[:3]) if "\n" in summary else (summary[:200] + "…" if len(summary) > 200 else summary)
            summary_short = re.sub(r"^#+\s*", "", summary_short, flags=re.MULTILINE).strip()
            if summary_short:
                st.write(summary_short)
        # Strategic Angle（若有）
        if strategic_angle:
            st.caption(f"**戰略視角**：{strategic_angle[:120]}{'…' if len(strategic_angle) > 120 else ''}")
        # 同溫層警示：僅單一立場來源時顯示
        if ref_url and _is_single_camp_source(ref_url):
            st.caption("⚠️ 同溫層警示")
        # 深度戰略分析按鈕
        if st.button("🔍 深度戰略分析 (Deep Dive)", key=f"btn_{news_id}", type="primary"):
            st.session_state["query"] = keywords
            st.session_state["current_page"] = "🚀 多元議題分析 (Deep Analysis)"
            # 重置分析相關狀態，避免沿用上一輪的關鍵字／報告／來源（Ghost Data）
            st.session_state["keyword_plan"] = None
            st.session_state["result"] = None
            st.session_state["scenario_result"] = None
            st.session_state["sources"] = None
            st.session_state["manipulation_signals"] = None
            st.session_state["cofacts_rumors"] = []
            st.session_state["volume_analysis"] = None
            st.session_state["stance_analysis"] = None
            st.rerun()


def parse_gemini_data(text: str) -> Dict[str, Any]:
    """
    解析 Gemini AI 返回的文本，提取時間軸和報告內容
    
    Args:
        text: AI 返回的原始文本（可能是字符串、None 或其他類型）
        
    Returns:
        包含 timeline 和 report_text 的字典
    """
    data = {"timeline": [], "report_text": ""}
    
    # 確保 text 是字符串類型
    if text is None:
        logger.warning("parse_gemini_data: 收到 None 值，返回空數據")
        return data
    
    # 如果不是字符串，嘗試轉換為字符串
    if not isinstance(text, str):
        logger.warning(f"parse_gemini_data: 收到非字符串類型 ({type(text).__name__})，嘗試轉換")
        try:
            # 處理 list 類型（Gemini API 有時返回 list，可能含 signature 等元數據）
            if isinstance(text, list):
                text = _extract_text_from_llm_content(text)
            else:
                text = str(text)
        except Exception as e:
            logger.error(f"parse_gemini_data: 無法轉換為字符串: {str(e)}")
            return data
    
    # 檢查字符串是否為空
    if not text.strip():
        logger.warning("parse_gemini_data: 收到空字符串，返回空數據")
        return data

    # === 關鍵修復：處理轉義字符問題 ===
    # 如果文本包含字面上的 \n 或 \"，說明它被雙重轉義了
    if "\\n" in text:
        logger.info("檢測到字面上的 \\n，執行反轉義處理")
        # 處理常見的轉義序列
        text = text.replace("\\n", "\n")
        text = text.replace("\\\"", "\"")
        text = text.replace("\\\'", "\'")
        text = text.replace("\\t", "\t")
        text = text.replace("\\r", "\r")

    # 先提取時間軸數據（從 [DATA_TIMELINE] 區塊）
    timeline_section = ""
    if "### [DATA_TIMELINE]" in text:
        parts = text.split("### [DATA_TIMELINE]")
        if len(parts) > 1:
            timeline_section = parts[1].split("### [REPORT_TEXT]")[0] if "### [REPORT_TEXT]" in parts[1] else parts[1]
    elif "[DATA_TIMELINE]" in text:
        parts = text.split("[DATA_TIMELINE]")
        if len(parts) > 1:
            timeline_section = parts[1].split("[REPORT_TEXT]")[0] if "[REPORT_TEXT]" in parts[1] else parts[1]
    
    # 解析時間軸
    if timeline_section:
        lines = timeline_section.split('\n')
        for line in lines:
            line = line.strip()
            # 檢查是否為時間軸行（包含 | 分隔符且至少有 3 個部分）
            if "|" in line and len(line.split("|")) >= 3:
                parts = line.split("|")
                try:
                    date = parts[0].strip()
                    name = parts[1].strip()
                    title = parts[2].strip()
                    source_id_str = "0"
                    if len(parts) >= 4: 
                        raw_id = parts[3].strip()
                        nums = re.findall(r'\d+', raw_id)
                        if nums: source_id_str = nums[0]
                    if "XX" in date or "xx" in date: 
                        date = "近期"
                    
                    # 驗證日期格式（YYYY-MM-DD 或 "近期"）
                    if re.match(r'^\d{4}-\d{2}-\d{2}$', date) or date == "近期":
                        data["timeline"].append({
                            "date": date,
                            "media": name,
                            "title": title,
                            "source_id": int(source_id_str)
                        })
                except Exception as e:
                    # 靜默跳過無效行
                    continue

    # 提取報告文本（優先順序：REPORT_TEXT 標記 > 摘要標記 > 全部文本）
    report_text = ""
    
    # 方法 1: 查找 [REPORT_TEXT] 標記
    if "### [REPORT_TEXT]" in text:
        parts = text.split("### [REPORT_TEXT]")
        if len(parts) > 1:
            report_text = parts[1].strip()
    elif "### REPORT_TEXT" in text:
        parts = text.split("### REPORT_TEXT")
        if len(parts) > 1:
            report_text = parts[1].strip()
    elif "[REPORT_TEXT]" in text:
        parts = text.split("[REPORT_TEXT]")
        if len(parts) > 1:
            report_text = parts[1].strip()
    
    # 方法 2: 如果沒有 REPORT_TEXT 標記，查找摘要或分析部分
    if not report_text:
        # 查找包含「摘要」、「分析」、「CLA」等關鍵字的標題
        patterns = [
            r"(#+\s*.*?摘要.*?\n.*?)(?=#+\s*|$)",
            r"(#+\s*.*?分析.*?\n.*?)(?=#+\s*|$)",
            r"(1\.\s*.*?摘要.*?\n.*?)(?=\d+\.\s*|$)",
            r"(#+\s*.*?CLA.*?\n.*?)(?=#+\s*|$)",
            r"(📊\s*.*?整體現況.*?\n.*?)(?=#+\s*|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                report_text = match.group(1).strip()
                break
    
    # 方法 3: 如果還是沒有，移除時間軸部分後，使用剩餘文本
    if not report_text:
        # 移除 [DATA_TIMELINE] 區塊
        if "### [DATA_TIMELINE]" in text:
            parts = text.split("### [DATA_TIMELINE]")
            if len(parts) > 1:
                remaining = parts[1].split("### [REPORT_TEXT]")[-1] if "### [REPORT_TEXT]" in parts[1] else ""
                if remaining.strip():
                    report_text = remaining.strip()
        else:
            # 如果沒有明確的標記，使用整個文本（但排除時間軸行）
            lines = text.split('\n')
            report_lines = []
            for line in lines:
                # 跳過時間軸行（包含 | 且格式為日期|媒體|標題）
                if "|" in line and len(line.split("|")) >= 3:
                    # 檢查是否為時間軸格式
                    parts = line.split("|")
                    if len(parts) >= 3 and (re.match(r'^\d{4}-\d{2}-\d{2}', parts[0].strip()) or "近期" in parts[0]):
                        continue
                report_lines.append(line)
            report_text = '\n'.join(report_lines).strip()
    
    # 清理報告文本：移除多餘的標記和空行
    if report_text:
        # 移除開頭的標記行
        report_text = re.sub(r'^###?\s*\[?REPORT_TEXT\]?\s*\n*', '', report_text, flags=re.MULTILINE)
        # 移除開頭的空行
        report_text = report_text.lstrip('\n').strip()
        
        # 清理過多的破折號（可能是 Markdown 渲染問題）
        # 移除連續超過 10 個破折號的行
        report_text = re.sub(r'^-{10,}\s*$', '', report_text, flags=re.MULTILINE)
        # 移除連續超過 5 個破折號但保留表格分隔符
        lines = report_text.split('\n')
        cleaned_lines = []
        for line in lines:
            # 保留表格分隔符（|:---:| 格式）
            if re.match(r'^\|[\s:-]+\|', line):
                cleaned_lines.append(line)
            # 移除只有破折號的行（超過 5 個）
            elif re.match(r'^-{5,}\s*$', line):
                continue  # 跳過這行
            else:
                cleaned_lines.append(line)
        report_text = '\n'.join(cleaned_lines)
    
    data["report_text"] = report_text if report_text else text  # 如果還是空的，使用原始文本
    
    return data

def create_full_html_report(data_result, scenario_result, sources, blind_mode) -> str:
    # [V37.3] 使用重構後的邏輯
    table_rows = process_timeline_rows(data_result.get("timeline", []), sources, blind_mode)
    
    timeline_html = ""
    if table_rows:
        timeline_html = f"""
        <h3>📅 關鍵發展時序</h3>
        <table class="custom-table" border="1" cellspacing="0" cellpadding="5" style="width:100%; border-collapse:collapse;">
            <thead><tr><th width="120">日期</th><th width="180">媒體來源 (Code Verified)</th><th>新聞標題 (點擊閱讀)</th></tr></thead>
            <tbody>{table_rows}</tbody>
        </table>
        <hr>
        """

    report_html_1 = ""
    if data_result:
        raw_md = data_result.get("report_text", "")
        html_content = markdown.markdown(raw_md, extensions=['tables'])
        final_html = format_citation_style(html_content)
        report_html_1 = f'<div class="report-paper"><h3>📝 平衡報導分析</h3>{final_html}</div>'

    report_html_2 = ""
    if scenario_result:
        raw_md_2 = scenario_result.get("report_text", "")
        html_content_2 = markdown.markdown(raw_md_2, extensions=['tables'])
        final_html_2 = format_citation_style(html_content_2)
        report_html_2 = f'<div class="report-paper"><h3>🔮 未來發展推演報告</h3>{final_html_2}</div>'

    sources_html = ""
    if sources:
        s_rows = ""
        for i, s in enumerate(sources):
            domain = get_domain_name(s.get('url'))
            media_name = domain
            for k, v in DOMAIN_NAME_MAP.items():
                if k in domain: media_name = v
                
            title = s.get('title', 'No Title')
            url = s.get('url')
            s_rows += f"<li><b>[{i+1}]</b> {media_name} - <a href='{url}' target='_blank'>{title}</a></li>"
        sources_html = f"<hr><h3>📚 引用文獻列表</h3><ul>{s_rows}</ul>"

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>多元觀點分析報告</title>
        {CSS_STYLE}
    </head>
    <body style="padding: 20px; max-width: 900px; margin: 0 auto;">
        <h1>多元觀點分析報告</h1>
        <p>生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        {timeline_html}
        {report_html_1}
        {report_html_2}
        {sources_html}
    </body>
    </html>
    """
    return full_html

def render_html_timeline(timeline_data, sources, blind_mode):
    # [V37.3] 直接呼叫重構後的邏輯
    table_rows = process_timeline_rows(timeline_data, sources, blind_mode)
    if not table_rows: return

    full_html = f"""
    <div class="scrollable-table-container">
    <table class="custom-table">
    <thead>
    <tr>
    <th style="width:120px;">日期</th>
    <th style="width:180px;">媒體</th>
    <th>新聞標題</th>
    </tr>
    </thead>
    <tbody>
    {table_rows}
    </tbody>
    </table>
    </div>
    """
    st.markdown("### 📅 關鍵發展時序")
    st.markdown(full_html, unsafe_allow_html=True)

def export_full_state():
    """匯出完整狀態為 JSON 格式
    
    安全地處理可能為 None 或非字典類型的 session_state 變數。
    
    Returns:
        str: JSON 格式的字串
    """
    data = {
        "result": st.session_state.result if isinstance(st.session_state.get('result'), dict) else None,
        "scenario_result": st.session_state.scenario_result if isinstance(st.session_state.get('scenario_result'), dict) else None,
        "sources": st.session_state.sources if isinstance(st.session_state.get('sources'), list) else []
    }
    return json.dumps(data, indent=2, ensure_ascii=False)

def convert_data_to_md(data):
    """將資料轉換為 Markdown 格式
    
    安全地處理可能為 None 或非字典類型的輸入。
    
    Args:
        data: 要轉換的資料（預期為字典，但可能是 None 或字串）
    
    Returns:
        str: Markdown 格式的字串
    """
    # 檢查輸入類型
    if data is None:
        return "# 多元觀點分析報告\n\n❌ 錯誤：無資料可匯出"
    
    if not isinstance(data, dict):
        logger.warning(f"convert_data_to_md 收到非字典類型輸入: {type(data)}")
        return f"# 多元觀點分析報告\n\n❌ 錯誤：資料格式不正確（收到 {type(data).__name__} 類型）"
    
    # 安全地取得 timeline 和 report_text
    timeline = data.get('timeline', [])
    if not isinstance(timeline, list):
        timeline = []
    
    report_text = data.get('report_text', '')
    if not isinstance(report_text, str):
        report_text = str(report_text) if report_text else ''
    
    timeline_df = pd.DataFrame(timeline)
    timeline_md = timeline_df.to_markdown(index=False) if not timeline_df.empty else "無時間軸資料"
    
    return f"""
# 多元觀點分析報告
產生時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 平衡報導分析
{report_text}

## 2. 時間軸
{timeline_md}
    """

def _plain_markdown_text(text: str) -> str:
    """將 Markdown/HTML 片段轉成適合 PDF 段落的純文字。"""
    text = re.sub(r'<span class="citation">(.*?)</span>', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)
    return text.strip()

def _download_pdf_cjk_font() -> Optional[str]:
    """下載可嵌入 PDF 的繁中文字型，供 Linux/雲端環境使用。"""
    font_dir = CACHE_DIR / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    font_path = font_dir / "NotoSansTC-Regular.ttf"
    if font_path.exists() and font_path.stat().st_size > 1024 * 1024:
        return str(font_path)

    font_urls = [
        "https://github.com/google/fonts/raw/main/ofl/notosanstc/NotoSansTC%5Bwght%5D.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanstc/NotoSansTC%5Bwght%5D.ttf",
    ]
    for url in font_urls:
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            if len(response.content) > 1024 * 1024:
                font_path.write_bytes(response.content)
                return str(font_path)
        except Exception as e:
            logger.warning(f"PDF 中文字型下載失敗 ({url}): {str(e)[:200]}")
    return None

def _find_pdf_cjk_font() -> Optional[str]:
    """尋找跨平台繁中文字型，供 ReportLab 嵌入 PDF。"""
    windows_dir = os.environ.get("WINDIR", r"C:\Windows")
    local_fonts = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
    system_fonts = Path(windows_dir) / "Fonts"
    linux_font_dirs = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local" / "share" / "fonts",
    ]
    candidates = [
        str(system_fonts / "msjh.ttc"),
        str(system_fonts / "msjh.ttf"),
        str(system_fonts / "msjhbd.ttc"),
        str(system_fonts / "mingliu.ttc"),
        str(system_fonts / "mingliu.ttf"),
        str(system_fonts / "pmingliu.ttc"),
        str(system_fonts / "kaiu.ttf"),
        str(system_fonts / "Microsoft JhengHei.ttf"),
        str(system_fonts / "Microsoft JhengHei UI.ttf"),
        str(local_fonts / "NotoSansCJKtc-Regular.otf"),
        str(local_fonts / "NotoSerifCJKtc-Regular.otf"),
        str(local_fonts / "NotoSansTC-Regular.otf"),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansTC-Regular.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for font_path in candidates:
        if os.path.exists(font_path):
            return font_path
    for font_dir in [system_fonts, local_fonts] + linux_font_dirs:
        if font_dir.exists():
            for pattern in [
                "**/*NotoSansTC*.ttf",
                "**/*NotoSansTC*.otf",
                "*NotoSansCJK*tc*.otf",
                "**/*NotoSansCJK*tc*.otf",
                "**/*NotoSansCJK*.ttc",
                "*NotoSansTC*.otf",
                "*SourceHanSans*TC*.otf",
                "**/*SourceHanSans*TC*.otf",
                "**/*SourceHanSans*.ttc",
                "**/*wqy*.ttc",
                "**/*uming*.ttc",
                "*JhengHei*.ttf",
                "*msjh*.ttc",
                "*mingliu*.ttc",
                "*kaiu*.ttf",
            ]:
                matches = list(font_dir.glob(pattern))
                if matches:
                    return str(matches[0])
    return _download_pdf_cjk_font()

def create_pdf_report(title: str, report_text: str, sources: Optional[List[Dict]] = None) -> Optional[bytes]:
    """將目前分析結果轉成 PDF bytes；若 ReportLab 或中文字型不可用則回傳 None。"""
    global LAST_PDF_EXPORT_ERROR
    LAST_PDF_EXPORT_ERROR = ""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from xml.sax.saxutils import escape
    except Exception as e:
        LAST_PDF_EXPORT_ERROR = (
            f"ReportLab 載入失敗：{str(e)}。"
            f"Streamlit 目前 Python：{sys.executable}。請在同一個環境執行 `python -m pip install reportlab`，並重新啟動 Streamlit。"
        )
        logger.warning(f"PDF 匯出不可用，{LAST_PDF_EXPORT_ERROR}")
        return None

    font_path = _find_pdf_cjk_font()
    pdf_font_name = "ReportCJK"
    if not font_path:
        LAST_PDF_EXPORT_ERROR = (
            "找不到可嵌入的 TrueType/OpenType 中文字型。"
            f"Streamlit 目前 Python：{sys.executable}。系統已嘗試 Windows/Linux 字型路徑與自動下載 Noto Sans TC；"
            "若仍失敗，請確認執行環境允許連線到 GitHub，或在環境中安裝 Noto Sans TC / Source Han Sans TC。"
        )
        logger.warning(f"PDF 匯出不可用：{LAST_PDF_EXPORT_ERROR}")
        return None
    try:
        pdfmetrics.registerFont(TTFont(pdf_font_name, font_path))
    except Exception as e:
        LAST_PDF_EXPORT_ERROR = f"中文字型註冊失敗：{font_path}；錯誤：{str(e)[:300]}"
        logger.warning(f"PDF 匯出不可用：{LAST_PDF_EXPORT_ERROR}")
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=title or "analysis_report",
    )
    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "ReportBase",
        parent=styles["Normal"],
        fontName=pdf_font_name,
        fontSize=10.5,
        leading=16,
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    heading1 = ParagraphStyle("ReportH1", parent=base, fontSize=18, leading=24, spaceAfter=12, textColor=colors.HexColor("#1f2937"))
    heading2 = ParagraphStyle("ReportH2", parent=base, fontSize=14, leading=20, spaceBefore=10, spaceAfter=8, textColor=colors.HexColor("#24304f"))
    heading3 = ParagraphStyle("ReportH3", parent=base, fontSize=12, leading=18, spaceBefore=8, spaceAfter=6, textColor=colors.HexColor("#374151"))
    small = ParagraphStyle("ReportSmall", parent=base, fontSize=8.5, leading=12, textColor=colors.HexColor("#4b5563"))

    def paragraph(raw: str, style=base):
        return Paragraph(escape(_plain_markdown_text(raw)).replace("\n", "<br/>"), style)

    story = [
        Paragraph(escape(title or "分析報告"), heading1),
        Paragraph(escape(f"產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), small),
        Spacer(1, 6),
    ]

    md_text = normalize_markdown_tables((report_text or "").replace("\\n", "\n"))
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            story.append(Spacer(1, 4))
            i += 1
            continue

        if _is_markdown_table_row(line):
            table_rows = []
            while i < len(lines) and (_is_markdown_table_row(lines[i].strip()) or not lines[i].strip()):
                current = lines[i].strip()
                if current and not _is_markdown_table_separator(current):
                    cells = [paragraph(cell, small) for cell in current.strip("|").split("|")]
                    table_rows.append(cells)
                i += 1
            if table_rows:
                table = Table(table_rows, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, -1), pdf_font_name),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                story.append(table)
                story.append(Spacer(1, 8))
            continue

        if line.startswith("# "):
            story.append(paragraph(line[2:], heading1))
        elif line.startswith("## "):
            story.append(paragraph(line[3:], heading2))
        elif line.startswith("### "):
            story.append(paragraph(line[4:], heading3))
        elif line.startswith(("- ", "* ")):
            story.append(paragraph("• " + line[2:], base))
        else:
            story.append(paragraph(line, base))
        i += 1

    if sources:
        story.append(Spacer(1, 10))
        story.append(Paragraph("引用來源", heading2))
        for idx, source in enumerate(sources, 1):
            domain = get_domain_name(source.get("url", ""))
            source_title = source.get("title", "No Title")
            evidence_level = source.get("evidence_level", "")
            url = source.get("url", "")
            story.append(paragraph(f"{idx}. {domain}｜{source_title}｜{evidence_level}｜{url}", small))

    try:
        doc.build(story)
        return buffer.getvalue()
    except Exception as e:
        LAST_PDF_EXPORT_ERROR = f"PDF 建置失敗：{str(e)[:500]}"
        logger.warning(f"PDF 匯出不可用：{LAST_PDF_EXPORT_ERROR}")
        return None

def _extract_meta_content(html_text: str, property_name: str) -> str:
    patterns = [
        rf'<meta[^>]+property=["\']{re.escape(property_name)}["\'][^>]+content=["\']([^"\']+)["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']{re.escape(property_name)}["\']',
        rf'<meta[^>]+name=["\']{re.escape(property_name)}["\'][^>]+content=["\']([^"\']+)["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']{re.escape(property_name)}["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return unescape(match.group(1)).strip()
    return ""

def _html_to_readable_text(html_text: str) -> str:
    """用標準函式庫做保守的新聞正文抽取，避免額外依賴。"""
    html_text = re.sub(r'(?is)<(script|style|noscript|svg|canvas|iframe).*?</\1>', ' ', html_text)
    candidates = []
    for tag in ["article", "main"]:
        candidates.extend(re.findall(rf'(?is)<{tag}[^>]*>(.*?)</{tag}>', html_text))
    if not candidates:
        paragraphs = re.findall(r'(?is)<p[^>]*>(.*?)</p>', html_text)
        candidates = ["\n".join(paragraphs)]
    best = max(candidates, key=len) if candidates else html_text
    best = re.sub(r'(?is)<br\s*/?>', '\n', best)
    best = re.sub(r'(?is)</(p|div|section|li|h[1-6])>', '\n', best)
    best = re.sub(r'(?is)<[^>]+>', ' ', best)
    best = unescape(best)
    lines = []
    seen = set()
    for raw_line in best.splitlines():
        line = re.sub(r'\s+', ' ', raw_line).strip()
        if len(line) < 12:
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n\n".join(lines).strip()

def fetch_news_article_from_url(url: str) -> Dict[str, str]:
    """從新聞網址擷取標題、來源與正文；失敗時回傳 error。"""
    clean_url = (url or "").strip()
    if not clean_url:
        return {"error": "未提供網址"}
    if not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url
    try:
        response = requests.get(
            clean_url,
            timeout=12,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.7",
            },
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding
        html_text = response.text
        title = (
            _extract_meta_content(html_text, "og:title")
            or _extract_meta_content(html_text, "twitter:title")
        )
        if not title:
            title_match = re.search(r'(?is)<title[^>]*>(.*?)</title>', html_text)
            title = unescape(re.sub(r'\s+', ' ', title_match.group(1)).strip()) if title_match else ""
        source_name = (
            _extract_meta_content(html_text, "og:site_name")
            or get_domain_name(clean_url)
            or "網址擷取"
        )
        content = _html_to_readable_text(html_text)
        if len(content) < 120:
            return {
                "error": "已連上網址，但無法擷取足夠正文；該網站可能使用動態載入、付費牆或阻擋爬取。",
                "title": title,
                "source_name": source_name,
                "url": clean_url,
                "content": content,
            }
        return {
            "title": title or "未提供標題",
            "source_name": source_name,
            "url": clean_url,
            "content": content,
        }
    except Exception as e:
        return {"error": f"網址擷取失敗：{str(e)[:300]}", "url": clean_url}

def build_news_text_context(title: str, source_name: str, source_url: str, content: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """將使用者貼上的單篇新聞整理成既有分析流程可讀的 Source context。"""
    clean_title = (title or "未提供標題").strip()
    clean_source_name = (source_name or "使用者貼上文本").strip()
    clean_source_url = (source_url or "#").strip() or "#"
    clean_content = (content or "").strip()
    source_category = classify_source(clean_source_url) if clean_source_url != "#" else "OTHER"
    evidence_level, evidence_score, evidence_details = calculate_academic_evidence_level(
        clean_source_url,
        source_category,
        clean_content,
        clean_title,
        all_sources=None,
    )
    language_style = evidence_details.get("language_style", analyze_language_style(clean_content, clean_title))
    content_quality = evidence_details.get("content_score", 0.0)
    language_flags = language_style.get("flags", [])
    language_flags_text = ", ".join(language_flags) if language_flags else "未偵測到明顯警示"
    source = {
        "title": clean_title,
        "url": clean_source_url,
        "content": clean_content,
        "source_category": source_category,
        "evidence_level": evidence_level,
        "evidence_score": evidence_score,
        "published_date": None,
    }
    context_text = f"""
【新聞文本分析 Context】
[Source 1]
標題：{clean_title}
來源：{clean_source_name}
網址：{clean_source_url}
來源分類：{source_category}
證據強度：{evidence_level} ({evidence_score:.2f})
內容品質分數：{content_quality:.2f}
語言風格警示：{language_flags_text}

新聞全文：
{clean_content}
"""
    diagnostics = {
        "source_category": source_category,
        "evidence_level": evidence_level,
        "evidence_score": evidence_score,
        "content_quality": content_quality,
        "language_style": language_style,
        "evidence_details": evidence_details,
    }
    return context_text.strip(), [source], diagnostics

def _is_markdown_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2

def _is_markdown_table_separator(line: str) -> bool:
    stripped = line.strip().strip("|")
    cells = [cell.strip() for cell in stripped.split("|")]
    return bool(cells) and all(re.match(r"^:?-{3,}:?$", cell or "") for cell in cells)

def normalize_markdown_tables(text: str) -> str:
    """修正 LLM 常見的鬆散表格輸出，避免表格列被當成一般段落顯示。"""
    lines = text.splitlines()
    normalized = []
    i = 0
    while i < len(lines):
        if not _is_markdown_table_row(lines[i]):
            normalized.append(lines[i])
            i += 1
            continue

        table_block = []
        while i < len(lines):
            current = lines[i].strip()
            if not current:
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and _is_markdown_table_row(lines[j]):
                    i = j
                    continue
                break
            if not _is_markdown_table_row(current):
                break
            table_block.append(current)
            i += 1

        if len(table_block) >= 2:
            if not _is_markdown_table_separator(table_block[1]):
                column_count = max(1, table_block[0].count("|") - 1)
                table_block.insert(1, "|" + "|".join(["---"] * column_count) + "|")
            if normalized and normalized[-1].strip():
                normalized.append("")
            normalized.extend(table_block)
            normalized.append("")
        else:
            normalized.extend(table_block)

    return "\n".join(normalized)

def render_report_paper(report_text: str) -> None:
    """以一致的報告樣式渲染 Markdown 報告。"""
    if not report_text or not str(report_text).strip():
        st.warning("⚠️ 報告內容為空，請重新執行分析。")
        return
    cleaned_report = str(report_text).replace('\\n', '\n').replace('\\"', '"')
    cleaned_report = re.sub(r'^-{10,}\s*$', '', cleaned_report, flags=re.MULTILINE)
    lines = cleaned_report.split('\n')
    cleaned_lines = []
    for line in lines:
        if re.match(r'^\|[\s:-]+\|', line):
            cleaned_lines.append(line)
        elif re.match(r'^-{5,}\s*$', line):
            continue
        else:
            cleaned_lines.append(line)
    cleaned_report = '\n'.join(cleaned_lines)
    cleaned_report = normalize_markdown_tables(cleaned_report)
    formatted_text = format_citation_style(cleaned_report)
    html_content = markdown.markdown(formatted_text, extensions=['tables'])
    st.markdown(f'<div class="report-paper">{html_content}</div>', unsafe_allow_html=True)

def extract_report_headings(report_text: str, max_items: int = 12) -> List[str]:
    """從 Markdown 報告抽取章節標題，作為閱讀導覽。"""
    if not report_text:
        return []
    headings = []
    for raw_line in report_text.replace("\\n", "\n").splitlines():
        line = raw_line.strip()
        match = re.match(r'^(#{1,3})\s+(.+)$', line)
        if match:
            title = _plain_markdown_text(match.group(2))
            if title and title not in headings:
                headings.append(title)
        elif re.match(r'^\d+\.\s+\*\*.+\*\*', line):
            title = _plain_markdown_text(re.sub(r'^\d+\.\s+', '', line))
            if title and title not in headings:
                headings.append(title)
        if len(headings) >= max_items:
            break
    return headings

def render_report_navigation(report_text: str, key_prefix: str) -> None:
    """顯示報告章節目錄與重新渲染控制。"""
    headings = extract_report_headings(report_text)
    if not headings:
        return
    with st.expander("🧭 報告章節導覽", expanded=False):
        for idx, heading in enumerate(headings, 1):
            st.write(f"{idx}. {heading}")
        st.caption("提示：章節導覽用於快速掌握報告結構；目前 Streamlit 不支援直接跳轉到報告內錨點。")
    if st.button("🔁 重新渲染目前報告", key=f"{key_prefix}_rerender_report"):
        st.toast("已重新套用報告清理與排版")
        st.rerun()

def render_analysis_summary_cards(data: Dict[str, Any], sources: Optional[List[Dict]], validation: Optional[Dict[str, Any]] = None) -> None:
    """在長報告前提供低負擔摘要，幫助使用者先判斷品質與風險。"""
    sources = sources or []
    validation = validation or data.get("validation", {}) if isinstance(data, dict) else {}
    evidence_counts = Counter(s.get("evidence_level", "未標註") for s in sources)
    strong_count = evidence_counts.get("極強", 0) + evidence_counts.get("強", 0)
    weak_count = evidence_counts.get("中弱", 0) + evidence_counts.get("弱", 0)
    validation_score = validation.get("score") if isinstance(validation, dict) else None
    score_text = f"{validation_score:.0f}/100" if isinstance(validation_score, (int, float)) else "未驗證"
    source_count = len(sources)
    timeline_count = len(data.get("timeline", [])) if isinstance(data, dict) else 0

    st.markdown("### 📌 報告快速檢視")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("來源數", source_count)
    with c2:
        st.metric("強證據來源", strong_count)
    with c3:
        st.metric("弱證據來源", weak_count)
    with c4:
        st.metric("格式驗證", score_text)
    st.caption(f"時間軸事件：{timeline_count} 筆；此區塊是閱讀長報告前的品質提示，不取代完整分析。")

def render_changelog_page() -> None:
    """本次改版與近期調整說明（維護者請同步更新此頁內容）。"""
    st.title("📋 本次修改內容")
    st.caption("記錄介面與行為的重要變更；不含未公開之實驗功能。")

    st.markdown("""
### 介面與導覽
- **已移除應用程式標題上的版本號**（側欄與瀏覽器分頁標題改為「多元觀點解析」）。
- **新增本頁**：集中說明近期修改，方便對照舊版行為。

### 模型與金鑰（簡化）
- **僅支援 Google Gemini**：已移除 Grok、Groq、OpenRouter 與 OpenAI 備援等選項，降低設定複雜度。
- **側欄「模型與金鑰」**：只保留 Gemini Key 與 Gemini 型號選擇。
- **API 驗證**：「多元議題分析」需一併驗證 Tavily；「全球情報」以 RSS 為主時可只驗證 Gemini。
- **全球情報**：預設 **RSS 訂閱**（快速、免 Tavily）；可選 **Gemini 綜整** 或 **進階 Tavily**。
- **匯出**：Markdown／HTML 報告標題已改用「多元觀點分析報告」（無版本號）。

### 先前已實作且仍適用
- 新聞網址擷取、文本分析、PDF 匯出（含跨平台中文字型）、Markdown 表格正規化、方法論專頁等仍沿用。

---
若您有建議更新項目，可於對話中註明以便納入本頁。
    """.strip())

def render_methodology_page() -> None:
    """主頁版方法論總覽，減少側欄長文對操作流程的干擾。"""
    st.title("📚 方法論與功能實裝狀態")
    st.caption("本頁整理系統目前採用的方法、實際有接線的功能，以及使用時應注意的限制。")

    overview_tab, pipeline_tab, evidence_tab, manipulation_tab, limits_tab = st.tabs([
        "總覽", "分析流程", "證據與查核", "資訊操作", "限制與建議"
    ])

    with overview_tab:
        st.markdown("""
        ### 三種分析模式
        - **多元議題分析**：適合需要搜尋多個來源、比較立場與查核聲量的議題。
        - **新聞文本分析**：適合貼上單篇新聞，檢查語言、框架、謬誤與資訊缺口。
        - **未來發展推演**：適合在已有分析報告後，進一步做 CLA、情境規劃與預警指標。

        ### 核心方法
        - **ACH**：用多個競爭假設避免只相信單一解釋。
        - **Entman Framing**：拆解問題定義、歸因、道德評價與解方暗示。
        - **GRADE / CERQual**：用來源、內容品質、語言風格與交叉驗證估計證據強度。
        - **Cui Bono**：檢查哪些行動者可能從特定敘事或框架中受益。
        """)

    with pipeline_tab:
        st.markdown("""
        ### 多源議題分析流程
        1. 產生或編輯搜尋策略。
        2. 依搜尋視角執行 Tavily 混合搜尋與保底搜尋。
        3. 對來源做分類、公信力與證據強度評估。
        4. 檢查立場缺口並進行最多兩輪補足搜尋。
        5. 執行可選 Google Fact Check、Cofacts 關聯查詢、共識分析與資訊操作訊號偵測。
        6. 將來源與操作訊號送入 LLM 產出報告。

        ### 單篇文本分析流程
        1. 將貼上的新聞包成 `Source 1`。
        2. 先做規則式證據強度與語言風格評估。
        3. 再交給 LLM 進行文本取證式分析。
        """)

    with evidence_tab:
        st.markdown("""
        ### 已接線功能
        - **來源公信力評分**：已在來源處理階段執行。
        - **內容品質與語言風格檢測**：已用於證據強度計算。
        - **Google Fact Check**：可選功能，需在側欄開啟，會增加等待時間與 API 消耗。
        - **Cofacts**：目前是依議題關鍵字做關聯查詢，不是逐條 claim 驗證。
        - **共識分析**：已使用實際來源；若沒有明確正反分桶，會以來源類型多樣性作保守近似。
        """)

    with manipulation_tab:
        st.markdown("""
        ### 資訊操作偵測
        - **跨網域聯播**：以正文相似度找出橫跨多網域的相似內容群。
        - **敘事擴散速度**：用網域數與發布時間估計擴散密度。
        - **語義旋轉**：若資料足夠，對比同一事實在不同來源中的框架差異。
        - **傳統協調指標**：檢查重複論述比例、來源集中度與時間集中度。

        這些結果會寫入操作訊號，供報告中的「敘事操縱與資訊操作風險」引用。
        """)

    with limits_tab:
        st.markdown("""
        ### 使用限制
        - 搜尋品質受 Tavily 回傳內容、關鍵字與時間範圍影響。
        - Google Fact Check 只查得到已被索引的公開查核結果，沒有結果不代表主張為真。
        - Cofacts 是關聯查詢，需人工判讀是否與本議題完全相同。
        - 社群帳號層級 CIB 仍需平台資料，現階段只能做內容與來源層級推估。

        ### 建議使用方式
        - 先用新聞文本分析檢查單篇文章。
        - 再用多元議題分析橫向查證。
        - 對重要結論，優先看官方、學術、外電與獨立調查來源是否互相支持。
        """)

# ==========================================
# 5. UI
# ==========================================
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "🚀 多元議題分析 (Deep Analysis)"

with st.sidebar:
    st.title("多元觀點解析")
    st.caption("✨ 多源搜尋 + 新聞文本分析 + 學術方法論")
    _nav_pages = [
        "🚀 多元議題分析 (Deep Analysis)",
        "🧾 新聞文本分析 (Text Analysis)",
        "📰 全球情報 (News Feed)",
        "📚 方法論 (Methodology)",
        "📋 本次修改 (Updates)",
    ]
    _cur_nav = st.session_state.get("current_page", _nav_pages[0])
    if _cur_nav not in _nav_pages:
        _cur_nav = _nav_pages[0]
        st.session_state["current_page"] = _cur_nav

    def _sidebar_nav_to(page: str) -> None:
        if st.session_state.get("current_page") != page:
            st.session_state["current_page"] = page
            st.rerun()

    st.markdown("##### 議題與文本分析")
    st.caption("多源查證、深度報告與單篇新聞結構化分析")
    if st.button(
        "🚀 多元議題分析 (Deep Analysis)",
        key="sidebar_nav_deep",
        use_container_width=True,
        type="primary" if _cur_nav == "🚀 多元議題分析 (Deep Analysis)" else "secondary",
    ):
        _sidebar_nav_to("🚀 多元議題分析 (Deep Analysis)")
    if st.button(
        "🧾 新聞文本分析 (Text Analysis)",
        key="sidebar_nav_text",
        use_container_width=True,
        type="primary" if _cur_nav == "🧾 新聞文本分析 (Text Analysis)" else "secondary",
    ):
        _sidebar_nav_to("🧾 新聞文本分析 (Text Analysis)")

    st.markdown("##### 全球情報與方法論")
    st.caption("要聞儀表與實裝方法說明")
    if st.button(
        "📰 全球情報 (News Feed)",
        key="sidebar_nav_feed",
        use_container_width=True,
        type="primary" if _cur_nav == "📰 全球情報 (News Feed)" else "secondary",
    ):
        _sidebar_nav_to("📰 全球情報 (News Feed)")
    if st.button(
        "📚 方法論 (Methodology)",
        key="sidebar_nav_meth",
        use_container_width=True,
        type="primary" if _cur_nav == "📚 方法論 (Methodology)" else "secondary",
    ):
        _sidebar_nav_to("📚 方法論 (Methodology)")

    st.markdown("##### 關於本應用")
    st.caption("改版紀錄與更新說明")
    if st.button(
        "📋 本次修改 (Updates)",
        key="sidebar_nav_updates",
        use_container_width=True,
        type="primary" if _cur_nav == "📋 本次修改 (Updates)" else "secondary",
    ):
        _sidebar_nav_to("📋 本次修改 (Updates)")

    current_page = st.session_state.get("current_page", _nav_pages[0])
    st.markdown("---")
    analysis_mode = st.radio(
        "選擇分析引擎：",
        options=["多元深度解析 (Fusion)", "未來發展推演 (Scenario)"],
        captions=["學術框架：框架 + 邏輯偵錯", "學術框架：CLA + 預警指標"],
        index=0
    )
    analysis_depth = st.selectbox(
        "分析詳盡度",
        options=["標準", "快速", "深度"],
        index=0,
        help="快速：較短、便於瀏覽；標準：平衡完整性與可讀性；深度：較完整但輸出較長。",
        key="analysis_depth",
    )
    st.markdown("---")
    
    blind_mode = st.toggle("🙈 盲測模式", value=False)
    is_deep_analysis_page = current_page == "🚀 多元議題分析 (Deep Analysis)"
    is_feed_page = current_page == "📰 全球情報 (News Feed)"
    is_text_page = current_page == "🧾 新聞文本分析 (Text Analysis)"
    is_methodology_page = current_page == "📚 方法論 (Methodology)"
    is_changelog_page = current_page == "📋 本次修改 (Updates)"
    tavily_key = st.session_state.get("tavily_key", "")
    search_days = 30
    max_results = 30
    selected_regions = ["🇹🇼 台灣 (Taiwan)"]
    use_cache = st.session_state.get("use_cache", True)
    enable_google_fact_check = st.session_state.get("enable_google_fact_check", False)
    enable_english_for_regions = st.session_state.get("enable_english_for_regions", True)
    past_report_input = ""

    st.markdown("#### 狀態")
    st.caption(f"模型：Gemini｜詳盡度：{st.session_state.get('analysis_depth', '標準')}")
    st.caption(f"Tavily：{'已啟用' if tavily_key else '未啟用'}｜Fact Check：{'開' if enable_google_fact_check else '關'}")
    
    with st.expander("🔑 模型與金鑰", expanded=not (is_methodology_page or is_changelog_page)):
        st.info("⚠️ API Key 不會永久儲存，重新整理後需再次輸入")
        google_key = st.text_input("Gemini Key", value="", type="password", placeholder="輸入 Google AI Studio API Key", help="全站分析僅使用 Google Gemini")
        model_name = st.selectbox(
            "Gemini 模型",
            GEMINI_MODEL_OPTIONS,
            index=0,
            help="Gemini 3.1 / 3.0 preview 系列；預設 3.1 Flash Preview。",
        )
        st.session_state["gemini_model"] = model_name
        st.session_state["llm_provider"] = "Gemini"
        if (google_key or "").strip():
            st.session_state["google_api_key"] = (google_key or "").strip()
        if "3.1-pro" in model_name:
            st.caption("🚀 **3.1 Pro**：適合複雜深度分析")
        elif "3.1-flash" in model_name:
            st.caption("⚡ **3.1 Flash**：預設推薦，一般分析與搜尋策略")
        elif "3-pro" in model_name:
            st.caption("🚀 **3.0 Pro**：深度備援")
        elif "3-flash" in model_name:
            st.caption("⚡ **3.0 Flash**：快速備援")

        needs_tavily_for_validate = is_deep_analysis_page
        if st.button("🔐 驗證 API Key", help="多元議題分析需一併驗證 Tavily；全球情報以 RSS 為主時僅需驗證 Gemini。"):
            tavily_key_for_check = st.session_state.get("tavily_key", "")
            if not (google_key or "").strip():
                st.warning("⚠️ 請先輸入 Gemini Key")
            elif needs_tavily_for_validate and not (tavily_key_for_check or "").strip():
                st.warning("⚠️ 「多元議題分析」需一併驗證 Tavily：請至下方「🔍 搜尋設定」輸入 Tavily Key。若只需驗證 Gemini，請切換到「全球情報」「新聞文本分析」「方法論」或「本次修改」頁再按驗證。")
            else:
                with st.spinner("正在驗證 API Key..."):
                    is_valid, message = validate_api_keys(
                        (google_key or "").strip(),
                        (tavily_key_for_check or "").strip(),
                        require_tavily=needs_tavily_for_validate,
                    )
                    if is_valid:
                        st.success(message)
                    else:
                        st.error(message)

    if is_deep_analysis_page:
        with st.expander("🔍 搜尋設定", expanded=is_deep_analysis_page):
            tavily_key = st.text_input("Tavily Key", value="", type="password", placeholder="輸入 Tavily API Key", help="用於「多元議題分析」新聞搜尋；「全球情報」進階 Tavily 載入時亦需要。", key="tavily_key")
            if tavily_key:
                st.success("✅ Tavily 搜尋已啟用")
            else:
                st.warning("⚠️ 請輸入 Tavily Key 以啟用新聞搜尋功能")

            st.info("ℹ️ 多元議題分析會使用 Tavily 搜尋、公信力評分、平衡檢索；Google Fact Check 可由下方開關啟用")
            search_days = st.number_input("搜尋時間範圍 (天數)", min_value=1, max_value=1825, value=30, step=1, help="設定要搜尋多少天內的新聞")
            max_results = st.slider("搜尋篇數上限", 10, 100, 30, help="設定最多搜尋多少篇新聞")
            use_cache = st.toggle("💾 啟用搜尋快取", value=True, help="啟用後會快取搜尋結果24小時，節省API配額")
            st.session_state.use_cache = use_cache
            enable_google_fact_check = st.toggle(
                "🧪 啟用 Google Fact Check 查核",
                value=st.session_state.get("enable_google_fact_check", False),
                help="開啟後會抽取來源聲明並呼叫 Google Fact Check Tools API，可能增加配額消耗與等待時間。",
                key="enable_google_fact_check",
            )
            if use_cache and st.button("🗑️ 清除快取", help="清除所有過期的快取資料"):
                deleted = clear_cache()
                st.success(f"✅ 已清除 {deleted} 條過期快取")
            selected_regions = st.multiselect(
                "搜尋視角",
                ["🇹🇼 台灣 (Taiwan)", "🌏 亞洲 (Asia)", "🌍 歐洲 (Europe)", "🌎 美洲 (Americas)", "🕵️ 獨立/自媒體 (Indie)"],
                default=["🇹🇼 台灣 (Taiwan)"],
                help="視角決定保底網域與是否觸發英文檢索。"
            )
            enable_english_for_regions = st.checkbox(
                "歐洲/美洲自動加入英文檢索",
                value=st.session_state.get("enable_english_for_regions", True),
                key="enable_english_for_regions",
                help="可關閉以僅使用中文查詢對歐美網域保底。",
            )

    with st.expander("📂 匯入舊情報 (JSON還原 / 文字貼上)", expanded=False):
        uploaded_file = st.file_uploader("上傳檔案", type=["json", "md", "txt"])
        default_text = ""
        is_json_upload = False
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".json"):
                    is_json_upload = True
                    st.success(f"✅ 完整存檔: {uploaded_file.name}")
                else:
                    default_text = uploaded_file.getvalue().decode("utf-8")
                    st.success(f"✅ 文字檔: {uploaded_file.name}")
            except Exception as e:
                logger.warning(f"檔案讀取失敗: {str(e)}")
                st.warning("⚠️ 檔案讀取失敗，請檢查檔案格式")

        past_report_input = st.text_area("或貼上內容：", value=default_text, height=150)
        
        if uploaded_file and st.button("🔄 確認載入/還原"):
            if is_json_upload:
                try:
                    state_data = json.load(uploaded_file)
                    st.session_state.result = state_data.get("result")
                    st.session_state.scenario_result = state_data.get("scenario_result")
                    st.session_state.sources = state_data.get("sources")
                    st.rerun()
                except: st.error("JSON 解析失敗")
            else:
                st.toast("✅ 文字已匯入")

    # ==================== 學術方法論詳解 ====================
    st.markdown("---")
    st.markdown("### 🧠 方法論速覽")
    st.caption("完整、較易閱讀的整理請切換到「📚 方法論」頁；下方保留詳細摺疊內容供快速查閱。")
    st.markdown("")
    
    # ---------- 0. 分析流程圖（透明公開分析流程）----------
    with st.expander("0. 📊 分析流程圖 (Analysis Pipeline)", expanded=False):
        st.markdown("**整體分析流程（依執行順序）**")
        st.markdown("""
        以下為系統從「查詢輸入」到「報告輸出」的完整流程，所有步驟均在送交 LLM 前或與 LLM 協同完成，確保方法透明可檢驗。
        """)
        flow_text = """
**步驟說明：**

1. **查詢輸入** → 使用者輸入議題關鍵字與搜尋參數（時間範圍、區域等）。
2. **議題類型判定** → 以 LLM 或關鍵字將議題分類為「台灣國內 / 兩岸 / 國際」，供後續立場平衡與補足邏輯動態調整。
3. **混和搜尋** → 執行 Tavily 多軌搜尋（事實/觀點/深度 + 平衡查詢），取得原始結果。
4. **來源處理與公信力評分** → 對每筆來源分類（BLUE/GREEN/INTL/CHINA 等）、計算證據強度與公信力、必要時事實查核。
5. **立場平衡評估（依議題類型）** → 依議題類型檢查缺口（國內：藍/綠/官方；國際：INTL 視角；兩岸：藍/綠 + 中國 + 國際）。
6. **Gap-Fill 補足** → 若平衡未達標，依缺失立場生成補充關鍵字（含 INTL_PERSPECTIVE、CHINA 等）並再次搜尋，最多 2 輪。
7. **跨網域聯播偵測** → 以正文相似度聚類，只保留「橫跨多個不同網域」的群組，識別聯播/PR 推送。
8. **敘事擴散速度** → 對每個聯播群組計算「網域數 / 時間跨度」，標記高擴散速度為高風險協調敘事。
9. **語義旋轉偵測** → 取最大聯播群組之代表來源與一篇對立陣營來源，以 LLM 辨識「共享事實」與雙方「框架/Spin」，產出 Spin Score。
10. **操作信號注入** → 將聯播群組摘要與語義旋轉結果寫入 [MANIPULATION_SIGNALS]，一併送入 LLM。
11. **LLM 戰略分析** → 依系統提示執行 ACH、Entman、邏輯謬誤偵測（12 類：滑坡、假兩難、預設謬誤、稻草人、訴諸人身、訴諸權威、訴諸情感、因果謬誤、以偏概全、訴諸無知、循環論證、紅鯡魚）、共識分析、敘事操縱風險等，產出 [DATA_TIMELINE] 與 [REPORT_TEXT]。
12. **報告輸出** → 解析時間軸與報告內文，呈現簡潔表格與引用。
        """
        st.markdown(flow_text.strip())
        st.markdown("---")
        st.markdown("**流程圖（Mermaid）** — 可將下方程式碼複製至 [Mermaid Live Editor](https://mermaid.live) 檢視圖形：")
        mermaid_code = """
flowchart LR
    A[查詢輸入] --> B[議題類型判定]
    B --> C[混和搜尋]
    C --> D[來源處理與公信力]
    D --> E[立場平衡評估]
    E --> F{Gap-Fill 補足}
    F -->|未達標| C
    F -->|達標| G[跨網域聯播偵測]
    G --> H[敘事擴散速度]
    H --> I[語義旋轉偵測]
    I --> J[操作信號注入]
    J --> K[LLM 戰略分析]
    K --> L[報告輸出]
"""
        st.code(mermaid_code.strip(), language="mermaid")
    
    # 強制顯示：確保這些區塊一定會顯示
    with st.expander("1. 資訊檢索：混和權重與多層次查詢擴展 (Hybrid Weighted Search)", expanded=False):
        st.markdown("""
        **核心機制：混和權重搜尋**
        - **分眾保底 (Safety Net)**：強制開啟專用通道，確保藍營、綠營、官方至少各抓取 8 篇代表性文章，保障弱勢觀點入場。
        - **熱度補完 (Volume Fill)**：剩餘名額開放給全網熱度排序，反映真實輿論聲量。
        
        **三軌搜尋架構 (Tri-Track via Dynamic Keywords)**
        將「通用搜尋 (General)」任務拆解為三組不同目的的指令，確保抓取內容的維度完整：
        1. **事實與時序 (Facts & Timeline)**
           - 指令：`{query} 新聞 事件 時間軸`
           - 目標：確保報告的骨架（人、事、時、地、物）是準確的。
        2. **觀點與爭議 (Opinions & Controversy)**
           - 指令：`{query} 評論 觀點 爭議 分析`
           - 目標：捕捉不同陣營（正方/反方）的論述邏輯，這是 Entman 框架分析的原料。
        3. **深度與結構 (Deep Dive)**
           - 指令：`{query} 懶人包 重點 影響`
           - 目標：快速獲取議題的全貌、背景知識與結構化資訊（如法規比較）。
        
        **多層次查詢擴展機制 (Multi-Level Query Expansion)**
        為提升搜尋覆蓋廣度 (Search Coverage & Recall)，系統採用三層擴展策略：
        - **詞彙級擴展**：基礎三軌（事實/觀點/深度）
        - **語義級擴展**：使用 LLM 生成 6-8 個語義相關的查詢變體，涵蓋不同表達方式和專業術語
        - **語境級擴展**：時間維度（最新發展、歷史背景）與觀點維度（支持觀點、反對觀點、中立分析）
        
        **多維度平衡檢索 (Multi-Perspective Retrieval)**
        為避免過濾氣泡 (Filter Bubble)，系統自動生成三組平衡查詢：
        - **正方觀點查詢**：生成 3-4 個支持該議題的觀點/論述查詢
        - **反方觀點查詢**：生成 3-4 個反對該議題的觀點/論述查詢
        - **中立學術分析查詢**：生成 3-4 個中立的學術研究、數據分析查詢
        
        **議題類型動態判定 (Issue Typology)**
        搜尋前先以 LLM 或關鍵字將查詢分類為：
        - **TAIWAN_DOMESTIC**：台灣選舉、藍綠、立委等 → 平衡標準為藍/綠/官方
        - **CROSS_STRAIT**：兩岸、台海、美中、ECFA 等 → 要求藍/綠 + 中國視角 + 國際視角
        - **INTERNATIONAL**：美國選舉、烏克蘭、NATO、國際經濟等 → 平衡標準為地理/地緣多元（INTL、CHINA 等）
        據此動態決定「缺失立場」定義與補足關鍵字，避免國際議題被誤判為缺藍/缺綠。
        
        **主動補足機制 (Active Gap-Filling)**
        當初次搜尋後檢測到立場缺口（balance_score < 0.7），系統會主動進行補充搜尋：
        - 最多執行 2 次補充迭代
        - 依議題類型生成補足關鍵字：**國內**缺 BLUE/GREEN/OFFICIAL；**國際**缺 INTL_PERSPECTIVE（如：global view, international news, 國際觀點）；**兩岸**缺 CHINA 或 INTL_PERSPECTIVE（如：中國觀點、北京立場、外電）
        - 每次補充最多 3 個查詢，每個查詢 5 筆結果
        - 動態調整直到達到平衡（balance_score ≥ 0.7）或達最大迭代次數
        
        **搜尋視角與關鍵字語言 (Region & Keyword Language)**
        - **台灣**：保底搜尋限定藍/綠/官方網域，並可設 Tavily `country: taiwan`。
        - **亞洲**：增加亞洲國際媒體保底（INTL_ASIA_WHITELIST）；**並自動產出日文與韓文關鍵字**，對日本/韓國網域執行日文/韓文檢索（各最多 2 條查詢），以覆蓋在地媒體。若查詢含日本/韓國關鍵字也會以中文再加日本/韓國媒體保底。
        - **歐洲/美洲（建議一+二已實作）**：勾選歐洲或美洲時，（1）**建議二**：各建一組**區域保底**任務（Europe_Guard / Americas_Guard），以主查詢（中文）+ 該區白名單**併入 INTL_WEST_WHITELIST（西方調查／國際權威）**、精簡 15 個網域、`search_depth=advanced`；（2）**建議一**：並以 LLM 翻譯為英文關鍵字，對歐洲/美洲/**西方調查**網域執行非中文檢索（最多 3 條英文查詢）；結果皆與中文搜尋**共用 URL 去重**。
        - **建議三（透明度與可控）**：側欄提供「搜尋視角說明」摺疊與核取方塊「當勾選歐洲/美洲時，自動加入英文關鍵字檢索」；可關閉以僅用中文查詢保底。策略表下方可顯示「英文檢索關鍵字（上次執行）」供檢視。
        - **視角→觸發對應**：台灣→藍/綠/官方保底 + country；亞洲→亞洲國際保底 + **日文/韓文關鍵字檢索（日本/韓國媒體）**（+ 查詢含日韓關鍵字時中文日韓保底）；歐洲→Europe_Guard + 可選英文檢索；美洲→Americas_Guard + 可選英文檢索；**獨立→Indie_Guard（獨立/自媒體白名單約 20 網域保底搜尋）+ 來源分類標為 INDIE**。
        - **若需更多非中文檢索**：可於「意圖導向」填寫「請產出英文關鍵字」或在「檢視與編輯搜尋策略」中手動加入英文查詢。
        
        **搜尋深度優化**
        - 通用搜尋使用 `basic` 深度（平衡速度與品質）
        - 保底搜尋使用 `advanced` 深度（確保最高相關性）
        - topic 設為 `general`（包含新聞、社群媒體如 YouTube/PTT/Dcard）
        """)
        
    with st.expander("2. 來源公信力評分系統 (Source Credibility Scoring)", expanded=False):
        st.markdown("""
        **四層評分架構 (Four-Tier Scoring System)**
        參考 GRADE 標準與 Media Bias/Fact Check 機制，建立系統化的來源公信力評分：
        
        **Tier 1：最高證據強度 (0.90-0.95)**
        - **學術機構** (.edu, .ac.uk, .ac.jp, .ac.tw)：權重係數 1.5
        - **政府機構** (.gov.tw, .gov, .gov.uk)：權重係數 1.4
        - 用途：官方原始文檔、同儕評審論文、權威機構正式報告
        
        **Tier 2：高證據強度 (0.75-0.85)**
        - **國際權威媒體** (BBC, Reuters, AP, Bloomberg, WSJ, NYT, The Guardian, DW)：權重係數 1.3
        - **獨立媒體機構** (報導者、端傳媒、關鍵評論網)：權重係數 1.2
        - 用途：專業媒體深度調查、獨立媒體機構報告
        
        **Tier 3：中等證據強度 (0.60-0.70)**
        - **商業媒體**：權重係數 1.0
        - 用途：一般媒體報導、專家評論、組織聲明
        
        **Tier 4：低證據強度 (0.30-0.40)**
        - **社群媒體** (Facebook, Twitter, YouTube, PTT, Dcard)：權重係數 0.5
        - 用途：個人觀點、未經查證的轉述
        
        **RAG 權重應用 (RAG Weighting)**
        在檢索增強生成 (RAG) 過程中，根據來源公信力動態調整：
        - **高公信力來源** (≥0.8)：完整展示內容，優先引用，詳細標註
        - **中等公信力來源** (0.6-0.8)：正常展示內容
        - **低公信力來源** (<0.6)：縮短內容至 50%，標註「⚠️ 低可信度，請謹慎參考」警告
        
        **評分計算公式（已上調文本內容權重，扎實內文易達強/中強）**
        ```
        證據強度 = 內容品質 (46%) + 語言風格 (17%) + 來源類型 (10%) + 公信力 (10%) + 交叉驗證 (12%) + 網站品質 (5%) + 利益衝突 (扣分)
        ※ 內容導向：高品質內文可彌補來源評級；content_score≥0.55 且 style_score≥0.55 時享有保底加分
        ```
        """)
        
    with st.expander("3. 事實查核與去謠言機制 (Fact-Checking Integration)", expanded=False):
        st.markdown("""
        **Google Fact Check Tools API（可選）**
        系統提供可選的 Google Fact Check Tools API 管線；只有在側欄開啟「啟用 Google Fact Check 查核」時，才會對搜尋結果進行二次驗證，以避免預設消耗 API 配額：
        
        **三階段處理流程**
        1. **聲明提取 (Claim Extraction)**
           - 使用 LLM 從標題和內容摘要中提取核心聲明（主張）
           - 每個聲明標註來源 ID 和上下文
           - 批次處理（每 5 個來源一批）以優化效能
        
        2. **API 驗證 (API Verification)**
           - 異步並行驗證（最多 5 個並發）
           - 查詢 Google Fact Check Tools API
           - 語言設定：zh-TW（繁體中文）
           - 時間範圍：最多查詢一年內的查核結果
        
        3. **結果標註與降權 (Result Tagging & Weighting)**
           - **VERIFIED_FALSE（已證偽）**：
             * 標籤：❌ 已證偽
             * 降權：證據強度降 2 級（A → C，中等 → 弱）
             * 在報告中突出顯示警告
        
           - **MISLEADING / PARTLY_FALSE（誤導性/部分錯誤）**：
             * 標籤：⚠️ 誤導性內容
             * 降權：證據強度降 1 級（A → B，強 → 中等）
        
           - **UNVERIFIED（未驗證）**：保持原狀
        
        **Cofacts 關聯查詢**
        - 依使用者輸入的議題關鍵字查詢 g0v Cofacts 謠言資料庫
        - 補充與該議題相關的社群查核紀錄
        - 此功能不是逐條 claim 驗證；逐條聲明查核需開啟 Google Fact Check
        
        **水平閱讀法 (Lateral Reading)**
        - 採用史丹佛歷史教育群 (SHEG) 提倡之方法
        - 不只深讀單一來源，而是橫向比對多個獨立來源以確認事實
        - 透過多來源搜尋實作交叉驗證
        """)
    
    with st.expander("4. 資訊操作與敘事操縱偵測 (CIB & Semantic Spin)", expanded=False):
        st.markdown("""
        **跨網域聯播偵測 (Cross-Domain Syndication Detection)**
        以**正文**（非僅標題）計算相似度，找出「同一內容橫跨多個不同網域」的群組：
        - **向量化與聚類**：TfidfVectorizer + 餘弦相似度（或 n-gram Jaccard 降級），相似度 ≥ 閾值（預設 0.75）者合併為同一群組
        - **關鍵篩選**：只保留**至少 2 個不同網域**的群組（同一站內多篇重複視為分頁/轉載，不計為聯播網）
        - **輸出**：每群組含 source_indices、domains、unique_domain_count、mean_similarity
        
        **敘事擴散速度 (Narrative Diffusion Velocity)**
        對每個跨網域聯播群組計算時間與擴散程度：
        - **time_span_hours** = 群組內最早與最晚發布時間差（小時）
        - **velocity** = unique_domain_count / time_span_hours（網域/小時）
        - **風險標記**：velocity ≥ 0.5 且至少 2 網域 → 高風險協調敘事；≥ 0.2 → 中風險
        
        **語義旋轉偵測 (Semantic Spin Detection)**
        當存在跨網域聯播群組時，取「最大群組」中一則為**敘事 A**，另選一則**不在群組內、優先不同 source_category** 的來源為**敘事 B**：
        - 將 A、B 的標題與正文前 500 字送交 LLM（預設 Gemini 3.1 Flash Preview）
        - 要求輸出：**共享客觀事實**、**A 的框架/Spin**、**B 的框架/Spin**、**Spin Score (0~1，>0.6 為高操弄)**
        - 結果寫入 [MANIPULATION_SIGNALS]，供報告中「敘事操縱與資訊操作風險」區塊引用
        
        **操作信號注入 (Manipulation Signals)**
        - 聯播群組摘要（篇數、網域數、擴散速度、風險等級）與語義旋轉結果（若 Spin Score > 0.6）合併為 [MANIPULATION_SIGNALS]
        - 一併送入 LLM 系統提示，使分析師能明確引用「跨網域洗稿」與「對立框架」證據
        """)
        
    with st.expander("5. False Balance 防範機制 (Evidence Weight Assessment)", expanded=False):
        st.markdown("""
        **證據權重評估（改進項目：新增）**
        避免「虛假平衡」（False Balance），根據證據強度而非觀點數量分配權重。
        參考 ABC 編採政策：「balance that follows the weight of evidence」
        
        **共識等級分類**
        - **強共識 (strong_consensus)**：權威來源比例 ≥70%，品質分 ≥0.7 → 權重 1.0
        - **中等共識 (moderate)**：權威來源比例 ≥50%，品質分 ≥0.6 → 權重 0.7
        - **分歧觀點 (divided)**：權威來源比例 ≥30% → 權重 0.5
        - **弱證據 (weak)**：權威來源比例 ≥15% → 權重 0.3，標註「⚠️ 證據薄弱，非主流觀點」
        - **邊緣觀點 (marginal)**：權威來源比例 <15% → 權重 0.15，標註「⚠️ 證據極弱，邊緣觀點」
        
        **權威來源定義**
        - Tier 1：官方機構（OFFICIAL）、學術機構（.edu, .ac.tw）
        - Tier 2：國際權威媒體（BBC, Reuters, AP, NYT 等）、獨立媒體（報導者、端傳媒等）
        - 計算權威來源數量和比例
        
        **品質分數計算**
        - 平均證據強度分數（evidence_score）
        - 考慮來源公信力（credibility_score）
        - 綜合評估來源品質
        
        **False Balance 防範**
        - 系統自動檢測邊緣觀點並標註警告
        - AI 分析時明確要求：「根據證據權重分配篇幅，避免 false balance」
        - 在報告中明確標示「主流共識」vs「邊緣觀點」
        - 避免將科學共識與邊緣主張等同處理
        
        **應用場景**
        - 在共識分析中為每個觀點計算證據權重
        - 在平衡報導中根據權重分配篇幅
        - 在 UI 中顯示 false_balance_warning 警示
        """)
        
    with st.expander("6. 學術級證據強度分級 (Academic Evidence Grading)", expanded=False):
        st.markdown("""
        **GRADE 標準參考 (Grading of Recommendations Assessment, Development and Evaluation)**
        參考 GRADE 系統與 CERQual（Confidence in the Evidence from Reviews of Qualitative research）方法：
        
        **證據類型分級（閾值已微調，使扎實文本易達強/中強）**
        - **A+ / 極強 (≥0.85)**：官方原始文檔 + 多源交叉驗證
        - **A / 強 (0.65-0.84)**：權威來源或扎實內容品質
        - **B+ / 中強 (0.50-0.64)**：一般媒體 + 基本品質
        - **B / 中等 (0.35-0.49)**：商業媒體報導
        - **C / 中弱 (0.22-0.34)**：低品質來源
        - **D / 弱 (<0.22)**：社群媒體、內容農場
        
        **多維度評分系統（內容權重已上調）**
        ```
        證據強度 = f(內容品質, 語言風格, 來源類型, 來源公信力, 交叉驗證, 網站品質, 利益衝突)
        ※ 以文章內容為主：長度、引用、事實密度、非聳動風格，高品質者可提升評級
        
        其中：
        - 內容品質權重：46%（長度、完整性、引用、相關性；已上調）
        - 語言風格權重：17%（非聳動、非標題黨）
        - 來源類型權重：10%（GRADE Tier）
        - 來源公信力權重：10%
        - 交叉驗證權重：12%（多個獨立來源確認）
        - 網站品質權重：5%；利益衝突：扣分
        ```
        
        **內容品質評估 (CERQual)**
        - **完整性 (Completeness)**：是否有完整的敘述、背景資訊、結論
        - **相關性 (Relevance)**：與議題核心的相關程度
        - **可信度 (Credibility)**：是否提供引用來源、是否有交叉驗證
        - **適用性 (Applicability)**：是否適用於當前情境
        
        **交叉驗證機制**
        - 檢查是否有其他獨立來源支持相同主張
        - 共識度計算：支持該主張的獨立來源數 / 總相關來源數
        - 高共識度 (>70%)：證據強度 +0.2
        - 中共識度 (40-70%)：證據強度 +0.1
        - 低共識度 (<40%)：可能為爭議點，證據強度不增加
        
        **利益衝突檢測**
        - 檢測內容中的利益關係標記（贊助、廣告、業配、合作、投資、股東）
        - 如果檢測到利益衝突，證據強度扣 0.15
        
        **報告表格與詳盡度原則**
        - 報告內表格（ACH、整體現況、爭議點、Entman 框架等）**可依需要詳盡填寫**：每格可多句或分點說明，以**完整論證與引用**為優先
        - 讀者偏好**詳細、可深入檢視**的分析；支持/反對證據、戰略影響等欄位請充分展開，勿僅一句話帶過
        """)
        
    with st.expander("7. 框架分析：Entman 理論與立場判定 (Framing Analysis)", expanded=False):
        st.markdown("""
        **Entman 框架理論 (Framing Theory)**
        分析文本如何透過「選擇 (Selection)」與「凸顯 (Salience)」來建構現實：
        
        **三維度分析框架**
        1. **問題定義 (Problem Definition)**
           - 不同陣營如何框定議題核心？
           - 使用什麼詞彙來描述議題？
           - 強調哪些面向（經濟/政治/社會/道德）？
           - 如何設定議題的邊界？
        
        2. **歸因分析 (Causal Attribution)**
           - **責任歸咎**：主要責任者（個人/組織/系統/外部因素）
           - **歸因傾向**：內因（個人能力/意圖）vs 外因（環境/結構）
           - **歸因明確性**：明確歸因 vs 模糊歸因
        
        3. **道德評價 (Moral Evaluation)**
           - **正面修辭**：讚揚、支持、正當化
           - **負面修辭**：批評、譴責、妖魔化
           - **道德框架**：正義/不義、公平/不公平、合法/非法
        
        **框架對比分析**
        - 針對**主要利益相關陣營**（依議題動態判斷：如親美/親中/在地保守/國際自由派等），非硬性限定藍/綠
        - 對比至少 2-3 個不同陣營的框架，識別共通點與差異點，標註框架衝突與共識
        
        **立場平衡檢測 (Stance Balance Assessment)**
        - **陣營覆蓋**：檢查各陣營（藍/綠/官方/中立/國際）的來源數量和質量
        - **觀點光譜**：將觀點放置在光譜上（支持/中立/反對），檢查分佈
        - **缺口檢測**：自動識別缺失的立場或觀點
        - **平衡建議**：生成平衡性報告，建議補充的搜尋方向
        
        **機構層次驗證**
        - 結合媒體所有權結構 (Ownership) 與過往政治傾向資料庫 (DB_MAP)
        - 對文章立場進行雙重驗證（來源層 + 內容層）
        - 靜態分類（媒體歷史立場）+ 動態評估（內容情感極性）
        """)
        
    with st.expander("8. ACH 競爭假設分析 (Analysis of Competing Hypotheses)", expanded=False):
        st.markdown("""
        **ACH 方法論 (Analysis of Competing Hypotheses)**
        ACH 是情報分析中避免認知偏誤的系統性方法，要求分析師同時考慮多個可能的解釋。
        
        **假設生成**
        - 必須列出所有可能的解釋假設（至少 3 個）
        - 包含「零假設」（沒有特殊情況）
        - 避免僅關注單一假設
        
        **證據評估**
        - 對每個假設，明確列出：
          * **支持證據**：Source ID + 證據描述
          * **反對證據**：Source ID + 證據描述
        - 評估證據的強度和可信度
        
        **可信度評估**
        - 使用量化或半量化方式評估每個假設的可信度
        - 標註「高/中/低」或使用 0-100 分數
        
        **不確定性標註**
        - 對於證據不足的情況，必須明確標註「證據不足」或「無法確定」
        - 嚴禁臆測，若證據不足直接標示
        
        **輸出格式與詳盡度**
        - 表格**可詳盡填寫**：每格可多句或分點，支持/反對證據欄請充分展開（多筆證據可分點或拆成多列），並標註 Source ID
        - 讀者偏好詳細分析；若有多筆證據，建議分別列出或合併成完整段落，勿僅一句摘要
        """)
        
    with st.expander("9. 邏輯謬誤偵測 (Logic Fallacy Detection)", expanded=False):
        st.markdown("""
        **系統性邏輯掃描 (Systematic Logic Scan)**
        AI 會自動掃描文本中的邏輯謬誤，識別論證缺陷。
        
        **常見謬誤類型（參考常見批判性思維教材）**
        - **滑坡謬誤 (Slippery Slope)**：誇大小事與大災難之間的因果連結，如「若 A 則必然 B、C、D…最終大禍臨頭」
        - **假兩難悖論 (False Dilemma)**：將複雜議題簡化為非黑即白的二元選項，如「不是盟友就是敵人」
        - **預設謬誤／乞題 (Begging the Question)**：論證的結論已預設在前提中，循環論證，如以「因為不該做 X」來論證「不該做 X」
        - **稻草人論證 (Straw Man)**：扭曲或誇大對手觀點以便攻擊，而非回應真實論點
        - **訴諸人身 (Ad Hominem)**：攻擊論者的人格、動機或背景，而非其論點本身
        - **訴諸權威 (Appeal to Authority)**：過度依賴權威身分而非證據或推理
        - **訴諸情感 (Appeal to Emotion)**：用恐懼、憤怒、同情等情感訴求替代理性論證
        - **因果謬誤 (Causal Fallacy)**：混淆相關性與因果關係，或倒果為因
        - **以偏概全 (Hasty Generalization)**：從少數案例或不具代表性樣本推論整體
        - **訴諸無知 (Appeal to Ignorance)**：主張「無法證明為假」即「為真」，或反之
        - **循環論證 (Circular Reasoning)**：前提與結論互相依賴，實質上未提供新論據
        - **紅鯡魚 (Red Herring)**：引入不相關話題轉移焦點，偏離原爭議
        
        **偵測要求**
        - 必須列出謬誤的具體位置（Source ID + 原文片段）
        - 說明謬誤類型和影響程度
        - 分析謬誤如何影響論述的可信度
        
        **輸出格式**
        | 謬誤類型 | 來源 (Source ID) | 原文片段（摘要） | 分析說明 |
        |:---|:---|:---|:---|
        """)
        
    with st.expander("10. 共識分析 (Consensus Analysis)", expanded=False):
        st.markdown("""
        **多維度平衡檢索後的共識分析**
        在執行多維度平衡檢索（正方/反方/中立）後，系統會進行共識分析：
        
        **共同事實識別**
        - 列出所有立場都認同的核心事實
        - 標註支持這些事實的來源（Source ID）
        - 評估事實的可信度（基於來源權威性）
        - 區分「事實」與「解釋」
        
        **分歧點分析**
        - 識別各方觀點的主要分歧點
        - 分析分歧的**根本原因**：
          * **價值觀差異**：例如經濟效率 vs 環境保護
          * **利益衝突**：例如產業利益 vs 公共利益
          * **資訊差異**：例如不同數據來源或解釋方式
        
        **共識度評估**
        - **高共識 (>70%)**：各方對核心事實有高度共識
        - **中共識 (40-70%)**：有部分共識，但存在明顯分歧
        - **低共識 (<40%)**：各方觀點高度對立
        
        **立場平衡度計算**
        - 計算各立場（正方/反方/中立）的來源比例
        - 評估觀點光譜分佈是否平衡
        - 識別話語權失衡（特定陣營聲音被過度放大或壓制）
        
        **輸出格式**
        **共同事實表**：
        | 事實描述 | 支持來源 (Source ID) | 可信度評估 |
        |:---|:---|:---|
        
        **分歧點分析表**：
        | 分歧點 | 正方立場 | 反方立場 | 根本原因 | 支持來源 |
        |:---|:---|:---|:---|:---|
        
        **共識度總結**：
        - 共識度等級：[高/中/低]
        - 共識分數：X%
        - 主要分歧領域：[列出 1-3 個]
        """)
        
    with st.expander("11. 網軍協調行為偵測 (Coordinated Behavior Detection)", expanded=False):
        st.markdown("""
        **協調行為偵測（整合 Phase 1～4 與既有邏輯）**
        檢測組織性資訊操作（Coordinated Inauthentic Behavior, CIB）特徵，分為**進階正文聯播偵測**與**傳統指標**兩層：
        
        **進階：跨網域聯播與敘事擴散（Phase 1）**
        - **正文相似度聚類**：以 TfidfVectorizer + 餘弦相似度（或 Jaccard）對**正文**做聚類，僅保留「橫跨至少 2 個不同網域」的群組，區分聯播網與站內轉載
        - **敘事擴散速度**：velocity = 網域數 / 時間跨度（小時），高 velocity 標記為高風險協調敘事，結果寫入 [MANIPULATION_SIGNALS]
        
        **進階：語義旋轉偵測（Phase 4）**
        - 取最大聯播群組之代表來源與一篇對立陣營來源，以 LLM 辨識「共享事實」與雙方「框架/Spin」，產出 Spin Score（>0.6 為高操弄），一併注入報告用操作信號
        
        **傳統指標（既有邏輯）**
        - **內容相似度**：標題相似度檢測，重複內容超過 30% 可能為協調發布
        - **來源集中度**：單一域名超過 50% 可能為組織性操作
        - **時間聚集**：同一天發布超過 40% 可能存在同步操作
        
        **協調性分數計算**
        - 重複內容 > 30%：+0.4
        - 域名集中度 > 50%：+0.3
        - 時間集中度 > 40%：+0.2
        - 總分 ≥ 0.6：標註「🚨 高風險：檢測到明顯的協調行為特徵」
        
        **限制說明**
        - 進階聯播/語義旋轉依正文與 LLM，傳統指標依標題、域名、日期
        - 完整 CIB 分析仍需社群媒體 API 進行帳號層級分析（co-tweet、網絡結構等）
        """)
        
    with st.expander("12. 聲量權重校正 (Volume Weight Analysis)", expanded=False):
        st.markdown("""
        **重複論述檢測 (Duplicate Narrative Detection)**
        識別「複讀機現象」，避免少數觀點被重複計算而放大影響：
        
        **相似度計算**
        - 使用加權組合的相似度算法：
          * **字元級相似度**：SequenceMatcher 相似度
          * **詞級相似度**：Jaccard 相似度（共同詞彙）
        - 綜合相似度 = 0.6 × 字元相似度 + 0.4 × 詞級相似度
        - 閾值：≥0.8 視為重複論述
        
        **快速過濾機制**
        - 先檢查標題字首相似度，快速過濾明顯不同的標題
        - 優化效能，避免不必要的計算
        
        **重複論述組識別**
        - 將相似度 ≥0.8 的標題分組
        - 每組至少包含 2 篇來源
        - 在報告中標註，避免重複計算聲量
        
        **獨特觀點識別**
        - 標註與主流論述不同的長尾觀點
        - 確保少數但重要的觀點不被淹沒
        - 特別關注獨立媒體和專家觀點
        
        **話語權失衡評估**
        - 評估是否有特定陣營的聲音被過度放大或壓制
        - 結合立場平衡分析，識別可能的偏見
        """)
        
    with st.expander("13. 戰略推演：CLA 層次分析與未來學方法 (Futures Studies)", expanded=False):
        st.markdown("""
        **CLA 層次分析法 (Causal Layered Analysis)**
        深入挖掘議題的四個層次：
        1. **表象 (Litany)**：公眾看到的事件與數據
           - 新聞標題、統計數字、表面現象
        
        2. **系統 (System)**：造成事件的社會結構與政策成因
           - 制度設計、政策框架、經濟結構、社會關係
        
        3. **世界觀 (Worldview)**：利益相關者的深層價值觀與意識形態
           - 信念體系、文化規範、哲學基礎
        
        4. **神話/隱喻 (Myth)**：潛意識中的集體焦慮或故事原型
           - 深層文化敘事、集體記憶、情感結構
        
        **場景規劃 (Scenario Planning)**
        為議題建立多種未來發展路徑：
        - **基準路徑 (Baseline Path)**：延續現有趨勢的發展
        - **轉折路徑 (Alternative Path)**：重要轉折點後的發展
        - **極端路徑 (Wild Card Path)**：極端情況下的發展
        
        **早期預警指標 (Signposts)**
        - 為每個未來情境設定具體的監測訊號
        - 可觀察、可測量的指標
        - 幫助提前識別趨勢轉折點
        
        **驗屍分析 (Pre-mortem Analysis)**
        - 假設預測失敗，反推可能的隱蔽變數
        - 識別可能導致分析失效的因素
        - 提高分析的韌性與可靠性
        
        **預警指標分類**
        - **量化指標**：經濟數據、民調數字、統計趨勢
        - **質化指標**：政策變動、重要人物表態、社會運動
        - **轉折點指標**：關鍵事件、決策時刻、外部衝擊
        """)
        
    with st.expander("14. 研究方法透明度要求 (Research Transparency)", expanded=False):
        st.markdown("""
        **方法論說明**
        - 報告必須說明使用的研究方法
        - 明確標註使用的框架（ACH、Entman、GRADE、邏輯謬誤偵測等）
        - 解釋為什麼選擇這些方法
        
        **資料來源標註**
        - 所有引用的來源都必須標註 Source ID
        - 明確列出所有資料來源的 URL
        - 標註來源的公信力等級和證據強度
        
        **局限性聲明**
        - 說明分析的局限性：
          * 資料不足：某些觀點可能未充分搜尋到
          * 時間限制：只涵蓋特定時間範圍的資料
          * 語言限制：主要搜尋中文資料
          * 來源限制：受限於可用的資料來源
        
        **不確定性標註**
        - 對於存疑或證據不足的部分，必須明確標註
        - 區分「事實」（可驗證）與「解釋」（需進一步證據）
        - 避免過度自信的結論
        
        **可重現性**
        - 報告生成時間、搜尋參數、使用模型都有記錄
        - 支援匯出完整狀態（JSON）以便後續分析
        - 快取機制確保相同查詢可重現結果
        
        **利益衝突聲明**
        - 系統會自動檢測並標註可能的利益衝突
        - 標註贊助、廣告、業配等關係
        - 降低可能受利益影響的來源權重
        """)
        
    st.markdown("### 📥 報告匯出")
    if st.session_state.get('result') or st.session_state.get('scenario_result'):
        html_report = create_full_html_report(st.session_state.result, st.session_state.scenario_result, st.session_state.sources, blind_mode)
        st.download_button("📥 列印用檔案 (HTML)", html_report, "Printable_Report.html", "text/html")
        full_state_json = export_full_state()
        st.download_button("📥 完整狀態 (JSON)", full_state_json, "Full_State.json", "application/json")
        
        result = st.session_state.get('result')
        export_data = None
        if result and isinstance(result, dict):
            export_data = result.copy()
            scenario_result = st.session_state.get('scenario_result')
            if scenario_result and isinstance(scenario_result, dict):
                report_text = export_data.get('report_text', '')
                scenario_text = scenario_result.get('report_text', '')
                if scenario_text:
                    export_data['report_text'] = report_text + "\n\n# 未來發展推演報告\n" + scenario_text
        else:
            st.error("❌ 無法匯出：結果資料格式錯誤")
        
        # 只有在 export_data 有效時才顯示下載按鈕
        if export_data is not None:
            st.download_button("📥 純文字 (Markdown)", convert_data_to_md(export_data), "report.md", "text/markdown")

if st.session_state["current_page"] == "📋 本次修改 (Updates)":
    render_changelog_page()
elif st.session_state["current_page"] == "📚 方法論 (Methodology)":
    render_methodology_page()
elif st.session_state["current_page"] == "📰 全球情報 (News Feed)":
    render_news_feed_page(google_key, tavily_key, model_name or st.session_state.get("gemini_model", DEFAULT_GEMINI_MODEL))
elif st.session_state["current_page"] == "🧾 新聞文本分析 (Text Analysis)":
    st.title("🧾 新聞文本分析")
    st.caption("貼上新聞網址即可分析；系統會自動擷取標題、來源與正文，再用 GRADE/CERQual、Entman、邏輯謬誤、深層偏見與 Cui Bono 框架進行文本取證式分析。")

    if "text_analysis_result" not in st.session_state:
        st.session_state.text_analysis_result = None
    if "text_analysis_sources" not in st.session_state:
        st.session_state.text_analysis_sources = None
    if "text_analysis_diagnostics" not in st.session_state:
        st.session_state.text_analysis_diagnostics = None

    news_url = st.text_input("新聞網址", placeholder="https://...", key="text_news_url")
    st.caption("多數公開新聞頁可直接擷取；若遇到付費牆、動態載入或網站阻擋，再於下方手動補全文。")
    text_col1, text_col2 = st.columns([2, 1])
    with text_col1:
        news_title = st.text_input("新聞標題（選填，網址擷取失敗時可補）", placeholder="可留空，系統會從網址擷取", key="text_news_title")
    with text_col2:
        news_source = st.text_input("來源 / 媒體名稱（選填）", placeholder="可留空，系統會從網址擷取", key="text_news_source")
    news_body = st.text_area(
        "新聞全文（選填）",
        height=220,
        placeholder="通常可留空。若網址擷取失敗或內容不完整，再手動貼上全文。",
        key="text_news_body",
    )

    st.info("此模式不需要 Tavily 搜尋 Key；貼網址會直接擷取該頁內容。若要進一步橫向查證，可把分析後的關鍵主張拿到「多元議題分析」搜尋。")
    analyze_text_btn = st.button("🧠 擷取網址並分析", type="primary")

    if analyze_text_btn:
        if not news_url.strip() and not news_body.strip():
            st.error("請先貼上新聞網址；若網址無法擷取，再手動貼上新聞全文。")
            st.stop()

        google_key_effective = (st.session_state.get("google_api_key") or "").strip() or (google_key or "").strip()
        effective_key = google_key_effective
        effective_model = model_name or st.session_state.get("gemini_model", DEFAULT_GEMINI_MODEL)

        if not effective_key:
            st.error("請先在側欄輸入可用的 LLM API Key。")
            st.stop()

        with st.status("🧠 正在進行新聞文本分析...", expanded=True) as status:
            extracted_title = news_title.strip()
            extracted_source = news_source.strip()
            extracted_url = news_url.strip()
            extracted_body = news_body.strip()
            if news_url.strip() and not extracted_body:
                st.write("1. 從新聞網址擷取標題、來源與正文...")
                fetched_article = fetch_news_article_from_url(news_url)
                if fetched_article.get("error"):
                    st.warning(f"網址擷取不完整：{fetched_article['error']}")
                    if not fetched_article.get("content"):
                        status.update(label="❌ 擷取失敗", state="error", expanded=False)
                        st.stop()
                extracted_title = extracted_title or fetched_article.get("title", "")
                extracted_source = extracted_source or fetched_article.get("source_name", "")
                extracted_url = fetched_article.get("url", extracted_url)
                extracted_body = fetched_article.get("content", extracted_body)
                st.write(f"   ↳ 已擷取約 {len(extracted_body)} 字，來源：{extracted_source or '未知'}")
            else:
                st.write("1. 使用手動提供的新聞文本...")
            if len(extracted_body.strip()) < 120:
                st.error("正文內容太短，無法進行可靠分析。請手動補上新聞全文。")
                status.update(label="❌ 內容不足", state="error", expanded=False)
                st.stop()

            context_text, text_sources, diagnostics = build_news_text_context(extracted_title, extracted_source, extracted_url, extracted_body)
            st.write("2. 建立單篇新聞 Source context...")
            st.write(f"   ↳ 證據強度：{diagnostics['evidence_level']} ({diagnostics['evidence_score']:.2f})")
            st.write("3. 執行文本取證、框架分析與邏輯謬誤掃描...")
            try:
                raw_report = run_strategic_analysis(
                    extracted_title or "新聞文本分析",
                    context_text,
                    effective_model,
                    effective_key,
                    mode="TEXT_ANALYSIS",
                    fast_mode=False,
                    manipulation_signals="單篇貼上新聞文本，未執行跨網域聯播偵測。",
                    analysis_depth=st.session_state.get("analysis_depth", "標準"),
                )
            except Exception as e:
                st.error(f"❌ 新聞文本分析失敗：{str(e)[:500]}")
                status.update(label="❌ 分析失敗", state="error", expanded=False)
                st.stop()

            validation = validate_ai_output_format(raw_report, "TEXT_ANALYSIS")
            parsed_data = parse_gemini_data(raw_report)
            parsed_data["validation"] = validation
            st.session_state.text_analysis_result = parsed_data
            st.session_state.text_analysis_sources = text_sources
            st.session_state.text_analysis_diagnostics = diagnostics
            st.session_state.text_analysis_article = {
                "title": extracted_title,
                "source": extracted_source,
                "url": extracted_url,
            }
            status.update(label="✅ 新聞文本分析完成", state="complete", expanded=False)
        st.rerun()

    if st.session_state.get("text_analysis_diagnostics"):
        diagnostics = st.session_state.text_analysis_diagnostics
        st.markdown("---")
        st.markdown("### 📌 文本初步評估")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("證據強度", diagnostics.get("evidence_level", "未知"))
        with m2:
            st.metric("證據分數", f"{diagnostics.get('evidence_score', 0):.2f}")
        with m3:
            st.metric("內容品質", f"{diagnostics.get('content_quality', 0):.2f}")
        flags = diagnostics.get("language_style", {}).get("flags", [])
        if flags:
            st.warning("；".join(flags))
        else:
            st.success("未偵測到明顯的標題黨、聳動或情緒操控警示。")

    if st.session_state.get("text_analysis_result"):
        st.markdown("---")
        st.markdown("### 📝 新聞文本分析報告")
        text_result = st.session_state.text_analysis_result
        validation = text_result.get("validation", {}) if isinstance(text_result, dict) else {}
        if validation and validation.get("score", 100) < 70:
            st.warning(f"⚠️ AI 輸出格式驗證分數：{validation.get('score', 0):.1f}/100")
            if validation.get("missing_sections"):
                st.caption("缺少章節：" + "、".join(validation.get("missing_sections", [])))
        render_analysis_summary_cards(text_result, st.session_state.get("text_analysis_sources"), validation)
        render_report_navigation(text_result.get("report_text", ""), "text_analysis")
        render_report_paper(text_result.get("report_text", ""))
        text_pdf = create_pdf_report(
            "新聞文本分析報告",
            text_result.get("report_text", ""),
            st.session_state.get("text_analysis_sources"),
        )
        if text_pdf:
            st.download_button(
                "📄 下載新聞文本分析 (PDF)",
                text_pdf,
                "news_text_analysis.pdf",
                "application/pdf",
            )
        else:
            st.warning(f"PDF 匯出不可用：{LAST_PDF_EXPORT_ERROR or '請確認已安裝 reportlab 並可讀取 Windows 中文字型。'}")
        st.download_button(
            "📥 下載新聞文本分析 (Markdown)",
            convert_data_to_md(text_result),
            "news_text_analysis.md",
            "text/markdown",
        )
        article_meta = st.session_state.get("text_analysis_article", {}) or {}
        cross_check_query = (article_meta.get("title") or "").strip()
        if cross_check_query:
            if st.button("🔎 用此新聞主題進行多源查證", type="secondary"):
                st.session_state["query"] = cross_check_query
                st.session_state.keyword_plan = None
                st.session_state["current_page"] = "🚀 多元議題分析 (Deep Analysis)"
                st.rerun()
else:
    st.title(f"{analysis_mode.split(' ')[0]}")
    query = st.text_input(
        "輸入議題關鍵字",
        value=st.session_state.get("query", ""),
        placeholder="例如：台積電美國設廠爭議",
        key="query_input",
    )
    focus_instruction = st.text_input(
        "意圖導向 (選填)",
        placeholder="例如：Focus on economic security, ignore gossip；或：著重法律影響、忽略八卦",
        help="引導關鍵字生成方向，留空則由系統自動生成。可填「請產出英文關鍵字」或「請同時給出英文搜尋詞」以加強非中文檢索。"
    )

    if 'result' not in st.session_state: st.session_state.result = None
    if 'scenario_result' not in st.session_state: st.session_state.scenario_result = None
    if 'sources' not in st.session_state: st.session_state.sources = None
    if 'cofacts_rumors' not in st.session_state: st.session_state.cofacts_rumors = []
    if 'volume_analysis' not in st.session_state: st.session_state.volume_analysis = None
    if 'stance_analysis' not in st.session_state: st.session_state.stance_analysis = None
    if 'keyword_plan' not in st.session_state: st.session_state.keyword_plan = None

    # ---------- Step 1: Generate Search Strategy ----------
    gen_btn = st.button("🧠 生成搜尋策略 (Generate Search Strategy)")
    if gen_btn and query:
        use_cache_enabled = st.session_state.get('use_cache', True)
        google_key_for_keywords = (st.session_state.get("google_api_key") or "").strip() or (google_key or "").strip()
        if google_key_for_keywords:
            with st.spinner("正在生成搜尋關鍵字..."):
                expanded = generate_expanded_queries(
                    query, google_key_for_keywords, max_expansions=15, use_cache=use_cache_enabled,
                    focus_instruction=focus_instruction.strip() or None
                )
            kw_list = [q["query"] for q in expanded]
        else:
            kw_list = [f"{query} 新聞 事件", f"{query} 爭議 評論", f"{query} 懶人包 分析", f"{query} 最新發展", f"{query} 分析"]
        st.session_state.keyword_plan = pd.DataFrame({"Keyword": kw_list, "Active": [True] * len(kw_list)})
        st.success(f"已生成 {len(kw_list)} 個關鍵字，請在下方檢視並編輯後再執行搜尋。")
        st.rerun()

    # ---------- Step 2: Review Strategy (editable table) ----------
    final_keywords = []
    if st.session_state.keyword_plan is not None:
        st.markdown("**🕵️ 檢視與編輯搜尋策略**（勾選保留、取消勾選排除；可新增/刪除列）")
        edited_df = st.data_editor(
            st.session_state.keyword_plan,
            num_rows="dynamic",
            column_config={"Keyword": st.column_config.TextColumn("關鍵字", width="large"), "Active": st.column_config.CheckboxColumn("使用", default=True)},
            use_container_width=True,
            key="keyword_plan_editor"
        )
        st.session_state.keyword_plan = edited_df
        final_keywords = edited_df[edited_df["Active"]]["Keyword"].astype(str).tolist()
        final_keywords = [k for k in final_keywords if k and k.strip()]
        # 建議三：顯示上次執行時使用的英文/日文/韓文檢索關鍵字（僅供參考，不可編輯）
        last_en = st.session_state.get("last_english_queries", [])
        last_ja = st.session_state.get("last_japanese_queries", [])
        last_ko = st.session_state.get("last_korean_queries", [])
        if last_en and isinstance(last_en, list) and len(last_en) > 0:
            with st.expander("🌐 英文檢索關鍵字（上次執行）", expanded=False):
                st.caption("勾選歐洲/美洲並執行分析後，系統會自動產出以下英文查詢用於非中文檢索；僅供參考。")
                for i, q in enumerate(last_en, 1):
                    st.text(f"{i}. {q}")
        if (last_ja and isinstance(last_ja, list) and len(last_ja) > 0) or (last_ko and isinstance(last_ko, list) and len(last_ko) > 0):
            with st.expander("🌏 日文／韓文檢索關鍵字（上次執行）", expanded=False):
                st.caption("勾選亞洲並執行分析後，系統會自動產出日文與韓文查詢用於日本/韓國媒體檢索；僅供參考。")
                if last_ja and len(last_ja) > 0:
                    st.markdown("**日文**")
                    for i, q in enumerate(last_ja, 1):
                        st.text(f"{i}. {q}")
                if last_ko and len(last_ko) > 0:
                    st.markdown("**韓文**")
                    for i, q in enumerate(last_ko, 1):
                        st.text(f"{i}. {q}")

    # ---------- Step 3: Execute Analysis ----------
    search_btn = st.button("🚀 執行搜尋與分析 (Execute Analysis)", type="primary")

    if search_btn and query and tavily_key:
        st.session_state.result = None
        st.session_state.scenario_result = None

        # 決定使用的關鍵字：若有檢視過的策略則用編輯後列表，否則即時生成
        if final_keywords:
            dynamic_keywords = final_keywords
        else:
            dynamic_keywords = None  # 將在 status 內生成

        with st.status("🚀 啟動多元觀點分析引擎...", expanded=True) as status:

            if dynamic_keywords is None:
                st.write("🧠 1. 生成動態搜尋策略（未預先檢視，即時生成）...")
                use_cache_enabled = st.session_state.get('use_cache', True)
                google_key_for_keywords = (st.session_state.get("google_api_key") or "").strip() or (google_key or "").strip()
                if google_key_for_keywords:
                    dynamic_keywords = generate_dynamic_keywords(query, google_key_for_keywords, use_cache=use_cache_enabled, focus_instruction=focus_instruction.strip() or None)
                else:
                    dynamic_keywords = [f"{query} 新聞 事件", f"{query} 爭議 評論", f"{query} 懶人包 分析"]
            else:
                st.write("🧠 1. 使用您檢視/編輯後的搜尋策略...")
            st.write(f"   ↳ 鎖定戰略關鍵字: {', '.join(dynamic_keywords[:10])}{' ...' if len(dynamic_keywords) > 10 else ''}")

            regions_label = ", ".join([r.split(" ")[1] for r in selected_regions])
            st.write(f"📡 2. 執行混和權重搜尋 (視角: {regions_label})...")
            _has_japan = any(kw in query for kw in ["日本", "自民黨", "岸田", "東京", "nhk", "日經"])
            _has_asia = "亞洲" in str(selected_regions)
            guard_desc = "分眾保底 (藍/綠/官方)"
            if _has_japan or _has_asia:
                guard_desc += " + 亞洲/日本國際媒體保底"
            st.write(f"   ↳ 啟動機制：{guard_desc} + 熱度補完 (動態三軌)")

            # 驗證 Tavily Key
            if not tavily_key:
                st.error("❌ 錯誤：未提供 Tavily API Key，無法執行搜尋")
                status.update(label="❌ 搜尋失敗", state="error", expanded=False)
                st.stop()

            # 檢查是否啟用快取（從 session_state 讀取，預設為 True）
            use_cache_enabled = st.session_state.get('use_cache', True)

            # === 測試搜尋功能（診斷用）===
            # 先執行一個簡單的測試搜尋，驗證 API 是否正常
            st.write("🧪 執行 API 連接測試...")
            try:
                test_tavily = TavilyClient(api_key=tavily_key)
                test_response = test_tavily.search(query="台灣新聞", max_results=3, search_depth="basic")
                if isinstance(test_response, dict) and 'results' in test_response:
                    test_count = len(test_response.get('results', []))
                    if test_count > 0:
                        st.success(f"✅ API 測試成功：找到 {test_count} 筆測試結果")
                        logger.info(f"API 測試成功：找到 {test_count} 筆結果")
                    else:
                        st.warning(f"⚠️ API 測試：查詢 '台灣新聞' 返回 0 筆結果（API 可能正常，但查詢無結果）")
                        logger.warning(f"API 測試：查詢 '台灣新聞' 返回 0 筆結果")
                else:
                    st.error(f"❌ API 測試失敗：回應格式異常 - {type(test_response).__name__}")
                    logger.error(f"API 測試失敗：回應格式異常 - {type(test_response).__name__}")
            except Exception as e:
                error_str = str(e)
                st.error(f"❌ API 測試失敗：{error_str[:200]}")
                logger.error(f"API 測試失敗：{error_str[:300]}")
                if "401" in error_str or "Unauthorized" in error_str:
                    st.error("認證失敗：請檢查 API Key 是否正確")
                elif "429" in error_str:
                    st.error("配額問題：API 配額可能已用完")
        
            # 執行搜尋（整合所有功能）
            st.write("🔍 開始執行完整搜尋...")
            google_key_for_analysis = (st.session_state.get("google_api_key") or "").strip() or (google_key or "").strip()
            if st.session_state.get("enable_google_fact_check", False) and not google_key_for_analysis:
                st.warning("⚠️ 已勾選 Google Fact Check，但未提供 Gemini/Google API Key；本輪將略過 Google Fact Check 查核。")
            search_result = get_search_context(
                query, tavily_key, search_days, selected_regions, max_results, dynamic_keywords,
                use_cache=use_cache_enabled, google_api_key=google_key_for_analysis,
                enable_english_for_regions=st.session_state.get("enable_english_for_regions", True),
                enable_google_fact_check=st.session_state.get("enable_google_fact_check", False),
            )
        
            if len(search_result) >= 8:
                context_text, sources, actual_query, is_strict_tw, stance_analysis, fact_check_results, consensus_analysis, manipulation_signals_text = search_result[:8]
            elif len(search_result) == 7:
                context_text, sources, actual_query, is_strict_tw, stance_analysis, fact_check_results, consensus_analysis = search_result
                manipulation_signals_text = ""
            else:
                # 向後相容
                context_text, sources, actual_query, is_strict_tw = search_result[:4]
                stance_analysis = None
                fact_check_results = None
                consensus_analysis = None
                manipulation_signals_text = ""
        
            st.write(f"   ↳ 搜尋完成：共獲取 {len(sources)} 篇資料 (已去重)。")
            if is_strict_tw:
                st.write(f"🛡️ 網域圍籬已啟動。")
        
            st.session_state.sources = sources
        
            # 顯示立場平衡分析結果
            if stance_analysis and isinstance(stance_analysis, dict) and stance_analysis.get('missing_stances'):
                st.warning(f"⚠️ 立場平衡警告：檢測到缺失立場 {', '.join(stance_analysis['missing_stances'])}")
                if isinstance(stance_analysis, dict) and stance_analysis.get('recommendations'):
                    with st.expander("🔍 查看平衡性建議", expanded=False):
                        for rec in stance_analysis['recommendations']:
                            st.write(f"**{rec['type']}**: {rec['reason']} (優先級: {rec['priority']})")
        
            st.session_state.stance_analysis = stance_analysis
            st.session_state.fact_check_results = fact_check_results
            st.session_state.consensus_analysis = consensus_analysis
        
            st.write("🛡️ 3. 查詢 Cofacts 謠言資料庫...")
            cofacts_txt, cofacts_rumors = search_cofacts(query)
            if cofacts_txt: 
                context_text += f"\n{cofacts_txt}\n"
                st.session_state.cofacts_rumors = cofacts_rumors
            else:
                st.session_state.cofacts_rumors = []
        
            # 事實查核結果顯示（方案 1）
            if fact_check_results and isinstance(fact_check_results, dict):
                false_count = len(fact_check_results.get('false_claims', []))
                misleading_count = len(fact_check_results.get('misleading_claims', []))
                if false_count > 0 or misleading_count > 0:
                    st.warning(f"⚠️ 事實查核警告：發現 {false_count} 項已證偽聲明，{misleading_count} 項誤導性內容")
            fact_check_status = "未啟用"
            if st.session_state.get("enable_google_fact_check", False):
                if fact_check_results and isinstance(fact_check_results, dict):
                    if fact_check_results.get("error"):
                        fact_check_status = "執行失敗"
                    else:
                        checked_count = sum(len(fact_check_results.get(k, [])) for k in ["verified_claims", "false_claims", "misleading_claims", "unverified_claims"])
                        fact_check_status = f"已執行（{checked_count} 項聲明）"
                else:
                    fact_check_status = "已勾選但未執行（缺少 Google API Key 或來源）"
        
            # 聲量權重分析
            if sources:
                st.write("📊 4. 執行聲量權重校正分析...")
                volume_analysis = analyze_volume_weight(sources)
                st.write(f"   ↳ 發現 {volume_analysis['duplicate_count']} 組重複論述，{volume_analysis['unique_count']} 篇獨特觀點")
                st.session_state.volume_analysis = volume_analysis
            else:
                st.session_state.volume_analysis = None
        
            # 共識分析結果顯示（方案 3.3）
            if consensus_analysis and isinstance(consensus_analysis, dict):
                consensus_score = consensus_analysis.get('consensus_score', 0)
                consensus_level = "高" if consensus_score > 0.7 else "中" if consensus_score > 0.4 else "低"
                st.info(f"📊 共識分析：共識度 {consensus_level} (分數: {consensus_score:.2f})")
            
            st.session_state.feature_status = {
                "Tavily 搜尋": f"已執行（{len(sources)} 篇來源）" if sources else "未取得來源",
                "Google Fact Check": fact_check_status,
                "Cofacts 關聯查詢": f"已查詢（{len(st.session_state.cofacts_rumors)} 筆相關）" if st.session_state.get('cofacts_rumors') else "已查詢，無相關結果",
                "共識分析": consensus_analysis.get("analysis_status", "未執行") if isinstance(consensus_analysis, dict) else "未執行",
                "資訊操作訊號": "已產生" if manipulation_signals_text else "未產生",
            }
        
            # === 檢查來源數量是否足夠進行分析 ===
            MIN_SOURCES_REQUIRED = 1  # 降低閾值：最少需要 1 篇來源即可嘗試分析
            sources_count = len(sources) if sources else 0
            context_length = len(context_text) if context_text else 0
        
            if sources_count < MIN_SOURCES_REQUIRED:
                # 檢查日誌以獲取更詳細的錯誤資訊
                debug_info = ""
                if sources_count == 0:
                    # 檢查查詢關鍵字是否可能太特定
                    query_words = query.split()
                    is_specific_query = len(query_words) >= 3
                
                    debug_info = f"""
                
                    **🔍 調試資訊：**
                    - 所有 Tavily API 搜尋任務都未返回結果
                    - API 測試：✅ 成功（說明 API 本身正常）
                    - 查詢關鍵字：`{query}` ({len(query_words)} 個關鍵字)
                    - 這可能表示：
                      * 查詢關鍵字過於特定（{len(query_words)} 個關鍵字），在指定時間範圍內找不到相關資料
                      * 網域過濾條件過於嚴格（已選定：{', '.join(selected_regions) if selected_regions else '無'}），排除了所有結果
                      * 搜尋時間範圍過短（目前：{search_days} 天），相關內容可能不在這個時間範圍內
                
                    **🧪 診斷步驟：**
                    1. 📅 **擴大搜尋時間範圍**：將「搜尋天數」改為 90 或 180 天
                    2. 🌐 **放寬區域限制**：暫時取消「僅限台灣來源」選項
                    3. 🔍 **簡化查詢關鍵字**：嘗試只使用主要關鍵字（例如：「周冠男」或「巨人傑」）
                    4. 📋 **檢查終端機日誌**：查看詳細的搜尋任務執行情況
                    """
            
                st.error(f"""
                ❌ **搜尋結果不足，無法進行深度分析**
            
                **目前狀況：**
                - 搜尋到 {sources_count} 篇來源（最少需要 {MIN_SOURCES_REQUIRED} 篇）
                - Context 長度：{context_length} 字元
                {debug_info}
            
                **可能原因：**
                1. 查詢關鍵字過於特定或冷門，找不到足夠的相關資料
                2. Tavily API 搜尋結果不足或 API 服務異常
                3. 搜尋時間範圍設定過短（目前：{search_days} 天）
                4. 網域圍籬過於嚴格（已選定區域：{', '.join(selected_regions) if selected_regions else '無'}）
                5. Tavily API Key 無效、配額用完或服務暫時不可用
            
                **建議解決方案：**
                1. 🔍 **調整查詢關鍵字**：使用更廣泛的關鍵字或同義詞
                2. 📅 **擴大搜尋時間範圍**：增加「搜尋天數」設定（例如：改為 90 天或 180 天）
                3. 🌐 **放寬區域限制**：取消「僅限台灣來源」選項，或選擇更多區域
                4. 🔑 **檢查 API Key**：
                   - 在側邊欄點擊「驗證 API Key」按鈕
                   - 確認 Tavily API Key 是否有效且有足夠配額
                   - 檢查 [Tavily Dashboard](https://app.tavily.com/) 查看配額使用情況
                5. 🔄 **重新搜尋**：點擊「開始分析」按鈕重新執行搜尋
                6. 🧪 **測試搜尋**：嘗試使用簡單的關鍵字（如「台灣新聞」）測試 API 是否正常
            
                **注意：** 即使來源不足，系統仍可嘗試分析，但結果可能不夠完整。
                """)
            
                # 詢問用戶是否仍要繼續分析
                continue_anyway = st.checkbox("⚠️ 我了解風險，仍要繼續分析（不建議）", value=False)
                if not continue_anyway:
                    status.update(label="❌ 分析已取消：來源不足", state="error", expanded=False)
                    st.stop()
                else:
                    st.warning("⚠️ 您選擇繼續分析，但結果可能不夠完整或準確。")
        
            elif context_length < 500:
                st.warning(f"""
                ⚠️ **Context 內容過短**
            
                **目前狀況：**
                - Context 長度：{context_length} 字元（建議至少 500 字元）
                - 來源數量：{sources_count} 篇
            
                **可能原因：**
                - 來源內容過短或無法取得完整內容
                - 搜尋結果品質不佳
            
                **建議：** 嘗試調整搜尋參數或使用不同的查詢關鍵字。
                """)
        
            st.write("🧠 5. AI 進行深度戰略分析 (ACH 競爭假設 + Entman 框架 + 邏輯偵錯 + 共識分析)...")
        
            mode_code = "DEEP_SCENARIO" if "未來" in analysis_mode else "FUSION"
            analysis_context = past_report_input if (mode_code == "DEEP_SCENARIO" and past_report_input) else context_text

            effective_key = (st.session_state.get("google_api_key") or "").strip() or (google_key or "").strip()
            effective_model = model_name or st.session_state.get("gemini_model", DEFAULT_GEMINI_MODEL)

            try:
                raw_report = run_strategic_analysis(
                    query, analysis_context, effective_model, effective_key,
                    mode=mode_code, fast_mode=False,
                    manipulation_signals=manipulation_signals_text,
                    analysis_depth=st.session_state.get("analysis_depth", "標準"),
                )
            except ChatGoogleGenerativeAIError as e:
                # Gemini API 特定錯誤（通常是配額相關）
                error_msg = str(e)
                st.error(f"""
                ❌ **API 錯誤**
            
                {error_msg}
            
                **建議：**
                1. 確認 Gemini API Key 與配額：https://ai.dev/rate-limit
                2. 改選側欄 **Flash** 系列模型以降低用量
                3. 稍後再試或升級 AI Studio 方案
                """)
                status.update(label="❌ 分析失敗：API 錯誤", state="error", expanded=False)
                logger.error(f"AI 分析失敗 (ChatGoogleGenerativeAIError): {error_msg}")
                st.stop()
            except Exception as e:
                # 其他錯誤（包括 RetryError）
                from tenacity import RetryError
                error_msg = str(e)
                error_type = type(e).__name__

                # 檢查是否為重試錯誤
                if isinstance(e, RetryError) or "RetryError" in error_type:
                    # 提取原始錯誤
                    last_attempt = None
                    original_error = None
                    if hasattr(e, 'last_attempt') and e.last_attempt:
                        try:
                            if hasattr(e.last_attempt, 'exception'):
                                original_error = e.last_attempt.exception()
                                last_attempt = str(original_error) if original_error else None
                            elif hasattr(e.last_attempt, 'result'):
                                original_error = e.last_attempt.result()
                                last_attempt = str(original_error) if original_error else None
                        except Exception:
                            pass

                    original_error_msg = last_attempt if last_attempt else error_msg

                    error_display = f"""
                    ❌ **API 調用失敗（重試後仍失敗）**

                    **錯誤類型**：{error_type}

                    **可能的原因：**
                    1. Gemini 配額已耗盡
                    2. API Key 無效
                    3. 網路連接問題
                    4. API 服務暫時不可用

                    **解決方案：**
                    1. 檢查 Gemini Key 是否正確
                    2. 檢查配額：https://ai.dev/rate-limit
                    3. 改選 **Flash** 型號或等待後重試
                    4. 原始錯誤：{original_error_msg[:500]}
                    """
                    st.error(error_display)
                    status.update(label="❌ 分析失敗：API 錯誤", state="error", expanded=False)
                    logger.error(f"AI 分析失敗 ({error_type}): {original_error_msg}")
                    st.stop()
                elif "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower() or "429" in error_msg:
                    st.error(f"""
                    ❌ **API 配額已耗盡**

                    **錯誤詳情**：{error_msg[:300]}

                    **解決方案：**
                    1. 檢查配額：https://ai.dev/rate-limit
                    2. 等待配額重置（通常每分鐘/每天重置）
                    3. 切換到 gemini-3.1-flash-preview 或 gemini-3-flash-preview
                    """)
                else:
                    st.error(f"❌ AI 分析失敗：{error_msg[:500]}")

                status.update(label="❌ 分析失敗", state="error", expanded=False)
                logger.error(f"AI 分析失敗 ({error_type}): {error_msg}")
                with st.expander("🔍 錯誤詳情", expanded=False):
                    st.code(f"錯誤類型: {error_type}\n錯誤訊息: {error_msg}")
                st.stop()
        
            # 驗證 AI 輸出格式
            validation = validate_ai_output_format(raw_report, mode_code)
            if validation['score'] < 70:
                st.warning(f"⚠️ AI 輸出格式驗證分數: {validation['score']:.1f}/100")
                if validation['missing_sections']:
                    st.warning(f"缺少章節: {', '.join(validation['missing_sections'])}")
                if not validation['has_timeline']:
                    st.warning("⚠️ 未檢測到時間軸區塊")
                if not validation['has_report']:
                    st.warning("⚠️ 未檢測到報告文本區塊")
        
            # 解析報告數據（確保 raw_report 不為 None）
            if raw_report is None:
                logger.error("raw_report 為 None，無法解析")
                st.error("❌ AI 分析返回空結果，請重試")
                status.update(label="❌ 分析失敗：返回空結果", state="error", expanded=False)
                st.stop()
        
            parsed_data = parse_gemini_data(raw_report)
            parsed_data['validation'] = validation  # 保存驗證結果
        
            # 驗證解析結果（安全檢查 report_text 類型）
            report_text = parsed_data.get("report_text", "")
            # 確保 report_text 是字符串類型
            if not isinstance(report_text, str):
                logger.warning(f"report_text 不是字符串類型: {type(report_text).__name__}，嘗試轉換")
                try:
                    report_text = str(report_text)
                    parsed_data["report_text"] = report_text
                except Exception as e:
                    logger.error(f"無法將 report_text 轉換為字符串: {str(e)}")
                    report_text = ""
        
            if not report_text or (isinstance(report_text, str) and not report_text.strip()):
                st.warning("⚠️ AI 返回的報告格式可能不符合預期，嘗試使用備用解析方法...")
                # 備用方法：如果沒有找到 REPORT_TEXT，使用整個文本（排除時間軸）
                if raw_report:
                    # 移除 [DATA_TIMELINE] 區塊
                    if "### [DATA_TIMELINE]" in raw_report:
                        parts = raw_report.split("### [DATA_TIMELINE]")
                        if len(parts) > 1:
                            remaining = parts[1]
                            if "### [REPORT_TEXT]" in remaining:
                                parsed_data["report_text"] = remaining.split("### [REPORT_TEXT]")[1].strip()
                            else:
                                # 移除時間軸行
                                lines = remaining.split('\n')
                                report_lines = []
                                for line in lines:
                                    if "|" in line and len(line.split("|")) >= 3:
                                        parts_line = line.split("|")
                                        if len(parts_line) >= 3 and (re.match(r'^\d{4}-\d{2}-\d{2}', parts_line[0].strip()) or "近期" in parts_line[0]):
                                            continue
                                    report_lines.append(line)
                                parsed_data["report_text"] = '\n'.join(report_lines).strip()
                    else:
                        # 如果完全沒有標記，使用整個文本
                        parsed_data["report_text"] = raw_report.strip()
        
            st.session_state.result = parsed_data

            # 顯示解析統計
            timeline_count = len(parsed_data.get("timeline", []))
            report_length = len(parsed_data.get("report_text", ""))
            st.write(f"   ↳ 解析完成：時間軸 {timeline_count} 筆，報告長度 {report_length} 字元")
            
            status.update(label="✅ 分析完成", state="complete", expanded=False)

        st.rerun()

    # Cofacts 謠言警告區塊
    if st.session_state.get('feature_status'):
        st.markdown("---")
        with st.expander("🧭 本輪功能執行狀態", expanded=False):
            for feature_name, status_text in st.session_state.feature_status.items():
                st.write(f"**{feature_name}**：{status_text}")

    if st.session_state.get('cofacts_rumors'):
        st.markdown("---")
        with st.container():
            st.markdown("### ⚠️ Cofacts 查核警告")
            st.warning("⚠️ 發現相關謠言或爭議訊息，請注意查證")
            for rumor in st.session_state.cofacts_rumors:
                rumor_type_emoji = "❌" if rumor.get('type') == 'NOT_ARTICLE' else "⚠️"
                with st.expander(f"{rumor_type_emoji} {rumor.get('text', '')[:100]}... (判定: {rumor.get('type', 'UNKNOWN')})"):
                    st.write(f"**謠言內容**：{rumor.get('text', '')}")
                    if rumor.get('reply'):
                        st.write(f"**查核回應**：{rumor.get('reply', '')}")

    # 聲量權重分析區塊
    if st.session_state.get('volume_analysis') and st.session_state.get('sources'):
        st.markdown("---")
        st.markdown("### 📊 聲量權重分析")
        vol_analysis = st.session_state.volume_analysis
        col1, col2 = st.columns(2)
        with col1:
            st.metric("重複論述組數", vol_analysis['duplicate_count'])
        with col2:
            st.metric("獨特觀點數", vol_analysis['unique_count'])
    
        if vol_analysis['duplicate_groups']:
            with st.expander("🔍 查看重複論述詳情", expanded=False):
                for group in vol_analysis['duplicate_groups']:
                    st.write(f"**重複組 #{len(group)} 篇相似報導：**")
                    for idx in group[:5]:  # 只顯示前5篇
                        source = st.session_state.sources[idx]
                        st.write(f"- [{idx+1}] {source.get('title', 'No Title')}")
                    if len(group) > 5:
                        st.caption(f"... 還有 {len(group)-5} 篇相似報導")

    if st.session_state.result:
        # 確保 result 是字典類型
        result = st.session_state.result
        if isinstance(result, dict):
            data = result
        else:
            logger.error(f"st.session_state.result 不是字典類型: {type(result).__name__}")
            st.error(f"❌ 資料格式錯誤：預期字典類型，但收到 {type(result).__name__}")
            st.stop()
    
        render_html_timeline(data.get("timeline", []), st.session_state.sources, blind_mode)

        st.markdown("---")
        st.markdown("### 📝 平衡報導分析")
    
        report_text = data.get("report_text", "")
        render_analysis_summary_cards(data, st.session_state.get("sources"), data.get("validation", {}))
        render_report_navigation(report_text, "fusion")
    
        # 檢查報告內容是否為空
        if not report_text or not report_text.strip():
            st.warning("⚠️ 報告內容為空。可能的原因：")
            st.info("""
            1. AI 返回的格式不符合預期
            2. 解析過程中出現錯誤
            3. 請檢查終端機/控制台的錯誤訊息
        
            **建議：**
            - 嘗試重新執行分析
            - 檢查 API 金鑰是否正確
            - 查看原始返回數據（可在調試模式下）
            """)
        
            # 顯示調試信息
            with st.expander("🔍 調試信息", expanded=False):
                st.write("**原始數據結構：**")
                st.json({
                    "has_timeline": len(data.get("timeline", [])) > 0,
                    "timeline_count": len(data.get("timeline", [])),
                    "report_text_length": len(report_text),
                    "report_text_preview": report_text[:500] if report_text else "（空）"
                })
        else:
            render_report_paper(report_text)
            st.markdown("### 📥 下載目前分析結果")
            current_pdf = create_pdf_report("多元觀點分析報告", report_text, st.session_state.get("sources"))
            if current_pdf:
                st.download_button(
                    "📄 下載分析結果 (PDF)",
                    current_pdf,
                    "analysis_report.pdf",
                    "application/pdf",
                )
            else:
                st.warning(f"PDF 匯出不可用：{LAST_PDF_EXPORT_ERROR or '請確認已安裝 reportlab 並可讀取 Windows 中文字型。'}")
            st.download_button(
                "📥 下載分析結果 (Markdown)",
                convert_data_to_md(data),
                "analysis_report.md",
                "text/markdown",
            )

        if "未來" not in analysis_mode and not st.session_state.scenario_result:
            st.markdown("---")
            if st.button("🚀 將此結果餵給未來發展推演 (資訊滾動)", type="secondary"):
                with st.spinner("🔮 正在讀取前次情報，啟動 CLA 層次分析與未來推演..."):
                    current_report = data.get("report_text", "")
                    effective_key = (st.session_state.get("google_api_key") or "").strip() or (google_key or "").strip()
                    effective_model = model_name or st.session_state.get("gemini_model", DEFAULT_GEMINI_MODEL)
                    raw_text = run_strategic_analysis(
                        query, current_report, effective_model, effective_key,
                        mode="DEEP_SCENARIO",
                        analysis_depth=st.session_state.get("analysis_depth", "標準"),
                    )
                    st.session_state.scenario_result = parse_gemini_data(raw_text) 
                    st.rerun()

    if st.session_state.scenario_result:
        st.markdown("---")
        st.markdown("### 🔮 未來發展推演報告")
        scenario_data = st.session_state.scenario_result
        formatted_scenario = format_citation_style(scenario_data.get("report_text", ""))
        html_scenario = markdown.markdown(formatted_scenario, extensions=['tables'])
        render_report_navigation(scenario_data.get("report_text", ""), "scenario")
        st.markdown(f'<div class="report-paper">{html_scenario}</div>', unsafe_allow_html=True)
        scenario_pdf = create_pdf_report("未來發展推演報告", scenario_data.get("report_text", ""), st.session_state.get("sources"))
        if scenario_pdf:
            st.download_button(
                "📄 下載未來推演報告 (PDF)",
                scenario_pdf,
                "scenario_report.pdf",
                "application/pdf",
            )

    if st.session_state.sources:
        st.markdown("---")
        st.markdown("### 📚 引用文獻列表")
        md_table = "| 編號 | 媒體/網域 | 標題摘要 | 證據強度 | 連結 |\n|:---:|:---|:---|:---|:---|\n"
        for i, s in enumerate(st.session_state.sources):
            domain = get_domain_name(s.get('url'))
        
            media_name = domain
            for k, v in DOMAIN_NAME_MAP.items():
                if k in domain: media_name = v
            
            if blind_mode: media_name = "*****"
        
            title = s.get('title', 'No Title')
            if len(title) > TITLE_TRUNCATE_LENGTH: title = title[:TITLE_TRUNCATE_LENGTH] + "..."
        
            # 顯示證據強度標記
            evidence_level = s.get('evidence_level', '中等')
            evidence_emoji = (
                "🟢" if evidence_level in ("強", "極強") else
                "🟡" if evidence_level in ("中等", "中強") else
                "🟠" if evidence_level == "中弱" else "🔴"
            )
            url = s.get('url')
            evidence_mark = f"{evidence_emoji} {evidence_level}" if 'evidence_level' in s else ""
            md_table += f"| **{i+1}** | `{media_name}` | {title} | {evidence_mark} | [點擊]({url}) |\n"
        st.markdown(md_table)
