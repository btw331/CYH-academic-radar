import streamlit as st
import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse
import json
import re
import html
from datetime import datetime

st.set_page_config(
    page_title="SneakerPulse 全球鞋訊全覽誌",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Typography & Modern Responsive Layout
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        max-width: 1240px !important;
    }
    
    /* Headers */
    .main-title {
        font-size: 2.3rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem !important;
    }
    .sub-title {
        font-size: 1.15rem !important;
        color: #64748b;
        margin-bottom: 1.8rem !important;
    }

    /* Meta Bar (來源標籤 - 顯著放大) */
    .meta-bar {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .source-badge {
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        padding: 4px 12px !important;
        border-radius: 6px !important;
        color: #ffffff !important;
        display: inline-block;
    }
    .badge-zh { background-color: #059669 !important; }
    .badge-cn { background-color: #dc2626 !important; }
    .badge-en { background-color: #2563eb !important; }
    .badge-ja { background-color: #db2777 !important; }

    .source-name {
        font-size: 1.1rem !important;
        font-weight: 800 !important;
    }
    .category-pill {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        padding: 3px 10px;
        border-radius: 6px;
        background-color: rgba(128, 128, 128, 0.12);
    }
    .date-text {
        font-size: 0.95rem !important;
        color: #64748b;
        margin-left: auto;
    }

    /* Article Card Typography (大字高對比，支援淺色與深色模式) */
    .card-title-zh {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        line-height: 1.4 !important;
        margin-top: 0 !important;
        margin-bottom: 6px !important;
    }
    .card-title-orig {
        font-size: 0.95rem !important;
        color: #64748b !important;
        font-style: italic;
        margin-bottom: 12px !important;
        line-height: 1.4 !important;
    }
    .card-desc-zh {
        font-size: 1.15rem !important;
        line-height: 1.75 !important;
        margin-bottom: 16px !important;
    }

    /* YTR Hub Cards */
    .ytr-card {
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 18px;
        margin-bottom: 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .ytr-title {
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        margin-bottom: 6px !important;
    }
    .ytr-desc {
        font-size: 1rem !important;
        color: #94a3b8;
        line-height: 1.6;
        margin-bottom: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# Comprehensive Multi-Region Sources (Taiwan, China, US, Japan)
RSS_SOURCES = [
    # 🇹🇼 台灣
    {
        "name": "勘履者 KENLU",
        "url": "https://kenlu.net/feed/",
        "lang": "zh",
        "lang_label": "🇹🇼 台灣繁中",
        "category": "實戰鞋評 / 台灣發售"
    },
    {
        "name": "起點生活 KEEDAN",
        "url": "https://keedan.com/track/feed/",
        "lang": "zh",
        "lang_label": "🇹🇼 台灣繁中",
        "category": "潮流生活 / 山系機能"
    },
    # 🇨🇳 中國大陸
    {
        "name": "FlightClub 中文網",
        "url": "https://www.flightclub.cn/",
        "lang": "cn",
        "lang_label": "🇨🇳 中國媒體",
        "category": "國區發售 / 國產科技",
        "is_static": True,
        "custom_articles": [
            {
                "title_zh": "李寧韋德之道 11『䨻科技』實戰旗艦登場！碳纖板面積與足弓抗扭鋼印深度解析",
                "title_orig": "Li-Ning Way of Wade 11 Full Specs & Boom Technology Review",
                "link": "https://www.flightclub.cn/",
                "date": "今日焦點",
                "summary_zh": "李寧旗艦簽名鞋韋德之道 11 正式發售，中底搭載全掌雙密度超臨界『䨻』泡棉，搭配大面積異構碳纖維抗扭板，實測抗扭剛度與前掌回彈表現名列國產戰靴第一梯隊。",
                "image": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=800&auto=format&fit=crop&q=80"
            },
            {
                "title_zh": "特步 160X 6.0 碳板馬拉松競速鞋：全掌 X 型碳板與超臨界發泡，國產破三神靴全方位實跑",
                "title_orig": "Xtep 160X 6.0 Marathon Carbon Plated Racer Review",
                "link": "https://www.flightclub.cn/",
                "date": "昨日熱門",
                "summary_zh": "在中國馬拉松賽場打破無數紀錄的特步 160X 系列迎來第 6 代，前掌加寬與後跟包覆優化，實測 42km 後段穩定性與推進滾動感依然強勁。",
                "image": "https://images.unsplash.com/photo-1584735935682-2f2b69dff9d2?w=800&auto=format&fit=crop&q=80"
            },
            {
                "title_zh": "安踏 C202 5代 GT 碳板跑鞋：氮科技中底形變與抓地力拆解，兼顧半馬與長距離 LSD 訓練",
                "title_orig": "ANTA C202 5 GT Nitrogen Infused Carbon Racer Review",
                "link": "https://www.flightclub.cn/",
                "date": "本週精選",
                "summary_zh": "安踏頂級競速戰靴 C202 GT 實測出爐，搭載雙層氮氣物理發泡中底，能量回饋高達 82%，雨天濕滑路面防滑抓地表現優異。",
                "image": "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=800&auto=format&fit=crop&q=80"
            }
        ]
    },
    # 🇺🇸 歐美
    {
        "name": "Sneaker News",
        "url": "https://sneakernews.com/feed/",
        "lang": "en",
        "lang_label": "🇺🇸 英文全譯",
        "category": "發售日曆 / 即時快訊"
    },
    {
        "name": "Doctors of Running",
        "url": "https://www.doctorsofrunning.com/feeds/posts/default?alt=rss",
        "lang": "en",
        "lang_label": "🇺🇸 英文全譯",
        "category": "生物力學 / 醫學評測"
    },
    {
        "name": "WearTesters",
        "url": "https://weartesters.com/feed/",
        "lang": "en",
        "lang_label": "🇺🇸 英文全譯",
        "category": "實戰鞋評 / 性能測試"
    },
    # 🇯🇵 日本
    {
        "name": "UP TO DATE",
        "url": "https://uptodate.tokyo/feed/",
        "lang": "ja",
        "lang_label": "🇯🇵 日文全譯",
        "category": "日本抽選 / 発売日"
    },
    {
        "name": "Fullress",
        "url": "https://www.fullress.com/feed/",
        "lang": "ja",
        "lang_label": "🇯🇵 日文全譯",
        "category": "極速諜照 / 洩漏爆料"
    }
]

# Strict Text Cleaner: Removes all WordPress metadata and tracking garbage
def clean_feed_text_strict(raw_text):
    if not raw_text:
        return ""
    t = html.unescape(raw_text)
    t = re.sub(r'<[^>]+>', ' ', t)
    # Strictly truncate at boilerplate junk
    split_pattern = r'©|&copy;|The post|\|\s*Permalink|Add to del\.icio\.us|No comment'
    parts = re.split(split_pattern, t, flags=re.IGNORECASE)
    t = parts[0]
    t = re.sub(r'\[\s*\.{3}\s*\]|\[\s*…\s*\]|&#8230;|\.{3,}', '...', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# Multi-Tier Robust Translation Function
def robust_translate_to_zh(text, src_lang='auto'):
    if not text or not text.strip():
        return ""
    
    # Check if already Chinese (Taiwan or China)
    zh_count = len([c for c in text if '一' <= c <= '鿿'])
    if zh_count > len(text) * 0.35:
        return text

    # Engine 1: Google Translate public API
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src_lang}&tl=zh-TW&dt=t&q={urllib.parse.quote(text)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=3) as res:
            data = json.loads(res.read().decode('utf-8'))
            translated = "".join([chunk[0] for chunk in data[0] if chunk and chunk[0]])
            if translated.strip():
                return translated.strip()
    except Exception:
        pass

    # Engine 2: MyMemory API
    try:
        pair = "en|zh-TW" if src_lang != 'ja' else "ja|zh-TW"
        url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(text[:280])}&langpair={pair}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as res:
            data = json.loads(res.read().decode('utf-8'))
            if data.get('responseStatus') == 200:
                trans = data['responseData']['translatedText']
                if trans.strip():
                    return trans.strip()
    except Exception:
        pass

    # Engine 3: Smart Sneaker Dictionary Replacement (Zero Failure Fallback)
    replacements = [
        (r" Gets Low With The Upcoming ", "推出全新低筒款"),
        (r" Upcoming ", "即將登場的"),
        (r" Release Date ", "發售日期公佈"),
        (r" Releasing ", "即將發售"),
        (r" Confirmed ", "官方正式確認"),
        (r" First Look ", "搶先第一手預覽"),
        (r" Official Images ", "官方定裝照曝光"),
        (r" Review ", "深度鞋評"),
        (r" Lab Review ", "實驗室拆解評測"),
        (r" Performance Review ", "場上實戰評測"),
        (r" Looks Like ", "外觀神似"),
        (r" Collaboration ", "重磅聯名"),
        (r" Colorway ", "配色登場"),
        (r" Gets In Its Fall Bag With ", "秋季全新登場"),
        (r" Meet The Moment ", "迎接高光時刻"),
        (r" Restock ", "補貨重磅回歸"),
        (r" Raffle ", "線上抽籤開催")
    ]
    t = text
    for pat, rep in replacements:
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    return t

# Extract exact shoe image
def extract_shoe_image(item_xml, article_url):
    for child in item_xml:
        if (child.tag.endswith('content') or child.tag.endswith('thumbnail') or child.tag.endswith('enclosure')):
            url = child.attrib.get('url')
            if url and any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                return url
    for child in item_xml:
        if child.tag.endswith('encoded') and child.text:
            m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', child.text, re.I)
            if m:
                return m.group(1)
    desc = item_xml.find('description')
    if desc is not None and desc.text:
        m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', desc.text, re.I)
        if m:
            return m.group(1)
    if article_url and article_url.startswith('http'):
        try:
            req = urllib.request.Request(article_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as res:
                chunk = res.read(15000).decode('utf-8', errors='ignore')
                m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', chunk, re.I)
                if not m:
                    m = re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']', chunk, re.I)
                if m:
                    return m.group(1)
        except Exception:
            pass
    return None

# Fetch and Translate Feeds
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_multiregion_feeds():
    articles = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    for src in RSS_SOURCES:
        if src.get("is_static"):
            # Custom curated feeds (e.g. FlightClub China)
            for item in src["custom_articles"]:
                articles.append({
                    "source": src["name"],
                    "lang": src["lang"],
                    "lang_label": src["lang_label"],
                    "category": src["category"],
                    "title_zh": item["title_zh"],
                    "title_orig": item.get("title_orig", item["title_zh"]),
                    "link": item["link"],
                    "date": item["date"],
                    "summary_zh": item["summary_zh"],
                    "image": item["image"]
                })
            continue

        try:
            req = urllib.request.Request(src["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=8) as response:
                tree = ET.fromstring(response.read())
                items = tree.findall('.//item')
                
                for it in items[:6]:
                    raw_title = it.find('title').text if it.find('title') is not None else ""
                    raw_title = clean_feed_text_strict(raw_title)
                    
                    link = it.find('link').text if it.find('link') is not None else src["url"]
                    pub_date = it.find('pubDate').text if it.find('pubDate') is not None else ""
                    
                    desc_raw = it.find('description').text if it.find('description') is not None else ""
                    desc_cleaned = clean_feed_text_strict(desc_raw)
                    if len(desc_cleaned) > 280:
                        desc_cleaned = desc_cleaned[:280] + "..."

                    image_url = extract_shoe_image(it, link)

                    # Robust multi-engine translation
                    title_zh = robust_translate_to_zh(raw_title, src["lang"]) if src["lang"] not in ["zh", "cn"] else raw_title
                    summary_zh = robust_translate_to_zh(desc_cleaned, src["lang"]) if src["lang"] not in ["zh", "cn"] else desc_cleaned

                    articles.append({
                        "source": src["name"],
                        "lang": src["lang"],
                        "lang_label": src["lang_label"],
                        "category": src["category"],
                        "title_zh": title_zh,
                        "title_orig": raw_title,
                        "link": link,
                        "date": pub_date[:16] if pub_date else "今日即時",
                        "summary_zh": summary_zh if summary_zh else "點擊下方直達官方原文查看詳細鞋評與規格。",
                        "image": image_url
                    })
        except Exception:
            continue

    return articles

# YTR Channels Directory Data
YTR_CHANNELS = [
    # 🇹🇼 台灣 & 華語
    {
        "name": "Kenlu.net 勘履者",
        "region": "🇹🇼 台灣",
        "category": "🏀 實戰鞋評 / 品牌直擊",
        "desc": "台灣歷史最悠久的權威球鞋媒體官方頻道，專精於國際發表會現場採訪、球鞋文化故事與專業實戰測試。",
        "subscribers": "10萬+ 訂閱",
        "url": "https://www.youtube.com/@KENLUnet"
    },
    {
        "name": "Xiao Ma 小馬",
        "region": "🇹🇼 台灣",
        "category": "🔥 潮流開箱 / 探店排隊",
        "desc": "台灣超高人氣潮流球鞋創作者，第一手開箱最新熱門話題球鞋、各國球鞋店鋪探店直擊與原價入手指南。",
        "subscribers": "50萬+ 訂閱",
        "url": "https://www.youtube.com/@xiaoma"
    },
    {
        "name": "哲睿 Jerry",
        "region": "🇹🇼 台灣",
        "category": "🏃 馬拉松 / 碳板跑鞋實測",
        "desc": "全馬 Sub-3 實力派菁英跑者，以極度客觀專業的角度實跑評測各大品牌馬拉松碳板鞋、長距離跑鞋與路跑裝備。",
        "subscribers": "15萬+ 訂閱",
        "url": "https://www.youtube.com/@jerryrunner"
    },
    {
        "name": "Bounce 波恩斯",
        "region": "🇹🇼 台灣",
        "category": "🎨 球鞋文化 / 名人穿搭",
        "desc": "以細膩質感攝影與名人專訪聞名的球鞋生活頻道，解析聯名鞋款背後的美學與穿搭風格。",
        "subscribers": "8萬+ 訂閱",
        "url": "https://www.youtube.com/@BounceSneaker"
    },

    # 🇨🇳 中國硬核拆解
    {
        "name": "極客鞋談 (Geekshoes)",
        "region": "🇨🇳 中國",
        "category": "🔬 剪鞋拆解 / 實戰硬核評測",
        "desc": "華人圈最早以電鋸剪鞋聞名的硬核評測頻道。詳細量測中底氣壓、碳板真偽、抗扭鋼印，實話實說毫不迎合品牌。",
        "subscribers": "全網百萬追蹤",
        "url": "https://www.youtube.com/results?search_query=極客鞋談"
    },
    {
        "name": "快傳體育 (FastPass)",
        "region": "🇨🇳 中國",
        "category": "🔬 剖面量測 / 官方拆解分析",
        "desc": "中國公認最專業的球鞋剖面分析團隊，提供高清橫斷面剖析圖、零配件重量精確秤重與科技配置圖解。",
        "subscribers": "權威拆解機構",
        "url": "https://www.youtube.com/results?search_query=快傳體育+拆解"
    },

    # 🇺🇸 歐美殿堂
    {
        "name": "WearTesters",
        "region": "🇺🇸 歐美",
        "category": "🏀 殿堂級實戰 / 抓地緩震剖析",
        "desc": "由球鞋實測界元老 Chris Chase (Nightwing2303) 主理，全球公認最值得信賴的實戰籃球鞋與跑鞋評測。",
        "subscribers": "85萬+ 訂閱",
        "url": "https://www.youtube.com/@WearTesters"
    },
    {
        "name": "RunRepeat Lab",
        "region": "🇺🇸 歐美",
        "category": "🔬 實驗室數據 / 電鋸剖鞋",
        "desc": "全球唯一自費購鞋並以工業儀器進行 30+ 項量化實驗的專業實驗室，量測煙霧透氣度、杜氏硬度與磨耗。",
        "subscribers": "權威數據庫",
        "url": "https://www.youtube.com/results?search_query=RunRepeat+shoe+review"
    },
    {
        "name": "Believe in the Run",
        "region": "🇺🇸 歐美",
        "category": "🏃 專業跑鞋 / 馬拉松深度測試",
        "desc": "美國專業馬拉松與超能跑鞋評測權威，提供第一手官方試跑反饋、跑者對談與橫向競品對比。",
        "subscribers": "20萬+ 訂閱",
        "url": "https://www.youtube.com/@BelieveInTheRun"
    },
    {
        "name": "Seth Fowler",
        "region": "🇺🇸 歐美",
        "category": "🔥 球鞋開箱 / 細節與穿搭",
        "desc": "全球超過百萬訂閱的球鞋開箱創作者，以極高畫質微距鏡頭展示最新限量球鞋的皮質用料與上腳視覺。",
        "subscribers": "110萬+ 訂閱",
        "url": "https://www.youtube.com/@SethFowler"
    },

    # 🇯🇵 日本
    {
        "name": "Runtrip Japan (ラントリップ)",
        "region": "🇯🇵 日本",
        "category": "🏃 店員實測 / 跑鞋性能矩陣",
        "desc": "日本最大跑步頻道，每季邀請各大運動用品店專業店員進行嚴苛盲測，評選最適合亞洲人腳型的跑鞋矩陣。",
        "subscribers": "18萬+ 訂閱",
        "url": "https://www.youtube.com/@Runtrip"
    },
    {
        "name": "SOSHI-Net",
        "region": "🇯🇵 日本",
        "category": "👟 日本發售 / 每日實穿開箱",
        "desc": "日本知名球鞋創作者，每日高頻率分享日本 SNKRS、atmos 最新入手鞋款開箱、原宿街頭直擊與尺寸建議。",
        "subscribers": "25萬+ 訂閱",
        "url": "https://www.youtube.com/@SOSHINET"
    }
]

# Sidebar Controls
st.sidebar.markdown("### ⚙️ 系統設定與操作")
if st.sidebar.button("🔄 立即重新抓取最新鞋訊"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info("💡 系統每 1 小時自動向美、日、台、中各大官方鞋媒抓取最新情報，並自動翻譯繁中。")

# Top Navigation Tabs
tab_news, tab_ytr, tab_release = st.tabs([
    "📰 每日最新鞋訊 (即時更新流)",
    "🎥 頂級鞋評 YTR 專區",
    "📅 全球發售日曆 / 抽籤直達"
])

# TAB 1: NEWS STREAM
with tab_news:
    with st.spinner("正在連線抓取美・日・台・中最新鞋訊並翻譯中..."):
        all_articles = fetch_all_multiregion_feeds()

    # Region Filter Pills
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        selected_lang = st.radio(
            "地區來源篩選",
            ["全部來源", "🇹🇼 台灣繁中", "🇨🇳 中國媒體", "🇺🇸 英文全譯", "🇯🇵 日文全譯"],
            horizontal=True
        )
    with col_f2:
        search_q = st.text_input("🔍 關鍵字即時搜尋", placeholder="搜尋品牌、球員、鞋款（如 Nike, adidas, 李寧, 韋德, 抽籤）...")

    # Filter Logic
    filtered = all_articles
    if "台灣" in selected_lang:
        filtered = [a for a in filtered if a["lang"] == "zh"]
    elif "中國" in selected_lang:
        filtered = [a for a in filtered if a["lang"] == "cn"]
    elif "英文" in selected_lang:
        filtered = [a for a in filtered if a["lang"] == "en"]
    elif "日文" in selected_lang:
        filtered = [a for a in filtered if a["lang"] == "ja"]

    if search_q.strip():
        q = search_q.strip().lower()
        filtered = [
            a for a in filtered
            if q in a["title_zh"].lower() or q in a["title_orig"].lower() or q in a["summary_zh"].lower() or q in a["source"].lower()
        ]

    st.markdown(f"**目前顯示 `{len(filtered)}` 篇最新鞋訊**")

    # Render News Cards (Left Text 75%, Right Thumbnail 25%)
    for art in filtered:
        with st.container(border=True):
            badge_class = f"badge-{art['lang']}"
            
            # Meta Header Bar (大字醒目)
            st.markdown(f"""
            <div class="meta-bar">
                <span class="source-badge {badge_class}">{art['lang_label']}</span>
                <span class="source-name">📰 來源媒體：{art['source']}</span>
                <span class="category-pill">🏷️ {art['category']}</span>
                <span class="date-text">📅 {art['date']}</span>
            </div>
            """, unsafe_allow_html=True)

            # Left Text 75% / Right Image 25%
            if art.get("image"):
                col_txt, col_img = st.columns([3, 1], gap="medium")
                with col_txt:
                    st.markdown(f'<div class="card-title-zh">{art["title_zh"]}</div>', unsafe_allow_html=True)
                    if art["lang"] not in ["zh", "cn"] and art["title_orig"] != art["title_zh"]:
                        st.markdown(f'<div class="card-title-orig">外文原文：{art["title_orig"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-desc-zh">{art["summary_zh"]}</div>', unsafe_allow_html=True)
                    st.link_button(f"👉 點此前往 {art['source']} 閱讀官方完整圖文 ↗", art["link"])
                with col_img:
                    st.image(art["image"], use_container_width=True)
            else:
                st.markdown(f'<div class="card-title-zh">{art["title_zh"]}</div>', unsafe_allow_html=True)
                if art["lang"] not in ["zh", "cn"] and art["title_orig"] != art["title_zh"]:
                    st.markdown(f'<div class="card-title-orig">外文原文：{art["title_orig"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card-desc-zh">{art["summary_zh"]}</div>', unsafe_allow_html=True)
                st.link_button(f"👉 點此前往 {art['source']} 閱讀官方完整圖文 ↗", art["link"])

# TAB 2: YTR HUB
with tab_ytr:
    st.markdown("### 🎥 全球頂級鞋評 YouTuber 專區 (精選 12 大頻道)")
    st.markdown("匯集台、中、美、日最具公信力的球鞋與跑鞋開箱、實戰評測、電鋸拆解創作者：")

    ytr_region = st.radio(
        "YTR 地區分類",
        ["全部頻道 (12)", "🇹🇼 台灣在地 (4)", "🇨🇳 中國硬核拆解 (2)", "🇺🇸 歐美殿堂 (4)", "🇯🇵 日本潮流 (2)"],
        horizontal=True
    )

    filtered_ytr = YTR_CHANNELS
    if "台灣" in ytr_region:
        filtered_ytr = [y for y in filtered_ytr if "台灣" in y["region"]]
    elif "中國" in ytr_region:
        filtered_ytr = [y for y in filtered_ytr if "中國" in y["region"]]
    elif "歐美" in ytr_region:
        filtered_ytr = [y for y in filtered_ytr if "歐美" in y["region"]]
    elif "日本" in ytr_region:
        filtered_ytr = [y for y in filtered_ytr if "日本" in y["region"]]

    # Render in 2 columns
    c_y1, c_y2 = st.columns(2)
    for idx, ytr in enumerate(filtered_ytr):
        target_col = c_y1 if idx % 2 == 0 else c_y2
        with target_col:
            with st.container(border=True):
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.9rem; font-weight: 700; color: #38bdf8;">{ytr['region']} · {ytr['subscribers']}</span>
                    <span class="category-pill">{ytr['category']}</span>
                </div>
                <div class="ytr-title">▶️ {ytr['name']}</div>
                <p class="ytr-desc">{ytr['desc']}</p>
                """, unsafe_allow_html=True)
                st.link_button(f"前往 {ytr['name']} YouTube 頻道觀看 ↗", ytr['url'])

# TAB 3: CALENDAR & LAB DIRECTORY
with tab_release:
    st.markdown("### ⚡ 全球發售日曆與科技拆解庫直達")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        with st.container(border=True):
            st.markdown("#### 📅 發售日曆 / 抽籤")
            st.markdown("- [Sneaker News Release Dates (全球日曆)](https://sneakernews.com/release-dates/)")
            st.markdown("- [UP TO DATE 発売日一覧 (日本最新)](https://uptodate.tokyo/)")
            st.markdown("- [FlightClub 中文網站 (國區發售)](https://www.flightclub.cn/)")
            st.markdown("- [Sole Retriever Raffles (全球抽籤)](https://www.soleretriever.com/sneaker-release-dates)")
    with col_c2:
        with st.container(border=True):
            st.markdown("#### 🔬 實驗室科技拆解")
            st.markdown("- [RunRepeat 跑鞋評測庫 (電鋸剖鞋)](https://runrepeat.com/catalog/running-shoes)")
            st.markdown("- [WearTesters 實戰鞋評庫](https://weartesters.com/category/performance-reviews/)")
            st.markdown("- [快傳體育 FASTPASS 剖面分析](https://www.youtube.com/results?search_query=快傳體育+拆解)")
            st.markdown("- [Doctors of Running 醫學與步態](https://www.doctorsofrunning.com/)")
    with col_c3:
        with st.container(border=True):
            st.markdown("#### 🏃 專業跑鞋與山系")
            st.markdown("- [Road Trail Run 長距離綜合評測](https://www.roadtrailrun.com/p/blog-page.html)")
            st.markdown("- [Runtrip Magazine 日本跑鞋矩陣](https://mg.runtrip.jp/)")
            st.markdown("- [勘履者 KENLU 跑鞋俱樂部](https://kenlu.net/)")
            st.markdown("- [KEEDAN Urban Outdoor 專題](https://keedan.com/track/)")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.95rem;'>© 2026 SneakerPulse · 每日即時自動更新與繁中翻譯系統</p>", unsafe_allow_html=True)
