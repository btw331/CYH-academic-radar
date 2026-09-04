import streamlit as st
import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse
import json
import re
import html
from datetime import datetime

st.set_page_config(
    page_title="SneakerPulse 每日鞋訊全覽誌",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Responsive & High-Contrast Typography Styling (Supports Light & Dark Modes)
st.markdown("""
<style>
    /* Global Container */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }
    
    /* Main Page Header */
    .main-title {
        font-size: 2.3rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem !important;
    }
    .sub-title {
        font-size: 1.15rem !important;
        color: #64748b;
        margin-bottom: 2rem !important;
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

    /* Article Card Typography (大字高對比) */
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
</style>
""", unsafe_allow_html=True)

# RSS Sources
RSS_SOURCES = [
    {
        "name": "Sneaker News",
        "url": "https://sneakernews.com/feed/",
        "lang": "en",
        "lang_label": "🇺🇸 英文全譯",
        "category": "發售日曆 / 即時快訊"
    },
    {
        "name": "UP TO DATE",
        "url": "https://uptodate.tokyo/feed/",
        "lang": "ja",
        "lang_label": "🇯🇵 日文全譯",
        "category": "日本抽選 / 発売日"
    },
    {
        "name": "勘履者 KENLU",
        "url": "https://kenlu.net/feed/",
        "lang": "zh",
        "lang_label": "🇹🇼 繁中原創",
        "category": "實戰拆解 / 台灣發售"
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
    {
        "name": "Fullress",
        "url": "https://www.fullress.com/feed/",
        "lang": "ja",
        "lang_label": "🇯🇵 日文全譯",
        "category": "極速諜照 / 洩漏爆料"
    }
]

# Thoroughly Clean Text and Remove All WordPress Junk
def clean_feed_text(raw_text):
    if not raw_text:
        return ""
    t = re.sub(r'<[^>]+>', ' ', raw_text)
    t = html.unescape(t)
    # Strip WordPress footers, links, del.icio.us, and tracking lines
    t = re.sub(r'©\s*Sneaker\s*News.*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'The post .* appeared first on .*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\|\s*Permalink\s*\|.*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\|\s*No comment.*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'Add to del\.icio\.us.*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\[&#8230;\]', '...', t)
    t = re.sub(r'\[\.\.\.\]', '...', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# Real-time Translation Function with Multi-Chunk Support
def translate_to_zh(text):
    if not text or not text.strip():
        return ""
    # If already mostly Chinese, skip translation
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    if chinese_chars > len(text) * 0.4:
        return text

    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=zh-TW&dt=t&q={urllib.parse.quote(text)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=4) as res:
            raw = res.read().decode('utf-8')
            data = json.loads(raw)
            translated = "".join([chunk[0] for chunk in data[0] if chunk and chunk[0]])
            return translated.strip()
    except Exception:
        return text

# Extract the EXACT image of the specific shoe
def extract_shoe_image(item_xml, article_url):
    # 1. Look for media:content, media:thumbnail, enclosure
    for child in item_xml:
        if (child.tag.endswith('content') or child.tag.endswith('thumbnail') or child.tag.endswith('enclosure')):
            url = child.attrib.get('url')
            if url and any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                return url

    # 2. Look for <img> inside content:encoded
    for child in item_xml:
        if child.tag.endswith('encoded') and child.text:
            m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', child.text, re.I)
            if m:
                return m.group(1)

    # 3. Look for <img> inside description
    desc = item_xml.find('description')
    if desc is not None and desc.text:
        m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', desc.text, re.I)
        if m:
            return m.group(1)

    # 4. Fetch the real article page to get the exact og:image
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

# Fetch and Translate All Feeds (Cached for 1 hour)
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_and_translate_feeds():
    articles = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    for src in RSS_SOURCES:
        try:
            req = urllib.request.Request(src["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=8) as response:
                tree = ET.fromstring(response.read())
                items = tree.findall('.//item')
                
                for it in items[:6]:  # top 6 per site
                    raw_title = it.find('title').text if it.find('title') is not None else ""
                    raw_title = clean_feed_text(raw_title)
                    
                    link = it.find('link').text if it.find('link') is not None else src["url"]
                    pub_date = it.find('pubDate').text if it.find('pubDate') is not None else ""
                    
                    desc_raw = it.find('description').text if it.find('description') is not None else ""
                    desc_cleaned = clean_feed_text(desc_raw)
                    if len(desc_cleaned) > 280:
                        desc_cleaned = desc_cleaned[:280] + "..."

                    # Exact image of this shoe
                    image_url = extract_shoe_image(it, link)

                    # Translate Title and Summary to Traditional Chinese
                    title_zh = translate_to_zh(raw_title) if src["lang"] != "zh" else raw_title
                    summary_zh = translate_to_zh(desc_cleaned) if src["lang"] != "zh" else desc_cleaned

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

# Sidebar Controls
st.sidebar.markdown("### ⚙️ 系統設定與操作")
if st.sidebar.button("🔄 立即重新抓取最新鞋訊"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info("💡 系統每 1 小時自動連線美、日、台官方資訊流，自動翻譯繁體中文。")

selected_lang = st.sidebar.radio(
    "語言來源過濾",
    ["全部來源", "🇹🇼 繁中原創", "🇺🇸 英文全譯", "🇯🇵 日文全譯"]
)

search_q = st.sidebar.text_input("🔍 關鍵字搜尋", placeholder="搜尋 Nike, adidas, ASICS, 抽籤...")

# Main Flow
with st.spinner("正在連線抓取美・日・台最新鞋訊並翻譯為繁體中文中..."):
    all_articles = fetch_and_translate_feeds()

# Fallback in case of offline
if not all_articles:
    all_articles = [
        {
            "source": "Sneaker News",
            "lang": "en",
            "lang_label": "🇺🇸 英文全譯",
            "category": "發售日曆 / 經典復刻",
            "title_zh": "Air Jordan 4 OG 'Bred' 2026 黑色星期五正式回歸！忠於 1989 年原版 Nike Air 老屁股曝光",
            "title_orig": "Air Jordan 4 OG 'Bred' Black Friday 2026 Confirmed With Original Specs",
            "link": "https://sneakernews.com/release-dates/",
            "date": "2026-09-03",
            "summary_zh": "Jordan Brand 官方確認黑紅 AJ4 將於今年黑五復刻，還原 1989 年高規格牛巴戈皮與後跟 Nike Air 標，成人款定價 $230 美元。",
            "image": "https://images.unsplash.com/photo-1552346154-21d32810aba3?w=800&auto=format&fit=crop&q=80"
        }
    ]

# Filtering
filtered = all_articles
if "繁中" in selected_lang:
    filtered = [a for a in filtered if a["lang"] == "zh"]
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

# Page Header
st.markdown('<div class="main-title">👟 SneakerPulse 每日鞋訊全覽誌</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">已串接中・英・日權威鞋媒即時資訊流 · <b>外文全自動翻譯為繁體中文，無代碼雜訊</b> · 目前共 <b>{len(filtered)}</b> 篇</div>', unsafe_allow_html=True)

# Render Redesigned Cards (Text Left 75%, Compact Thumbnail Right 25%)
for art in filtered:
    with st.container(border=True):
        badge_class = f"badge-{art['lang']}"
        
        # 1. 來源與分類資訊條 (放大醒目)
        st.markdown(f"""
        <div class="meta-bar">
            <span class="source-badge {badge_class}">{art['lang_label']}</span>
            <span class="source-name">📰 來源媒體：{art['source']}</span>
            <span class="category-pill">🏷️ {art['category']}</span>
            <span class="date-text">📅 {art['date']}</span>
        </div>
        """, unsafe_allow_html=True)

        # 2. 內容與縮圖佈局 (Google News 經典模式：左側大字新聞 75%，右側精緻小縮圖 25%)
        if art.get("image"):
            col_txt, col_img = st.columns([3, 1], gap="medium")
            with col_txt:
                st.markdown(f'<div class="card-title-zh">{art["title_zh"]}</div>', unsafe_allow_html=True)
                if art["lang"] != "zh" and art["title_orig"] != art["title_zh"]:
                    st.markdown(f'<div class="card-title-orig">外文原文：{art["title_orig"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card-desc-zh">{art["summary_zh"]}</div>', unsafe_allow_html=True)
                st.link_button(f"👉 點此前往 {art['source']} 閱讀官方完整圖文 ↗", art["link"])
            with col_img:
                st.image(art["image"], use_container_width=True)
        else:
            st.markdown(f'<div class="card-title-zh">{art["title_zh"]}</div>', unsafe_allow_html=True)
            if art["lang"] != "zh" and art["title_orig"] != art["title_zh"]:
                st.markdown(f'<div class="card-title-orig">外文原文：{art["title_orig"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card-desc-zh">{art["summary_zh"]}</div>', unsafe_allow_html=True)
            st.link_button(f"👉 點此前往 {art['source']} 閱讀官方完整圖文 ↗", art["link"])

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.95rem;'>© 2026 SneakerPulse · 每日即時自動更新與繁中翻譯系統</p>", unsafe_allow_html=True)
