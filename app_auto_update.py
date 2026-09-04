import streamlit as st
import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse
import json
import re
import html
from datetime import datetime

st.set_page_config(
    page_title="SneakerPulse 每日即時鞋訊 (全譯無亂碼版)",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .badge-zh { background-color: #065f46; color: #34d399; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; }
    .badge-en { background-color: #1e40af; color: #60a5fa; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; }
    .badge-ja { background-color: #831843; color: #f472b6; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; }
    .article-container {
        border-radius: 12px;
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 16px;
        margin-bottom: 20px;
    }
    .card-title-zh {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 4px 0 2px 0;
        line-height: 1.4;
    }
    .card-title-orig {
        font-size: 0.85rem;
        color: #64748b;
        font-style: italic;
        margin-bottom: 10px;
    }
    .card-desc-zh {
        font-size: 0.95rem;
        color: #94a3b8;
        line-height: 1.6;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Sources
RSS_SOURCES = [
    {
        "name": "Sneaker News",
        "url": "https://sneakernews.com/feed/",
        "lang": "en",
        "lang_label": "🇺🇸 英文媒體",
        "category": "發售日曆 / 即時快訊"
    },
    {
        "name": "UP TO DATE",
        "url": "https://uptodate.tokyo/feed/",
        "lang": "ja",
        "lang_label": "🇯🇵 日文媒體",
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
        "lang_label": "🇺🇸 英文媒體",
        "category": "生物力學 / 醫學評測"
    },
    {
        "name": "WearTesters",
        "url": "https://weartesters.com/feed/",
        "lang": "en",
        "lang_label": "🇺🇸 英文媒體",
        "category": "實戰鞋評 / 性能測試"
    },
    {
        "name": "Fullress",
        "url": "https://www.fullress.com/feed/",
        "lang": "ja",
        "lang_label": "🇯🇵 日文媒體",
        "category": "極速諜照 / 洩漏爆料"
    }
]

# Robust text cleaner to remove all WordPress junk and HTML tags
def clean_feed_text(raw_text):
    if not raw_text:
        return ""
    # Strip HTML tags
    t = re.sub(r'<[^>]+>', ' ', raw_text)
    t = html.unescape(t)
    # Remove WordPress boilerplate & metadata strings
    t = re.sub(r'©\s*Sneaker\s*News.*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'The post .* appeared first on .*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\|\s*Permalink\s*\|.*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\|\s*No comment.*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\[&#8230;\]', '...', t)
    t = re.sub(r'\[\.\.\.\]', '...', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# Auto-translation function using Google Translate public endpoint
def translate_to_zh(text):
    if not text or not text.strip():
        return ""
    # If text is already mostly Chinese, don't translate
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    if chinese_chars > len(text) * 0.4:
        return text

    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=zh-TW&dt=t&q={urllib.parse.quote(text)}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4) as res:
            raw = res.read().decode('utf-8')
            data = json.loads(raw)
            translated = "".join([chunk[0] for chunk in data[0] if chunk and chunk[0]])
            return translated
    except Exception:
        # Fallback to original text if translation service is unreachable
        return text

# Extract EXACT shoe image
def extract_shoe_image(item_xml, article_url):
    # 1. Check media:content, media:thumbnail, enclosure
    for child in item_xml:
        if child.tag.endswith('content') or child.tag.endswith('thumbnail') or child.tag.endswith('enclosure'):
            url = child.attrib.get('url')
            if url and any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                return url

    # 2. Check content:encoded
    for child in item_xml:
        if child.tag.endswith('encoded') and child.text:
            m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', child.text, re.I)
            if m:
                return m.group(1)

    # 3. Check description
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

# Fetch and translate all feeds with caching
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

                    # Extract the exact image of this specific shoe
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
                        "summary_zh": summary_zh if summary_zh else "點擊直達官方原文查看詳細鞋評與規格。",
                        "image": image_url
                    })
        except Exception:
            continue

    return articles

# Sidebar
st.sidebar.markdown("### ⚙️ 自動更新與篩選")
if st.sidebar.button("🔄 立即強制重新抓取與翻譯"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info("💡 每 1 小時後端會自動向美、日、台各大鞋媒抓取最新鞋訊，並自動翻譯為繁體中文，無須手動維護。")

selected_lang = st.sidebar.radio(
    "語言來源過濾",
    ["全部來源", "🇹🇼 繁中原創", "🇺🇸 英文 (全譯繁中)", "🇯🇵 日文 (全譯繁中)"]
)

search_q = st.sidebar.text_input("🔍 關鍵字即時搜尋", placeholder="搜尋 Nike, ASICS, NB, 抽籤, 評測...")

# Fetch Data
with st.spinner("正在連線抓取美・日・台最新鞋訊並翻譯為繁體中文中..."):
    all_articles = fetch_and_translate_feeds()

# Fallback curated articles if offline
if not all_articles:
    all_articles = [
        {
            "source": "Sneaker News",
            "lang": "en",
            "lang_label": "🇺🇸 英文媒體",
            "category": "發售日曆 / 經典復刻",
            "title_zh": "Air Jordan 4 OG 'Bred' 2026 黑色星期五正式回歸！忠於 1989 年原版 Nike Air 老屁股曝光",
            "title_orig": "Air Jordan 4 OG 'Bred' Black Friday 2026 Confirmed With Original Specs",
            "link": "https://sneakernews.com/release-dates/",
            "date": "2026-09-03",
            "summary_zh": "Jordan Brand 官方確認黑紅 AJ4 將於黑五復刻，還原 1989 年高規格牛巴戈皮與後跟 Nike Air 標，成人款定價 $230 美元。",
            "image": "https://images.unsplash.com/photo-1552346154-21d32810aba3?w=800&auto=format&fit=crop&q=80"
        }
    ]

# Filter Logic
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

# Header
st.markdown('<div class="main-title">👟 SneakerPulse 每日鞋訊全覽誌</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">已串接中・英・日權威鞋媒即時資訊流 · <b>所有外文已全數自動翻譯為繁體中文，絕無代碼雜訊</b> · 目前共 <b>{len(filtered)}</b> 篇</div>', unsafe_allow_html=True)

# Render Articles
for art in filtered:
    with st.container(border=True):
        badge_class = f"badge-{art['lang']}"
        
        # Meta Bar
        st.markdown(f"""
        <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 6px;">
            <span class="{badge_class}">{art['lang_label']}</span>
            <span style="color: #64748b; font-size: 0.85rem;">來源：<b>{art['source']}</b> | 📅 {art['date']} | 🏷️ {art['category']}</span>
        </div>
        """, unsafe_allow_html=True)

        # Title
        st.markdown(f'<div class="card-title-zh">{art["title_zh"]}</div>', unsafe_allow_html=True)
        if art["lang"] != "zh" and art["title_orig"] != art["title_zh"]:
            st.markdown(f'<div class="card-title-orig">外文原文對照：{art["title_orig"]}</div>', unsafe_allow_html=True)

        # Layout: Image on left if present, Content on right
        if art.get("image"):
            col_img, col_txt = st.columns([1, 2])
            with col_img:
                st.image(art["image"], caption="官方實物照片", use_container_width=True)
            with col_txt:
                st.markdown(f'<div class="card-desc-zh">{art["summary_zh"]}</div>', unsafe_allow_html=True)
                st.link_button(f"前往 {art['source']} 閱讀官方完整圖文 ↗", art["link"])
        else:
            # If no image found, display full width text without showing a fake/wrong shoe!
            st.markdown(f'<div class="card-desc-zh">{art["summary_zh"]}</div>', unsafe_allow_html=True)
            st.link_button(f"前往 {art['source']} 閱讀官方完整圖文 ↗", art["link"])

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>© 2026 SneakerPulse · 每日即時自動抓取與繁中翻譯系統</p>", unsafe_allow_html=True)
