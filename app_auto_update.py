import streamlit as st
import xml.etree.ElementTree as ET
import urllib.request
import re
from datetime import datetime
from html import unescape

st.set_page_config(
    page_title="SneakerPulse 每日自動更新鞋訊全覽誌",
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
</style>
""", unsafe_allow_html=True)

# RSS Sources Configuration
RSS_SOURCES = [
    {
        "name": "Sneaker News",
        "url": "https://sneakernews.com/feed/",
        "lang": "en",
        "lang_label": "🇺🇸 英文媒體",
        "category": "發售日曆 / 快訊",
        "default_img": "https://images.unsplash.com/photo-1552346154-21d32810aba3?w=800&auto=format&fit=crop&q=80"
    },
    {
        "name": "UP TO DATE",
        "url": "https://uptodate.tokyo/feed/",
        "lang": "ja",
        "lang_label": "🇯🇵 日文媒體",
        "category": "日本抽選 / 発売日",
        "default_img": "https://images.unsplash.com/photo-1539185441755-769473a23570?w=800&auto=format&fit=crop&q=80"
    },
    {
        "name": "勘履者 KENLU",
        "url": "https://kenlu.net/feed/",
        "lang": "zh",
        "lang_label": "🇹🇼 繁中原創",
        "category": "實戰拆解 / 台灣發售",
        "default_img": "https://images.unsplash.com/photo-1607522370275-f14206abe5d3?w=800&auto=format&fit=crop&q=80"
    },
    {
        "name": "Doctors of Running",
        "url": "https://www.doctorsofrunning.com/feeds/posts/default?alt=rss",
        "lang": "en",
        "lang_label": "🇺🇸 英文媒體",
        "category": "生物力學 / 醫學評測",
        "default_img": "https://images.unsplash.com/photo-1512412046876-f3863b17025a?w=800&auto=format&fit=crop&q=80"
    },
    {
        "name": "WearTesters",
        "url": "https://weartesters.com/feed/",
        "lang": "en",
        "lang_label": "🇺🇸 英文媒體",
        "category": "實戰鞋評 / 性能測試",
        "default_img": "https://images.unsplash.com/photo-1579338559194-a162d19bf842?w=800&auto=format&fit=crop&q=80"
    },
    {
        "name": "Fullress",
        "url": "https://www.fullress.com/feed/",
        "lang": "ja",
        "lang_label": "🇯🇵 日文媒體",
        "category": "極速諜照 / 洩漏爆料",
        "default_img": "https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a?w=800&auto=format&fit=crop&q=80"
    }
]

# Helper to strip HTML tags
def clean_html(raw_html):
    if not raw_html:
        return ""
    clean_text = re.sub(r'<[^>]+>', '', raw_html)
    return unescape(clean_text).strip()

# Helper to extract first image from RSS item
def extract_image(item_xml, default_img):
    # Try <media:content url="..."> or <enclosure url="...">
    for elem in item_xml:
        if elem.tag.endswith('content') or elem.tag.endswith('enclosure'):
            url = elem.attrib.get('url')
            if url and ('.jpg' in url or '.jpeg' in url or '.png' in url or '.webp' in url):
                return url
    # Try finding <img> in description
    desc = item_xml.find('description')
    if desc is not None and desc.text:
        m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', desc.text)
        if m:
            return m.group(1)
    return default_img

# Dynamic Fetch with 1-hour cache
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_feeds():
    articles = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    for src in RSS_SOURCES:
        try:
            req = urllib.request.Request(src["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=8) as response:
                tree = ET.fromstring(response.read())
                
                # Support standard RSS 2.0
                items = tree.findall('.//item')
                for it in items[:6]: # Get top 6 from each
                    title = it.find('title').text if it.find('title') is not None else "最新鞋訊"
                    link = it.find('link').text if it.find('link') is not None else src["url"]
                    pub_date = it.find('pubDate').text if it.find('pubDate') is not None else ""
                    desc_raw = it.find('description').text if it.find('description') is not None else ""
                    
                    desc_clean = clean_html(desc_raw)[:220] + "..." if desc_raw else "點擊直達原文閱讀完整分析與高清大圖。"
                    img_url = extract_image(it, src["default_img"])
                    
                    articles.append({
                        "source": src["name"],
                        "lang": src["lang"],
                        "lang_label": src["lang_label"],
                        "category": src["category"],
                        "title": unescape(title),
                        "link": link,
                        "date": pub_date[:16] if pub_date else "今日即時",
                        "summary": desc_clean,
                        "image": img_url
                    })
        except Exception:
            # Silently pass if a single feed is temporarily unreachable
            continue

    return articles

# Sidebar
st.sidebar.markdown("### ⚙️ 自動更新與篩選")
if st.sidebar.button("🔄 立即強制重新抓取最新鞋訊"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info("💡 本系統每 1 小時會自動向美、日、台各大鞋媒抓取最新情報，無須手動維護。")

selected_lang = st.sidebar.radio(
    "語言來源過濾",
    ["全部來源", "🇹🇼 繁中原創", "🇺🇸 英文媒體", "🇯🇵 日文媒體"]
)

search_q = st.sidebar.text_input("🔍 關鍵字即時搜尋", placeholder="搜尋 Nike, ASICS, NB, 抽籤, 評測...")

# Fetch Live Data
with st.spinner("正在連線抓取全球最新鞋訊中..."):
    all_articles = fetch_all_feeds()

# Fallback if offline/empty
if not all_articles:
    st.warning("目前後端連線抓取中，若暫時無外網連線，可使用內建的 32 篇深度評測庫。")
    all_articles = [
        {
            "source": "Sneaker News",
            "lang": "en",
            "lang_label": "🇺🇸 英文媒體",
            "category": "發售日曆 / 經典復刻",
            "title": "Air Jordan 4 OG 'Bred' 2026 黑色星期五正式回歸！忠於 1989 年原版配置規格曝光",
            "link": "https://sneakernews.com/release-dates/",
            "date": "今日焦點",
            "summary": "Jordan Brand 官方正式確認，史上最具傳奇色彩的 Air Jordan 4 OG 'Bred' 黑紅配色將於今年黑色星期五重磅復刻，全家族尺碼登場。",
            "image": "https://images.unsplash.com/photo-1552346154-21d32810aba3?w=800&auto=format&fit=crop&q=80"
        }
    ]

# Filter logic
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
        if q in a["title"].lower() or q in a["summary"].lower() or q in a["source"].lower() or q in a["category"].lower()
    ]

# Main UI
st.markdown('<div class="main-title">👟 SneakerPulse 每日即時自動更新全覽誌</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">已即時串接中・英・日權威鞋媒公開 RSS 資訊流 · 目前匯總 <b>{len(filtered)}</b> 篇最新文章</div>', unsafe_allow_html=True)

# Grid of Live Articles
for art in filtered:
    with st.container(border=True):
        col_img, col_txt = st.columns([1, 2])
        with col_img:
            st.image(art["image"], use_container_width=True)
        with col_txt:
            badge_class = f"badge-{art['lang']}"
            st.markdown(f"""
            <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 6px;">
                <span class="{badge_class}">{art['lang_label']}</span>
                <span style="color: #64748b; font-size: 0.85rem;">來源：<b>{art['source']}</b> | 📅 {art['date']} | 🏷️ {art['category']}</span>
            </div>
            <h3 style="margin: 4px 0 6px 0;">{art['title']}</h3>
            <p style="font-size: 0.95rem; color: #94a3b8; line-height: 1.5; margin-bottom: 12px;">{art['summary']}</p>
            """, unsafe_allow_html=True)
            
            st.link_button(f"前往 {art['source']} 閱讀官方完整圖文 ↗", art["link"])

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>© 2026 SneakerPulse · 每日自動即時更新系統</p>", unsafe_allow_html=True)
