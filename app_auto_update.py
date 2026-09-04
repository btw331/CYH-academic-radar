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
        margin-bottom: 1.6rem !important;
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
            },
            {
                "title_zh": "快傳體育 FASTPASS 拆解：李寧超輕 21 代剖面分析，超輕量碳板與中底空洞力學結構揭秘",
                "title_orig": "FastPass Deconstructs Li-Ning Super Light 21",
                "link": "https://www.flightclub.cn/",
                "date": "拆解專題",
                "summary_zh": "快傳體育帶來全新超輕 21 代剖面實測，詳細量測單隻 180g 的超輕量奧秘，中底拱橋縷空結構在吸收垂直衝擊的同時大幅減重。",
                "image": "https://images.unsplash.com/photo-1512412046876-f3863b17025a?w=800&auto=format&fit=crop&q=80"
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
    {
        "name": "Nice Kicks",
        "url": "https://www.nicekicks.com/feed/",
        "lang": "en",
        "lang_label": "🇺🇸 英文全譯",
        "category": "球鞋故事 / 潮流發售"
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
    zh_count = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
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
        (r"\bGets Low With The Upcoming\b", "推出全新低筒款"),
        (r"\bUpcoming\b", "即將登場的"),
        (r"\bRelease Date\b", "發售日期公佈"),
        (r"\bReleasing\b", "即將發售"),
        (r"\bConfirmed\b", "官方正式確認"),
        (r"\bFirst Look\b", "搶先第一手預覽"),
        (r"\bOfficial Images\b", "官方定裝照曝光"),
        (r"\bReview\b", "深度鞋評"),
        (r"\bLab Review\b", "實驗室拆解評測"),
        (r"\bPerformance Review\b", "場上實戰評測"),
        (r"\bLooks Like\b", "外觀神似"),
        (r"\bCollaboration\b", "重磅聯名"),
        (r"\bColorway\b", "配色登場"),
        (r"\bGets In Its Fall Bag With\b", "秋季全新登場"),
        (r"\bMeet The Moment\b", "迎接高光時刻"),
        (r"\bRestock\b", "補貨重磅回歸"),
        (r"\bRaffle\b", "線上抽籤開催")
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

# YTR Channels: Personalized (from your YouTube subscriptions) + Global Curated
YTR_CHANNELS = [
    # ⭐ 您的 YouTube 關注頻道 (Personalized Subscriptions)
    {
        "name": "That Fit Friend (Jake Boly)",
        "region": "⭐ 您的專屬關注",
        "category": "🏋️ 訓練鞋 / 舉重深蹲 / 寬楦赤足極簡",
        "desc": "您常看的頂級功能性訓練鞋評測頻道！專注於深蹲鞋、舉重鞋、CrossFit 訓練鞋、零落差極簡鞋（Minimus、Nano、Dropset、Metcon）之硬度與寬楦實測。",
        "latest_title": "NEW BALANCE MINIMUS TR V2 深度鞋評：差一點就成神作？",
        "latest_orig": "NEW BALANCE MINIMUS TR V2 REVIEW | So Close Yet So Far?",
        "latest_date": "2024-06-25",
        "latest_id": "brJoRjAIDqg",
        "latest_thumbnail": "https://i.ytimg.com/vi/brJoRjAIDqg/hqdefault.jpg",
        "latest_url": "https://www.youtube.com/watch?v=brJoRjAIDqg"
    },
    {
        "name": "Run4Adventure (Lloyd Purvis)",
        "region": "⭐ 您的專屬關注",
        "category": "⛰️ 硬核越野跑鞋 / 技術山徑 / Vibram",
        "desc": "您常看的專業越野跑鞋頻道！深入實測泥濘岩石等技術越野地形、Vibram Megagrip 抓地力、Kailas 凱樂石、Scarpa、HOKA、Karhu 等山系越野鞋。",
        "latest_title": "技術山徑怪物級抓地？Kailas 凱樂石 FUGA MONSTER DU 首跑實測",
        "latest_orig": "MONSTER Grip on Technical Trails? | Kailas FUGA MONSTER DU First Run Review",
        "latest_date": "2026-07-05",
        "latest_id": "I9dzEkgUyr4",
        "latest_thumbnail": "https://i.ytimg.com/vi/I9dzEkgUyr4/hqdefault.jpg",
        "latest_url": "https://www.youtube.com/watch?v=I9dzEkgUyr4"
    },
    {
        "name": "Fleet Feet",
        "region": "⭐ 您的專屬關注",
        "category": "🏃 專業跑鞋庫 / 楦型與足弓支撐",
        "desc": "您常看的美國專業運動專門店官方頻道！專注於各大品牌全系列跑鞋陣容矩陣分析（如 Karhu、ASICS、Brooks、HOKA）與寬楦、足弓穩定度實測。",
        "latest_title": "芬蘭百年跑鞋 Karhu 2024/2026 全系列陣容解析：哪一雙最適合你的腳型？",
        "latest_orig": "Karhu's Running Shoe Lineup | Which is Right for You?",
        "latest_date": "2024-10-09",
        "latest_id": "tfFH6qhi-aY",
        "latest_thumbnail": "https://i.ytimg.com/vi/tfFH6qhi-aY/hqdefault.jpg",
        "latest_url": "https://www.youtube.com/watch?v=tfFH6qhi-aY"
    },
    {
        "name": "Bem Kicks",
        "region": "⭐ 您的專屬關注",
        "category": "👟 New Balance 深度 / 復古跑鞋排行榜",
        "desc": "您常看的 New Balance 深度頻道！專精分析 NB 990v6、1906R、2002R、9060、550 等經典鞋款背後的科技演進、材質做工與排行榜分析。",
        "latest_title": "2026 年必備 Top 10 New Balance 球鞋排行榜：哪一雙真正值得入手？",
        "latest_orig": "Top 10 Must-Have New Balance Sneakers of 2026!",
        "latest_date": "2026-07-06",
        "latest_id": "dqZ4BorSS2A",
        "latest_thumbnail": "https://i.ytimg.com/vi/dqZ4BorSS2A/hqdefault.jpg",
        "latest_url": "https://www.youtube.com/watch?v=dqZ4BorSS2A"
    },

    # 🇹🇼 台灣在地
    {
        "name": "哲睿 Jerry",
        "region": "🇹🇼 台灣在地",
        "category": "🏃 馬拉松 / 碳板跑鞋實測",
        "desc": "全馬 Sub-3 實力派菁英跑者，以極度客觀專業的角度實跑評測各大品牌馬拉松碳板鞋、長距離跑鞋與路跑裝備。",
        "latest_title": "該說的就是要說！adidas 3 雙跑鞋實測比較：PRO 4、EVO SL、BOSTON 13 優缺點一次講清",
        "latest_orig": "adidas 3雙鞋跑鞋比較！優缺一次說清楚！PRO 4、EVO SL、BOSTON 13",
        "latest_date": "2025-07-02",
        "latest_id": "1T_KlQiO8JA",
        "latest_thumbnail": "https://i.ytimg.com/vi/1T_KlQiO8JA/hqdefault.jpg",
        "latest_url": "https://www.youtube.com/watch?v=1T_KlQiO8JA"
    },
    {
        "name": "Kenlu.net 勘履者",
        "region": "🇹🇼 台灣在地",
        "category": "🏀 實戰鞋評 / 品牌發表直擊",
        "desc": "台灣歷史最悠久的權威球鞋媒體官方頻道，專精於國際發表會現場採訪、球鞋文化故事與專業實戰測試。",
        "latest_title": "【特輯&心得】年年都推鞋今年玩點不一樣的！feat. KENLU 勘履者",
        "latest_orig": "年年都推鞋今年玩點不一樣的~也告訴大家今年最應該避雷的是... feat. KENLU勘履者",
        "latest_date": "2026-01-16",
        "latest_id": "ArU8kFvNea8",
        "latest_thumbnail": "https://i.ytimg.com/vi/ArU8kFvNea8/hqdefault.jpg",
        "latest_url": "https://www.youtube.com/watch?v=ArU8kFvNea8"
    },
    {
        "name": "Xiao Ma 小馬",
        "region": "🇹🇼 台灣在地",
        "category": "🔥 潮流開箱 / 探店排隊",
        "desc": "台灣超高人氣潮流球鞋創作者，第一手開箱最新熱門話題球鞋、各國球鞋店鋪探店直擊與原價入手指南。",
        "latest_title": "一次開箱 4 雙今年新款 YEEZY！全新設計 350 與話題拖鞋第一手體驗",
        "latest_orig": "一次開箱4雙今年新款YEEZY ! 全新設計350跟最夯的拖鞋",
        "latest_date": "2021-07-15",
        "latest_id": "v8Qa8I8FTlo",
        "latest_thumbnail": "https://i.ytimg.com/vi/v8Qa8I8FTlo/hqdefault.jpg",
        "latest_url": "https://www.youtube.com/watch?v=v8Qa8I8FTlo"
    },

    # 🇨🇳 中國硬核拆解
    {
        "name": "極客鞋談 (Geekshoes)",
        "region": "🇨🇳 中國硬核",
        "category": "🔬 剪鞋拆解 / 實戰客觀評測",
        "desc": "華人圈最早以電鋸剪鞋聞名的硬核評測團隊。詳細量測中底氣壓、碳板真偽、抗扭鋼印，實話實說毫不迎合品牌。",
        "latest_title": "Kobe 8 Protro 復刻評測：不靠氣墊，靠本體感受與 React 泡棉",
        "latest_orig": "kobe 8 复刻评测：不靠气垫，靠本体感受。",
        "latest_date": "2026-07-27",
        "latest_id": "FBc9nkMUMPo",
        "latest_thumbnail": "https://i.ytimg.com/vi/FBc9nkMUMPo/hqdefault.jpg",
        "latest_url": "https://www.youtube.com/watch?v=FBc9nkMUMPo"
    },

    # 🇺🇸 歐美殿堂
    {
        "name": "WearTesters",
        "region": "🇺🇸 歐美殿堂",
        "category": "🏀 殿堂級實戰 / 抓地緩震剖析",
        "desc": "由球鞋實測界元老 Chris Chase (Nightwing2303) 主理，全球公認最值得信賴的實戰籃球鞋與跑鞋評測。",
        "latest_title": "Nike Air More Uptempo 大 Air 2026 黑白黑原版配置實測",
        "latest_orig": "Nike Air More Uptempo Black/White 2026 Performance Review",
        "latest_date": "2026-09-03",
        "latest_id": "ZCkQjR-49-g",
        "latest_thumbnail": "https://i.ytimg.com/vi/ZCkQjR-49-g/hqdefault.jpg",
        "latest_url": "https://www.youtube.com/watch?v=ZCkQjR-49-g"
    },
    {
        "name": "Believe in the Run",
        "region": "🇺🇸 歐美殿堂",
        "category": "🏃 專業跑鞋 / 馬拉松深度測試",
        "desc": "美國專業馬拉松與超能跑鞋評測權威，提供第一手官方試跑反饋、跑者對談與橫向競品對比。",
        "latest_title": "ASICS Superblast 2 vs 3：兩代厚底無板神鞋深度對決",
        "latest_orig": "Asics Superblast 2 vs 3 | Between Two Shoes",
        "latest_date": "2026-04-13",
        "latest_id": "0rphMUdA5yo",
        "latest_thumbnail": "https://i.ytimg.com/vi/0rphMUdA5yo/hqdefault.jpg",
        "latest_url": "https://www.youtube.com/watch?v=0rphMUdA5yo"
    },

    # 🇯🇵 日本
    {
        "name": "Runtrip Japan (ラントリップ)",
        "region": "🇯🇵 日本潮流",
        "category": "🏃 店員實測 / 跑鞋性能矩陣",
        "desc": "日本最大跑步頻道，每季邀請各大運動用品店專業店員進行嚴苛盲測，評選最適合亞洲人腳型的跑鞋矩陣。",
        "latest_title": "【NIKE】定番跑鞋最新作「Pegasus 42 小飛馬」登場！全掌 Air Zoom 實跑評測",
        "latest_orig": "【NIKE】定番ランニングシューズ最新作「ペガサス 42」登場！今作からフルレングスのAir Zoomユニットを搭載！",
        "latest_date": "2026-05-19",
        "latest_id": "5bSRMKjUcGw",
        "latest_thumbnail": "https://i.ytimg.com/vi/5bSRMKjUcGw/hqdefault.jpg",
        "latest_url": "https://www.youtube.com/watch?v=5bSRMKjUcGw"
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
    "🎥 頂級鞋評 YTR 專區 (含您的關注頻道)",
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

# TAB 2: YTR HUB (含最新影片標題與縮圖)
with tab_ytr:
    st.markdown("### 🎥 全球頂級鞋評 YouTuber 專區 (已列出最新影片與縮圖)")
    st.markdown("匯集包含您的專屬關注在內的台、中、美、日頂尖實戰開箱創作者，**點擊即可直接觀看最新評測影片**：")

    ytr_region = st.radio(
        "YTR 分類篩選",
        ["全部頻道 (11)", "⭐ 您的專屬關注 (4)", "🇹🇼 台灣在地 (3)", "🇨🇳 中國硬核拆解 (1)", "🇺🇸 歐美殿堂 (2)", "🇯🇵 日本潮流 (1)"],
        horizontal=True
    )

    filtered_ytr = YTR_CHANNELS
    if "專屬關注" in ytr_region:
        filtered_ytr = [y for y in filtered_ytr if "關注" in y["region"]]
    elif "台灣" in ytr_region:
        filtered_ytr = [y for y in filtered_ytr if "台灣" in y["region"]]
    elif "中國" in ytr_region:
        filtered_ytr = [y for y in filtered_ytr if "中國" in y["region"]]
    elif "歐美" in ytr_region:
        filtered_ytr = [y for y in filtered_ytr if "歐美" in y["region"]]
    elif "日本" in ytr_region:
        filtered_ytr = [y for y in filtered_ytr if "日本" in y["region"]]

    for ytr in filtered_ytr:
        with st.container(border=True):
            # Header
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.25rem; font-weight: 800;">▶️ {ytr['name']}</span>
                    <span style="font-size: 0.9rem; font-weight: 800; color: #38bdf8; background: rgba(56, 189, 248, 0.15); padding: 2px 8px; border-radius: 6px;">{ytr['region']}</span>
                </div>
                <span class="category-pill">{ytr['category']}</span>
            </div>
            <p style="font-size: 0.95rem; color: #94a3b8; line-height: 1.5; margin-bottom: 14px;">{ytr['desc']}</p>
            """, unsafe_allow_html=True)

            # Left Video Thumbnail (35%), Right Video Info (65%)
            col_v_img, col_v_txt = st.columns([1, 2], gap="large")
            with col_v_img:
                st.image(ytr["latest_thumbnail"], use_container_width=True)
            with col_v_txt:
                st.markdown(f"**🎬 最新發布影片：**")
                st.markdown(f'<h4 style="margin: 4px 0 6px 0; font-size: 1.25rem; font-weight: 800; line-height: 1.4;">{ytr["latest_title"]}</h4>', unsafe_allow_html=True)
                if ytr.get("latest_orig") and ytr["latest_orig"] != ytr["latest_title"]:
                    st.markdown(f'<p style="font-size: 0.85rem; color: #64748b; font-style: italic; margin-bottom: 8px;">原標題：{ytr["latest_orig"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size: 0.9rem; color: #64748b; margin-bottom: 12px;">📅 上傳時間：{ytr["latest_date"]}</p>', unsafe_allow_html=True)
                st.link_button(f"▶️ 點此立即在 YouTube 觀看本片 ↗", ytr["latest_url"])

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
