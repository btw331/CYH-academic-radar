    "📋 本次修改 (Updates)",
)

NAVIGATION_LABELS = {
    "📰 全球情報 (News Feed)": "全球情報",
    "🚀 多元議題分析 (Deep Analysis)": "多元分析",
    "🧾 新聞文本分析 (Text Analysis)": "新聞文本",
    "📚 方法論 (Methodology)": "方法論",
    "📋 本次修改 (Updates)": "更新說明",
}


def sidebar_navigation_label(page: str) -> str:
    """回傳側欄使用的短標籤，避免技術名稱干擾選擇。"""
    return NAVIGATION_LABELS.get(page, page)


def resolve_sidebar_navigation(selected_page: str, current_page: str) -> str:
    """只接受已知頁面；無效選項保留目前頁面。"""
    if selected_page in NAVIGATION_PAGES:
        return selected_page
    if current_page in NAVIGATION_PAGES:
        return current_page
    return NAVIGATION_PAGES[0]


def enhance_report_html(html_content: str, key_prefix: str, variant: str = "analysis") -> str:

def extract_report_headings(report_text: str, max_items: int = 12) -> List[str]:
    """從 Markdown 報告抽取章節標題，作為閱讀導覽。"""
def extract_report_navigation_items(
    report_text: str,
    max_items: int = 6,
    max_level: int = 2,
) -> List[Dict[str, Any]]:
    """抽取主要章節並保留其在完整報告中的真實錨點序號。"""
    if not report_text:
        return []
    headings = []
    items: List[Dict[str, Any]] = []
    seen_titles = set()
    anchor_index = 0
    for raw_line in report_text.replace("\\n", "\n").splitlines():
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
        if not match:
            continue
        anchor_index += 1
        level = len(match.group(1))
        title = _plain_markdown_text(match.group(2))
        if level <= max_level and title and title not in seen_titles:
            items.append({"title": title, "anchor_index": anchor_index, "level": level})
            seen_titles.add(title)
        if len(items) >= max_items:
            break
    return headings
    return items


def extract_report_headings(report_text: str, max_items: int = 6) -> List[str]:
    """從 Markdown 報告抽取精簡的主要章節標題。"""
    return [item["title"] for item in extract_report_navigation_items(report_text, max_items=max_items)]

def build_report_navigation_html(report_text: str, key_prefix: str) -> str:
    """建立與報告錨點一一對應的可點擊章節導覽。"""
    headings = extract_report_headings(report_text)
    if not headings:
    items = extract_report_navigation_items(report_text)
    if not items:
        return ""
    items = "".join(
        f'<li><a href="#{prefix}-section-{idx}">{escape(heading)}</a></li>'
        for idx, heading in enumerate(headings, 1)
        f'<li><a href="#{prefix}-section-{item["anchor_index"]}">{escape(item["title"])}</a></li>'
        for item in items
    )
    st.title("多元觀點解析")
    st.caption("✨ 多源搜尋 + 新聞文本分析 + 學術方法論")
    _nav_pages = [
        "📰 全球情報 (News Feed)",
        "🚀 多元議題分析 (Deep Analysis)",
        "🧾 新聞文本分析 (Text Analysis)",
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

    st.markdown("##### 全球情報")
    st.caption("AllSides 亞洲／台灣平衡報導與重點整理")
    if st.button(
        "📰 全球情報 (News Feed)",
        key="sidebar_nav_feed",
        use_container_width=True,
        type="primary" if _cur_nav == "📰 全球情報 (News Feed)" else "secondary",
    ):
        _sidebar_nav_to("📰 全球情報 (News Feed)")

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

    st.markdown("##### 方法論")
    st.caption("實裝方法與限制說明")
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
    st.caption("多源查證與閱讀報告")
    _nav_pages = list(NAVIGATION_PAGES)
    _cur_nav = resolve_sidebar_navigation(
        st.session_state.get("current_page", _nav_pages[0]),
        _nav_pages[0],
    )
    st.session_state["current_page"] = _cur_nav
    if st.session_state.get("sidebar_page_selector") != _cur_nav:
        st.session_state["sidebar_page_selector"] = _cur_nav
    selected_page = st.selectbox(
        "功能",
