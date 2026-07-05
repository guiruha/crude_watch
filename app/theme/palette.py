"""Corporate black & emerald palette plus the app's global CSS.

Kept deliberately restrained (deep charcoal surfaces, a single emerald accent)
for a professional look, and colour-matched to the Plotly charts via ``ACCENT``.
"""
from __future__ import annotations

import streamlit as st

# Palette --------------------------------------------------------------------
BACKGROUND = "#0B0E0D"   # near-black, faint green tint
SURFACE = "#141A17"      # cards / panels
SURFACE_2 = "#1B221E"    # hover / elevated
BORDER = "#26302A"       # hairline borders
ACCENT = "#10B981"       # emerald (corporate green, matches charts)
ACCENT_MUTED = "#0E9E6E"
TEXT = "#E7ECEA"
SUBTEXT = "#8B9691"

# Signal tones (traffic-light panel & indicator overlays).
BULL = ACCENT            # bullish bias
BEAR = "#E5484D"         # bearish bias (restrained red)
FLAT = "#8B9691"         # neutral
AMBER = "#F5A623"        # secondary line / mean

# Chart accent alias so screens import a single source of truth.
CHART_ACCENT = ACCENT

# Brand mark: an oil droplet (crude) with a price-pulse line running through it
# (monitoring) — a professional, on-theme glyph tinted with the brand accent.
BRAND_MARK_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M12 2.2c2.7 3.4 7 7.6 7 11.8a7 7 0 0 1-14 0c0-4.2 4.3-8.4 7-11.8z" '
    'fill="currentColor" fill-opacity="0.13"/>'
    '<path d="M7.5 14.4h2l1.3-2.7 1.7 4 1.1-1.9h2.9" stroke-width="1.7"/>'
    '</svg>'
)


def inject_css() -> None:
    """Inject global styling: hide chrome, style cards, sidebar, and controls."""
    st.markdown(
        f"""
        <style>
        /* Keep the top bar minimal but preserve the sidebar toggle ("menu" icon):
           the header is where Streamlit renders the collapse/expand control, so we
           only strip the rainbow decoration + toolbar instead of hiding it all. */
        header[data-testid="stHeader"] {{
            background: transparent !important;
            box-shadow: none !important;
            pointer-events: none !important;
        }}
        /* Strip only the rainbow decoration and the Deploy button — keep the
           hamburger main menu (Settings / Rerun / etc.) visible and clickable. */
        [data-testid="stDecoration"], [data-testid="stAppDeployButton"] {{ display: none !important; }}
        [data-testid="stToolbar"],
        [data-testid="stToolbarActions"],
        [data-testid="stMainMenu"] {{ pointer-events: auto !important; }}
        [data-testid="stMainMenu"] {{
            visibility: visible !important;
            opacity: 1 !important;
            z-index: 1000 !important;
        }}
        [data-testid="stMainMenu"] button {{ color: {ACCENT} !important; }}
        [data-testid="stMainMenu"] button * {{ color: {ACCENT} !important; fill: {ACCENT} !important; }}
        [data-testid="stMainMenu"] span[data-testid="stIconMaterial"] {{ font-size: 1.6rem !important; }}

        /* Sidebar collapse (« in the sidebar header) and expand (» when collapsed)
           controls. Streamlit hides the collapse arrow until hover and paints both
           in a faint grey — force them permanently visible and emerald so the menu
           toggle is always obvious. The glyphs are Material font spans, not SVGs. */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stExpandSidebarButton"] {{
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
            z-index: 1000 !important;
        }}
        [data-testid="stSidebarCollapseButton"] *,
        [data-testid="stExpandSidebarButton"] * {{
            color: {ACCENT} !important;
            fill: {ACCENT} !important;
            pointer-events: auto !important;
        }}
        [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"],
        [data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"] {{
            font-size: 1.7rem !important;
        }}

        /* Nudge the whole app's type scale up a touch (rem-based widgets follow). */
        html {{ font-size: 17.5px; }}
        .stApp {{ background: {BACKGROUND}; }}
        div[data-testid="stAppViewBlockContainer"] {{ padding-top: 2.2rem; }}
        /* Body copy & control labels a little larger. */
        div[data-testid="stMarkdownContainer"] p {{ font-size: 15.5px; }}
        label[data-testid="stWidgetLabel"] p {{ font-size: 14.5px; }}

        /* Sidebar shell */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {SURFACE} 0%, {BACKGROUND} 100%);
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {{
            padding-top: 1.4rem;
        }}

        /* Brand lockup */
        .cw-side-brand {{
            display: flex; align-items: center; gap: 10px; margin: 0 0 2px 2px;
        }}
        .cw-side-mark {{
            width: 46px; height: 46px; border-radius: 12px; flex: none;
            display: flex; align-items: center; justify-content: center;
            background: radial-gradient(120% 120% at 30% 20%, {SURFACE_2} 0%, {BACKGROUND} 100%);
            color: {ACCENT}; line-height: 0;
            box-shadow: inset 0 0 0 1px {ACCENT}66, 0 6px 18px -8px {ACCENT};
        }}
        .cw-side-mark svg {{ width: 28px; height: 28px; display: block; }}
        .cw-side-word {{
            font-size: 30px; font-weight: 800; letter-spacing: .5px;
            text-transform: uppercase; color: {ACCENT}; line-height: 1;
        }}
        .cw-side-word span {{ color: {TEXT}; }}
        .cw-side-tag {{
            color: {SUBTEXT}; font-size: 13px; letter-spacing: .3px;
            margin: 8px 0 2px 2px;
        }}

        /* Nav section label */
        .cw-nav-label {{
            color: {SUBTEXT}; font-size: 13px; font-weight: 700;
            letter-spacing: 1.4px; text-transform: uppercase;
            margin: 10px 0 4px 4px;
        }}

        /* Turn the sidebar radio into a vertical nav */
        section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap: 2px; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
            display: flex; align-items: center; width: 100%;
            padding: 12px 14px; margin: 2px 0; border-radius: 9px;
            border: 1px solid transparent; cursor: pointer;
            transition: background .15s ease, border-color .15s ease;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
            background: {SURFACE_2}; border-color: {BORDER};
        }}
        /* hide the actual radio dot */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{
            display: none;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label p {{
            font-size: 20px; font-weight: 600; color: {TEXT};
        }}
        /* active item: emerald tint + left accent bar */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
            background: {SURFACE_2};
            border-color: {BORDER};
            box-shadow: inset 3px 0 0 0 {ACCENT};
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {{
            color: {ACCENT}; font-weight: 700;
        }}

        /* Dataset status card */
        .cw-side-card {{
            background: {SURFACE}; border: 1px solid {BORDER};
            border-radius: 10px; padding: 12px 14px; margin: 4px 0;
        }}
        .cw-side-card .row {{
            display: flex; justify-content: space-between; align-items: baseline;
            padding: 3px 0; font-size: 12px;
        }}
        .cw-side-card .row .k {{ color: {SUBTEXT}; }}
        .cw-side-card .row .v {{ color: {TEXT}; font-weight: 600; }}

        /* Sidebar footer */
        .cw-side-foot {{
            color: {SUBTEXT}; font-size: 10.5px; letter-spacing: .3px;
            margin-top: 6px; opacity: .8;
        }}
        /* Copyright under the navigation menu */
        .cw-side-copy {{
            color: {SUBTEXT}; font-size: 11px; letter-spacing: .3px;
            margin: 10px 0 2px 2px; opacity: .7;
        }}

        /* Skeleton-chapter "coming next" banner */
        .cw-phase {{
            display: flex; align-items: center; gap: 10px;
            background: {SURFACE}; border: 1px dashed {BORDER};
            border-radius: 8px; padding: 8px 12px; margin: 2px 0 14px 0;
        }}
        .cw-phase-tag {{
            background: {ACCENT}; color: {BACKGROUND};
            font-size: 10px; font-weight: 800; letter-spacing: .5px;
            text-transform: uppercase; padding: 2px 8px; border-radius: 6px;
        }}
        .cw-phase-txt {{ color: {SUBTEXT}; font-size: 12.5px; }}

        /* Signal-panel chips (traffic light) */
        .cw-sig {{
            border: 1px solid {BORDER}; border-radius: 10px;
            padding: 10px 12px; background: {SURFACE};
            border-left-width: 4px;
        }}
        .cw-sig .fam {{
            color: {SUBTEXT}; font-size: 11px; font-weight: 700;
            letter-spacing: .8px; text-transform: uppercase;
        }}
        .cw-sig .bias {{ font-size: 20px; font-weight: 800; margin-top: 3px; }}
        .cw-sig .conv {{ color: {SUBTEXT}; font-size: 12.5px; margin-top: 2px; }}
        .cw-sig.bull {{ border-left-color: {BULL}; }}
        .cw-sig.bull .bias {{ color: {BULL}; }}
        .cw-sig.bear {{ border-left-color: {BEAR}; }}
        .cw-sig.bear .bias {{ color: {BEAR}; }}
        .cw-sig.flat {{ border-left-color: {FLAT}; }}
        .cw-sig.flat .bias {{ color: {TEXT}; }}

        /* Fancy theme dropdowns (expanders) */
        [data-testid="stExpander"] {{ margin: 8px 0; }}
        [data-testid="stExpander"] details {{
            background: linear-gradient(180deg, {SURFACE} 0%, {BACKGROUND} 100%);
            border: 1px solid {BORDER};
            border-radius: 14px;
            overflow: hidden;
            transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
        }}
        [data-testid="stExpander"] details:hover {{
            border-color: {ACCENT}55;
            box-shadow: 0 10px 26px -16px {ACCENT};
            transform: translateY(-1px);
        }}
        [data-testid="stExpander"] details[open] {{
            border-color: {ACCENT}77;
            box-shadow: inset 3px 0 0 0 {ACCENT}, 0 8px 22px -18px {ACCENT};
        }}
        [data-testid="stExpander"] summary {{
            padding: 15px 18px;
            list-style: none;
            border-radius: 14px;
            transition: background .18s ease;
        }}
        [data-testid="stExpander"] summary:hover {{ background: {SURFACE_2}; }}
        [data-testid="stExpander"] details[open] summary {{
            background: {SURFACE_2};
            border-bottom: 1px solid {BORDER};
            border-radius: 14px 14px 0 0;
        }}
        [data-testid="stExpander"] summary p {{
            font-size: 17px !important; font-weight: 700 !important;
            letter-spacing: .2px; color: {TEXT};
        }}
        [data-testid="stExpander"] summary:hover p {{ color: {ACCENT}; }}
        [data-testid="stExpanderToggleIcon"] {{ color: {ACCENT}; }}
        [data-testid="stExpander"] summary svg {{ fill: {ACCENT}; width: 1.35rem; height: 1.35rem; }}

        /* Metric cards */
        div[data-testid="stMetric"] {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 14px 16px;
        }}
        div[data-testid="stMetricValue"] {{ color: {TEXT}; font-weight: 600; }}
        div[data-testid="stMetricLabel"] p {{
            color: {SUBTEXT};
            text-transform: uppercase;
            letter-spacing: 0.4px;
            font-size: 11px;
        }}

        /* Segmented control: emerald active tab */
        button[data-testid="stBaseButton-segmented_controlActive"] {{
            background: {ACCENT} !important;
            color: {BACKGROUND} !important;
            border-color: {ACCENT} !important;
        }}
        button[data-testid="stBaseButton-segmented_controlActive"] p {{
            color: {BACKGROUND} !important; font-weight: 700 !important;
        }}
        /* Larger option text in the structure/maturity selectors */
        button[data-testid^="stBaseButton-segmented_control"] {{ padding: 7px 14px !important; }}
        button[data-testid^="stBaseButton-segmented_control"] p {{ font-size: 17px !important; }}
        button[data-testid^="stBaseButton-segmented_control"] div {{ font-size: 17px !important; }}
        div[data-testid="stSelectbox"] div[data-baseweb="select"] {{ font-size: 16px !important; }}
        div[data-testid="stSelectbox"] div[data-baseweb="select"] * {{ font-size: 16px !important; }}

        /* Maturity selector: bigger + centered (scoped to its keyed container so the
           Level/Structure/Vintage controls keep their one-line column layout) */
        div[class*="st-key-seas_bucket_"],
        div[class*="st-key-tech_bucket_"] {{
            display: flex !important; flex-direction: column !important; align-items: center !important;
        }}
        div[class*="st-key-seas_bucket_"] label,
        div[class*="st-key-tech_bucket_"] label {{
            width: 100% !important; text-align: center !important; justify-content: center !important;
        }}
        div[class*="st-key-seas_bucket_"] label p,
        div[class*="st-key-tech_bucket_"] label p {{
            font-size: 18px !important; font-weight: 600 !important;
        }}
        div[class*="st-key-seas_bucket_"] div[data-testid="stButtonGroup"],
        div[class*="st-key-tech_bucket_"] div[data-testid="stButtonGroup"] {{
            width: 100% !important; max-width: 100% !important; justify-content: center !important;
        }}
        div[class*="st-key-seas_bucket_"] button[data-testid^="stBaseButton-segmented_control"],
        div[class*="st-key-tech_bucket_"] button[data-testid^="stBaseButton-segmented_control"] {{
            padding: 10px 24px !important;
        }}
        div[class*="st-key-seas_bucket_"] button[data-testid^="stBaseButton-segmented_control"] p,
        div[class*="st-key-tech_bucket_"] button[data-testid^="stBaseButton-segmented_control"] p,
        div[class*="st-key-seas_bucket_"] button[data-testid^="stBaseButton-segmented_control"] div,
        div[class*="st-key-tech_bucket_"] button[data-testid^="stBaseButton-segmented_control"] div {{
            font-size: 20px !important;
        }}

        /* Headings accent rule */
        .cw-title {{
            color: {TEXT}; font-size: 32px; font-weight: 700;
            border-left: 5px solid {ACCENT}; padding-left: 14px; margin: 0 0 6px 0;
        }}
        .cw-sub {{ color: {SUBTEXT}; font-size: 15px; margin: 0 0 18px 16px; }}
        .cw-brand {{
            color: {ACCENT}; font-size: 20px; font-weight: 800;
            letter-spacing: 1px; text-transform: uppercase;
        }}
        .cw-brand span {{ color: {TEXT}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def title_block(title: str, subtitle: str = "") -> None:
    """Render a page title with the emerald accent rule and optional subtitle."""
    st.markdown(f'<div class="cw-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="cw-sub">{subtitle}</div>', unsafe_allow_html=True)


def sidebar_brand() -> None:
    """Render the sidebar brand lockup: emerald mark and wordmark."""
    st.markdown(
        f"""
        <div class="cw-side-brand">
            <div class="cw-side-mark">{BRAND_MARK_SVG}</div>
            <div class="cw-side-word">Crude<span>Watch</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_card(rows: dict[str, str]) -> None:
    """Render a small key/value status card in the sidebar."""
    body = "".join(
        f'<div class="row"><span class="k">{k}</span><span class="v">{v}</span></div>'
        for k, v in rows.items()
    )
    st.markdown(f'<div class="cw-side-card">{body}</div>', unsafe_allow_html=True)


def sidebar_footer(text: str) -> None:
    """Render muted footer text pinned under the sidebar content."""
    st.markdown(f'<div class="cw-side-foot">{text}</div>', unsafe_allow_html=True)


def nav_label(text: str) -> None:
    """Render an uppercase section label for a sidebar nav group."""
    st.markdown(f'<div class="cw-nav-label">{text}</div>', unsafe_allow_html=True)


def signal_chip(family: str, bias_label: str, conviction: float, tone: str) -> None:
    """Render one traffic-light chip; ``tone`` is 'bull' | 'bear' | 'flat'."""
    conv = f"{conviction:.0%} agreement" if conviction > 0 else "—"
    st.markdown(
        f"""
        <div class="cw-sig {tone}">
            <div class="fam">{family}</div>
            <div class="bias">{bias_label}</div>
            <div class="conv">{conv}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def phase_note(phase: str, upcoming: str) -> None:
    """Render a subtle 'coming in <phase>' banner used by skeleton chapters."""
    st.markdown(
        f"""
        <div class="cw-phase">
            <span class="cw-phase-tag">{phase}</span>
            <span class="cw-phase-txt">{upcoming}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
