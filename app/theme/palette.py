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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        /* Load the icon font with display=block so its ligature text ("keyboard_…")
           stays invisible until the glyph is ready — kills the raw-text flash. */
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');

        :root {{
            --cw-font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --cw-font-display: 'Space Grotesk', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        /* Body typeface everywhere; display typeface for titles + numeric readouts. */
        html, body, .stApp, [class^="st-"], [class*=" st-"],
        button, input, select, textarea,
        div[data-testid="stMarkdownContainer"] {{
            font-family: var(--cw-font-body) !important;
        }}
        h1, h2, h3, h4, h5, h6,
        div[data-testid="stMarkdownContainer"] h1, div[data-testid="stMarkdownContainer"] h2,
        div[data-testid="stMarkdownContainer"] h3, div[data-testid="stMarkdownContainer"] h4,
        div[data-testid="stMarkdownContainer"] h5,
        .cw-title, .cw-side-word, .cw-brand, .cw-fc-val, .cw-fc-label,
        div[data-testid="stMetricValue"] {{
            font-family: var(--cw-font-display) !important;
        }}
        /* Tabular figures so digits align in cards, tables and metrics. */
        .cw-fc-val, .cw-fc-ends, .cw-ftab .v, .cw-ftab .pct .num,
        div[data-testid="stMetricValue"] {{
            font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1;
        }}
        /* CRITICAL: never let the body font win over the Material Symbols icon
           font — otherwise ligature glyphs render as raw text ("keyboard_arrow…"
           and overlap). Higher specificity + !important beats the base override. */
        span[data-testid="stIconMaterial"] {{
            font-family: 'Material Symbols Rounded' !important;
        }}

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
        /* Bulletproof toggle glyphs: some environments briefly (or persistently)
           fail to swap the Material ligature "keyboard_double_arrow_*" into its
           icon glyph, leaving raw text. Hide the ligature text entirely and draw
           the chevron with a plain Unicode character in the body font, so the
           toggle never depends on the icon font loading. */
        [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"],
        [data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"] {{
            font-size: 0 !important;
            line-height: 1 !important;
        }}
        [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"]::after,
        [data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"]::after {{
            font-family: var(--cw-font-body) !important;
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            line-height: 1 !important;
            color: {ACCENT} !important;
        }}
        [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"]::after {{
            content: "\\00AB"; /* « collapse */
        }}
        [data-testid="stExpandSidebarButton"] span[data-testid="stIconMaterial"]::after {{
            content: "\\00BB"; /* » expand */
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
        /* Hide the native Streamlit radio control; the label row is the nav UI. */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{
            display: none !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {{
            opacity: 0 !important;
            position: absolute !important;
            pointer-events: none !important;
            width: 0 !important;
            height: 0 !important;
            margin: 0 !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] + div {{
            display: none !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] div:has(> input[type="radio"]) {{
            display: none !important;
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
        .cw-sig .bias .cw-sig-ico {{ font-size: 13px; margin-right: 7px; vertical-align: .1em; }}
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

        /* Global top selection bar — matched to the sidebar surface for a
           cohesive, professional look (charcoal gradient, hairline border,
           emerald left-accent like the active nav item, uppercase micro-labels). */
        .st-key-cw_topbar {{
            background: linear-gradient(180deg, {SURFACE} 0%, {BACKGROUND} 100%);
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 12px 18px 15px 18px;
            margin: 0 0 20px 0;
            box-shadow: inset 3px 0 0 0 {ACCENT}, 0 12px 30px -22px {ACCENT};
        }}
        .cw-topbar-head {{
            display: flex; align-items: center; gap: 9px;
            color: {SUBTEXT}; font-size: 13px; font-weight: 700;
            letter-spacing: 1.4px; text-transform: uppercase;
            margin: 0 0 10px 2px;
        }}
        .cw-topbar-head span {{
            width: 8px; height: 8px; border-radius: 50%;
            background: {ACCENT}; box-shadow: 0 0 0 3px {ACCENT}22;
        }}
        /* Field labels: uppercase micro-labels like the sidebar nav labels. */
        .st-key-cw_topbar label[data-testid="stWidgetLabel"] p {{
            color: {SUBTEXT} !important; text-transform: uppercase;
            letter-spacing: .8px; font-size: 11.5px !important; font-weight: 700;
        }}
        /* Select + date controls: elevated surface, hairline border, emerald focus. */
        .st-key-cw_topbar div[data-baseweb="select"] > div,
        .st-key-cw_topbar div[data-testid="stDateInput"] div[data-baseweb="input"] {{
            background: {SURFACE_2} !important;
            border-color: {BORDER} !important;
            border-radius: 9px !important;
        }}
        .st-key-cw_topbar div[data-baseweb="select"] > div:hover,
        .st-key-cw_topbar div[data-testid="stDateInput"] div[data-baseweb="input"]:hover {{
            border-color: {ACCENT}66 !important;
        }}
        .st-key-cw_topbar div[data-testid="stDateInput"] input {{
            background: transparent !important;
        }}

        /* Caveat panel — measured limits of the score. Amber, always visible. */
        .cw-caveat {{
            display: flex; gap: 13px; align-items: flex-start;
            background: {SURFACE}; border: 1px solid {BORDER};
            border-left: 4px solid {AMBER}; border-radius: 14px;
            padding: 15px 20px; margin: 2px 0 18px;
        }}
        .cw-caveat .ico {{ color: {AMBER}; flex: none; font-size: 19px; line-height: 1.2; }}
        .cw-caveat .txt {{ color: {SUBTEXT}; font-size: 15px; line-height: 1.6; flex: 1; }}
        .cw-caveat .txt p {{ margin: 0 0 10px; }}
        .cw-caveat .txt p:last-child {{ margin-bottom: 0; }}
        .cw-caveat .txt b {{ color: {TEXT}; font-weight: 700; }}
        .cw-caveat .txt code {{
            background: {SURFACE_2}; border-radius: 4px; padding: 1px 5px;
            font-size: .92em; color: {TEXT};
        }}

        /* Block intro panel — the block's explanation as a carded info note. */
        .cw-intro {{
            display: flex; gap: 13px; align-items: flex-start;
            background: {SURFACE}; border: 1px solid {BORDER};
            border-left: 4px solid {ACCENT}; border-radius: 14px;
            padding: 15px 20px; margin: 2px 0 18px;
        }}
        .cw-intro .ico {{ color: {ACCENT}; flex: none; line-height: 0; margin-top: 3px; }}
        .cw-intro .ico svg {{ width: 20px; height: 20px; }}
        .cw-intro .txt {{ color: {SUBTEXT}; font-size: 15.5px; line-height: 1.62; flex: 1; }}
        .cw-intro .txt p {{ margin: 0 0 11px; }}
        .cw-intro .txt p:last-child {{ margin-bottom: 0; }}
        .cw-intro .txt b {{ color: {TEXT}; font-weight: 700; }}
        .cw-intro .txt i {{ color: {TEXT}; font-style: italic; }}
        /* "Cómo leerlo" footer: a quieter, accented practical-reading strip. */
        .cw-intro-read {{
            margin-top: 13px; padding-top: 12px; border-top: 1px dashed {BORDER};
            font-size: 14.5px; line-height: 1.6; color: {SUBTEXT};
        }}
        .cw-intro-read .lbl {{
            display: inline-block; margin-right: 9px; padding: 2px 10px;
            border-radius: 999px; background: {ACCENT}22; color: {ACCENT};
            font-size: 11.5px; font-weight: 800; letter-spacing: .6px;
            text-transform: uppercase; vertical-align: 1px;
        }}
        .cw-intro-read b {{ color: {TEXT}; font-weight: 700; }}
        .cw-intro-read i {{ color: {TEXT}; font-style: italic; }}

        /* Metric context readout — larger, carded, with the key figures in bold. */
        .cw-mctx {{
            background: linear-gradient(180deg, {SURFACE} 0%, {BACKGROUND} 100%);
            border: 1px solid {BORDER}; border-left: 4px solid {ACCENT};
            border-radius: 14px; padding: 15px 20px 16px; margin: 4px 0 12px;
        }}
        .cw-mctx-line {{ font-size: 18px; color: {TEXT}; margin: 4px 0; letter-spacing: .1px; }}
        .cw-mctx-line .k {{ color: {SUBTEXT}; font-weight: 600; }}
        .cw-mctx-line b {{ font-weight: 800; color: {TEXT}; font-variant-numeric: tabular-nums; }}
        .cw-mctx-tag {{ color: {ACCENT}; font-weight: 700; }}
        .cw-mctx-bar {{
            position: relative; display: inline-block; vertical-align: middle;
            width: 120px; height: 7px; margin: 0 2px; border-radius: 5px;
            background: {BACKGROUND}; border: 1px solid {BORDER}; overflow: hidden;
        }}
        .cw-mctx-bar > span {{
            position: absolute; left: 0; top: 0; bottom: 0;
            background: {ACCENT}; border-radius: 5px;
        }}
        .cw-mctx-desc {{
            color: {SUBTEXT}; font-size: 14px; margin-top: 12px;
            border-top: 1px solid {BORDER}; padding-top: 11px;
        }}
        .cw-mctx-desc b {{ color: {TEXT}; }}

        /* Score flashcard — visual headline for a block (big value + tag + gauge). */
        .cw-fc {{
            background: linear-gradient(180deg, {SURFACE} 0%, {BACKGROUND} 100%);
            border: 1px solid {BORDER}; border-left: 4px solid {ACCENT};
            border-radius: 16px; padding: 18px 22px 20px; margin: 2px 0 16px;
        }}
        .cw-fc-top {{
            display: flex; align-items: center; justify-content: space-between; gap: 12px;
        }}
        .cw-fc-label {{
            color: {SUBTEXT}; font-size: 12px; font-weight: 700;
            letter-spacing: 1.3px; text-transform: uppercase;
        }}
        .cw-fc-tag {{
            font-size: 13px; font-weight: 700; border: 1px solid {BORDER};
            border-radius: 999px; padding: 3px 13px; white-space: nowrap;
        }}
        .cw-fc-ico {{ margin-right: 6px; font-size: 11px; vertical-align: .04em; }}
        .cw-fc-val {{
            font-size: 48px; font-weight: 800; line-height: 1.02;
            margin: 8px 0 16px; font-variant-numeric: tabular-nums;
        }}
        .cw-fc-track {{
            position: relative; height: 10px; border-radius: 6px;
            background: {BACKGROUND}; border: 1px solid {BORDER};
        }}
        .cw-fc-fill {{
            position: absolute; top: 0; bottom: 0; border-radius: 6px; opacity: .9;
        }}
        .cw-fc-mid {{
            position: absolute; top: -4px; bottom: -4px; width: 2px;
            background: {SUBTEXT}; opacity: .7; transform: translateX(-1px); z-index: 2;
        }}
        .cw-fc-dot {{
            position: absolute; top: 50%; width: 15px; height: 15px; border-radius: 50%;
            transform: translate(-50%, -50%); box-shadow: 0 0 0 3px {BACKGROUND}; z-index: 3;
        }}
        .cw-fc-ends {{
            display: flex; justify-content: space-between; margin-top: 8px;
            color: {SUBTEXT}; font-size: 11.5px; letter-spacing: .3px;
        }}

        /* Feature table — bespoke card (native st.dataframe can't be themed:
           its grid is canvas-rendered and its ProgressColumn uses the default
           red primary). This matches the palette: hairline rows, emerald
           percentile bars, muted descriptions. */
        .cw-ftab-wrap {{
            border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden;
            background: {SURFACE}; margin: 6px 0 4px;
        }}
        .cw-ftab {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        .cw-ftab thead th {{
            text-align: left; color: {SUBTEXT}; font-size: 11px; font-weight: 700;
            letter-spacing: .6px; text-transform: uppercase; padding: 10px 14px;
            background: {SURFACE_2}; border-bottom: 1px solid {BORDER};
        }}
        .cw-ftab tbody td {{
            padding: 11px 14px; border-bottom: 1px solid {BORDER}; vertical-align: middle;
        }}
        .cw-ftab tbody tr:last-child td {{ border-bottom: none; }}
        .cw-ftab tbody tr:hover td {{ background: {SURFACE_2}; }}
        .cw-ftab .m {{ color: {TEXT}; font-weight: 600; white-space: nowrap; }}
        .cw-ftab .m .info {{
            color: {SUBTEXT}; cursor: help; margin-left: 7px;
            display: inline-flex; vertical-align: -0.12em;
        }}
        .cw-ftab .m .info:hover {{ color: {ACCENT}; }}
        .cw-ftab .m .info svg {{ width: 15px; height: 15px; }}
        .cw-ftab .v {{
            color: {TEXT}; font-variant-numeric: tabular-nums;
            text-align: right; white-space: nowrap;
        }}
        .cw-ftab .r {{ color: {TEXT}; }}
        .cw-ftab .f {{ color: {SUBTEXT}; }}
        .cw-ftab .d {{ color: {SUBTEXT}; font-size: 12.5px; }}
        .cw-ftab .pct {{ display: flex; align-items: center; gap: 9px; min-width: 130px; }}
        .cw-ftab .pct .track {{
            position: relative; flex: 1; height: 8px; border-radius: 5px;
            background: {BACKGROUND}; border: 1px solid {BORDER}; overflow: hidden;
        }}
        .cw-ftab .pct .fill {{
            position: absolute; left: 0; top: 0; bottom: 0;
            background: {ACCENT}; border-radius: 5px;
        }}
        .cw-ftab .pct .num {{
            color: {TEXT}; font-variant-numeric: tabular-nums;
            width: 26px; text-align: right; font-size: 13px;
        }}

        /* Metric context sub-menu (pills): a menu *panel* (same charcoal gradient
           + hairline border as the top selection bar / sidebar) holding enlarged,
           title-like tabs with a filled surface and emerald active state. */
        div[class*="st-key-ctxsel_"] {{
            background: linear-gradient(180deg, {SURFACE} 0%, {BACKGROUND} 100%);
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 12px 16px;
            margin: 4px 0 12px;
        }}
        div[class*="st-key-ctxsel_"] button[data-testid^="stBaseButton-pills"] {{
            padding: 8px 15px !important;
            border-radius: 10px !important;
            background: {SURFACE_2} !important;
            border-color: {BORDER} !important;
        }}
        div[class*="st-key-ctxsel_"] button[data-testid^="stBaseButton-pills"]:hover {{
            border-color: {ACCENT}66 !important;
        }}
        div[class*="st-key-ctxsel_"] button[data-testid^="stBaseButton-pills"] p,
        div[class*="st-key-ctxsel_"] button[data-testid^="stBaseButton-pills"] div {{
            font-size: 16px !important; font-weight: 700 !important; letter-spacing: .2px;
        }}
        div[class*="st-key-ctxsel_"] button[data-testid^="stBaseButton-pills"]
            span[data-testid="stIconMaterial"] {{
            font-size: 15px !important;
        }}
        div[class*="st-key-ctxsel_"] button[data-testid="stBaseButton-pillsActive"] {{
            background: {SURFACE_2} !important;
            border-color: {ACCENT} !important;
            box-shadow: inset 0 0 0 1px {ACCENT}, 0 8px 22px -18px {ACCENT};
        }}
        div[class*="st-key-ctxsel_"] button[data-testid="stBaseButton-pillsActive"] p,
        div[class*="st-key-ctxsel_"] button[data-testid="stBaseButton-pillsActive"] div,
        div[class*="st-key-ctxsel_"] button[data-testid="stBaseButton-pillsActive"]
            span[data-testid="stIconMaterial"] {{
            color: {ACCENT} !important;
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


INFO_MARK_SVG = (
    '<svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/>'
    '<line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
)


def title_block(title: str, subtitle: str = "", icon: str | None = None) -> None:
    """Render a page title with the emerald accent rule and optional subtitle/icon."""
    ico = (
        f'<span style="display:inline-flex;align-items:center;color:{ACCENT};'
        f'margin-right:10px;vertical-align:-0.06em;">{icon}</span>'
        if icon
        else ""
    )
    st.markdown(f'<div class="cw-title">{ico}{title}</div>', unsafe_allow_html=True)
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
    """Render one traffic-light chip; ``tone`` is 'bull' | 'bear' | 'flat'.

    A leading glyph (▲/▼/■) encodes the bias redundantly with colour so the chip
    is readable in grayscale / for colour-vision deficiency (WCAG 1.4.1).
    """
    conv = f"{conviction:.0%} agreement" if conviction > 0 else "—"
    glyph = {"bull": "▲", "bear": "▼"}.get(tone, "■")
    st.markdown(
        f"""
        <div class="cw-sig {tone}">
            <div class="fam">{family}</div>
            <div class="bias"><span class="cw-sig-ico">{glyph}</span>{bias_label}</div>
            <div class="conv">{conv}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def caveat_note(html: str) -> None:
    """Render an amber-accented caveat panel — the honest-limits counterpart to
    ``cw-intro``. Always visible, never behind an expander: a limitation the user
    has to open a disclosure to find is a limitation they will not read."""
    st.markdown(
        f"""
        <div class="cw-caveat">
            <div class="ico">&#9888;</div>
            <div class="txt">{html}</div>
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
