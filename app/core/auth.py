"""Optional Google (OIDC) login gate for the hosted app.

Uses Streamlit's native authentication (``st.login`` / ``st.user``). Login is
enforced only when an ``[auth]`` block is present in ``st.secrets`` — so local
development (no secrets) runs ungated, while the deployed app requires sign-in.

Access is further restricted to an allowlist of emails set in secrets::

    [auth]
    redirect_uri = "https://<your-app-url>/oauth2callback"
    cookie_secret = "<random-long-string>"
    client_id = "<google-oauth-client-id>"
    client_secret = "<google-oauth-client-secret>"
    server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

    [access]
    allowed_emails = ["alice@example.com", "bob@example.com"]
"""
from __future__ import annotations

import streamlit as st

from theme.palette import ACCENT, BORDER, SUBTEXT, SURFACE, TEXT, BRAND_MARK_SVG


def _auth_configured() -> bool:
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def _auth_supported() -> bool:
    return all(hasattr(st, name) for name in ("user", "login", "logout"))


def _allowed_emails() -> set[str]:
    try:
        raw = st.secrets.get("access", {}).get("allowed_emails", [])
    except Exception:
        raw = []
    return {str(e).strip().lower() for e in raw}


def require_login() -> None:
    """Block the app until a permitted user is signed in (no-op when unconfigured)."""
    if not _auth_configured():
        return
    if not _auth_supported():
        version = getattr(st, "__version__", "desconocida")
        st.error(
            "Google login está configurado, pero esta sesión está usando "
            f"Streamlit {version}, que no soporta autenticación nativa. "
            "Arranca la app con `uv run streamlit run app/main.py` para usar "
            "el entorno del proyecto."
        )
        st.stop()

    if not getattr(st.user, "is_logged_in", False):
        _login_screen()
        st.stop()

    allow = _allowed_emails()
    email = (getattr(st.user, "email", "") or "").lower()
    if allow and email not in allow:
        _denied_screen(email)
        st.stop()


def _auth_styles() -> None:
    st.markdown(
        f"""
        <style>
        /* Hide chrome on the gate so it reads as a clean sign-in page. */
        header[data-testid="stHeader"], section[data-testid="stSidebar"] {{ display: none !important; }}
        .block-container {{ padding-top: 4vh !important; }}
        .cw-auth {{ max-width: 620px; margin: 6vh auto 0; text-align: center; }}
        .cw-auth-card {{
            background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 22px;
            padding: 56px 56px 44px; box-shadow: 0 30px 80px -40px #000;
        }}
        .cw-auth-mark {{ color: {ACCENT}; line-height: 0; }}
        .cw-auth-mark svg {{ width: 76px; height: 76px; }}
        .cw-auth-word {{
            font-size: 40px; font-weight: 800; letter-spacing: .4px; color: {ACCENT};
            margin-top: 16px;
        }}
        .cw-auth-word span {{ color: {TEXT}; }}
        .cw-auth-sub {{ color: {SUBTEXT}; font-size: 16px; margin-top: 8px; }}
        .cw-auth-rule {{ height: 1px; background: {BORDER}; margin: 30px 0 24px; }}
        .cw-auth-note {{ color: {TEXT}; font-size: 20px; opacity: .9; line-height: 1.65; }}
        .cw-auth-note b {{ color: {ACCENT}; }}
        /* Sign-in button: large, centred, and clearly primary. */
        div[class*="st-key-cw_login_btn"], div[class*="st-key-cw_denied_btn"] {{
            width: min(100%, 460px) !important; margin: 42px auto 0 !important;
        }}
        div[class*="st-key-cw_login_btn"] button, div[class*="st-key-cw_denied_btn"] button {{
            width: 100% !important; min-height: 66px !important;
            background: {SURFACE} !important;
            border: 1px solid {ACCENT}88 !important;
            border-radius: 16px !important;
            padding: 0 28px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 14px !important;
            font-weight: 800 !important;
            box-shadow: inset 0 0 0 1px {ACCENT}22, 0 18px 42px -28px {ACCENT} !important;
            transition: border-color .15s ease, box-shadow .15s ease;
        }}
        div[class*="st-key-cw_login_btn"] button:hover, div[class*="st-key-cw_denied_btn"] button:hover {{
            border-color: {ACCENT} !important; background: {SURFACE} !important;
            box-shadow: inset 0 0 0 1px {ACCENT}66, 0 20px 46px -24px {ACCENT} !important;
        }}
        div[class*="st-key-cw_login_btn"] button p, div[class*="st-key-cw_denied_btn"] button p {{
            color: {TEXT} !important; font-size: 21px !important; letter-spacing: .1px;
            line-height: 1 !important;
        }}
        div[class*="st-key-cw_login_btn"] button span, div[class*="st-key-cw_denied_btn"] button span {{
            color: {ACCENT} !important; font-size: 1.55rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _auth_shell(inner_html: str) -> None:
    _auth_styles()
    st.markdown(
        f"""
        <div class="cw-auth">
          <div class="cw-auth-card">
            <div class="cw-auth-mark">{BRAND_MARK_SVG}</div>
            <div class="cw-auth-word">Crude<span>Watch</span></div>
            <div class="cw-auth-sub">Contract exploration for WTI crude futures</div>
            <div class="cw-auth-rule"></div>
            {inner_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _login_screen() -> None:
    _auth_shell(
        '<div class="cw-auth-note">This is a <b>private</b> tool. '
        "Please sign in with an authorised Google account to continue.</div>"
    )
    if st.button("Sign in with Google", key="cw_login_btn", icon=":material/login:"):
        st.login()


def _denied_screen(email: str) -> None:
    _auth_shell(
        '<div class="cw-auth-note">The account '
        f"<b>{email or 'you used'}</b> is not authorised to access CrudeWatch.<br>"
        "Ask the owner to add your email, or sign in with a different account.</div>"
    )
    if st.button("Sign out", key="cw_denied_btn", icon=":material/logout:"):
        st.logout()


def sidebar_account() -> None:
    """Show the signed-in user and a sign-out control (when auth is active)."""
    if (
        not _auth_configured()
        or not _auth_supported()
        or not getattr(st.user, "is_logged_in", False)
    ):
        return
    email = getattr(st.user, "email", "") or "signed in"
    st.caption(f"Signed in as {email}")
    if st.button("Sign out"):
        st.logout()
