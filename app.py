"""
app.py
──────
Single Streamlit entry point.

Flow:
  * page_config + global CSS injected ONCE here
  * DB bootstrapped
  * If not logged in -> auth screen and stop
  * Once authenticated -> render_dashboard(user) takes over the page,
    sidebar (incl. logout), and all tabs (incl. History).
"""

from __future__ import annotations

import streamlit as st

from auth import AuthError, authenticate_user, register_user
from db import init_db
from skin_lesion_dashboard import inject_css, render_dashboard

# Page config — MUST be the first Streamlit call, exactly once
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Shared theme CSS — applied to both the login screen and the dashboard
inject_css()

# Bootstrap DB (idempotent)
init_db()

# Session defaults
if "user" not in st.session_state:
    st.session_state.user = None


# ════════════════════════════════════════════════════════════════════════
# UNAUTHENTICATED VIEW
# ════════════════════════════════════════════════════════════════════════
def render_auth_screen() -> None:
    st.markdown(
        """
        <div style="padding:.5rem 0 .2rem">
          <span style="font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;
                       letter-spacing:-.03em;color:#541A1A">Skin Lesion Analysis</span>
        </div>
        <div style="font-size:.78rem;color:#8b4a4a;letter-spacing:.1em;text-transform:uppercase;
                    margin-bottom:1.4rem">
          Secure Access &nbsp;·&nbsp; Capstone 2026 &nbsp;·&nbsp; Bahçeşehir University
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Narrow, centred auth card on the wide layout
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                if st.form_submit_button("Login", use_container_width=True):
                    try:
                        user = authenticate_user(username, password)
                    except AuthError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.user = user
                        st.rerun()

        with tab_register:
            with st.form("register_form"):
                new_username = st.text_input(
                    "Username",
                    help="3–32 chars: letters, digits, underscores.",
                    key="reg_username",
                )
                new_password = st.text_input(
                    "Password",
                    type="password",
                    help="Minimum 8 characters.",
                    key="reg_password",
                )
                confirm = st.text_input(
                    "Confirm password",
                    type="password",
                    key="reg_password_confirm",
                )
                if st.form_submit_button("Create account", use_container_width=True):
                    try:
                        uid = register_user(new_username, new_password, confirm)
                    except AuthError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.user = {"id": uid, "username": new_username.strip()}
                        st.rerun()

        st.markdown(
            """
            <div class="disc" style="margin-top:1.2rem">
              <strong>⚠ Disclaimer:</strong> Educational and research use only.
              Not a medical device. Always consult a qualified dermatologist.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════════════════
def main() -> None:
    if st.session_state.user is None:
        render_auth_screen()
    else:
        render_dashboard(st.session_state.user)


main()
