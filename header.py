"""
Header UI Component
"""

import streamlit as st

def render_enhanced_header():
    """Render enhanced header with branding"""
    st.markdown("""
    <div class="main-header">
        <h1>AI-Powered BRD Generator Pro</h1>
        <p>Transform regulatory documents into comprehensive Business Requirements Documents with advanced AI intelligence, real-time collaboration, and compliance tracking</p>
    </div>
    """, unsafe_allow_html=True)
