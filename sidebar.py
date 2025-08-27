"""
Sidebar UI Component
"""

import streamlit as st
from typing import Tuple, Dict

def render_sidebar() -> Tuple[any, Dict[str, any]]:
    """Render sidebar with upload and options"""
    st.sidebar.title("BRD Generator Pro")
    st.sidebar.markdown("Advanced AI-powered document transformation")
    
    # User profile section
    st.sidebar.subheader("User Profile")
    user = st.session_state.current_user
    st.sidebar.info(f"**{user.name}**\n{user.role}\n{user.email}")
    
    # File upload section
    st.sidebar.subheader("Document Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Regulatory Document",
        type=['pdf', 'docx', 'txt'],
        help="Support for documents up to 500 pages with advanced AI analysis"
    )
    
    # Enhanced extraction options
    st.sidebar.subheader("AI Enhancement Options")
    extract_images = st.sidebar.checkbox("Extract & Analyze Images", value=True)
    extract_formulas = st.sidebar.checkbox("Detect Mathematical Formulas", value=True)
    intelligent_analysis = st.sidebar.checkbox("Advanced Document Intelligence", value=True)
    stakeholder_detection = st.sidebar.checkbox("Auto-detect Stakeholders", value=True)
    risk_assessment = st.sidebar.checkbox("AI Risk Assessment", value=True)
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        max_images = st.slider("Max Images to Extract", 1, 100, 30)
        quality_threshold = st.slider("Quality Threshold", 0.0, 1.0, 0.7)
        collaboration_mode = st.checkbox("Enable Real-time Collaboration", value=False)
    
    extraction_options = {
        'extract_images': extract_images,
        'extract_formulas': extract_formulas,
        'intelligent_analysis': intelligent_analysis,
        'stakeholder_detection': stakeholder_detection,
        'risk_assessment': risk_assessment,
        'max_images': max_images,
        'quality_threshold': quality_threshold,
        'collaboration_mode': collaboration_mode
    }
    
    return uploaded_file, extraction_options
