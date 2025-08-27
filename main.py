"""
Main Application Entry Point
AI-Powered BRD Generator Pro
"""

import streamlit as st
from config.app_config import configure_app, init_session_state
from ui.header import render_enhanced_header
from ui.sidebar import render_sidebar
from ui.tabs import render_main_tabs
from utils.logger import setup_logger

def main():
    """Main application function"""
    logger = setup_logger()
    
    try:
        # Configure Streamlit app
        configure_app()
        
        # Initialize session state
        init_session_state()
        
        # Render UI components
        render_enhanced_header()
        
        # Main content with sidebar and tabs
        uploaded_file, extraction_options = render_sidebar()
        render_main_tabs(uploaded_file, extraction_options)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
