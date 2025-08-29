"""
Enhanced Sidebar UI Component with Table Extraction
"""

import streamlit as st
from typing import Tuple, Dict

def render_sidebar() -> Tuple[any, Dict[str, any]]:
    """Render sidebar with upload and enhanced options"""
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
    
    # Core extraction options
    extract_images = st.sidebar.checkbox(
        "Extract & Analyze Images", 
        value=True,
        help="Extract diagrams, charts, and visual elements from documents"
    )
    
    extract_formulas = st.sidebar.checkbox(
        "Detect Mathematical Formulas", 
        value=True,
        help="Extract mathematical expressions, equations, and regulatory formulas"
    )
    
    extract_tables = st.sidebar.checkbox(
        "Extract Data Tables", 
        value=True,
        help="Extract structured tables including correlation matrices, risk weights, and regulatory data"
    )
    
    # Advanced AI analysis options
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Advanced Analysis**")
    
    intelligent_analysis = st.sidebar.checkbox(
        "Advanced Document Intelligence", 
        value=True,
        help="AI-powered analysis of document structure, complexity, and regulatory framework"
    )
    
    stakeholder_detection = st.sidebar.checkbox(
        "Auto-detect Stakeholders", 
        value=True,
        help="Automatically identify stakeholders, roles, and responsibilities"
    )
    
    risk_assessment = st.sidebar.checkbox(
        "AI Risk Assessment", 
        value=True,
        help="Identify and assess regulatory risks and compliance requirements"
    )
    
    # Enhanced extraction settings
    with st.sidebar.expander("📊 Table & Formula Settings"):
        st.markdown("**Table Extraction**")
        table_confidence_threshold = st.slider(
            "Table Detection Confidence", 
            0.1, 1.0, 0.5,
            help="Minimum confidence level for table detection"
        )
        
        max_tables = st.slider(
            "Max Tables to Extract", 
            1, 50, 20,
            help="Maximum number of tables to extract per document"
        )
        
        st.markdown("**Formula Extraction**")
        formula_confidence_threshold = st.slider(
            "Formula Detection Confidence", 
            0.1, 1.0, 0.3,
            help="Minimum confidence level for mathematical formula detection"
        )
        
        max_formulas = st.slider(
            "Max Formulas to Extract", 
            1, 100, 50,
            help="Maximum number of formulas to extract per document"
        )
        
        # Formula type filters
        st.markdown("**Formula Types to Extract**")
        extract_correlations = st.checkbox("Correlation Formulas (ρ)", value=True)
        extract_risk_weights = st.checkbox("Risk Weight Formulas (RW)", value=True)
        extract_capital_formulas = st.checkbox("Capital Requirement Formulas (K)", value=True)
        extract_sensitivity = st.checkbox("Sensitivity Formulas (PV01, CS01)", value=True)
        extract_greek_letters = st.checkbox("Greek Letter Expressions", value=True)
        extract_summations = st.checkbox("Summation Formulas (∑)", value=True)
    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced Options"):
        max_images = st.slider("Max Images to Extract", 1, 100, 30)
        quality_threshold = st.slider("Overall Quality Threshold", 0.0, 1.0, 0.7)
        collaboration_mode = st.checkbox("Enable Real-time Collaboration", value=False)
        
        # Document processing options
        st.markdown("**Document Processing**")
        preserve_formatting = st.checkbox("Preserve Original Formatting", value=True)
        extract_metadata = st.checkbox("Extract Document Metadata", value=True)
        
        # Performance options
        st.markdown("**Performance Settings**")
        parallel_processing = st.checkbox("Enable Parallel Processing", value=True)
        cache_results = st.checkbox("Cache Extraction Results", value=True)
    
    # Document type hints
    with st.sidebar.expander("📋 Document Type Optimization"):
        document_type = st.selectbox(
            "Document Type (optional)",
            ["Auto-detect", "Basel Regulatory", "Financial Policy", "Technical Specification", 
             "Business Requirements", "Compliance Manual", "Risk Assessment"],
            help="Optimize extraction algorithms for specific document types"
        )
        
        regulatory_framework = st.multiselect(
            "Expected Regulatory Framework",
            ["Basel III", "Basel IV", "Solvency II", "MiFID II", "GDPR", "SOX", "Dodd-Frank"],
            help="Pre-configure extraction for known regulatory frameworks"
        )
    
    # Extraction preview settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Preview Options**")
    show_extraction_preview = st.sidebar.checkbox(
        "Show Extraction Preview", 
        value=True,
        help="Display preview of extracted elements during processing"
    )
    
    # Compile all extraction options
    extraction_options = {
        # Core extraction flags
        'extract_images': extract_images,
        'extract_formulas': extract_formulas,
        'extract_tables': extract_tables,
        
        # Advanced analysis flags
        'intelligent_analysis': intelligent_analysis,
        'stakeholder_detection': stakeholder_detection,
        'risk_assessment': risk_assessment,
        
        # Table extraction settings
        'table_confidence_threshold': table_confidence_threshold,
        'max_tables': max_tables,
        
        # Formula extraction settings
        'formula_confidence_threshold': formula_confidence_threshold,
        'max_formulas': max_formulas,
        'extract_correlations': extract_correlations,
        'extract_risk_weights': extract_risk_weights,
        'extract_capital_formulas': extract_capital_formulas,
        'extract_sensitivity': extract_sensitivity,
        'extract_greek_letters': extract_greek_letters,
        'extract_summations': extract_summations,
        
        # General settings
        'max_images': max_images,
        'quality_threshold': quality_threshold,
        'collaboration_mode': collaboration_mode,
        'preserve_formatting': preserve_formatting,
        'extract_metadata': extract_metadata,
        'parallel_processing': parallel_processing,
        'cache_results': cache_results,
        
        # Document optimization
        'document_type': document_type,
        'regulatory_framework': regulatory_framework,
        'show_extraction_preview': show_extraction_preview
    }
    
    # Display extraction summary
    if uploaded_file:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Extraction Summary**")
        enabled_features = []
        if extract_images:
            enabled_features.append("🖼️ Images")
        if extract_formulas:
            enabled_features.append("📐 Formulas")
        if extract_tables:
            enabled_features.append("📊 Tables")
        if intelligent_analysis:
            enabled_features.append("🧠 AI Analysis")
        
        if enabled_features:
            st.sidebar.success(f"**Enabled:** {', '.join(enabled_features)}")
        
        # Show expected extraction counts
        file_size_mb = uploaded_file.size / (1024 * 1024)
        estimated_elements = int(file_size_mb * 10)  # Rough estimate
        
        st.sidebar.info(f"""
        **Document:** {uploaded_file.name}  
        **Size:** {file_size_mb:.1f} MB  
        **Est. Elements:** ~{estimated_elements}
        """)
    
    return uploaded_file, extraction_options
