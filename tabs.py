"""
Enhanced Tab UI Components with Sophisticated Basel Document Processing
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime
from typing import Dict, Any

from utils.document_processor import process_document, display_image_from_base64, render_content_with_images
from utils.ai_processor import generate_enhanced_brd, parse_table_content
from utils.export_utils import export_to_word_docx, export_to_pdf, export_to_excel, export_to_json
from ui.analytics import create_compliance_dashboard, create_stakeholder_matrix, create_risk_heatmap
from ui.collaboration import render_collaboration_hub
from utils.logger import get_logger

logger = get_logger(__name__)

def render_main_tabs(uploaded_file, extraction_options: Dict[str, Any]):
    """Render main application tabs with enhanced processing"""
    tab1, tab2, tab3, tab4 = st.tabs(["Document Analysis", "BRD Generation", "Analytics", "Collaboration"])
    
    with tab1:
        render_sophisticated_document_analysis_tab(uploaded_file, extraction_options)
    
    with tab2:
        render_sophisticated_brd_generation_tab(uploaded_file)
    
    with tab3:
        render_analytics_tab()
    
    with tab4:
        render_collaboration_hub()

def render_sophisticated_document_analysis_tab(uploaded_file, extraction_options: Dict[str, Any]):
    """Enhanced document analysis with sophisticated Basel-specific processing"""
    if uploaded_file is not None:
        # Document processing and analysis
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"**{uploaded_file.name}** uploaded successfully ({file_size_mb:.2f} MB)")
        
        # Sophisticated document extraction
        with st.spinner("Performing sophisticated AI analysis with Basel-specific intelligence..."):
            try:
                document_text, extracted_images, extracted_formulas, document_analysis = process_document(
                    uploaded_file, 
                    extract_images=extraction_options.get('extract_images', True),
                    extract_formulas=extraction_options.get('extract_formulas', True)
                )
                
                # Store in session state
                st.session_state.document_text = document_text
                st.session_state.extracted_images = extracted_images
                st.session_state.extracted_formulas = extracted_formulas
                st.session_state.document_analysis = document_analysis
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                st.error(f"Error processing document: {str(e)}")
                return
        
        # Enhanced metrics display with sophisticated analysis
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Content", f"{len(document_text):,} chars")
        with col2:
            st.metric("Images", len(extracted_images))
        with col3:
            st.metric("Formulas", len(extracted_formulas))
        with col4:
            complexity = document_analysis.get('complexity_score', 0)
            st.metric("Complexity", f"{complexity:.2f}")
        with col5:
            math_complexity = document_analysis.get('mathematical_complexity', 'Low')
            st.metric("Math Level", math_complexity)
        
        # Sophisticated document intelligence insights
        if document_analysis:
            st.subheader("Sophisticated AI Document Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Document Classification:** {document_analysis.get('document_type', 'Unknown')}")
                
                frameworks = document_analysis.get('regulatory_framework', [])
                if frameworks:
                    framework_text = ', '.join(frameworks)
                    st.info(f"**Regulatory Frameworks:** {framework_text}")
                
                math_complexity = document_analysis.get('mathematical_complexity', 'Unknown')
                st.info(f"**Mathematical Sophistication:** {math_complexity}")
                
                if 'mathematical_formulas' in document_analysis:
                    formula_data = document_analysis['mathematical_formulas']
                    st.success(f"**Formula Analysis:** {formula_data.get('total_count', 0)} mathematical elements")
            
            with col2:
                key_entities = document_analysis.get('key_entities', {})
                if isinstance(key_entities, dict) and 'by_type' in key_entities:
                    total_entities = sum(len(entities) for entities in key_entities['by_type'].values())
                    st.success(f"**Regulatory Entities:** {total_entities} identified")
                elif isinstance(key_entities, list):
                    st.success(f"**Key Entities:** {len(key_entities)} found")
                
                regulatory_sections = document_analysis.get('regulatory_sections', [])
                if regulatory_sections:
                    st.success(f"**Regulatory Sections:** {len(regulatory_sections)} identified")
                
                table_count = document_analysis.get('table_count', 0)
                st.success(f"**Tables Detected:** {table_count}")
            
            # Sophisticated mathematical content analysis
            if 'mathematical_formulas' in document_analysis and document_analysis['mathematical_formulas'].get('total_count', 0) > 0:
                st.subheader("Advanced Mathematical Content Analysis")
                
                formula_data = document_analysis['mathematical_formulas']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Formulas", formula_data.get('total_count', 0))
                with col2:
                    by_type = formula_data.get('by_type', {})
                    st.metric("Formula Types", len(by_type))
                with col3:
                    key_formulas = len(formula_data.get('key_formulas', []))
                    st.metric("Key Formulas", key_formulas)
                with col4:
                    complexity_dist = formula_data.get('complexity_distribution', {})
                    high_complex = complexity_dist.get('High', 0) + complexity_dist.get('Very High', 0)
                    st.metric("Complex Formulas", high_complex)
                
                # Sophisticated formula type breakdown
                if by_type:
                    st.write("**Mathematical Formula Categories:**")
                    for ftype, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:10]:
                        type_display = ftype.replace('_', ' ').title()
                        st.write(f"• **{type_display}**: {count} instances")
                
                # Key formulas preview
                key_formulas = formula_data.get('key_formulas', [])
                if key_formulas:
                    with st.expander("Key Mathematical Formulas Preview"):
                        for i, formula in enumerate(key_formulas[:5]):
                            st.write(f"**{i+1}. {formula.get('type', '').replace('_', ' ').title()}**")
                            st.code(formula.get('text', '')[:150], language='text')
                            st.caption(f"Confidence: {formula.get('confidence', 0):.1%} | Page: {formula.get('page', 'Unknown')}")
                            st.markdown("---")
            
            # Enhanced regulatory sections preview
            if regulatory_sections:
                with st.expander("Regulatory Structure Analysis"):
                    st.write("**Major Regulatory Sections Identified:**")
                    for i, section in enumerate(regulatory_sections[:15]):
                        section_text = section[:120] + "..." if len(str(section)) > 120 else str(section)
                        st.write(f"{i+1}. {section_text}")
            
            # Sophisticated content preview with mathematical context
            with st.expander("Document Content Preview with Mathematical Context"):
                preview_text = document_text[:3000] + "..." if len(document_text) > 3000 else document_text
                
                # Highlight mathematical content in preview
                highlighted_text = preview_text
                for formula in extracted_formulas[:10]:  # Highlight top 10 formulas
                    if isinstance(formula, dict):
                        formula_text = formula.get('text', '')
                        if len(formula_text) > 3 and formula_text in highlighted_text:
                            highlighted_text = highlighted_text.replace(formula_text, f"**{formula_text}**")
                
                st.text_area("Enhanced Document Preview", highlighted_text, height=250, disabled=True)
            
            # Sophisticated media gallery with classification
            if extracted_images or extracted_formulas:
                with st.expander("Advanced Content Gallery"):
                    if extracted_images:
                        st.subheader("Extracted Visual Content")
                        
                        # Classify images by type
                        math_images = [k for k in extracted_images.keys() if 'math' in k.lower() or 'formula' in k.lower()]
                        table_images = [k for k in extracted_images.keys() if 'table' in k.lower()]
                        other_images = [k for k
