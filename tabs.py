"""
Main Tab UI Components
"""

import streamlit as st
import pandas as pd
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
    """Render main application tabs with enhanced extraction options"""
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Analysis", "🚀 BRD Generation", "📊 Analytics", "👥 Collaboration"])
    
    with tab1:
        render_enhanced_document_analysis_tab(uploaded_file, extraction_options)
    
    with tab2:
        render_brd_generation_tab(uploaded_file)
    
    with tab3:
        render_analytics_tab()
    
    with tab4:
        render_collaboration_hub()

def render_enhanced_document_analysis_tab(uploaded_file, extraction_options: Dict[str, Any]):
    """Enhanced document analysis tab with comprehensive extraction options"""
    if uploaded_file is not None:
        # Document processing and analysis
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"**{uploaded_file.name}** uploaded successfully ({file_size_mb:.2f} MB)")
        
        # Show extraction configuration
        with st.expander("📋 Extraction Configuration", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("**Core Extraction**")
                st.write(f"🖼️ Images: {'✅' if extraction_options.get('extract_images') else '❌'}")
                st.write(f"📐 Formulas: {'✅' if extraction_options.get('extract_formulas') else '❌'}")
                st.write(f"📊 Tables: {'✅' if extraction_options.get('extract_tables') else '❌'}")
            
            with col2:
                st.info("**AI Analysis**")
                st.write(f"🧠 Intelligence: {'✅' if extraction_options.get('intelligent_analysis') else '❌'}")
                st.write(f"👥 Stakeholders: {'✅' if extraction_options.get('stakeholder_detection') else '❌'}")
                st.write(f"⚠️ Risk Assessment: {'✅' if extraction_options.get('risk_assessment') else '❌'}")
            
            with col3:
                st.info("**Thresholds**")
                st.write(f"Table Confidence: {extraction_options.get('table_confidence_threshold', 0.5):.1%}")
                st.write(f"Formula Confidence: {extraction_options.get('formula_confidence_threshold', 0.3):.1%}")
                st.write(f"Max Tables: {extraction_options.get('max_tables', 20)}")
                st.write(f"Max Formulas: {extraction_options.get('max_formulas', 50)}")
        
        # Enhanced document extraction with progress tracking
        with st.spinner("Performing advanced AI analysis with enhanced extraction..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Update progress
                status_text.text("🔍 Processing document structure...")
                progress_bar.progress(20)
                
                # Process document with enhanced extraction
                document_text, extracted_images, extracted_formulas, document_analysis = process_document_enhanced(
                    uploaded_file, 
                    extract_images=extraction_options.get('extract_images', True),
                    extract_formulas=extraction_options.get('extract_formulas', True),
                    extract_tables=extraction_options.get('extract_tables', True)
                )
                
                progress_bar.progress(40)
                status_text.text("📊 Extracting tables and formulas...")
                
                # Apply extraction filters based on options
                if extraction_options.get('extract_formulas'):
                    filtered_formulas = []
                    for formula in extracted_formulas:
                        # Apply confidence threshold
                        if formula.get('confidence', 0) >= extraction_options.get('formula_confidence_threshold', 0.3):
                            # Apply formula type filters
                            formula_type = formula.get('type', '')
                            include_formula = False
                            
                            if extraction_options.get('extract_correlations') and 'correlation' in formula_type:
                                include_formula = True
                            elif extraction_options.get('extract_risk_weights') and 'risk' in formula_type:
                                include_formula = True
                            elif extraction_options.get('extract_capital_formulas') and 'capital' in formula_type:
                                include_formula = True
                            elif extraction_options.get('extract_sensitivity') and 'sensitivity' in formula_type:
                                include_formula = True
                            elif extraction_options.get('extract_greek_letters') and 'greek' in formula_type:
                                include_formula = True
                            elif extraction_options.get('extract_summations') and 'summation' in formula_type:
                                include_formula = True
                            elif formula_type in ['extracted_table', 'docx_table']:
                                # Always include tables if table extraction is enabled
                                include_formula = extraction_options.get('extract_tables', True)
                            else:
                                # Include other formula types by default
                                include_formula = True
                            
                            if include_formula:
                                filtered_formulas.append(formula)
                    
                    # Limit number of formulas
                    max_formulas = extraction_options.get('max_formulas', 50)
                    extracted_formulas = filtered_formulas[:max_formulas]
                
                progress_bar.progress(60)
                status_text.text("🧠 Performing advanced analysis...")
                
                # Apply table confidence filtering
                if document_analysis.get('extracted_tables'):
                    table_threshold = extraction_options.get('table_confidence_threshold', 0.5)
                    filtered_tables = [
                        table for table in document_analysis['extracted_tables'] 
                        if table.get('confidence', 0) >= table_threshold
                    ]
                    max_tables = extraction_options.get('max_tables', 20)
                    document_analysis['extracted_tables'] = filtered_tables[:max_tables]
                
                progress_bar.progress(80)
                status_text.text("💾 Storing results...")
                
                # Store in session state
                st.session_state.document_text = document_text
                st.session_state.extracted_images = extracted_images
                st.session_state.extracted_formulas = extracted_formulas
                st.session_state.document_analysis = document_analysis
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Clear progress indicators after a moment
                import time
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
                st.error(f"Error processing document: {str(e)}")
                progress_bar.empty()
                status_text.empty()
                return
        
        # Enhanced metrics display with detailed breakdown
        st.subheader("📈 Extraction Results")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            char_count = len(document_text)
            st.metric("Content", f"{char_count:,} chars")
        
        with col2:
            image_count = len(extracted_images)
            st.metric("Images", image_count)
        
        with col3:
            formula_count = document_analysis.get('formula_analysis', {}).get('total_formulas', 0)
            st.metric("Formulas", formula_count)
        
        with col4:
            table_count = document_analysis.get('table_count', 0)
            st.metric("Tables", table_count)
        
        with col5:
            complexity = document_analysis.get('complexity_score', 0)
            st.metric("Complexity", f"{complexity:.1f}")
        
        # Enhanced document intelligence insights
        if document_analysis:
            st.subheader("AI Document Intelligence")
            
            # Document classification and framework detection
            col1, col2 = st.columns(2)
            
            with col1:
                doc_type = document_analysis.get('document_type', 'Unknown')
                st.info(f"**Document Type:** {doc_type}")
                
                frameworks = document_analysis.get('regulatory_framework', [])
                if frameworks:
                    st.info(f"**Regulatory Frameworks:** {', '.join(frameworks[:3])}")
                
                math_complexity = document_analysis.get('mathematical_complexity', 'Unknown')
                complexity_colors = {
                    'Very High': '🔴',
                    'High': '🟠', 
                    'Medium': '🟡',
                    'Low': '🟢',
                    'Unknown': '⚪'
                }
                st.info(f"**Mathematical Complexity:** {complexity_colors.get(math_complexity, '⚪')} {math_complexity}")
            
            with col2:
                key_entities = document_analysis.get('key_entities', [])
                if key_entities:
                    st.success(f"**Key Entities Found:** {len(key_entities)}")
                
                reg_sections = document_analysis.get('regulatory_sections', [])
                if reg_sections:
                    st.success(f"**Regulatory Sections:** {len(reg_sections)}")
                
                total_data_points = document_analysis.get('table_analysis', {}).get('structural_info', {}).get('total_data_points', 0)
                if total_data_points > 0:
                    st.success(f"**Total Data Points:** {total_data_points:,}")
            
            # Render enhanced extraction results
            render_enhanced_extraction_results(document_analysis, extracted_images)
            
            # Advanced insights section
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if extraction_options.get('show_extraction_preview'):
                    with st.expander("Content Preview", expanded=False):
                        preview_text = document_text[:3000] + "..." if len(document_text) > 3000 else document_text
                        st.text_area("Document Content", preview_text, height=200, disabled=True)
            
            with col2:
                if extracted_images and extraction_options.get('extract_images'):
                    with st.expander("Image Gallery", expanded=False):
                        st.subheader("Extracted Images")
                        cols = st.columns(3)
                        for idx, (img_key, img_b64) in enumerate(list(extracted_images.items())[:9]):
                            with cols[idx % 3]:
                                display_image_from_base64(img_b64, caption=img_key, max_width=150)
            
            with col3:
                if key_entities:
                    with st.expander("Key Entities", expanded=False):
                        st.subheader("Identified Entities")
                        for entity in key_entities[:15]:
                            st.write(f"• {entity}")
            
            # Regulatory sections preview
            if reg_sections and len(reg_sections) > 5:
                with st.expander("Regulatory Sections Detected", expanded=False):
                    st.subheader("Document Structure")
                    for i, section in enumerate(reg_sections[:20]):
                        section_text = section[:100] + "..." if len(section) > 100 else section
                        st.write(f"{i+1}. {section_text}")
            
            # Quality and confidence metrics
            formula_analysis = document_analysis.get('formula_analysis', {})
            table_analysis = document_analysis.get('table_analysis', {})
            
            if formula_analysis or table_analysis:
                st.subheader("Extraction Quality Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if formula_analysis:
                        avg_formula_conf = formula_analysis.get('avg_confidence', 0)
                        conf_color = "🟢" if avg_formula_conf > 0.7 else "🟡" if avg_formula_conf > 0.5 else "🔴"
                        st.metric("Formula Confidence", f"{conf_color} {avg_formula_conf:.1%}")
                
                with col2:
                    if table_analysis:
                        avg_table_conf = table_analysis.get('avg_confidence', 0)
                        conf_color = "🟢" if avg_table_conf > 0.7 else "🟡" if avg_table_conf > 0.5 else "🔴"
                        st.metric("Table Confidence", f"{conf_color} {avg_table_conf:.1%}")
                
                with col3:
                    if formula_analysis:
                        high_conf_formulas = formula_analysis.get('high_confidence_formulas', 0)
                        total_formulas = formula_analysis.get('total_formulas', 1)
                        high_conf_ratio = high_conf_formulas / total_formulas
                        st.metric("High Quality Formulas", f"{high_conf_ratio:.1%}")
                
                with col4:
                    if table_analysis:
                        large_tables = table_analysis.get('large_tables', 0)
                        total_tables = table_analysis.get('total_tables', 1)
                        large_table_ratio = large_tables / total_tables if total_tables > 0 else 0
                        st.metric("Large Tables", f"{large_table_ratio:.1%}")
        
        # Document optimization suggestions
        if extraction_options.get('intelligent_analysis'):
            st.subheader("Optimization Recommendations")
            
            suggestions = []
            
            # Analysis-based suggestions
            if document_analysis.get('complexity_score', 0) > 0.8:
                suggestions.append("🔍 Consider using higher confidence thresholds due to document complexity")
            
            if len(extracted_formulas) > 40:
                suggestions.append("📐 Large number of formulas detected - consider formula type filtering")
            
            if document_analysis.get('table_count', 0) > 15:
                suggestions.append("📊 Many tables found - review table extraction settings for optimal results")
            
            if document_analysis.get('mathematical_complexity') == 'Very High':
                suggestions.append("🧮 Very high mathematical complexity - enable all formula types for complete extraction")
            
            regulatory_frameworks = document_analysis.get('regulatory_framework', [])
            if 'basel' in ' '.join(regulatory_frameworks).lower():
                suggestions.append("🏦 Basel framework detected - optimize for correlation matrices and risk weights")
            
            # Performance suggestions
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 10:
                suggestions.append("⚡ Large document - consider enabling parallel processing for faster extraction")
            
            if suggestions:
                for suggestion in suggestions[:4]:  # Show top 4 suggestions
                    st.info(suggestion)
            else:
                st.success("✅ Current extraction settings are optimal for this document")
    
    else:
        # Enhanced sample document showcase when no file is uploaded
        st.info("Please upload a document to begin AI-powered analysis")
        
        st.subheader("Enhanced Document Processing Capabilities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔧 Advanced Extraction Features:**")
            st.write("• Mathematical formula detection with confidence scoring")
            st.write("• Structured table extraction with type classification") 
            st.write("• Regulatory framework auto-detection")
            st.write("• Risk weight and correlation matrix parsing")
            st.write("• Greek letter and mathematical symbol recognition")
            st.write("• Document complexity analysis")
        
        with col2:
            st.markdown("**📋 Supported Document Types:**")
            st.write("• Basel Committee regulations (MAR21, etc.)")
            st.write("• Financial risk management documents")
            st.write("• Regulatory compliance manuals")
            st.write("• Technical specifications with formulas")
            st.write("• Business requirements with data tables")
            st.write("• Policy documents with structured content")
        
        # Sample document showcase with enhanced details
        st.subheader("Sample Document Analysis")
        sample_docs = [
            {
                "name": "Basel MAR21 - Sensitivities Method", 
                "type": "Basel Regulatory", 
                "pages": 46, 
                "complexity": "Very High",
                "formulas": "50+ mathematical formulas",
                "tables": "12 correlation/risk weight tables",
                "features": ["Greek letter formulas", "Correlation matrices", "Risk calculations"]
            },
            {
                "name": "GDPR Compliance Framework", 
                "type": "Privacy Regulatory", 
                "pages": 89, 
                "complexity": "High",
                "formulas": "Legal definitions and clauses",
                "tables": "5 compliance requirement tables",
                "features": ["Stakeholder matrices", "Process flows", "Risk assessments"]
            },
            {
                "name": "SOX Internal Controls Guide", 
                "type": "Financial Control", 
                "pages": 156, 
                "complexity": "Medium",
                "formulas": "Financial calculation formulas",
                "tables": "8 control framework tables",
                "features": ["Control matrices", "Testing procedures", "Audit trails"]
            }
        ]
        
        for doc in sample_docs:
            with st.expander(f"📄 {doc['name']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Type:** {doc['type']}")
                    st.write(f"**Pages:** {doc['pages']}")
                    st.write(f"**Complexity:** {doc['complexity']}")
                
                with col2:
                    st.write(f"**Formulas:** {doc['formulas']}")
                    st.write(f"**Tables:** {doc['tables']}")
                
                with col3:
                    st.write("**Key Features:**")
                    for feature in doc['features']:
                        st.write(f"• {feature}")

def render_document_analysis_tab(uploaded_file, extraction_options: Dict[str, Any]):
    """Render document analysis tab"""
    if uploaded_file is not None:
        # Document processing and analysis
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"**{uploaded_file.name}** uploaded successfully ({file_size_mb:.2f} MB)")
        
        # Enhanced document extraction
        with st.spinner("Performing advanced AI analysis..."):
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
        
        # Enhanced metrics display
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Content", f"{len(document_text):,} chars")
        with col2:
            st.metric("Images", len(extracted_images))
        with col3:
            st.metric("Formulas", len(extracted_formulas))
        with col4:
            complexity = document_analysis.get('complexity_score', 0)
            st.metric("Complexity", f"{complexity:.1f}")
        
        # Document intelligence insights
        if document_analysis:
            st.subheader("AI Document Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Document Type:** {document_analysis.get('document_type', 'Unknown')}")
                if document_analysis.get('regulatory_framework'):
                    frameworks = ', '.join(document_analysis['regulatory_framework'])
                    st.info(f"**Regulatory Frameworks:** {frameworks}")
                
                math_complexity = document_analysis.get('mathematical_complexity', 'Unknown')
                st.info(f"**Mathematical Complexity:** {math_complexity}")
            
            with col2:
                if document_analysis.get('key_entities'):
                    st.success(f"**Key Entities Found:** {len(document_analysis['key_entities'])}")
                if document_analysis.get('regulatory_sections'):
                    st.success(f"**Regulatory Sections:** {len(document_analysis['regulatory_sections'])}")
                
                table_count = document_analysis.get('table_count', 0)
                st.success(f"**Tables Detected:** {table_count}")
            
            # Enhanced regulatory insights
            if document_analysis.get('formula_types'):
                st.subheader("Mathematical Content Analysis")
                formula_types = document_analysis['formula_types']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Formula Types", len(formula_types))
                with col2:
                    complexity_score = document_analysis.get('complexity_score', 0)
                    st.metric("Complexity Score", f"{complexity_score:.2f}")
                with col3:
                    total_formulas = len(extracted_formulas)
                    st.metric("Total Elements", total_formulas)
                
                # Formula type breakdown
                if formula_types:
                    st.write("**Formula Types Detected:**")
                    for ftype in formula_types[:10]:  # Show top 10 types
                        type_name = ftype.replace('_', ' ').title()
                        count = len([f for f in extracted_formulas if isinstance(f, dict) and f.get('type') == ftype])
                        st.write(f"• {type_name}: {count} instances")
            
            # Regulatory sections preview
            if document_analysis.get('regulatory_sections'):
                with st.expander("Regulatory Sections Detected"):
                    for section in document_analysis['regulatory_sections'][:15]:
                        st.write(f"• {section[:100]}...")
            
            # Preview extracted content
            with st.expander("Content Preview", expanded=False):
                preview_text = document_text[:2000] + "..." if len(document_text) > 2000 else document_text
                st.text_area("Document Content", preview_text, height=200, disabled=True)
            
            # Media preview with enhanced display
            if extracted_images or extracted_formulas:
                with st.expander("Extracted Media Gallery", expanded=False):
                    if extracted_images:
                        st.subheader("Images")
                        cols = st.columns(4)
                        for idx, (img_key, img_b64) in enumerate(extracted_images.items()):
                            with cols[idx % 4]:
                                display_image_from_base64(img_b64, caption=img_key, max_width=150)
                    
                    if extracted_formulas:
                        st.subheader("Mathematical Formulas and Regulatory Elements")
                        
                        # Group formulas by type for better organization
                        formula_groups = {}
                        for formula in extracted_formulas[:20]:  # Show top 20
                            if isinstance(formula, dict):
                                formula_type = formula.get('type', 'unknown')
                                confidence = formula.get('confidence', 0)
                                
                                if formula_type not in formula_groups:
                                    formula_groups[formula_type] = []
                                formula_groups[formula_type].append(formula)
                        
                        # Display grouped formulas
                        for formula_type, formulas_list in formula_groups.items():
                            type_name = formula_type.replace('_', ' ').title()
                            with st.expander(f"{type_name} ({len(formulas_list)} items)"):
                                for i, formula in enumerate(formulas_list):
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    with col1:
                                        st.code(formula.get('text', ''), language="text")
                                    with col2:
                                        confidence = formula.get('confidence', 0)
                                        st.metric("Confidence", f"{confidence:.1%}")
                                    with col3:
                                        if formula.get('page'):
                                            st.write(f"Page {formula['page']}")
                                    
                                    # Show context if available
                                    if formula.get('context'):
                                        st.caption(f"Context: {formula['context'][:100]}...")
                                    st.markdown("---")
                        
                        # Summary statistics
                        st.subheader("Formula Analysis Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            total_formulas = len(extracted_formulas)
                            st.metric("Total Elements", total_formulas)
                        with col2:
                            high_conf = len([f for f in extracted_formulas if isinstance(f, dict) and f.get('confidence', 0) > 0.7])
                            st.metric("High Confidence", high_conf)
                        with col3:
                            formula_types = len(set(f.get('type', 'unknown') for f in extracted_formulas if isinstance(f, dict)))
                            st.metric("Formula Types", formula_types)
                        with col4:
                            math_complexity = document_analysis.get('mathematical_complexity', 'Unknown')
                            st.metric("Complexity", math_complexity)
    else:
        st.info("Please upload a document to begin AI-powered analysis")
        
        # Sample document showcase
        st.subheader("Sample Documents")
        sample_docs = [
            {"name": "GDPR Compliance Guide", "type": "Regulatory", "pages": 89, "complexity": "High"},
            {"name": "SOX Internal Controls", "type": "Financial", "pages": 156, "complexity": "Medium"},
            {"name": "API Security Standards", "type": "Technical", "pages": 45, "complexity": "Medium"},
        ]
        
        for doc in sample_docs:
            with st.expander(f"{doc['name']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Type:** {doc['type']}")
                with col2:
                    st.write(f"**Pages:** {doc['pages']}")
                with col3:
                    st.write(f"**Complexity:** {doc['complexity']}")

def render_brd_generation_tab(uploaded_file):
    """Render BRD generation tab"""
    if uploaded_file is not None and st.session_state.get('document_text'):
        # Enhanced BRD generation
        st.subheader("AI-Powered BRD Generation")
        
        # Generation options
        col1, col2, col3 = st.columns(3)
        with col1:
            template_type = st.selectbox(
                "BRD Template",
                ["Standard Enterprise", "Regulatory Compliance", "Technical Integration", "Business Process"]
            )
        with col2:
            quality_level = st.selectbox(
                "Quality Level",
                ["Standard", "Premium", "Enterprise"]
            )
        with col3:
            stakeholder_focus = st.selectbox(
                "Stakeholder Focus",
                ["Balanced", "Business-Heavy", "Technical-Heavy", "Compliance-Heavy"]
            )
        
        # Advanced generation options
        with st.expander("Advanced Generation Settings"):
            col1, col2 = st.columns(2)
            with col1:
                include_risk_analysis = st.checkbox("Include Risk Analysis", value=True)
                include_timeline = st.checkbox("Include Implementation Timeline", value=True)
                include_kpis = st.checkbox("Include Success Metrics", value=True)
            with col2:
                auto_stakeholder_mapping = st.checkbox("Auto-map Stakeholders", value=True)
                compliance_validation = st.checkbox("Compliance Validation", value=True)
                generate_appendices = st.checkbox("Generate Appendices", value=True)
        
        # Generate button
        if st.button("Generate Enhanced BRD", type="primary", use_container_width=True):
            with st.spinner("Generating comprehensive BRD..."):
                try:
                    result = generate_enhanced_brd(
                        st.session_state.document_text,
                        st.session_state.extracted_images,
                        st.session_state.extracted_formulas,
                        st.session_state.document_analysis
                    )
                    
                    st.session_state.brd_content = result['brd_content']
                    st.session_state.quality_scores = result['quality_scores']
                    st.session_state.compliance_checks = result['compliance_checks']
                    st.session_state.generated = True
                    
                    st.success("Enhanced BRD Generation Complete!")
                    st.balloons()
                    
                except Exception as e:
                    logger.error(f"Error generating BRD: {e}")
                    st.error(f"Error generating BRD: {str(e)}")
        
        # Display generated content with export options
        if st.session_state.get('generated') and st.session_state.get('brd_content'):
            st.markdown("---")
            st.header("Enhanced BRD - Review & Edit")
            
            render_export_section()
            render_brd_content_editor()
            
    else:
        st.info("Please upload and analyze a document first in the Document Analysis tab")

def render_export_section():
    """Render export options section"""
    st.subheader("Export Options")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Export to Word", type="secondary"):
            try:
                word_doc = export_to_word_docx(st.session_state.brd_content)
                st.download_button(
                    label="Download Word Document",
                    data=word_doc.getvalue(),
                    file_name=f"BRD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except ImportError:
                st.error("Word export requires python-docx. Please install it.")
            except Exception as e:
                st.error(f"Error exporting to Word: {str(e)}")
    
    with col2:
        if st.button("Export to PDF", type="secondary"):
            try:
                pdf_doc = export_to_pdf(st.session_state.brd_content)
                st.download_button(
                    label="Download PDF Document",
                    data=pdf_doc.getvalue(),
                    file_name=f"BRD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            except ImportError:
                st.error("PDF export requires ReportLab. Please install it.")
            except Exception as e:
                st.error(f"Error exporting to PDF: {str(e)}")
    
    with col3:
        if st.button("Export to Excel", type="secondary"):
            try:
                excel_doc = export_to_excel(st.session_state.brd_content)
                st.download_button(
                    label="Download Excel File",
                    data=excel_doc.getvalue(),
                    file_name=f"BRD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error exporting to Excel: {str(e)}")
    
    with col4:
        if st.button("Export to JSON", type="secondary"):
            try:
                json_content = export_to_json(st.session_state.brd_content)
                st.download_button(
                    label="Download JSON File",
                    data=json_content,
                    file_name=f"BRD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error exporting to JSON: {str(e)}")

def render_brd_content_editor():
    """Render BRD content editor"""
    # Quality overview
    if st.session_state.quality_scores:
        avg_quality = sum(st.session_state.quality_scores.values()) / len(st.session_state.quality_scores)
        st.success(f"**Overall Quality Score: {avg_quality:.1f}%**")
    
    # Section tabs for editing
    section_names = list(st.session_state.brd_content.keys())
    section_tabs = st.tabs([name.split('.')[0] + "." for name in section_names])
    
    for i, (section_name, content) in enumerate(st.session_state.brd_content.items()):
        with section_tabs[i]:
            render_section_editor(section_name, content)

def render_section_editor(section_name: str, content):
    """Render individual section editor"""
    # Section header with quality indicator
    quality_score = st.session_state.quality_scores.get(section_name, 0)
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.subheader(section_name)
    with col2:
        st.metric("Quality", f"{quality_score:.0f}%")
    with col3:
        status = "Good" if quality_score >= 80 else "Fair" if quality_score >= 60 else "Needs Work"
        st.write(f"Status: {status}")
    
    # Section-specific quality checks
    section_checks = [c for c in st.session_state.compliance_checks if c.section == section_name]
    if section_checks:
        with st.expander("Quality Insights"):
            for check in section_checks:
                if check.status == "PASS":
                    st.success(f"✅ **{check.check_type.title()}:** {check.message}")
                elif check.status == "WARNING":
                    st.warning(f"⚠️ **{check.check_type.title()}:** {check.message}")
                else:
                    st.error(f"❌ **{check.check_type.title()}:** {check.message}")
    
    # Content editing
    if isinstance(content, dict):
        for subsection_name, subcontent in content.items():
            st.write(f"**{subsection_name}**")
            
            if isinstance(subcontent, pd.DataFrame):
                # Enhanced table editor
                st.write("Interactive Table Editor:")
                edited_df = st.data_editor(
                    subcontent,
                    use_container_width=True,
                    num_rows="dynamic",
                    key=f"table_{section_name}_{subsection_name}"
                )
                st.session_state.brd_content[section_name][subsection_name] = edited_df
            else:
                # Enhanced text editor
                render_content_with_images(subcontent, st.session_state.extracted_images)
                
                edited_text = st.text_area(
                    f"Edit {subsection_name}",
                    value=str(subcontent),
                    height=250,
                    key=f"text_{section_name}_{subsection_name}"
                )
                st.session_state.brd_content[section_name][subsection_name] = edited_text
            
            st.markdown("---")
    else:
        # Single content editing
        if isinstance(content, pd.DataFrame):
            st.write("Interactive Table Editor:")
            edited_df = st.data_editor(
                content,
                use_container_width=True,
                num_rows="dynamic",
                key=f"table_{section_name}"
            )
            st.session_state.brd_content[section_name] = edited_df
        else:
            # Enhanced text content editing
            render_content_with_images(str(content), st.session_state.extracted_images)
            
            edited_text = st.text_area(
                f"Edit {section_name}",
                value=str(content),
                height=300,
                key=f"text_{section_name}"
            )
            st.session_state.brd_content[section_name] = edited_text

def render_analytics_tab():
    """Render analytics dashboard tab"""
    st.subheader("Analytics Dashboard")
    
    if st.session_state.get('generated') and st.session_state.get('brd_content'):
        # Create analytics dashboards
        create_compliance_dashboard()
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            create_stakeholder_matrix()
        with col2:
            create_risk_heatmap()
        
        # Workflow timeline
        st.markdown("---")
        render_workflow_timeline()
        
    else:
        st.info("Generate a BRD first to view analytics")

def render_workflow_timeline():
    """Render workflow timeline"""
    st.subheader("Workflow Timeline")
    
    timeline_steps = [
        {"step": "Document Upload", "status": "completed", "date": "Today"},
        {"step": "AI Analysis", "status": "completed", "date": "Today"},
        {"step": "BRD Generation", "status": "in_progress", "date": "Today"},
        {"step": "Quality Review", "status": "pending", "date": "Tomorrow"},
        {"step": "Stakeholder Approval", "status": "pending", "date": "Next Week"},
        {"step": "Final Sign-off", "status": "pending", "date": "TBD"}
    ]
    
    for step in timeline_steps:
        status_color = {
            "completed": "#10B981",
            "in_progress": "#F59E0B", 
            "pending": "#6B7280"
        }[step["status"]]
        
        status_icon = {
            "completed": "✅",
            "in_progress": "⏳",
            "pending": "⭕"
        }[step["status"]]
        
        st.markdown(f"""
        <div class="timeline-item" style="border-left-color: {status_color};">
            {status_icon} <strong>{step['step']}</strong> - {step['date']}<br>
            <small style="color: {status_color};">{step['status'].replace('_', ' ').title()}</small>
        </div>
        """, unsafe_allow_html=True)
