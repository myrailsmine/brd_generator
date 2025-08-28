"""
Updated Main Tab UI Components with Enhanced Image and Formula Processing
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime
from typing import Dict, Any

# Updated imports to use the enhanced processors
from utils.document_processor import process_document_enhanced, display_image_from_base64, render_content_with_images
from utils.ai_processor import generate_enhanced_brd_with_images, parse_table_content_enhanced
from utils.export_utils import export_to_word_docx, export_to_pdf, export_to_excel, export_to_json
from ui.analytics import create_compliance_dashboard, create_stakeholder_matrix, create_risk_heatmap
from ui.collaboration import render_collaboration_hub
from utils.logger import get_logger

logger = get_logger(__name__)

def render_main_tabs(uploaded_file, extraction_options: Dict[str, Any]):
    """Render main application tabs with enhanced processing"""
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Analysis", "🚀 BRD Generation", "📊 Analytics", "👥 Collaboration"])
    
    with tab1:
        render_document_analysis_tab_enhanced(uploaded_file, extraction_options)
    
    with tab2:
        render_brd_generation_tab_enhanced(uploaded_file)
    
    with tab3:
        render_analytics_tab()
    
    with tab4:
        render_collaboration_hub()

def render_document_analysis_tab_enhanced(uploaded_file, extraction_options: Dict[str, Any]):
    """Enhanced document analysis tab with improved extraction capabilities"""
    if uploaded_file is not None:
        # Document processing and analysis
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"**{uploaded_file.name}** uploaded successfully ({file_size_mb:.2f} MB)")
        
        # Enhanced document extraction with progress bar
        progress_bar = st.progress(0, text="Initializing advanced AI analysis...")
        
        try:
            progress_bar.progress(20, text="Extracting text content...")
            
            document_text, extracted_images, extracted_elements, document_analysis = process_document_enhanced(
                uploaded_file, 
                extract_images=extraction_options.get('extract_images', True),
                extract_formulas=extraction_options.get('extract_formulas', True)
            )
            
            progress_bar.progress(60, text="Processing mathematical formulas...")
            
            # Store in session state
            st.session_state.document_text = document_text
            st.session_state.extracted_images = extracted_images
            st.session_state.extracted_formulas = extracted_elements  # Now contains all elements
            st.session_state.document_analysis = document_analysis
            
            progress_bar.progress(100, text="Analysis complete!")
            progress_bar.empty()
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            st.error(f"Error processing document: {str(e)}")
            progress_bar.empty()
            return
        
        # Enhanced metrics display with new categories
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Content", f"{len(document_text):,} chars")
        with col2:
            st.metric("Total Images", len(extracted_images))
        with col3:
            formula_count = len([e for e in extracted_elements if 'formula' in e.get('type', '')])
            st.metric("Formulas", formula_count)
        with col4:
            table_count = len([e for e in extracted_elements if 'table' in e.get('type', '')])
            st.metric("Tables", table_count)
        with col5:
            complexity = document_analysis.get('complexity_score', 0)
            st.metric("Complexity", f"{complexity:.1f}")
        
        # Enhanced extraction summary
        extraction_summary = document_analysis.get('extraction_summary', {})
        if extraction_summary:
            st.subheader("Extraction Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Text Formulas:** {extraction_summary.get('text_formulas', 0)}")
                st.info(f"**Formula Images:** {extraction_summary.get('formula_images', 0)}")
            with col2:
                st.success(f"**Table Images:** {extraction_summary.get('table_images', 0)}")
                st.success(f"**Diagrams:** {extraction_summary.get('diagrams', 0)}")
            with col3:
                st.warning(f"**Embedded Images:** {extraction_summary.get('embedded_images', 0)}")
                st.warning(f"**Pages Processed:** {extraction_summary.get('pages_processed', 0)}")
        
        # Document intelligence insights with enhanced categorization
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
                
                total_elements = len(extracted_elements)
                st.success(f"**Total Elements:** {total_elements}")
            
            # Enhanced visual content analysis
            st.subheader("Visual Content Analysis")
            
            # Categorize elements for display
            element_categories = {
                'formula_images': [e for e in extracted_elements if e.get('type') == 'formula_image'],
                'table_images': [e for e in extracted_elements if e.get('type') == 'table_image'],
                'diagrams': [e for e in extracted_elements if 'diagram' in e.get('type', '')],
                'text_formulas': [e for e in extracted_elements if 'formula' in e.get('type', '') and 'image_data' not in e]
            }
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Formula Images", len(element_categories['formula_images']))
            with col2:
                st.metric("Table Images", len(element_categories['table_images']))
            with col3:
                st.metric("Diagrams", len(element_categories['diagrams']))
            with col4:
                st.metric("Text Formulas", len(element_categories['text_formulas']))
            
            # Enhanced media preview with categorized display
            if extracted_images or extracted_elements:
                with st.expander("Enhanced Media Gallery", expanded=False):
                    
                    # Formula Images Section
                    if element_categories['formula_images']:
                        st.subheader("🧮 Mathematical Formula Images")
                        cols = st.columns(3)
                        for idx, formula_img in enumerate(element_categories['formula_images']):
                            with cols[idx % 3]:
                                key = formula_img.get('key', '')
                                if key in extracted_images:
                                    formula_text = formula_img.get('formula_text', 'Mathematical Formula')
                                    confidence = formula_img.get('confidence', 0)
                                    page = formula_img.get('page', 'Unknown')
                                    
                                    st.write(f"**Page {page}** (Confidence: {confidence:.1%})")
                                    display_image_from_base64(extracted_images[key], 
                                                            caption=f"Formula: {formula_text[:50]}...", 
                                                            max_width=200)
                                    st.caption(f"Key: {key}")
                    
                    # Table Images Section
                    if element_categories['table_images']:
                        st.subheader("📊 Table Structure Images")
                        cols = st.columns(3)
                        for idx, table_img in enumerate(element_categories['table_images']):
                            with cols[idx % 3]:
                                key = table_img.get('key', '')
                                if key in extracted_images:
                                    lines = table_img.get('lines', 0)
                                    confidence = table_img.get('confidence', 0)
                                    page = table_img.get('page', 'Unknown')
                                    
                                    st.write(f"**Page {page}** ({lines} rows, Confidence: {confidence:.1%})")
                                    display_image_from_base64(extracted_images[key], 
                                                            caption=f"Table with {lines} rows", 
                                                            max_width=200)
                                    st.caption(f"Key: {key}")
                    
                    # Diagrams Section
                    if element_categories['diagrams']:
                        st.subheader("📈 Diagrams and Charts")
                        cols = st.columns(3)
                        for idx, diagram in enumerate(element_categories['diagrams']):
                            with cols[idx % 3]:
                                key = diagram.get('key', '')
                                if key in extracted_images:
                                    elements_count = diagram.get('elements_count', 0)
                                    confidence = diagram.get('confidence', 0)
                                    page = diagram.get('page', 'Unknown')
                                    
                                    st.write(f"**Page {page}** ({elements_count} elements, Confidence: {confidence:.1%})")
                                    display_image_from_base64(extracted_images[key], 
                                                            caption=f"Diagram with {elements_count} elements", 
                                                            max_width=200)
                                    st.caption(f"Key: {key}")
                    
                    # Regular Embedded Images
                    embedded_images = [e for e in extracted_elements if e.get('type') == 'embedded_image']
                    if embedded_images:
                        st.subheader("🖼️ Embedded Images")
                        cols = st.columns(4)
                        for idx, emb_img in enumerate(embedded_images):
                            with cols[idx % 4]:
                                key = emb_img.get('key', '')
                                if key in extracted_images:
                                    img_format = emb_img.get('format', 'image')
                                    page = emb_img.get('page', 'Unknown')
                                    
                                    display_image_from_base64(extracted_images[key], 
                                                            caption=f"Page {page} ({img_format})", 
                                                            max_width=150)
                                    st.caption(f"Key: {key}")
                    
                    # Text-based Formulas Section
                    if element_categories['text_formulas']:
                        st.subheader("📝 Text-Based Mathematical Formulas")
                        
                        # Group by formula type for better organization
                        formula_groups = {}
                        for formula in element_categories['text_formulas'][:20]:  # Show top 20
                            formula_type = formula.get('type', 'unknown')
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
                                    if i < len(formulas_list) - 1:
                                        st.markdown("---")
            
            # Enhanced preview with image integration awareness
            with st.expander("Content Preview with Image Integration", expanded=False):
                preview_text = document_text[:2000] + "..." if len(document_text) > 2000 else document_text
                st.text_area("Document Content", preview_text, height=200, disabled=True)
                
                st.write("**Available for BRD Integration:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success(f"✅ {len(element_categories['formula_images'])} Formula Images")
                    st.success(f"✅ {len(element_categories['text_formulas'])} Text Formulas")
                with col2:
                    st.success(f"✅ {len(element_categories['table_images'])} Table Images")
                    st.success(f"✅ {len(embedded_images)} Embedded Images")
                with col3:
                    st.success(f"✅ {len(element_categories['diagrams'])} Diagrams")
                    st.success(f"✅ Total: {len(extracted_images)} Visual Elements")
    else:
        st.info("Please upload a document to begin AI-powered analysis with enhanced image and formula extraction")
        
        # Enhanced sample document showcase
        st.subheader("Enhanced Processing Capabilities")
        
        capabilities = [
            {
                "feature": "Mathematical Formula Extraction",
                "description": "Extracts both text-based formulas and renders formula regions as high-quality images",
                "types": "Equations, Risk calculations, Statistical formulas, Greek symbols"
            },
            {
                "feature": "Table Image Capture", 
                "description": "Identifies table structures and captures them as images for visual reference",
                "types": "Regulatory tables, Data matrices, Compliance frameworks"
            },
            {
                "feature": "Diagram Recognition",
                "description": "Detects and extracts process diagrams, charts, and flowcharts",
                "types": "Process flows, Organizational charts, Technical diagrams"
            },
            {
                "feature": "Integrated BRD Generation",
                "description": "Incorporates all visual elements directly into BRD sections with proper references",
                "types": "Image references, Formula integration, Table context"
            }
        ]
        
        for capability in capabilities:
            with st.expander(f"🚀 {capability['feature']}"):
                st.write(capability['description'])
                st.write(f"**Supports:** {capability['types']}")

def render_brd_generation_tab_enhanced(uploaded_file):
    """Enhanced BRD generation tab with image integration"""
    if uploaded_file is not None and st.session_state.get('document_text'):
        # Enhanced BRD generation interface
        st.subheader("AI-Powered BRD Generation with Visual Integration")
        
        # Show extraction summary
        if st.session_state.get('extracted_images'):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_images = len(st.session_state.extracted_images)
                st.metric("Total Images", total_images)
            with col2:
                formula_imgs = len([e for e in st.session_state.get('extracted_formulas', []) if e.get('type') == 'formula_image'])
                st.metric("Formula Images", formula_imgs)
            with col3:
                table_imgs = len([e for e in st.session_state.get('extracted_formulas', []) if e.get('type') == 'table_image'])
                st.metric("Table Images", table_imgs)
            with col4:
                diagrams = len([e for e in st.session_state.get('extracted_formulas', []) if 'diagram' in e.get('type', '')])
                st.metric("Diagrams", diagrams)
        
        # Enhanced generation options
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
            visual_integration = st.selectbox(
                "Visual Integration",
                ["Full Integration", "Selective References", "Text Only"]
            )
        
        # Advanced generation options with image controls
        with st.expander("Advanced Generation Settings"):
            col1, col2 = st.columns(2)
            with col1:
                include_risk_analysis = st.checkbox("Include Risk Analysis", value=True)
                include_timeline = st.checkbox("Include Implementation Timeline", value=True)
                include_kpis = st.checkbox("Include Success Metrics", value=True)
                integrate_formula_images = st.checkbox("Integrate Formula Images", value=True)
            with col2:
                auto_stakeholder_mapping = st.checkbox("Auto-map Stakeholders", value=True)
                compliance_validation = st.checkbox("Compliance Validation", value=True)
                integrate_table_images = st.checkbox("Integrate Table Images", value=True)
                integrate_diagrams = st.checkbox("Integrate Diagrams", value=True)
        
        # Enhanced generate button with visual integration info
        if st.button("🚀 Generate Enhanced BRD with Visual Integration", type="primary", use_container_width=True):
            with st.spinner("Generating comprehensive BRD with integrated images..."):
                try:
                    # Progress indicator
                    progress = st.progress(0, text="Initializing enhanced AI generation...")
                    
                    progress.progress(25, text="Processing visual elements...")
                    
                    # Use the enhanced generation function
                    result = generate_enhanced_brd_with_images(
                        st.session_state.document_text,
                        st.session_state.extracted_images,
                        st.session_state.extracted_formulas,  # Now contains all elements
                        st.session_state.document_analysis
                    )
                    
                    progress.progress(75, text="Integrating images and formulas...")
                    
                    st.session_state.brd_content = result['brd_content']
                    st.session_state.quality_scores = result['quality_scores']
                    st.session_state.compliance_checks = result['compliance_checks']
                    st.session_state.image_integration_stats = result.get('image_integration_stats', {})
                    st.session_state.generated = True
                    
                    progress.progress(100, text="Generation complete!")
                    progress.empty()
                    
                    # Show integration statistics
                    st.success("Enhanced BRD Generation Complete with Visual Integration!")
                    
                    if st.session_state.image_integration_stats:
                        stats = st.session_state.image_integration_stats
                        st.info(f"**Visual Integration Summary:** "
                               f"{stats.get('total_images', 0)} total images, "
                               f"{stats.get('formula_images', 0)} formula images, "
                               f"{stats.get('table_images', 0)} table images, "
                               f"{stats.get('diagrams', 0)} diagrams integrated")
                    
                    st.balloons()
                    
                except Exception as e:
                    logger.error(f"Error generating enhanced BRD: {e}")
                    st.error(f"Error generating enhanced BRD: {str(e)}")
        
        # Display generated content with enhanced image integration
        if st.session_state.get('generated') and st.session_state.get('brd_content'):
            st.markdown("---")
            st.header("Enhanced BRD with Visual Integration - Review & Edit")
            
            # Show visual integration summary
            if st.session_state.get('image_integration_stats'):
                with st.expander("Visual Integration Summary"):
                    stats = st.session_state.image_integration_stats
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Images", stats.get('total_images', 0))
                    with col2:
                        st.metric("Formula Images", stats.get('formula_images', 0))
                    with col3:
                        st.metric("Table Images", stats.get('table_images', 0))
                    with col4:
                        st.metric("Diagrams", stats.get('diagrams', 0))
            
            render_export_section()
            render_brd_content_editor_enhanced()
            
    else:
        st.info("Please upload and analyze a document first in the Document Analysis tab")
        
        # Show enhanced capabilities
        st.subheader("Enhanced BRD Generation Features")
        
        features = [
            {
                "icon": "🧮",
                "title": "Mathematical Formula Integration",
                "desc": "Automatically includes formula images with contextual references in relevant BRD sections"
            },
            {
                "icon": "📊", 
                "title": "Table Image Integration",
                "desc": "Captures table structures as images and references them in data requirement sections"
            },
            {
                "icon": "📈",
                "title": "Diagram Integration", 
                "desc": "Incorporates process diagrams and charts into workflow and process sections"
            },
            {
                "icon": "🔗",
                "title": "Smart Reference System",
                "desc": "Creates intelligent cross-references between text content and visual elements"
            }
        ]
        
        cols = st.columns(2)
        for i, feature in enumerate(features):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <h4>{feature['icon']} {feature['title']}</h4>
                    <p>{feature['desc']}</p>
                </div>
                """, unsafe_allow_html=True)

def render_brd_content_editor_enhanced():
    """Enhanced BRD content editor with visual integration support"""
    # Quality overview with visual integration metrics
    if st.session_state.quality_scores:
        avg_quality = sum(st.session_state.quality_scores.values()) / len(st.session_state.quality_scores)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"**Overall Quality Score: {avg_quality:.1f}%**")
        with col2:
            image_refs_count = sum(str(content).count('[IMAGE:') for content in st.session_state.brd_content.values() 
                                 if isinstance(content, str))
            st.info(f"**Image References: {image_refs_count}**")
        with col3:
            visual_checks = len([c for c in st.session_state.compliance_checks if c.check_type == 'visual_integration'])
            st.warning(f"**Visual Integration Checks: {visual_checks}**")
    
    # Enhanced section tabs for editing
    section_names = list(st.session_state.brd_content.keys())
    section_tabs = st.tabs([name.split('.')[0] + "." for name in section_names])
    
    for i, (section_name, content) in enumerate(st.session_state.brd_content.items()):
        with section_tabs[i]:
            render_section_editor_enhanced(section_name, content)

def render_section_editor_enhanced(section_name: str, content):
    """Enhanced section editor with image integration support"""
    # Section header with quality and visual integration indicators
    quality_score = st.session_state.quality_scores.get(section_name, 0)
    
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        st.subheader(section_name)
    with col2:
        st.metric("Quality", f"{quality_score:.0f}%")
    with col3:
        # Count image references in this section
        content_str = str(content)
        image_refs = len(re.findall(r'\[IMAGE:\s*[^\]]+\]', content_str))
        st.metric("Images", image_refs)
    with col4:
        status = "Good" if quality_score >= 80 else "Fair" if quality_score >= 60 else "Needs Work"
        st.write(f"Status: {status}")
    
    # Enhanced quality checks with visual integration
    section_checks = [c for c in st.session_state.compliance_checks if c.section == section_name]
    if section_checks:
        with st.expander("Quality & Visual Integration Insights"):
            for check in section_checks:
                if check.status == "PASS":
                    st.success(f"✅ **{check.check_type.replace('_', ' ').title()}:** {check.message}")
                elif check.status == "WARNING":
                    st.warning(f"⚠️ **{check.check_type.replace('_', ' ').title()}:** {check.message}")
                else:
                    st.error(f"❌ **{check.check_type.replace('_', ' ').title()}:** {check.message}")
    
    # Enhanced content editing with image preview
    if isinstance(content, dict):
        for subsection_name, subcontent in content.items():
            st.write(f"**{subsection_name}**")
            
            if isinstance(subcontent, pd.DataFrame):
                # Enhanced table editor with image reference support
                st.write("Interactive Table Editor with Image References:")
                
                # Show any image references in table data
                table_str = subcontent.to_string()
                table_image_refs = re.findall(r'\[IMAGE:\s*([^\]]+)\]', table_str)
                if table_image_refs:
                    with st.expander(f"Referenced Images in Table ({len(table_image_refs)} found)"):
                        cols = st.columns(min(3, len(table_image_refs)))
                        for idx, img_ref in enumerate(table_image_refs):
                            with cols[idx % 3]:
                                if img_ref.strip() in st.session_state.extracted_images:
                                    display_image_from_base64(
                                        st.session_state.extracted_images[img_ref.strip()], 
                                        caption=img_ref.strip(), 
                                        max_width=150
                                    )
                
                edited_df = st.data_editor(
                    subcontent,
                    use_container_width=True,
                    num_rows="dynamic",
                    key=f"table_{section_name}_{subsection_name}"
                )
                st.session_state.brd_content[section_name][subsection_name] = edited_df
            else:
                # Enhanced text editor with image preview
                text_content = str(subcontent)
                
                # Show referenced images
                image_refs = re.findall(r'\[IMAGE:\s*([^\]]+)\]', text_content)
                if image_refs:
                    with st.expander(f"Referenced Images ({len(image_refs)} found)"):
                        cols = st.columns(min(4, len(image_refs)))
                        for idx, img_ref in enumerate(image_refs):
                            with cols[idx % 4]:
                                img_key = img_ref.strip()
                                if img_key in st.session_state.extracted_images:
                                    display_image_from_base64(
                                        st.session_state.extracted_images[img_key], 
                                        caption=img_key, 
                                        max_width=120
                                    )
                                    st.caption(f"Key: {img_key}")
                                else:
                                    st.warning(f"Image not found: {img_key}")
                
                # Text editor with image integration helper
                col1, col2 = st.columns([3, 1])
                with col1:
                    edited_text = st.text_area(
                        f"Edit {subsection_name}",
                        value=text_content,
                        height=300,
                        key=f"text_{section_name}_{subsection_name}",
                        help="Use [IMAGE: key] format to reference images"
                    )
                    st.session_state.brd_content[section_name][subsection_name] = edited_text
                
                with col2:
                    st.write("**Available Images:**")
                    if st.session_state.extracted_images:
                        for img_key in list(st.session_state.extracted_images.keys())[:10]:
                            if st.button(f"Insert {img_key}", key=f"insert_{section_name}_{subsection_name}_{img_key}"):
                                # This would need JavaScript integration for actual insertion
                                st.info(f"Copy: [IMAGE: {img_key}]")
            
            st.markdown("---")
    else:
        # Single content editing with enhanced image support
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
            # Enhanced single text content editing
            text_content = str(content)
            
            # Show referenced images for single content
            image_refs = re.findall(r'\[IMAGE:\s*([^\]]+)\]', text_content)
            if image_refs:
                with st.expander(f"Referenced Images in Section ({len(image_refs)} found)"):
                    cols = st.columns(min(4, len(image_refs)))
                    for idx, img_ref in enumerate(image_refs):
                        with cols[idx % 4]:
                            img_key = img_ref.strip()
                            if img_key in st.session_state.extracted_images:
                                display_image_from_base64(
                                    st.session_state.extracted_images[img_key], 
                                    caption=img_key, 
                                    max_width=120
                                )
            
            edited_text = st.text_area(
                f"Edit {section_name}",
                value=text_content,
                height=350,
                key=f"text_{section_name}",
                help="Use [IMAGE: key] format to reference images"
            )
            st.session_state.brd_content[section_name] = edited_text

# Keep the existing functions for export and analytics tabs
def render_export_section():
    """Render export options section (unchanged)"""
    st.subheader("Export Options")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Export to Word", type="secondary"):
            try:
                word_doc = export_to_word_docx(st.session_state.brd_content)
                st.download_button(
                    label="Download Word Document",
                    data=word_doc.getvalue(),
                    file_name=f"Enhanced_BRD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
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
                    file_name=f"Enhanced_BRD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
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
                    file_name=f"Enhanced_BRD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
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
                    file_name=f"Enhanced_BRD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error exporting to JSON: {str(e)}")

def render_analytics_tab():
    """Render analytics dashboard tab (unchanged)"""
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
        
        # Enhanced workflow timeline with visual integration milestones
        st.markdown("---")
        render_workflow_timeline_enhanced()
        
    else:
        st.info("Generate a BRD first to view analytics")

def render_workflow_timeline_enhanced():
    """Enhanced workflow timeline with visual integration tracking"""
    st.subheader("Enhanced Workflow Timeline")
    
    timeline_steps = [
        {"step": "Document Upload", "status": "completed", "date": "Today", "visual_elements": "✓"},
        {"step": "AI Analysis & Image Extraction", "status": "completed", "date": "Today", "visual_elements": "✓"},
        {"step": "Enhanced BRD Generation", "status": "in_progress", "date": "Today", "visual_elements": "⏳"},
        {"step": "Visual Integration Review", "status": "pending", "date": "Tomorrow", "visual_elements": "⭕"},
        {"step": "Stakeholder Approval", "status": "pending", "date": "Next Week", "visual_elements": "⭕"},
        {"step": "Final Sign-off", "status": "pending", "date": "TBD", "visual_elements": "⭕"}
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
            <small style="color: {status_color};">Status: {step['status'].replace('_', ' ').title()}</small><br>
            <small>Visual Elements: {step['visual_elements']}</small>
        </div>
        """, unsafe_allow_html=True)
