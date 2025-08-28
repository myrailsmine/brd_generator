"""
Enhanced AI Processing Utilities
Updated to handle formula images and table images in BRD generation
"""

import streamlit as st
import pandas as pd
import re
from typing import List, Dict, Any, Tuple
from config.app_config import ENHANCED_BRD_STRUCTURE, QualityCheck
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. AI features will be limited.")

@st.cache_resource
def init_enhanced_llm():
    """Initialize ChatOpenAI with custom configuration"""
    if not LANGCHAIN_AVAILABLE:
        logger.error("LangChain not available for AI processing")
        return None
        
    try:
        config = st.session_state.get('llm_config', {})
        return ChatOpenAI(
            base_url=config.get('base_url', "http://localhost:8123/v1"),
            api_key=config.get('api_key', "dummy"),
            model=config.get('model', "llama3"),
            temperature=config.get('temperature', 0.3),
            max_tokens=config.get('max_tokens', 4000),
            streaming=False
        )
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return None

def categorize_extracted_elements(extracted_elements: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize extracted elements by type for better processing"""
    categories = {
        'text_formulas': [],
        'formula_images': [],
        'table_images': [],
        'diagrams': [],
        'embedded_images': [],
        'docx_tables': []
    }
    
    for element in extracted_elements:
        element_type = element.get('type', '')
        
        if 'formula' in element_type and 'image_data' in element:
            categories['formula_images'].append(element)
        elif 'formula' in element_type:
            categories['text_formulas'].append(element)
        elif 'table_image' in element_type:
            categories['table_images'].append(element)
        elif 'diagram' in element_type:
            categories['diagrams'].append(element)
        elif 'embedded_image' in element_type:
            categories['embedded_images'].append(element)
        elif 'docx_table' in element_type:
            categories['docx_tables'].append(element)
    
    return categories

def create_image_references_for_brd(images: Dict[str, str], element_categories: Dict[str, List[Dict[str, Any]]]) -> str:
    """Create image reference text for BRD sections"""
    image_refs = []
    
    # Add formula images
    for formula_img in element_categories.get('formula_images', []):
        key = formula_img.get('key')
        formula_text = formula_img.get('formula_text', 'Mathematical Formula')
        if key in images:
            image_refs.append(f"[IMAGE: {key}] - Mathematical Formula: {formula_text[:100]}...")
    
    # Add table images
    for table_img in element_categories.get('table_images', []):
        key = table_img.get('key')
        lines = table_img.get('lines', 0)
        if key in images:
            image_refs.append(f"[IMAGE: {key}] - Table Structure ({lines} rows detected)")
    
    # Add diagrams
    for diagram in element_categories.get('diagrams', []):
        key = diagram.get('key')
        elements_count = diagram.get('elements_count', 0)
        if key in images:
            image_refs.append(f"[IMAGE: {key}] - Diagram/Chart ({elements_count} visual elements)")
    
    # Add embedded images
    for emb_img in element_categories.get('embedded_images', []):
        key = emb_img.get('key')
        img_format = emb_img.get('format', 'image')
        if key in images:
            image_refs.append(f"[IMAGE: {key}] - Embedded Image ({img_format} format)")
    
    return "\n".join(image_refs)

def generate_mathematical_content_summary(element_categories: Dict[str, List[Dict[str, Any]]]) -> str:
    """Generate enhanced summary of mathematical content"""
    text_formulas = element_categories.get('text_formulas', [])
    formula_images = element_categories.get('formula_images', [])
    
    summary_parts = []
    
    if text_formulas or formula_images:
        total_formulas = len(text_formulas) + len(formula_images)
        summary_parts.append(f"Mathematical Content Analysis:")
        summary_parts.append(f"- Total formulas detected: {total_formulas}")
        summary_parts.append(f"- Text-based formulas: {len(text_formulas)}")
        summary_parts.append(f"- Formula images: {len(formula_images)}")
        
        # Formula type distribution
        formula_types = set()
        for formula in text_formulas:
            formula_types.add(formula.get('type', 'unknown'))
        for formula in formula_images:
            formula_types.add(formula.get('formula_type', 'unknown'))
        
        if formula_types:
            summary_parts.append(f"- Formula types: {', '.join(formula_types)}")
        
        # High-confidence formulas
        high_conf_formulas = []
        for formula in text_formulas + formula_images:
            if formula.get('confidence', 0) > 0.7:
                formula_text = formula.get('text') or formula.get('formula_text', '')
                if formula_text:
                    high_conf_formulas.append(formula_text[:50])
        
        if high_conf_formulas:
            summary_parts.append("High-Confidence Mathematical Elements:")
            for i, formula in enumerate(high_conf_formulas[:5]):
                summary_parts.append(f"- {formula}...")
    
    return "\n".join(summary_parts)

def generate_table_content_summary(element_categories: Dict[str, List[Dict[str, Any]]]) -> str:
    """Generate summary of table content"""
    table_images = element_categories.get('table_images', [])
    docx_tables = element_categories.get('docx_tables', [])
    
    summary_parts = []
    
    if table_images or docx_tables:
        total_tables = len(table_images) + len(docx_tables)
        summary_parts.append(f"Table Structure Analysis:")
        summary_parts.append(f"- Total tables detected: {total_tables}")
        summary_parts.append(f"- Table images: {len(table_images)}")
        summary_parts.append(f"- DOCX tables: {len(docx_tables)}")
        
        # Table complexity analysis
        if table_images:
            avg_rows = sum(t.get('lines', 0) for t in table_images) / len(table_images)
            summary_parts.append(f"- Average rows per table: {avg_rows:.1f}")
        
        # Sample table content from DOCX
        if docx_tables:
            sample_table = docx_tables[0]
            rows = sample_table.get('rows', 0)
            cols = sample_table.get('cols', 0)
            summary_parts.append(f"- Sample table structure: {rows}x{cols}")
    
    return "\n".join(summary_parts)

def generate_intelligent_brd_section_enhanced(
    llm: Any, 
    section_name: str, 
    section_config: Dict[str, Any], 
    document_text: str, 
    images: Dict[str, str], 
    extracted_elements: List[Dict[str, Any]], 
    document_analysis: Dict[str, Any]
) -> str:
    """Enhanced BRD section generation with formula and table image integration"""
    
    if llm is None:
        logger.warning("LLM not available, generating enhanced placeholder content")
        return generate_enhanced_placeholder_content_with_images(section_name, section_config, extracted_elements, document_analysis, images)
    
    # Categorize extracted elements
    element_categories = categorize_extracted_elements(extracted_elements)
    
    # Generate content summaries
    mathematical_summary = generate_mathematical_content_summary(element_categories)
    table_summary = generate_table_content_summary(element_categories)
    
    # Create image references for this section
    image_references = create_image_references_for_brd(images, element_categories)
    
    # Enhanced regulatory context
    regulatory_context = f"""
    Enhanced Document Analysis:
    - Document Type: {document_analysis.get('document_type', 'Unknown')}
    - Regulatory Frameworks: {', '.join(document_analysis.get('regulatory_framework', []))}
    - Complexity Score: {document_analysis.get('complexity_score', 0):.2f}
    - Mathematical Complexity: {document_analysis.get('mathematical_complexity', 'Unknown')}
    - Total Extracted Elements: {len(extracted_elements)}
    - Images Available: {len(images)}
    """
    
    # Visual content context
    visual_context = f"""
    Visual Content Available:
    {mathematical_summary}
    
    {table_summary}
    
    Available Images and Diagrams: {len(images)} items
    - Formula Images: {len(element_categories.get('formula_images', []))}
    - Table Images: {len(element_categories.get('table_images', []))}
    - Diagrams: {len(element_categories.get('diagrams', []))}
    - Embedded Images: {len(element_categories.get('embedded_images', []))}
    """
    
    # Enhanced prompts based on section type with image integration
    if section_config.get("type") == "table":
        columns = section_config.get("columns", [])
        quality_criteria = section_config.get("quality_criteria", [])
        
        user_prompt = f"""
        You are an expert regulatory business analyst creating a comprehensive "{section_name}" section for a Business Requirements Document.
        
        {regulatory_context}
        {visual_content}
        
        Create a detailed regulatory-compliant table with exactly these columns: {' | '.join(columns)}
        
        Quality Criteria to Address: {', '.join(quality_criteria)}
        
        Document Context (first 3500 chars): {document_text[:3500]}
        
        IMPORTANT - Include Visual References:
        {image_references}
        
        Requirements:
        1. Generate 6-10 detailed, regulatory-compliant rows
        2. Reference extracted mathematical formulas and tables where relevant
        3. Include image references using [IMAGE: key] format for relevant visual content
        4. Incorporate mathematical concepts from extracted formulas
        5. Reference table structures found in the document
        6. Use precise regulatory terminology
        7. Include risk weights, parameters, and compliance thresholds
        8. Ensure all entries are specific, actionable, and audit-ready
        9. When referencing mathematical formulas, include the corresponding [IMAGE: formula_key] reference
        10. When discussing table structures, reference [IMAGE: table_key] for visual context
        
        Return in pipe-separated format:
        {' | '.join(columns)}
        Row1Value1 | Row1Value2 | Row1Value3...
        Row2Value1 | Row2Value2 | Row2Value3...
        """
    else:
        description = section_config.get("description", f"Generate content for {section_name}")
        required_elements = section_config.get("required_elements", [])
        quality_criteria = section_config.get("quality_criteria", [])
        
        user_prompt = f"""
        You are an expert regulatory business analyst creating a comprehensive "{section_name}" section for a Business Requirements Document.
        
        {regulatory_context}
        {visual_content}
        
        Section Purpose: {description}
        
        Required Elements to Include: {', '.join(required_elements)}
        Quality Criteria to Address: {', '.join(quality_criteria)}
        
        Document Context (first 3500 chars): {document_text[:3500]}
        
        IMPORTANT - Visual Content Integration:
        {image_references}
        
        Requirements:
        1. Provide comprehensive, regulatory-compliant content (minimum 400 words)
        2. Address all required elements explicitly with regulatory precision
        3. **MUST include relevant image references using [IMAGE: key] format**
        4. Reference mathematical formulas with both text description AND image reference
        5. Include table references where discussing data structures
        6. Reference diagrams when explaining complex processes or relationships
        7. Use clear, banking industry-appropriate language and terminology
        8. Include specific regulatory references and compliance requirements
        9. Structure with appropriate regulatory headings and detailed subsections
        10. Provide specific metrics, thresholds, and measurement criteria
        11. When discussing mathematical concepts, format as: "The risk calculation [IMAGE: formula_p1_f1] shows..."
        12. When referencing tables, format as: "As shown in the regulatory table [IMAGE: table_p2_t1]..."
        13. For diagrams, format as: "The process flow [IMAGE: diagram_p1_d1] illustrates..."
        
        CRITICAL: Always include at least 3-5 image references in the content to provide visual context and support.
        """
    
    try:
        # Create message objects for ChatOpenAI
        system_message = SystemMessage(
            content="You are a senior regulatory compliance expert and business analyst with 15+ years experience in Basel banking supervision, market risk frameworks, and regulatory implementation. You specialize in creating detailed, audit-ready Business Requirements Documents that integrate mathematical formulas, tables, and diagrams for comprehensive visual context. Always include relevant image references to support your analysis."
        )
        human_message = HumanMessage(content=user_prompt)
        
        # Get response from ChatOpenAI using invoke method
        messages = [system_message, human_message]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error generating {section_name}: {str(e)}")
        return generate_enhanced_placeholder_content_with_images(section_name, section_config, extracted_elements, document_analysis, images)

def generate_enhanced_placeholder_content_with_images(
    section_name: str, 
    section_config: Dict[str, Any], 
    extracted_elements: List[Dict[str, Any]], 
    document_analysis: Dict[str, Any],
    images: Dict[str, str]
) -> str:
    """Generate enhanced placeholder content with image references"""
    
    element_categories = categorize_extracted_elements(extracted_elements)
    
    if section_config.get("type") == "table":
        columns = section_config.get("columns", ["Column1", "Column2"])
        placeholder_rows = []
        
        # Add regulatory-specific sample data with image references
        if "Risk" in section_name or "Assessment" in section_name:
            for i in range(5):
                image_ref = ""
                if i < len(element_categories.get('formula_images', [])):
                    formula_key = element_categories['formula_images'][i].get('key', '')
                    image_ref = f"[IMAGE: {formula_key}]"
                
                row = [f"REG-RISK-{i+1:03d}"] + [f"Basel III Risk Item {i+1} {image_ref}" if j == 1 else f"Value {i+1}" for j, _ in enumerate(columns[1:])]
                placeholder_rows.append(" | ".join(row))
        elif "Requirements" in section_name:
            for i in range(5):
                image_ref = ""
                if i < len(element_categories.get('table_images', [])):
                    table_key = element_categories['table_images'][i].get('key', '')
                    image_ref = f"[IMAGE: {table_key}]"
                
                row = [f"REQ-{i+1:03d}"] + [f"Regulatory Requirement {i+1} {image_ref}" if j == 1 else f"Specification {i+1}" for j, _ in enumerate(columns[1:])]
                placeholder_rows.append(" | ".join(row))
        else:
            for i in range(3):
                row = [f"ITEM-{i+1:03d}"] + [f"Regulatory Item {i+1}" for _ in columns[1:]]
                placeholder_rows.append(" | ".join(row))
        
        return "\n".join(placeholder_rows)
    else:
        # Enhanced text placeholder with image integration
        complexity = document_analysis.get('mathematical_complexity', 'Unknown')
        frameworks = ', '.join(document_analysis.get('regulatory_framework', ['Basel III']))
        
        # Create sample image references
        sample_image_refs = []
        for category, elements in element_categories.items():
            if elements and len(sample_image_refs) < 3:
                for element in elements[:2]:  # Take up to 2 from each category
                    key = element.get('key', '')
                    if key:
                        if 'formula' in category:
                            sample_image_refs.append(f"Mathematical calculation [IMAGE: {key}] demonstrates...")
                        elif 'table' in category:
                            sample_image_refs.append(f"Regulatory table [IMAGE: {key}] shows...")
                        elif 'diagram' in category:
                            sample_image_refs.append(f"Process diagram [IMAGE: {key}] illustrates...")
        
        image_integration_text = "\n".join(sample_image_refs) if sample_image_refs else "No visual references available for integration."
        
        total_elements = len(extracted_elements)
        
        return f"""ENHANCED PLACEHOLDER CONTENT FOR {section_name}
        
        This section addresses regulatory requirements under {frameworks} framework with {complexity.lower()} mathematical complexity.
        
        KEY REGULATORY CONSIDERATIONS:
        - Compliance Framework: {frameworks}
        - Mathematical Elements: {total_elements} formulas and visual elements identified
        - Document Type: {document_analysis.get('document_type', 'Regulatory')}
        - Complexity Score: {document_analysis.get('complexity_score', 0):.2f}
        
        VISUAL CONTENT INTEGRATION:
        {image_integration_text}
        
        IMPLEMENTATION REQUIREMENTS:
        - Regulatory approval processes must be established with visual documentation
        - Mathematical models require validation as shown in extracted formula images
        - Compliance monitoring systems need to reference table structures found in documentation
        - Risk management controls must incorporate diagram-based process flows
        
        MATHEMATICAL AND VISUAL COMPONENTS:
        {"- Advanced risk calculations with visual formula references required" if len(element_categories.get('formula_images', [])) > 0 else "- Standard regulatory calculations apply"}
        {"- Complex table structures require detailed analysis" if len(element_categories.get('table_images', [])) > 0 else "- Standard tabular data processing sufficient"}
        {"- Process diagrams provide essential workflow context" if len(element_categories.get('diagrams', [])) > 0 else "- Standard process documentation adequate"}
        
        COMPLIANCE INTEGRATION:
        The regulatory framework requires comprehensive documentation that integrates mathematical formulas, tabular data, and process diagrams. This section must reference all relevant visual elements to ensure complete compliance coverage.
        
        AI processing is currently unavailable. Please configure your AI model properly or manually complete this section with appropriate regulatory content addressing the specific requirements of {section_name}, including integration of the {total_elements} visual elements extracted from the source document.
        """

def parse_table_content_enhanced(content: str, columns: List[str]) -> pd.DataFrame:
    """Enhanced table parsing with image reference preservation"""
    try:
        if not content or not columns:
            return pd.DataFrame(columns=columns)
            
        lines = content.strip().split('\n')
        data_rows = []
        
        for line in lines:
            if '|' in line and line.strip():
                # Clean and split the line, preserving image references
                row = [cell.strip() for cell in line.split('|')]
                # Remove empty cells at the beginning/end
                row = [cell for cell in row if cell]
                
                # Skip header separators like |---|---|
                if all(cell in ['', '---', '--', '-'] or set(cell) <= {'-', ' '} for cell in row):
                    continue
                
                # Skip the column header row if it matches our expected columns
                if len(row) == len(columns) and all(col.lower() in ' '.join(row).lower() for col in columns[:3]):
                    continue
                
                # Pad or trim to match column count
                if len(row) < len(columns):
                    row.extend([''] * (len(columns) - len(row)))
                elif len(row) > len(columns):
                    row = row[:len(columns)]
                
                if any(cell.strip() for cell in row):  # Only add non-empty rows
                    data_rows.append(row)
        
        # If no data rows found, create sample data with image references
        if not data_rows:
            data_rows = []
            for i in range(3):
                sample_row = [f'Sample {i+1}']
                for j in range(1, len(columns)):
                    if j == 1 and i == 0:  # Add sample image reference
                        sample_row.append('Sample content [IMAGE: sample_ref]')
                    else:
                        sample_row.append('')
                data_rows.append(sample_row)
        
        df = pd.DataFrame(data_rows, columns=columns)
        return df
    
    except Exception as e:
        logger.error(f"Error parsing table content: {str(e)}")
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=columns)

def calculate_quality_score_enhanced(section_name: str, content: Any, structure_config: Dict[str, Any], images: Dict[str, str]) -> Tuple[float, List[QualityCheck]]:
    """Enhanced quality scoring that considers image integration"""
    checks = []
    score = 0.0
    max_score = 100.0
    
    try:
        # Basic completeness check
        if content and str(content).strip():
            score += 30
            checks.append(QualityCheck(section_name, "completeness", "PASS", "Section has content", "info"))
        else:
            checks.append(QualityCheck(section_name, "completeness", "FAIL", "Section is empty", "error"))
        
        # Image integration check
        content_str = str(content)
        image_refs = re.findall(r'\[IMAGE:\s*([^\]]+)\]', content_str)
        if image_refs:
            valid_refs = [ref.strip() for ref in image_refs if ref.strip() in images]
            if valid_refs:
                score += 20
                checks.append(QualityCheck(section_name, "visual_integration", "PASS", f"Contains {len(valid_refs)} valid image references", "info"))
            else:
                checks.append(QualityCheck(section_name, "visual_integration", "WARNING", "Image references found but not valid", "warning"))
        
        # Structure-specific checks
        if structure_config.get("type") == "table":
            if isinstance(content, pd.DataFrame) and not content.empty:
                score += 30
                checks.append(QualityCheck(section_name, "format", "PASS", "Proper table format", "info"))
                
                # Check for minimum rows
                if len(content) >= 3:
                    score += 10
                    checks.append(QualityCheck(section_name, "content_depth", "PASS", "Sufficient detail provided", "info"))
                else:
                    checks.append(QualityCheck(section_name, "content_depth", "WARNING", "Consider adding more detail", "warning"))
                    
                # Check for required columns
                required_cols = structure_config.get("columns", [])
                if all(col in content.columns for col in required_cols):
                    score += 10
                    checks.append(QualityCheck(section_name, "column_compliance", "PASS", "All required columns present", "info"))
            else:
                checks.append(QualityCheck(section_name, "format", "FAIL", "Should be in table format", "error"))
        
        elif structure_config.get("type") == "text":
            if isinstance(content, str) and len(content) > 100:
                score += 30
                checks.append(QualityCheck(section_name, "detail_level", "PASS", "Adequate detail provided", "info"))
                
                # Check for required elements
                required_elements = structure_config.get("required_elements", [])
                elements_found = 0
                for element in required_elements:
                    if element.replace("_", " ").lower() in content.lower():
                        elements_found += 1
                
                if required_elements:
                    element_score = (elements_found / len(required_elements)) * 10
                    score += element_score
                    if elements_found == len(required_elements):
                        checks.append(QualityCheck(section_name, "required_elements", "PASS", "All required elements present", "info"))
                    else:
                        missing_count = len(required_elements) - elements_found
                        checks.append(QualityCheck(section_name, "required_elements", "WARNING", f"Missing {missing_count} required elements", "warning"))
            else:
                checks.append(QualityCheck(section_name, "detail_level", "WARNING", "Consider adding more detail", "warning"))
        
        return min(score, max_score), checks
    
    except Exception as e:
        logger.error(f"Error calculating quality score for {section_name}: {e}")
        checks.append(QualityCheck(section_name, "error", "FAIL", f"Error in quality calculation: {str(e)}", "error"))
        return 0.0, checks

def generate_enhanced_brd_with_images(
    document_text: str, 
    extracted_images: Dict[str, str], 
    extracted_elements: List[Dict[str, Any]], 
    document_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate complete enhanced BRD with integrated images and visual elements"""
    logger.info("Starting enhanced BRD generation with image integration")
    
    brd_content = {}
    quality_scores = {}
    compliance_checks = []
    
    # Initialize LLM
    llm = init_enhanced_llm()
    
    total_sections = len(ENHANCED_BRD_STRUCTURE)
    section_count = 0
    
    logger.info(f"Processing {len(extracted_elements)} extracted elements and {len(extracted_images)} images")
    
    for section_name, section_config in ENHANCED_BRD_STRUCTURE.items():
        try:
            if section_config.get("type") == "parent":
                brd_content[section_name] = {}
                for subsection_name, subsection_config in section_config.get("subsections", {}).items():
                    logger.info(f"Generating {subsection_name} with image integration")
                    
                    content = generate_intelligent_brd_section_enhanced(
                        llm, subsection_name, subsection_config, document_text,
                        extracted_images, extracted_elements, document_analysis
                    )
                    
                    if subsection_config.get("type") == "table":
                        df = parse_table_content_enhanced(content, subsection_config.get("columns", []))
                        brd_content[section_name][subsection_name] = df
                    else:
                        brd_content[section_name][subsection_name] = content
                    
                    # Calculate quality score for subsection
                    score, checks = calculate_quality_score_enhanced(subsection_name, content, subsection_config, extracted_images)
                    quality_scores[subsection_name] = score
                    compliance_checks.extend(checks)
            else:
                logger.info(f"Generating {section_name} with image integration")
                
                content = generate_intelligent_brd_section_enhanced(
                    llm, section_name, section_config, document_text,
                    extracted_images, extracted_elements, document_analysis
                )
                
                if section_config.get("type") == "table":
                    df = parse_table_content_enhanced(content, section_config.get("columns", []))
                    brd_content[section_name] = df
                else:
                    brd_content[section_name] = content
                
                # Calculate quality score for section
                score, checks = calculate_quality_score_enhanced(section_name, content, section_config, extracted_images)
                quality_scores[section_name] = score
                compliance_checks.extend(checks)
            
            section_count += 1
            logger.info(f"Completed {section_count}/{total_sections} sections")
            
        except Exception as e:
            logger.error(f"Error generating section {section_name}: {e}")
            # Add placeholder content for failed sections
            if section_config.get("type") == "parent":
                brd_content[section_name] = {"error": f"Error generating section: {str(e)}"}
            else:
                brd_content[section_name] = f"Error generating section: {str(e)}"
            
            quality_scores[section_name] = 0.0
            compliance_checks.append(QualityCheck(section_name, "generation_error", "FAIL", str(e), "error"))
    
    logger.info("Enhanced BRD generation with image integration completed")
    
    return {
        'brd_content': brd_content,
        'quality_scores': quality_scores,
        'compliance_checks': compliance_checks,
        'image_integration_stats': {
            'total_images': len(extracted_images),
            'formula_images': len([e for e in extracted_elements if 'formula_image' in e.get('type', '')]),
            'table_images': len([e for e in extracted_elements if 'table_image' in e.get('type', '')]),
            'diagrams': len([e for e in extracted_elements if 'diagram' in e.get('type', '')]),
            'embedded_images': len([e for e in extracted_elements if 'embedded_image' in e.get('type', '')])
        }
    }
