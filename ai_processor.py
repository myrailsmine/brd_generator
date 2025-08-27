"""
AI Processing Utilities
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

def parse_table_content(content: str, columns: List[str]) -> pd.DataFrame:
    """Parse AI-generated table content into DataFrame"""
    try:
        if not content or not columns:
            return pd.DataFrame(columns=columns)
            
        lines = content.strip().split('\n')
        data_rows = []
        
        for line in lines:
            if '|' in line and line.strip():
                # Clean and split the line
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
        
        # If no data rows found, create sample data
        if not data_rows:
            data_rows = [['Sample ' + str(i+1)] + [''] * (len(columns) - 1) for i in range(3)]
        
        df = pd.DataFrame(data_rows, columns=columns)
        return df
    
    except Exception as e:
        logger.error(f"Error parsing table content: {str(e)}")
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=columns)

def generate_intelligent_brd_section(
    llm: Any, 
    section_name: str, 
    section_config: Dict[str, Any], 
    document_text: str, 
    images: Dict[str, str], 
    formulas: List[Any], 
    document_analysis: Dict[str, Any]
) -> str:
    """Generate BRD section with enhanced AI intelligence for regulatory documents"""
    
    if llm is None:
        logger.warning("LLM not available, generating enhanced placeholder content")
        return generate_enhanced_placeholder_content(section_name, section_config, formulas, document_analysis)
    
    # Handle both old format (list of strings) and new format (list of dicts)
    formula_summary = ""
    if formulas:
        formula_types = set()
        high_priority_formulas = []
        formula_count = len(formulas)
        
        for formula in formulas:
            if isinstance(formula, dict):
                formula_types.add(formula.get('type', 'unknown'))
                if formula.get('confidence', 0) > 0.7:
                    high_priority_formulas.append(formula)
            else:
                # Handle old string format
                formula_types.add('basic_formula')
                high_priority_formulas.append({'text': str(formula), 'type': 'basic_formula'})
        
        formula_summary = f"""
        Mathematical Content Analysis:
        - Total formulas extracted: {formula_count}
        - Formula types: {', '.join(formula_types)}
        - Mathematical complexity: {document_analysis.get('mathematical_complexity', 'Unknown')}
        - Key formulas: {len(high_priority_formulas)} high-confidence formulas
        """
        
        if high_priority_formulas:
            formula_summary += "\nHigh-Priority Formulas:\n"
            for i, formula in enumerate(high_priority_formulas[:5]):  # Top 5 formulas
                if isinstance(formula, dict):
                    formula_text = formula.get('text', '')
                else:
                    formula_text = str(formula)
                formula_summary += f"- {formula_text[:100]}...\n"
    
    regulatory_context = f"""
    Regulatory Document Analysis:
    - Document Type: {document_analysis.get('document_type', 'Unknown')}
    - Regulatory Frameworks: {', '.join(document_analysis.get('regulatory_framework', []))}
    - Complexity Score: {document_analysis.get('complexity_score', 0):.2f}
    - Table Count: {document_analysis.get('table_count', 0)}
    - Regulatory Sections: {len(document_analysis.get('regulatory_sections', []))}
    """
    
    media_context = ""
    if images:
        media_context += f"\nAvailable Images: {', '.join(list(images.keys())[:5])}\n"
    
    # Enhanced prompts based on section type with regulatory focus
    if section_config.get("type") == "table":
        columns = section_config.get("columns", [])
        quality_criteria = section_config.get("quality_criteria", [])
        
        user_prompt = f"""
        You are an expert regulatory business analyst creating a comprehensive "{section_name}" section for a Business Requirements Document based on Basel Committee banking supervision standards.
        
        {regulatory_context}
        {formula_summary}
        
        Create a detailed regulatory-compliant table with exactly these columns: {' | '.join(columns)}
        
        Quality Criteria to Address: {', '.join(quality_criteria)}
        
        Document Context (first 3500 chars): {document_text[:3500]}
        {media_context}
        
        Requirements:
        1. Generate 6-10 detailed, Basel-compliant rows that reflect actual regulatory requirements
        2. Include specific references to regulatory sections (e.g., MAR21.x) where applicable
        3. Incorporate mathematical risk concepts and formulas where relevant
        4. Use precise regulatory terminology and banking industry standards
        5. Include risk weights, correlation parameters, and compliance thresholds as appropriate
        6. Reference extracted mathematical formulas and tables where relevant
        7. Ensure all entries are specific, actionable, and audit-ready
        
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
        You are an expert regulatory business analyst creating a comprehensive "{section_name}" section for a Business Requirements Document based on Basel Committee banking supervision standards.
        
        {regulatory_context}
        {formula_summary}
        
        Section Purpose: {description}
        
        Required Elements to Include: {', '.join(required_elements)}
        Quality Criteria to Address: {', '.join(quality_criteria)}
        
        Document Context (first 3500 chars): {document_text[:3500]}
        {media_context}
        
        Requirements:
        1. Provide comprehensive, regulatory-compliant content (minimum 400 words)
        2. Address all required elements explicitly with regulatory precision
        3. Use clear, banking industry-appropriate language and terminology
        4. Include specific regulatory references and compliance requirements
        5. Incorporate mathematical concepts and risk management principles
        6. Reference extracted formulas and calculations where relevant
        7. Structure with appropriate regulatory headings and detailed subsections
        8. Include compliance checkpoints and audit requirements
        9. Address both current requirements and implementation timelines
        10. Provide specific metrics, thresholds, and measurement criteria
        """
    
    try:
        # Create message objects for ChatOpenAI
        system_message = SystemMessage(
            content="You are a senior regulatory compliance expert and business analyst with 15+ years experience in Basel banking supervision, market risk frameworks, and regulatory implementation. You specialize in creating detailed, audit-ready Business Requirements Documents that meet the highest regulatory standards and include precise mathematical risk calculations."
        )
        human_message = HumanMessage(content=user_prompt)
        
        # Get response from ChatOpenAI using invoke method
        messages = [system_message, human_message]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error generating {section_name}: {str(e)}")
        return generate_enhanced_placeholder_content(section_name, section_config, formulas, document_analysis)

def generate_placeholder_content(section_name: str, section_config: Dict[str, Any]) -> str:
    """Generate placeholder content when AI is not available"""
    if section_config.get("type") == "table":
        columns = section_config.get("columns", ["Column1", "Column2"])
        # Create placeholder table content
        placeholder_rows = []
        for i in range(3):
            row = [f"Item {i+1}"] + [f"Value {i+1}" for _ in columns[1:]]
            placeholder_rows.append(" | ".join(row))
        return "\n".join(placeholder_rows)
    else:
        return f"Placeholder content for {section_name}. AI processing is not available. Please configure your AI model properly."

def generate_enhanced_placeholder_content(section_name: str, section_config: Dict[str, Any], formulas: List[Any], document_analysis: Dict[str, Any]) -> str:
    """Generate enhanced placeholder content with regulatory and mathematical context"""
    
    if section_config.get("type") == "table":
        columns = section_config.get("columns", ["Column1", "Column2"])
        # Create enhanced placeholder table content
        placeholder_rows = []
        
        # Add regulatory-specific sample data
        if "Risk" in section_name or "Assessment" in section_name:
            for i in range(5):
                row = [f"REG-RISK-{i+1:03d}"] + [f"Basel III Compliance Item {i+1}" if j == 1 else f"Value {i+1}" for j, _ in enumerate(columns[1:])]
                placeholder_rows.append(" | ".join(row))
        elif "Requirements" in section_name:
            for i in range(5):
                row = [f"REQ-{i+1:03d}"] + [f"Regulatory Requirement {i+1}" if j == 1 else f"Specification {i+1}" for j, _ in enumerate(columns[1:])]
                placeholder_rows.append(" | ".join(row))
        else:
            for i in range(3):
                row = [f"ITEM-{i+1:03d}"] + [f"Regulatory Item {i+1}" for _ in columns[1:]]
                placeholder_rows.append(" | ".join(row))
        
        return "\n".join(placeholder_rows)
    else:
        # Enhanced text placeholder with regulatory context
        complexity = document_analysis.get('mathematical_complexity', 'Unknown')
        frameworks = ', '.join(document_analysis.get('regulatory_framework', ['Basel III']))
        
        # Handle both old and new formula formats
        formula_count = 0
        if formulas:
            formula_count = len(formulas)
        
        return f"""PLACEHOLDER CONTENT FOR {section_name}
        
        This section addresses regulatory requirements under {frameworks} framework with {complexity.lower()} mathematical complexity.
        
        KEY REGULATORY CONSIDERATIONS:
        - Compliance Framework: {frameworks}
        - Mathematical Elements: {formula_count} formulas and calculations identified
        - Document Type: {document_analysis.get('document_type', 'Regulatory')}
        - Complexity Score: {document_analysis.get('complexity_score', 0):.2f}
        
        IMPLEMENTATION REQUIREMENTS:
        - Regulatory approval processes must be established
        - Mathematical models require validation and testing
        - Compliance monitoring and reporting systems needed
        - Risk management controls and governance frameworks
        
        MATHEMATICAL COMPONENTS:
        {"- Advanced risk calculations and correlation models required" if formula_count > 10 else "- Standard regulatory calculations apply"}
        {"- Complex mathematical validation processes needed" if complexity == "Very High" else "- Standard validation procedures sufficient"}
        
        AI processing is currently unavailable. Please configure your AI model properly or manually complete this section with appropriate regulatory content addressing the specific requirements of {section_name}.
        
        This placeholder incorporates analysis of {formula_count} mathematical formulas and regulatory structures from the source document to ensure comprehensive coverage of all regulatory requirements."""

def calculate_quality_score(section_name: str, content: Any, structure_config: Dict[str, Any]) -> Tuple[float, List[QualityCheck]]:
    """Calculate quality score and generate quality checks"""
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
        
        # Structure-specific checks
        if structure_config.get("type") == "table":
            if isinstance(content, pd.DataFrame) and not content.empty:
                score += 30
                checks.append(QualityCheck(section_name, "format", "PASS", "Proper table format", "info"))
                
                # Check for minimum rows
                if len(content) >= 3:
                    score += 20
                    checks.append(QualityCheck(section_name, "content_depth", "PASS", "Sufficient detail provided", "info"))
                else:
                    checks.append(QualityCheck(section_name, "content_depth", "WARNING", "Consider adding more detail", "warning"))
                    
                # Check for required columns
                required_cols = structure_config.get("columns", [])
                if all(col in content.columns for col in required_cols):
                    score += 20
                    checks.append(QualityCheck(section_name, "column_compliance", "PASS", "All required columns present", "info"))
            else:
                checks.append(QualityCheck(section_name, "format", "FAIL", "Should be in table format", "error"))
        
        elif structure_config.get("type") == "text":
            if isinstance(content, str) and len(content) > 100:
                score += 40
                checks.append(QualityCheck(section_name, "detail_level", "PASS", "Adequate detail provided", "info"))
                
                # Check for required elements
                required_elements = structure_config.get("required_elements", [])
                elements_found = 0
                for element in required_elements:
                    if element.replace("_", " ").lower() in content.lower():
                        elements_found += 1
                
                if required_elements:
                    element_score = (elements_found / len(required_elements)) * 30
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

def generate_enhanced_brd(document_text: str, extracted_images: Dict[str, str], extracted_formulas: List[Any], document_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate complete enhanced BRD with quality scoring"""
    logger.info("Starting enhanced BRD generation")
    
    brd_content = {}
    quality_scores = {}
    compliance_checks = []
    
    # Initialize LLM
    llm = init_enhanced_llm()
    
    total_sections = len(ENHANCED_BRD_STRUCTURE)
    section_count = 0
    
    for section_name, section_config in ENHANCED_BRD_STRUCTURE.items():
        try:
            if section_config.get("type") == "parent":
                brd_content[section_name] = {}
                for subsection_name, subsection_config in section_config.get("subsections", {}).items():
                    logger.info(f"Generating {subsection_name}")
                    
                    content = generate_intelligent_brd_section(
                        llm, subsection_name, subsection_config, document_text,
                        extracted_images, extracted_formulas, document_analysis
                    )
                    
                    if subsection_config.get("type") == "table":
                        df = parse_table_content(content, subsection_config.get("columns", []))
                        brd_content[section_name][subsection_name] = df
                    else:
                        brd_content[section_name][subsection_name] = content
                    
                    # Calculate quality score for subsection
                    score, checks = calculate_quality_score(subsection_name, content, subsection_config)
                    quality_scores[subsection_name] = score
                    compliance_checks.extend(checks)
            else:
                logger.info(f"Generating {section_name}")
                
                content = generate_intelligent_brd_section(
                    llm, section_name, section_config, document_text,
                    extracted_images, extracted_formulas, document_analysis
                )
                
                if section_config.get("type") == "table":
                    df = parse_table_content(content, section_config.get("columns", []))
                    brd_content[section_name] = df
                else:
                    brd_content[section_name] = content
                
                # Calculate quality score for section
                score, checks = calculate_quality_score(section_name, content, section_config)
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
    
    logger.info("Enhanced BRD generation completed")
    
    return {
        'brd_content': brd_content,
        'quality_scores': quality_scores,
        'compliance_checks': compliance_checks
    }
