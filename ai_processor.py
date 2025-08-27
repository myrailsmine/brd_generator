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
    formulas: List[str], 
    document_analysis: Dict[str, Any]
) -> str:
    """Generate BRD section with enhanced AI intelligence"""
    
    if llm is None:
        logger.warning("LLM not available, generating placeholder content")
        return generate_placeholder_content(section_name, section_config)
    
    # Context enhancement based on document analysis
    context_enhancement = f"""
    Document Analysis Context:
    - Document Type: {document_analysis.get('document_type', 'Unknown')}
    - Regulatory Frameworks: {', '.join(document_analysis.get('regulatory_framework', []))}
    - Complexity Level: {document_analysis.get('complexity_score', 0):.1f}/1.0
    - Key Entities: {', '.join(document_analysis.get('key_entities', [])[:5])}
    """
    
    media_context = ""
    if images:
        media_context += f"\nAvailable Images: {', '.join(list(images.keys())[:5])}\n"
    if formulas:
        media_context += f"\nExtracted Formulas:\n" + "\n".join(formulas[:5])
    
    # Enhanced prompts based on section type
    if section_config.get("type") == "table":
        columns = section_config.get("columns", [])
        quality_criteria = section_config.get("quality_criteria", [])
        
        user_prompt = f"""
        You are an expert business analyst creating a high-quality "{section_name}" section for a Business Requirements Document.
        
        {context_enhancement}
        
        Create a comprehensive table with exactly these columns: {' | '.join(columns)}
        
        Quality Criteria to Address: {', '.join(quality_criteria)}
        
        Based on this regulatory text: {document_text[:3000]}
        {media_context}
        
        Requirements:
        1. Generate 5-8 detailed, realistic rows
        2. Ensure each entry is specific and actionable
        3. Use proper business terminology
        4. Reference images using [IMAGE: image_key] format where relevant
        5. Include risk assessments and priorities where applicable
        
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
        You are an expert business analyst creating a high-quality "{section_name}" section for a Business Requirements Document.
        
        {context_enhancement}
        
        Section Purpose: {description}
        
        Required Elements to Include: {', '.join(required_elements)}
        Quality Criteria to Address: {', '.join(quality_criteria)}
        
        Based on this regulatory text: {document_text[:2500]}
        {media_context}
        
        Requirements:
        1. Provide comprehensive, professional content (minimum 300 words)
        2. Address all required elements explicitly
        3. Use clear, business-appropriate language
        4. Include specific examples and metrics where applicable
        5. Reference images using [IMAGE: image_key] format where relevant
        6. Structure with appropriate headings and bullet points
        """
    
    try:
        # Create message objects for ChatOpenAI
        system_message = SystemMessage(
            content="You are an expert business analyst with deep knowledge of regulatory compliance, business process optimization, and stakeholder management. Create professional, detailed, and actionable BRD content."
        )
        human_message = HumanMessage(content=user_prompt)
        
        # Get response from ChatOpenAI using invoke method
        messages = [system_message, human_message]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error generating {section_name}: {str(e)}")
        return generate_placeholder_content(section_name, section_config)

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

def generate_enhanced_brd(document_text: str, extracted_images: Dict[str, str], extracted_formulas: List[str], document_analysis: Dict[str, Any]) -> Dict[str, Any]:
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
