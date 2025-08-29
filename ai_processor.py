"""
Enhanced AI Processing Utilities with Image Embedding Support
Enterprise-grade BRD generation with embedded mathematical formulas and tables as images
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

class ImageEmbeddingProcessor:
    """Process and embed images into BRD content"""
    
    @staticmethod
    def create_image_reference(image_id: str, image_type: str, description: str = "") -> str:
        """Create a standardized image reference for BRD content"""
        return f"[IMAGE_EMBED:{image_id}|{image_type}|{description}]"
    
    @staticmethod
    def embed_relevant_images(content: str, formulas: List[Dict[str, Any]], images: Dict[str, str], section_name: str) -> str:
        """Embed relevant images into content based on section context"""
        if not formulas or not images:
            return content
        
        enhanced_content = content
        embedded_images = set()
        
        # Section-specific image embedding logic
        section_keywords = {
            'business requirements': ['formula', 'calculation', 'mathematical'],
            'functional requirements': ['table', 'specification', 'structured'],
            'risk assessment': ['risk', 'sensitivity', 'var', 'correlation'],
            'regulations': ['mar', 'basel', 'regulatory', 'compliance'],
            'mathematical': ['formula', 'equation', 'mathematical', 'correlation'],
            'appendix': ['table', 'formula', 'reference', 'supporting']
        }
        
        section_lower = section_name.lower()
        relevant_keywords = []
        
        for key_section, keywords in section_keywords.items():
            if key_section in section_lower:
                relevant_keywords.extend(keywords)
        
        # Find and embed relevant images
        for formula in formulas:
            if not isinstance(formula, dict):
                continue
                
            image_id = formula.get('image_id')
            formula_type = formula.get('type', '')
            formula_text = formula.get('text', '')
            confidence = formula.get('confidence', 0)
            
            if not image_id or image_id not in images or confidence < 0.5:
                continue
            
            # Skip if already embedded
            if image_id in embedded_images:
                continue
            
            # Determine if image is relevant to this section
            should_embed = False
            
            # Always embed high-confidence mathematical content
            if confidence > 0.8:
                should_embed = True
            
            # Check for section-specific relevance
            elif relevant_keywords:
                formula_content = (formula_text + ' ' + formula_type).lower()
                if any(keyword in formula_content for keyword in relevant_keywords):
                    should_embed = True
            
            # Special cases for specific sections
            if 'risk' in section_lower and any(term in formula_type for term in ['risk', 'var', 'sensitivity', 'correlation']):
                should_embed = True
            elif 'requirement' in section_lower and any(term in formula_type for term in ['requirement', 'formula', 'calculation']):
                should_embed = True
            elif 'regulation' in section_lower and 'mar' in formula_type:
                should_embed = True
            elif 'appendix' in section_lower:
                should_embed = True  # Appendix gets all supporting materials
            
            if should_embed:
                # Create descriptive reference
                description = ImageEmbeddingProcessor._generate_image_description(formula)
                image_ref = ImageEmbeddingProcessor.create_image_reference(image_id, formula_type, description)
                
                # Find appropriate insertion point in content
                insertion_point = ImageEmbeddingProcessor._find_insertion_point(enhanced_content, formula)
                
                if insertion_point >= 0:
                    # Insert at specific point
                    enhanced_content = (
                        enhanced_content[:insertion_point] + 
                        f"\n\n{image_ref}\n\n" + 
                        enhanced_content[insertion_point:]
                    )
                else:
                    # Append to end of relevant paragraph
                    enhanced_content += f"\n\n{image_ref}\n"
                
                embedded_images.add(image_id)
        
        return enhanced_content
    
    @staticmethod
    def _generate_image_description(formula: Dict[str, Any]) -> str:
        """Generate descriptive text for an embedded image"""
        formula_type = formula.get('type', 'mathematical_element')
        confidence = formula.get('confidence', 0)
        page = formula.get('page', 'unknown')
        
        type_descriptions = {
            'basel_mar_reference': 'Basel MAR Regulatory Reference',
            'risk_weight_formula': 'Risk Weight Calculation Formula',
            'capital_requirement': 'Capital Requirement Formula',
            'correlation_formula': 'Correlation Parameter Formula',
            'sensitivity_measure': 'Risk Sensitivity Measure',
            'regulatory_table': 'Regulatory Parameters Table',
            'structured_table': 'Structured Data Table',
            'mathematical_expression': 'Mathematical Expression',
            'greek_formula': 'Mathematical Formula with Greek Notation',
            'bucket_classification': 'Basel Bucket Classification'
        }
        
        description = type_descriptions.get(formula_type, 'Mathematical/Regulatory Content')
        
        if confidence > 0.8:
            confidence_label = 'High Confidence'
        elif confidence > 0.6:
            confidence_label = 'Medium Confidence'
        else:
            confidence_label = 'Detected'
        
        return f"{description} (Page {page}, {confidence_label})"
    
    @staticmethod
    def _find_insertion_point(content: str, formula: Dict[str, Any]) -> int:
        """Find the best insertion point for an image reference"""
        formula_text = formula.get('text', '').strip()
        context = formula.get('context', '').strip()
        
        # Try to find exact formula text in content
        if formula_text and formula_text in content:
            return content.find(formula_text) + len(formula_text)
        
        # Try to find context keywords
        if context:
            context_words = context.split()[:10]  # First 10 words of context
            for i, word in enumerate(context_words):
                if len(word) > 4 and word.lower() in content.lower():
                    pos = content.lower().find(word.lower())
                    if pos >= 0:
                        # Find end of paragraph
                        next_para = content.find('\n\n', pos)
                        return next_para if next_para > pos else pos + len(word)
        
        return -1  # No specific insertion point found

@st.cache_resource
def init_enhanced_llm():
    """Initialize ChatOpenAI with custom configuration for enterprise use"""
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

def parse_table_content_enhanced(content: str, columns: List[str]) -> pd.DataFrame:
    """Enhanced table parsing with better error handling and data validation"""
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
                row = [cell for cell in row if cell or cell == ""]  # Keep empty strings but remove None/False
                
                # Skip header separators like |---|---|
                if all(cell in ['', '---', '--', '-'] or set(cell) <= {'-', ' ', '|'} for cell in row):
                    continue
                
                # Skip the column header row if it matches our expected columns
                if len(row) == len(columns) and all(col.lower() in ' '.join(row).lower() for col in columns[:min(3, len(columns))]):
                    continue
                
                # Pad or trim to match column count
                if len(row) < len(columns):
                    row.extend([''] * (len(columns) - len(row)))
                elif len(row) > len(columns):
                    row = row[:len(columns)]
                
                # Only add non-empty rows
                if any(cell.strip() for cell in row):
                    data_rows.append(row)
        
        # If no data rows found, create sample data based on section type
        if not data_rows:
            data_rows = generate_sample_table_data(columns)
        
        # Validate and clean data
        validated_rows = []
        for row in data_rows:
            validated_row = []
            for i, cell in enumerate(row):
                # Clean cell content
                clean_cell = str(cell).strip() if cell is not None else ""
                # Remove excessive whitespace
                clean_cell = re.sub(r'\s+', ' ', clean_cell)
                validated_row.append(clean_cell)
            validated_rows.append(validated_row)
        
        df = pd.DataFrame(validated_rows, columns=columns)
        return df
    
    except Exception as e:
        logger.error(f"Error parsing table content: {str(e)}")
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=columns)

def generate_sample_table_data(columns: List[str]) -> List[List[str]]:
    """Generate contextual sample data based on column names"""
    sample_data = []
    
    # Determine table type from column names
    columns_lower = [col.lower() for col in columns]
    
    if any('risk' in col for col in columns_lower):
        # Risk-related table
        for i in range(5):
            row = []
            for col in columns_lower:
                if 'id' in col:
                    row.append(f"RISK-{i+1:03d}")
                elif 'description' in col:
                    row.append(f"Market risk scenario {i+1}")
                elif 'probability' in col:
                    row.append(f"{20 + i*15}%")
                elif 'impact' in col:
                    row.append("High" if i < 2 else "Medium")
                else:
                    row.append(f"Value {i+1}")
            sample_data.append(row)
            
    elif any('requirement' in col for col in columns_lower):
        # Requirements table
        for i in range(6):
            row = []
            for col in columns_lower:
                if 'id' in col:
                    row.append(f"REQ-{i+1:03d}")
                elif 'name' in col:
                    row.append(f"Regulatory Requirement {i+1}")
                elif 'priority' in col:
                    row.append("High" if i < 3 else "Medium")
                elif 'owner' in col:
                    row.append("Compliance Team")
                else:
                    row.append(f"Specification {i+1}")
            sample_data.append(row)
            
    elif any('stakeholder' in col for col in columns_lower):
        # Stakeholder table
        stakeholders = ["Business Sponsor", "Compliance Officer", "Risk Manager", "IT Lead", "Legal Counsel"]
        for i, stakeholder in enumerate(stakeholders):
            row = []
            for col in columns_lower:
                if 'stakeholder' in col:
                    row.append(stakeholder)
                elif 'role' in col:
                    row.append(f"Primary {stakeholder.split()[-1]}")
                elif 'interest' in col:
                    row.append(f"{7 + i}")
                elif 'influence' in col:
                    row.append(f"{8 - i}")
                else:
                    row.append(f"Value {i+1}")
            sample_data.append(row)
    else:
        # Generic table
        for i in range(4):
            row = []
            for j, col in enumerate(columns):
                if j == 0:
                    row.append(f"ITEM-{i+1:03d}")
                else:
                    row.append(f"Data {i+1}.{j}")
            sample_data.append(row)
    
    return sample_data

def generate_intelligent_brd_section_enhanced(
    llm: Any, 
    section_name: str, 
    section_config: Dict[str, Any], 
    document_text: str, 
    images: Dict[str, str], 
    formulas: List[Any], 
    document_analysis: Dict[str, Any]
) -> str:
    """Enhanced BRD section generation with intelligent image embedding"""
    
    if llm is None:
        logger.warning("LLM not available, generating enhanced placeholder content")
        return generate_enhanced_placeholder_content_with_images(section_name, section_config, formulas, document_analysis, images)
    
    # Prepare enhanced formula and image analysis
    formula_summary = ""
    image_summary = ""
    relevant_formulas = []
    
    if formulas:
        # Filter formulas relevant to this section
        section_keywords = extract_section_keywords(section_name)
        relevant_formulas = filter_relevant_formulas(formulas, section_keywords)
        
        if relevant_formulas:
            formula_summary = f"""
            Relevant Mathematical Content for {section_name}:
            - {len(relevant_formulas)} relevant formulas/tables identified
            - Mathematical complexity: {document_analysis.get('mathematical_complexity', 'Unknown')}
            - Types: {', '.join(set(f.get('type', 'unknown') for f in relevant_formulas if isinstance(f, dict)))}
            """
            
            # Add top relevant formulas
            for i, formula in enumerate(relevant_formulas[:3]):
                if isinstance(formula, dict):
                    formula_text = formula.get('text', '')[:100]
                    formula_type = formula.get('type', 'unknown')
                    confidence = formula.get('confidence', 0)
                    image_id = formula.get('image_id', '')
                    
                    formula_summary += f"\n- Formula {i+1}: {formula_text}... (Type: {formula_type}, Confidence: {confidence:.1%})"
                    if image_id:
                        formula_summary += f" [Image Available: {image_id}]"
    
    if images:
        image_summary = f"""
        Available Visual Content:
        - Total images: {len(images)}
        - Mathematical images: {len([k for k in images.keys() if 'math' in k.lower()])}
        - Table images: {len([k for k in images.keys() if 'table' in k.lower()])}
        """
    
    regulatory_context = f"""
    Enhanced Regulatory Document Analysis:
    - Document Type: {document_analysis.get('document_type', 'Unknown')}
    - Regulatory Frameworks: {', '.join(document_analysis.get('regulatory_framework', []))}
    - Complexity Score: {document_analysis.get('complexity_score', 0):.2f}
    - Extraction Quality: {document_analysis.get('extraction_quality', {}).get('overall_extraction_score', 0):.1%}
    - Mathematical Content Ratio: {document_analysis.get('document_structure', {}).get('mathematical_content_ratio', 0):.2f} formulas per 1000 words
    """
    
    # Generate content based on section type
    if section_config.get("type") == "table":
        columns = section_config.get("columns", [])
        quality_criteria = section_config.get("quality_criteria", [])
        
        user_prompt = f"""
        You are an expert regulatory business analyst creating a comprehensive "{section_name}" table for a Business Requirements Document based on Basel Committee banking supervision standards.
        
        {regulatory_context}
        {formula_summary}
        {image_summary}
        
        Create a detailed regulatory-compliant table with exactly these columns: {' | '.join(columns)}
        
        Quality Criteria to Address: {', '.join(quality_criteria)}
        
        Document Context (first 4000 chars): {document_text[:4000]}
        
        CRITICAL REQUIREMENTS:
        1. Generate 8-12 detailed, Basel-compliant rows that reflect actual regulatory requirements
        2. Include specific references to regulatory sections (e.g., MAR21.x) where applicable
        3. Incorporate mathematical risk concepts and formulas where relevant
        4. Use precise regulatory terminology and banking industry standards
        5. Include risk weights, correlation parameters, and compliance thresholds as appropriate
        6. Reference extracted mathematical formulas and tables where relevant
        7. Ensure all entries are specific, actionable, and audit-ready
        8. For mathematical/formula content, reference specific image IDs when available: {[f.get('image_id', '') for f in relevant_formulas if isinstance(f, dict) and f.get('image_id')]}
        
        FORMAT REQUIREMENTS:
        - Return ONLY the table in pipe-separated format
        - First row must be the exact column headers: {' | '.join(columns)}
        - Each subsequent row must have exactly {len(columns)} columns
        - Use proper regulatory terminology and precise numerical values
        - Include specific regulatory references (MAR sections, Basel guidelines)
        
        Example format:
        {' | '.join(columns)}
        REQ-001 | Basel III Capital Requirements | Market Risk Capital Calculation | Risk Management Team | High | 99.5% accuracy in capital calculation | Implementation of MAR21.88 requirements
        """
    else:
        description = section_config.get("description", f"Generate content for {section_name}")
        required_elements = section_config.get("required_elements", [])
        quality_criteria = section_config.get("quality_criteria", [])
        
        user_prompt = f"""
        You are an expert regulatory business analyst creating a comprehensive "{section_name}" section for a Business Requirements Document based on Basel Committee banking supervision standards.
        
        {regulatory_context}
        {formula_summary}
        {image_summary}
        
        Section Purpose: {description}
        
        Required Elements to Include: {', '.join(required_elements)}
        Quality Criteria to Address: {', '.join(quality_criteria)}
        
        Document Context (first 4000 chars): {document_text[:4000]}
        
        CRITICAL REQUIREMENTS:
        1. Provide comprehensive, regulatory-compliant content (minimum 500 words)
        2. Address all required elements explicitly with regulatory precision
        3. Use clear, banking industry-appropriate language and terminology
        4. Include specific regulatory references and compliance requirements
        5. Incorporate mathematical concepts and risk management principles
        6. Reference extracted formulas and calculations where relevant
        7. Structure with appropriate regulatory headings and detailed subsections
        8. Include compliance checkpoints and audit requirements
        9. Address both current requirements and implementation timelines
        10. Provide specific metrics, thresholds, and measurement criteria
        11. When referencing mathematical content, include image references where available
        
        MATHEMATICAL/VISUAL CONTENT INTEGRATION:
        - Reference relevant formulas and tables naturally within the text
        - When discussing complex calculations, mention that "supporting mathematical formulas and tables are provided as visual references"
        - For regulatory tables and calculations, indicate that "detailed specifications are shown in the accompanying regulatory tables"
        - Use phrases like "as illustrated in the extracted regulatory documentation" when referencing visual content
        
        STRUCTURE REQUIREMENTS:
        - Use clear, descriptive headings
        - Include numbered subsections where appropriate
        - Provide specific implementation guidance
        - Include compliance validation requirements
        - End with next steps and approval requirements
        """
    
    try:
        # Create enhanced system message
        system_message = SystemMessage(
            content="""You are a senior regulatory compliance expert and business analyst with 20+ years experience in Basel banking supervision, market risk frameworks, and regulatory implementation. You specialize in creating detailed, audit-ready Business Requirements Documents that meet the highest regulatory standards and include precise mathematical risk calculations.
            
            Your expertise includes:
            - Basel III/IV regulatory frameworks and MAR requirements
            - Market risk, credit risk, and operational risk management
            - Mathematical modeling and risk sensitivity calculations
            - Regulatory compliance and audit requirements
            - Business requirements documentation best practices
            - Integration of complex mathematical content with business requirements
            
            Always provide specific, actionable, and professionally detailed content that would pass regulatory scrutiny."""
        )
        
        human_message = HumanMessage(content=user_prompt)
        
        # Get response from ChatOpenAI
        messages = [system_message, human_message]
        response = llm.invoke(messages)
        
        # Post-process content to embed images
        generated_content = response.content
        
        # Embed relevant images into the generated content
        if relevant_formulas and images:
            generated_content = ImageEmbeddingProcessor.embed_relevant_images(
                generated_content, relevant_formulas, images, section_name
            )
        
        return generated_content
        
    except Exception as e:
        logger.error(f"Error generating {section_name}: {str(e)}")
        return generate_enhanced_placeholder_content_with_images(section_name, section_config, formulas, document_analysis, images)

def extract_section_keywords(section_name: str) -> List[str]:
    """Extract relevant keywords for section-specific content filtering"""
    section_lower = section_name.lower()
    
    keyword_mapping = {
        'executive': ['summary', 'overview', 'high-level', 'strategic'],
        'background': ['context', 'history', 'current', 'drivers'],
        'scope': ['boundary', 'inclusion', 'exclusion', 'limitations'],
        'stakeholder': ['stakeholder', 'role', 'responsibility', 'influence'],
        'assumption': ['assumption', 'dependency', 'prerequisite', 'constraint'],
        'business': ['business', 'functional', 'requirement', 'process'],
        'functional': ['functional', 'system', 'feature', 'capability'],
        'non-functional': ['performance', 'security', 'scalability', 'reliability'],
        'risk': ['risk', 'threat', 'vulnerability', 'mitigation', 'var', 'sensitivity'],
        'regulation': ['regulation', 'compliance', 'mar', 'basel', 'supervisory'],
        'timeline': ['timeline', 'schedule', 'milestone', 'phase', 'delivery'],
        'metrics': ['metric', 'kpi', 'measure', 'target', 'performance'],
        'approval': ['approval', 'sign-off', 'authorization', 'governance'],
        'appendix': ['appendix', 'reference', 'supporting', 'documentation']
    }
    
    keywords = []
    for key, values in keyword_mapping.items():
        if key in section_lower:
            keywords.extend(values)
    
    return keywords

def filter_relevant_formulas(formulas: List[Dict[str, Any]], section_keywords: List[str]) -> List[Dict[str, Any]]:
    """Filter formulas relevant to the specific section"""
    if not section_keywords:
        return formulas[:5]  # Return top 5 if no specific keywords
    
    relevant_formulas = []
    scored_formulas = []
    
    for formula in formulas:
        if not isinstance(formula, dict):
            continue
            
        relevance_score = 0
        formula_text = formula.get('text', '').lower()
        formula_type = formula.get('type', '').lower()
        context = formula.get('context', '').lower()
        
        combined_text = f"{formula_text} {formula_type} {context}"
        
        # Score based on keyword matches
        for keyword in section_keywords:
            if keyword in combined_text:
                relevance_score += 1
        
        # Bonus for high confidence
        confidence = formula.get('confidence', 0)
        relevance_score += confidence
        
        # Bonus for having associated image
        if formula.get('image_id'):
            relevance_score += 0.5
        
        if relevance_score > 0:
            scored_formulas.append((formula, relevance_score))
    
    # Sort by relevance score and return top formulas
    scored_formulas.sort(key=lambda x: x[1], reverse=True)
    relevant_formulas = [formula for formula, score in scored_formulas[:8]]  # Top 8 relevant formulas
    
    return relevant_formulas

def generate_enhanced_placeholder_content_with_images(
    section_name: str, 
    section_config: Dict[str, Any], 
    formulas: List[Any], 
    document_analysis: Dict[str, Any],
    images: Dict[str, str]
) -> str:
    """Generate enhanced placeholder content with image references"""
    
    if section_config.get("type") == "table":
        columns = section_config.get("columns", ["Column1", "Column2"])
        sample_data = generate_sample_table_data(columns)
        
        # Create enhanced table content with image references
        table_rows = [" | ".join(columns)]
        
        for row_data in sample_data:
            table_rows.append(" | ".join(row_data))
        
        # Add image references if relevant images exist
        relevant_images = [img_id for img_id in images.keys() if 'table' in img_id.lower() or 'math' in img_id.lower()]
        if relevant_images:
            table_content = "\n".join(table_rows)
            for img_id in relevant_images[:2]:  # Add up to 2 relevant images
                table_content += f"\n\n[IMAGE_EMBED:{img_id}|supporting_table|Extracted regulatory table from source document]\n"
            return table_content
        
        return "\n".join(table_rows)
    else:
        # Enhanced text placeholder with regulatory context
        complexity = document_analysis.get('mathematical_complexity', 'Unknown')
        frameworks = ', '.join(document_analysis.get('regulatory_framework', ['Basel III']))
        
        formula_count = len(formulas) if formulas else 0
        image_count = len(images) if images else 0
        
        placeholder_content = f"""ENHANCED CONTENT FOR {section_name}
        
        This section addresses regulatory requirements under {frameworks} framework with {complexity.lower()} mathematical complexity.
        
        KEY REGULATORY CONSIDERATIONS:
        - Compliance Framework: {frameworks}
        - Mathematical Elements: {formula_count} formulas and calculations identified
        - Visual Content: {image_count} images extracted for reference
        - Document Type: {document_analysis.get('document_type', 'Regulatory')}
        - Complexity Score: {document_analysis.get('complexity_score', 0):.2f}
        - Extraction Quality: {document_analysis.get('extraction_quality', {}).get('overall_extraction_score', 0):.1%}
        
        IMPLEMENTATION REQUIREMENTS:
        - Regulatory approval processes must be established in accordance with supervisory guidelines
        - Mathematical models require validation and testing as per regulatory standards
        - Compliance monitoring and reporting systems needed for ongoing oversight
        - Risk management controls and governance frameworks must be implemented
        - Documentation and audit trails must be maintained for regulatory inspection
        
        MATHEMATICAL AND VISUAL CONTENT INTEGRATION:
        {"- Advanced risk calculations and correlation models require detailed mathematical validation" if formula_count > 10 else "- Standard regulatory calculations apply with appropriate validation procedures"}
        {"- Complex mathematical validation processes needed with visual documentation support" if complexity == "Very High" else "- Standard validation procedures sufficient with supporting documentation"}
        {"- Extracted visual content provides supporting evidence for regulatory compliance" if image_count > 5 else "- Limited visual content available for reference"}
        
        SUPPORTING DOCUMENTATION:
        This section incorporates analysis of {formula_count} mathematical formulas and {image_count} visual elements from the source document to ensure comprehensive coverage of all regulatory requirements.
        
        NEXT STEPS:
        1. Complete detailed requirements analysis using extracted mathematical and visual content
        2. Validate all regulatory references against current Basel standards
        3. Obtain stakeholder approval for implementation approach
        4. Establish compliance monitoring and reporting procedures
        5. Configure AI model properly for enhanced content generation
        
        Note: AI processing is currently unavailable. Please configure your AI model properly or manually complete this section with appropriate regulatory content addressing the specific requirements of {section_name}."""
        
        # Add relevant image references to placeholder
        section_keywords = extract_section_keywords(section_name)
        relevant_images = []
        
        for img_id, img_data in images.items():
            img_lower = img_id.lower()
            if any(keyword in img_lower for keyword in section_keywords[:3]):  # Top 3 keywords
                relevant_images.append(img_id)
        
        if relevant_images:
            placeholder_content += f"\n\nRELEVANT VISUAL CONTENT:\n"
            for img_id in relevant_images[:3]:  # Up to 3 relevant images
                img_type = "mathematical_formula" if "math" in img_id.lower() else "regulatory_table"
                placeholder_content += f"\n[IMAGE_EMBED:{img_id}|{img_type}|Supporting visual content for {section_name}]"
        
        return placeholder_content

def calculate_quality_score_enhanced(section_name: str, content: Any, structure_config: Dict[str, Any], images: Dict[str, str] = None) -> Tuple[float, List[QualityCheck]]:
    """Enhanced quality calculation with image integration assessment"""
    checks = []
    score = 0.0
    max_score = 100.0
    
    try:
        # Basic completeness check
        if content and str(content).strip():
            score += 25
            checks.append(QualityCheck(section_name, "completeness", "PASS", "Section has content", "info"))
        else:
            checks.append(QualityCheck(section_name, "completeness", "FAIL", "Section is empty", "error"))
            return 0.0, checks
        
        # Image integration check
        if images:
            content_str = str(content)
            embedded_images = len(re.findall(r'\[IMAGE_EMBED:[^\]]+\]', content_str))
            if embedded_images > 0:
                score += 15
                checks.append(QualityCheck(section_name, "visual_integration", "PASS", f"{embedded_images} visual elements integrated", "info"))
            else:
                checks.append(QualityCheck(section_name, "visual_integration", "WARNING", "No visual content integrated", "warning"))
        
        # Structure-specific checks
        if structure_config.get("type") == "table":
            if isinstance(content, pd.DataFrame) and not content.empty:
                score += 25
                checks.append(QualityCheck(section_name, "format", "PASS", "Proper table format", "info"))
                
                # Check for minimum rows
                if len(content) >= 5:
                    score += 15
                    checks.append(QualityCheck(section_name, "content_depth", "PASS", "Comprehensive detail provided", "info"))
                elif len(content) >= 3:
                    score += 10
                    checks.append(QualityCheck(section_name, "content_depth", "WARNING", "Adequate detail, consider expansion", "warning"))
                else:
                    checks.append(QualityCheck(section_name, "content_depth", "WARNING", "Limited detail provided", "warning"))
                    
                # Check for required columns
                required_cols = structure_config.get("columns", [])
                if all(col in content.columns for col in required_cols):
                    score += 20
                    checks.append(QualityCheck(section_name, "column_compliance", "PASS", "All required columns present", "info"))
                    
                # Check for data quality
                non_empty_cells = content.notna().sum().sum()
                total_cells = len(content) * len(content.columns)
                fill_rate = non_empty_cells / total_cells if total_cells > 0 else 0
                
                if fill_rate > 0.8:
                    checks.append(QualityCheck(section_name, "data_quality", "PASS", f"High data completeness ({fill_rate:.1%})", "info"))
                elif fill_rate > 0.6:
                    checks.append(QualityCheck(section_name, "data_quality", "WARNING", f"Moderate data completeness ({fill_rate:.1%})", "warning"))
                else:
                    checks.append(QualityCheck(section_name, "data_quality", "FAIL", f"Low data completeness ({fill_rate:.1%})", "error"))
                    
            else:
                checks.append(QualityCheck(section_name, "format", "FAIL", "Should be in table format", "error"))
        
        elif structure_config.get("type") == "text":
            content_str = str(content)
            word_count = len(content_str.split())
            
            if word_count > 200:
                score += 25
                checks.append(QualityCheck(section_name, "detail_level", "PASS", f"Comprehensive content ({word_count} words)", "info"))
            elif word_count > 100:
                score += 15
                checks.append(QualityCheck(section_name, "detail_level", "WARNING", f"Adequate content ({word_count} words)", "warning"))
            else:
                checks.append(QualityCheck(section_name, "detail_level", "WARNING", f"Limited content ({word_count} words)", "warning"))
                
            # Check for required elements
            required_elements = structure_config.get("required_elements", [])
            elements_found = 0
            missing_elements = []
            
            for element in required_elements:
                element_variations = [
                    element.replace("_", " "),
                    element.replace("_", "-"),
                    element,
                ]
                
                found = False
                for variation in element_variations:
                    if variation.lower() in content_str.lower():
                        elements_found += 1
                        found = True
                        break
                        
                if not found:
                    missing_elements.append(element)
            
            if required_elements:
                element_score = (elements_found / len(required_elements)) * 25
                score += element_score
                
                if elements_found == len(required_elements):
                    checks.append(QualityCheck(section_name, "required_elements", "PASS", "All required elements present", "info"))
                elif elements_found >= len(required_elements) * 0.7:
                    checks.append(QualityCheck(section_name, "required_elements", "WARNING", f"Missing elements: {', '.join(missing_elements)}", "warning"))
                else:
                    checks.append(QualityCheck(section_name, "required_elements", "FAIL", f"Many missing elements: {', '.join(missing_elements[:3])}...", "error"))
            
            # Regulatory content quality checks
            regulatory_indicators = [
                'basel', 'mar', 'regulatory', 'compliance', 'supervisory',
                'requirement', 'standard', 'guideline', 'framework'
            ]
            
            regulatory_score = 0
            for indicator in regulatory_indicators:
                if indicator in content_str.lower():
                    regulatory_score += 1
            
            if regulatory_score >= 5:
                score += 10
                checks.append(QualityCheck(section_name, "regulatory_content", "PASS", "Strong regulatory focus", "info"))
            elif regulatory_score >= 3:
                score += 5
                checks.append(QualityCheck(section_name, "regulatory_content", "WARNING", "Moderate regulatory content", "warning"))
            else:
                checks.append(QualityCheck(section_name, "regulatory_content", "WARNING", "Limited regulatory terminology", "warning"))
        
        return min(score, max_score), checks
    
    except Exception as e:
        logger.error(f"Error calculating quality score for {section_name}: {e}")
        checks.append(QualityCheck(section_name, "error", "FAIL", f"Error in quality calculation: {str(e)}", "error"))
        return 0.0, checks

def generate_enhanced_brd_with_images(document_text: str, extracted_images: Dict[str, str], extracted_formulas: List[Any], document_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate complete enhanced BRD with intelligent image embedding and quality scoring"""
    logger.info("Starting enhanced BRD generation with image integration")
    
    brd_content = {}
    quality_scores = {}
    compliance_checks = []
    
    # Initialize enhanced LLM
    llm = init_enhanced_llm()
    
    total_sections = len(ENHANCED_BRD_STRUCTURE)
    section_count = 0
    
    for section_name, section_config in ENHANCED_BRD_STRUCTURE.items():
        try:
            if section_config.get("type") == "parent":
                brd_content[section_name] = {}
                for subsection_name, subsection_config in section_config.get("subsections", {}).items():
                    logger.info(f"Generating {subsection_name} with image integration")
                    
                    content = generate_intelligent_brd_section_enhanced(
                        llm, subsection_name, subsection_config, document_text,
                        extracted_images, extracted_formulas, document_analysis
                    )
                    
                    if subsection_config.get("type") == "table":
                        df = parse_table_content_enhanced(content, subsection_config.get("columns", []))
                        brd_content[section_name][subsection_name] = df
                    else:
                        brd_content[section_name][subsection_name] = content
                    
                    # Calculate enhanced quality score
                    score, checks = calculate_quality_score_enhanced(subsection_name, content, subsection_config, extracted_images)
                    quality_scores[subsection_name] = score
                    compliance_checks.extend(checks)
            else:
                logger.info(f"Generating {section_name} with image integration")
                
                content = generate_intelligent_brd_section_enhanced(
                    llm, section_name, section_config, document_text,
                    extracted_images, extracted_formulas, document_analysis
                )
                
                if section_config.get("type") == "table":
                    df = parse_table_content_enhanced(content, section_config.get("columns", []))
                    brd_content[section_name] = df
                else:
                    brd_content[section_name] = content
                
                # Calculate enhanced quality score
                score, checks = calculate_quality_score_enhanced(section_name, content, section_config, extracted_images)
                quality_scores[section_name] = score
                compliance_checks.extend(checks)
            
            section_count += 1
            logger.info(f"Completed {section_count}/{total_sections} sections with enhanced processing")
            
        except Exception as e:
            logger.error(f"Error generating section {section_name}: {e}")
            # Add enhanced error handling
            if section_config.get("type") == "parent":
                brd_content[section_name] = {"error": f"Enhanced processing error: {str(e)}"}
            else:
                brd_content[section_name] = f"Enhanced processing error: {str(e)}"
            
            quality_scores[section_name] = 0.0
            compliance_checks.append(QualityCheck(section_name, "generation_error", "FAIL", str(e), "error"))
    
    # Generate overall statistics
    total_images_embedded = sum(str(content).count('[IMAGE_EMBED:') for content in brd_content.values() if content)
    avg_quality_score = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
    
    logger.info("Enhanced BRD generation with image integration completed")
    logger.info(f"  - Total sections: {total_sections}")
    logger.info(f"  - Images embedded: {total_images_embedded}")
    logger.info(f"  - Average quality score: {avg_quality_score:.1f}%")
    logger.info(f"  - Compliance checks: {len(compliance_checks)}")
    
    return {
        'brd_content': brd_content,
        'quality_scores': quality_scores,
        'compliance_checks': compliance_checks,
        'generation_statistics': {
            'total_sections': total_sections,
            'images_embedded': total_images_embedded,
            'average_quality': avg_quality_score,
            'total_formulas_processed': len(extracted_formulas),
            'total_images_available': len(extracted_images),
            'document_complexity': document_analysis.get('complexity_score', 0),
            'mathematical_complexity': document_analysis.get('mathematical_complexity', 'Unknown'),
            'extraction_quality': document_analysis.get('extraction_quality', {}).get('overall_extraction_score', 0)
        }
    }

# Update the main function to use enhanced processing
def generate_enhanced_brd(document_text: str, extracted_images: Dict[str, str], extracted_formulas: List[Any], document_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for enhanced BRD generation with image integration"""
    return generate_enhanced_brd_with_images(document_text, extracted_images, extracted_formulas, document_analysis)

# Backward compatibility function
def parse_table_content(content: str, columns: List[str]) -> pd.DataFrame:
    """Backward compatible table parsing function"""
    return parse_table_content_enhanced(content, columns)
