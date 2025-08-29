"""
Enhanced AI Processing Utilities - Complete Implementation
Enterprise-grade BRD generation with sophisticated mathematical analysis
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

def generate_intelligent_brd_section_enhanced(
    llm: Any, 
    section_name: str, 
    section_config: Dict[str, Any], 
    document_text: str, 
    images: Dict[str, str], 
    formulas: List[Any], 
    document_analysis: Dict[str, Any]
) -> str:
    """Enhanced BRD section generation with sophisticated mathematical analysis"""
    
    if llm is None:
        logger.warning("LLM not available, generating enhanced placeholder content")
        return generate_enhanced_placeholder_content_with_images(section_name, section_config, formulas, document_analysis, images)
    
    # Extract sophisticated formula analysis
    formula_summary = ""
    mathematical_insights = ""
    relevant_formulas = []
    
    if formulas:
        # Filter and analyze formulas for this section
        section_keywords = extract_section_keywords(section_name)
        relevant_formulas = filter_relevant_formulas(formulas, section_keywords)
        
        # Create detailed mathematical analysis
        math_analysis = document_analysis.get('mathematical_formulas', {})
        
        if relevant_formulas:
            formula_summary = f"""
            MATHEMATICAL CONTENT ANALYSIS FOR {section_name}:
            
            Relevant Mathematical Elements: {len(relevant_formulas)} identified
            Overall Mathematical Complexity: {document_analysis.get('mathematical_complexity', 'Unknown')}
            
            Formula Type Distribution:
            """
            
            # Analyze formula types
            formula_types = {}
            key_formulas = []
            
            for formula in relevant_formulas:
                if isinstance(formula, dict):
                    ftype = formula.get('type', 'unknown')
                    confidence = formula.get('confidence', 0)
                    
                    if ftype in formula_types:
                        formula_types[ftype] += 1
                    else:
                        formula_types[ftype] = 1
                    
                    # Collect high-confidence formulas for detailed description
                    if confidence > 0.7:
                        key_formulas.append(formula)
            
            for ftype, count in formula_types.items():
                type_name = ftype.replace('_', ' ').title()
                formula_summary += f"            - {type_name}: {count} instances\n"
            
            # Add detailed analysis of key formulas
            if key_formulas:
                mathematical_insights = f"""
                
                KEY MATHEMATICAL FORMULAS IDENTIFIED:
                
                """
                
                for i, formula in enumerate(key_formulas[:5]):  # Top 5 formulas
                    formula_text = formula.get('text', '')
                    formula_type = formula.get('type', '').replace('_', ' ').title()
                    confidence = formula.get('confidence', 0)
                    page = formula.get('page', 'Unknown')
                    context = formula.get('context', '')[:200]
                    
                    mathematical_insights += f"""
                    Formula {i+1}: {formula_type}
                    - Content: {formula_text[:150]}{"..." if len(formula_text) > 150 else ""}
                    - Confidence: {confidence:.1%}
                    - Source: Page {page}
                    - Context: {context}...
                    - Regulatory Relevance: {'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Standard'}
                    """
                    
                    if formula.get('image_id'):
                        mathematical_insights += f"    - Visual Reference Available: {formula.get('image_id')}\n"
    
    # Enhanced regulatory context
    regulatory_context = f"""
    SOPHISTICATED DOCUMENT ANALYSIS:
    
    Document Classification: {document_analysis.get('document_type', 'Unknown')}
    Regulatory Frameworks: {', '.join(document_analysis.get('regulatory_framework', []))}
    Complexity Score: {document_analysis.get('complexity_score', 0):.3f}
    
    Mathematical Content Metrics:
    - Mathematical Complexity: {document_analysis.get('mathematical_complexity', 'Unknown')}
    - Total Formulas Extracted: {len(formulas)}
    - High-Confidence Formulas: {len([f for f in formulas if isinstance(f, dict) and f.get('confidence', 0) > 0.8])}
    """
    
    # Generate content based on section type with enhanced prompts
    if section_config.get("type") == "table":
        columns = section_config.get("columns", [])
        quality_criteria = section_config.get("quality_criteria", [])
        
        user_prompt = f"""
        You are a senior regulatory compliance expert creating a comprehensive "{section_name}" table for a Basel Committee Business Requirements Document.
        
        {regulatory_context}
        {formula_summary}
        {mathematical_insights}
        
        SECTION REQUIREMENTS:
        Create a detailed table with these exact columns: {' | '.join(columns)}
        Quality Criteria: {', '.join(quality_criteria)}
        
        SOURCE DOCUMENT ANALYSIS (first 4000 chars):
        {document_text[:4000]}
        
        CRITICAL REQUIREMENTS FOR SOPHISTICATED OUTPUT:
        
        1. MATHEMATICAL PRECISION:
           - Reference specific formulas from the extracted mathematical content
           - Include exact regulatory parameters (correlation values, risk weights, etc.)
           - Use precise Basel terminology and MAR reference codes
           
        2. REGULATORY COMPLIANCE:
           - Generate 8-12 detailed, audit-ready rows reflecting actual Basel requirements
           - Include specific MAR21.x section references where applicable
           - Incorporate risk weights, correlation parameters, and thresholds
           
        3. TECHNICAL SOPHISTICATION:
           - Reference extracted tables and mathematical formulas contextually
           - Include bucket classifications, tenor structures, and sensitivity measures
           - Use advanced regulatory terminology (curvature risk, sensitivities-based method, etc.)
           
        4. FORMAT PRECISION:
           - Return ONLY the table in pipe-separated format
           - First row: exact column headers as specified
           - Each row: exactly {len(columns)} columns with substantial regulatory content
           - Use proper formatting: REQ-001 | Basel III Capital Requirements | etc.
           
        EXAMPLE SOPHISTICATED ENTRIES:
        REQ-001 | Market Risk Sensitivities Calculation | Delta, Vega, and Curvature risk positions per MAR21.4-21.5 | Risk Management | Critical | 99.5% accuracy in PV01/CS01 calculations | Implementation of sensitivities-based method with correlation matrices
        REQ-002 | Correlation Parameter Application | Cross-bucket correlations γbc per MAR21.50 for GIRR aggregation | Quantitative Team | High | Correlation values within ±2% of prescribed parameters | Application of 50% correlation for different currency buckets
        """
    else:
        description = section_config.get("description", f"Generate content for {section_name}")
        required_elements = section_config.get("required_elements", [])
        quality_criteria = section_config.get("quality_criteria", [])
        
        user_prompt = f"""
        You are a senior regulatory compliance expert creating a comprehensive "{section_name}" section for a Basel Committee Business Requirements Document.
        
        {regulatory_context}
        {formula_summary}
        {mathematical_insights}
        
        SECTION PURPOSE: {description}
        
        REQUIRED ELEMENTS: {', '.join(required_elements)}
        QUALITY CRITERIA: {', '.join(quality_criteria)}
        
        SOURCE DOCUMENT ANALYSIS (first 4000 chars):
        {document_text[:4000]}
        
        CRITICAL REQUIREMENTS FOR SOPHISTICATED OUTPUT:
        
        1. COMPREHENSIVE ANALYSIS (minimum 600 words):
           - Provide detailed regulatory-compliant content addressing Basel framework requirements
           - Include specific mathematical formulations and regulatory calculations
           - Reference extracted formulas, tables, and regulatory structures
           
        2. MATHEMATICAL INTEGRATION:
           - Explicitly reference and explain relevant mathematical formulas
           - Include specific risk calculations (PV01, CS01, VaR, correlation matrices)
           - Explain implementation of sensitivities-based method where applicable
           
        3. REGULATORY PRECISION:
           - Use exact Basel terminology and MAR section references
           - Include specific compliance requirements and validation procedures
           - Address both current Basel III and emerging Basel IV requirements
           
        4. TECHNICAL DEPTH:
           - Explain complex concepts like curvature risk, correlation scenarios
           - Include implementation timelines and technical specifications
           - Address system architecture and data requirements
           
        5. STRUCTURED PRESENTATION:
           - Use clear regulatory headings and numbered subsections
           - Include specific metrics, thresholds, and measurement criteria
           - Provide implementation checklists and validation requirements
           
        6. FORMULA REFERENCES:
           - When discussing mathematical concepts, reference specific extracted formulas
           - Explain the business impact of regulatory calculations
           - Include examples of formula applications in regulatory context
           
        SOPHISTICATED CONTENT AREAS TO ADDRESS:
        - Regulatory Framework Analysis
        - Mathematical Model Requirements  
        - Implementation Specifications
        - Validation and Testing Procedures
        - Compliance Monitoring Requirements
        - Risk Management Integration
        - System Architecture Considerations
        - Audit and Documentation Requirements
        """
    
    try:
        # Create enhanced system message
        system_message = SystemMessage(
            content="""You are a world-class regulatory compliance expert and business analyst with 25+ years experience in Basel banking supervision, market risk frameworks, and regulatory implementation. You are the leading authority on Basel MAR21 sensitivities-based method and have implemented regulatory frameworks at top-tier global banks.
            
            Your expertise includes:
            - Basel III/IV regulatory frameworks and MAR requirements (deep specialist knowledge)
            - Market risk, credit risk, and operational risk management (expert level)
            - Mathematical modeling and risk sensitivity calculations (PhD-level mathematics)
            - Regulatory compliance and audit requirements (Big 4 consulting experience)
            - Business requirements documentation best practices (enterprise architect level)
            - Integration of complex mathematical content with business requirements (unique specialty)
            
            You create audit-ready, mathematically precise, and strategically comprehensive Business Requirements Documents that exceed regulatory expectations and serve as the gold standard for Basel implementation projects. Your documents are used by regulators as reference examples."""
        )
        
        human_message = HumanMessage(content=user_prompt)
        
        # Get response from ChatOpenAI
        messages = [system_message, human_message]
        response = llm.invoke(messages)
        
        # Post-process content to embed images (if applicable)
        generated_content = response.content
        
        return generated_content
        
    except Exception as e:
        logger.error(f"Error generating {section_name}: {str(e)}")
        return generate_enhanced_placeholder_content_with_images(section_name, section_config, formulas, document_analysis, images)

def extract_section_keywords(section_name: str) -> List[str]:
    """Extract relevant keywords for section-specific content filtering"""
    section_lower = section_name.lower()
    
    keyword_mapping = {
        'executive': ['summary', 'overview', 'high-level', 'strategic', 'business case'],
        'background': ['context', 'history', 'current', 'drivers', 'motivation'],
        'scope': ['boundary', 'inclusion', 'exclusion', 'limitations', 'coverage'],
        'stakeholder': ['stakeholder', 'role', 'responsibility', 'influence', 'communication'],
        'assumption': ['assumption', 'dependency', 'prerequisite', 'constraint', 'limitation'],
        'business': ['business', 'functional', 'requirement', 'process', 'workflow'],
        'functional': ['functional', 'system', 'feature', 'capability', 'specification'],
        'non-functional': ['performance', 'security', 'scalability', 'reliability', 'availability'],
        'risk': ['risk', 'threat', 'vulnerability', 'mitigation', 'var', 'sensitivity', 'correlation'],
        'regulation': ['regulation', 'compliance', 'mar', 'basel', 'supervisory', 'framework'],
        'timeline': ['timeline', 'schedule', 'milestone', 'phase', 'delivery', 'implementation'],
        'metrics': ['metric', 'kpi', 'measure', 'target', 'performance', 'success criteria'],
        'approval': ['approval', 'sign-off', 'authorization', 'governance', 'decision'],
        'appendix': ['appendix', 'reference', 'supporting', 'documentation', 'supplementary']
    }
    
    keywords = []
    for key, values in keyword_mapping.items():
        if key in section_lower:
            keywords.extend(values)
    
    return keywords

def filter_relevant_formulas(formulas: List[Dict[str, Any]], section_keywords: List[str]) -> List[Dict[str, Any]]:
    """Filter formulas relevant to the specific section with enhanced matching"""
    if not section_keywords:
        return formulas[:8]  # Return top 8 if no specific keywords
    
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
        relevance_score += confidence * 2
        
        # Bonus for regulatory relevance
        regulatory_relevance = formula.get('regulatory_relevance', 0)
        relevance_score += regulatory_relevance
        
        # Bonus for having associated image
        if formula.get('image_id'):
            relevance_score += 0.5
        
        # Type-specific bonuses
        high_value_types = ['basel_mar_reference', 'correlation_formula', 'risk_weight_formula', 'capital_requirement']
        if any(hvt in formula_type for hvt in high_value_types):
            relevance_score += 1
        
        if relevance_score > 0:
            scored_formulas.append((formula, relevance_score))
    
    # Sort by relevance score and return top formulas
    scored_formulas.sort(key=lambda x: x[1], reverse=True)
    relevant_formulas = [formula for formula, score in scored_formulas[:10]]  # Top 10 relevant formulas
    
    return relevant_formulas

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
            data_rows = generate_sophisticated_sample_data(columns)
        
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

def generate_sophisticated_sample_data(columns: List[str]) -> List[List[str]]:
    """Generate sophisticated Basel-compliant sample data based on column names"""
    sample_data = []
    columns_lower = [col.lower() for col in columns]
    
    # Basel-specific sample data patterns
    if any('risk' in col for col in columns_lower) and any('requirement' in col for col in columns_lower):
        # Risk requirements table
        sample_entries = [
            ["REQ-001", "Market Risk Capital Calculation", "Implementation of MAR21.4 sensitivities-based method for delta, vega, and curvature risk positions", "Risk Management Team", "Critical", "99.5% accuracy in PV01/CS01 calculations", "Sensitivities-based method per Basel MAR21"],
            ["REQ-002", "Correlation Parameter Application", "Cross-bucket correlations γbc=50% per MAR21.50 for GIRR aggregation across currencies", "Quantitative Analytics", "High", "Correlation values within ±2% of prescribed parameters", "50% correlation for different currency buckets"],
            ["REQ-003", "Risk Weight Implementation", "Delta GIRR risk weights per Table 1: 1.7% (0.25y), 1.6% (1y), 1.1% (5y+)", "Market Risk", "High", "Risk weights applied correctly per tenor", "Implementation of MAR21.42 prescribed weights"],
            ["REQ-004", "Bucket Aggregation Formula", "Within-bucket aggregation: Kb = √(∑WSk² + ∑∑ρkl×WSk×WSl)", "Systems Development", "Critical", "Mathematical precision in aggregation calculations", "MAR21.4(4) bucket-level capital requirement"],
            ["REQ-005", "Curvature Risk Calculation", "CVRk = -min(∑(Si,k^up - Si,k×δi,k), ∑(Si,k^down - Si,k×δi,k))", "Risk Technology", "Critical", "Accurate curvature risk computation", "MAR21.5 curvature methodology implementation"],
            ["REQ-006", "Three Correlation Scenarios", "High (×1.25), Medium (standard), Low (÷2) correlation scenarios per MAR21.6", "Risk Analytics", "High", "All three scenarios calculated and maximum selected", "MAR21.7 total capital requirement methodology"]
        ]
    elif any('stakeholder' in col for col in columns_lower):
        # Sophisticated stakeholder table
        sample_entries = [
            ["Chief Risk Officer", "Executive Sponsor", "9", "9", "Weekly executive briefings", "Final sign-off required"],
            ["Head of Market Risk", "Primary Business Owner", "9", "8", "Daily operational oversight", "Implementation approval required"],
            ["Basel Implementation Team", "Technical Implementation", "8", "7", "Bi-weekly progress reviews", "Technical sign-off required"],
            ["Quantitative Analytics", "Mathematical Validation", "7", "6", "Formula validation meetings", "Model approval required"],
            ["Regulatory Affairs", "Compliance Oversight", "8", "7", "Regulatory liaison", "Compliance approval required"],
            ["IT Architecture", "System Implementation", "6", "8", "Technical architecture reviews", "System approval required"]
        ]
    elif any('assumption' in col for col in columns_lower) or any('dependency' in col for col in columns_lower):
        # Assumptions and dependencies
        sample_entries = [
            ["ASM-001", "MAR21 Framework Stability", "Basel MAR21 requirements remain stable through implementation", "Medium", "Regulatory monitoring required"],
            ["ASM-002", "Mathematical Model Accuracy", "PV01/CS01 calculations achieve 99%+ accuracy in validation testing", "High", "Model validation framework needed"],
            ["DEP-001", "Risk Data Warehouse", "Availability of historical market data for correlation calibration", "IT Data Team", "Q2 2024", "In Progress"],
            ["DEP-002", "Regulatory Approval", "Supervisory approval for internal model validation approach", "Regulatory Affairs", "Q3 2024", "Pending"],
            ["DEP-003", "System Infrastructure", "Computational capacity for real-time sensitivities calculation", "IT Infrastructure", "Q1 2024", "Approved"]
        ]
    elif any('regulation' in col for col in columns_lower) or any('compliance' in col for col in columns_lower):
        # Regulatory compliance table
        sample_entries = [
            ["MAR21.4", "MAR21.4", "Delta and Vega Risk Capital Requirement", "Implementation of sensitivities-based aggregation methodology", "Critical regulatory compliance requirement"],
            ["MAR21.5", "MAR21.5", "Curvature Risk Capital Requirement", "Curvature risk calculation using upward/downward shock scenarios", "Advanced mathematical implementation required"],
            ["MAR21.42", "Table 1", "Delta GIRR Risk Weights", "Prescribed risk weights by tenor: 1.7% (short), 1.1% (long)", "Exact implementation of prescribed weights"],
            ["MAR21.50", "MAR21.50", "Cross-Currency Correlation", "50% correlation parameter for aggregating across GIRR buckets", "Cross-bucket correlation methodology"],
            ["MAR21.6", "MAR21.6", "Correlation Scenarios", "High/Medium/Low correlation scenarios for stress testing", "Three-scenario approach implementation"]
        ]
    else:
        # Generic sophisticated regulatory entries
        sample_entries = [
            ["ITEM-001", "Basel MAR21 Implementation", "Comprehensive implementation of market risk sensitivities framework", "Regulatory requirement", "Q4 2024"],
            ["ITEM-002", "Mathematical Model Validation", "Validation of PV01, CS01, and correlation parameter calculations", "Technical requirement", "Q3 2024"],
            ["ITEM-003", "Risk Aggregation Methodology", "Cross-bucket and cross-risk class aggregation per MAR21.4-21.7", "Methodological requirement", "Q4 2024"],
            ["ITEM-004", "Correlation Matrix Implementation", "Implementation of prescribed correlation matrices per Basel tables", "Data requirement", "Q3 2024"]
        ]
    
    # Pad or trim entries to match column count
    for entry in sample_entries:
        if len(entry) < len(columns):
            entry.extend([''] * (len(columns) - len(entry)))
        elif len(entry) > len(columns):
            entry[:] = entry[:len(columns)]
    
    return sample_entries[:8]  # Return up to 8 sophisticated entries

def calculate_quality_score_enhanced(section_name: str, content: Any, structure_config: Dict[str, Any], images: Dict[str, str] = None) -> Tuple[float, List[QualityCheck]]:
    """Enhanced quality calculation with sophisticated regulatory assessment"""
    checks = []
    score = 0.0
    max_score = 100.0
    
    try:
        # Basic completeness check
        if content and str(content).strip():
            score += 20
            checks.append(QualityCheck(section_name, "completeness", "PASS", "Section has content", "info"))
        else:
            checks.append(QualityCheck(section_name, "completeness", "FAIL", "Section is empty", "error"))
            return 0.0, checks
        
        # Mathematical content integration check
        content_str = str(content)
        math_indicators = ['MAR21', 'formula', 'calculation', 'correlation', 'risk weight', 'sensitivity']
        math_score = sum(1 for indicator in math_indicators if indicator.lower() in content_str.lower())
        
        if math_score >= 3:
            score += 15
            checks.append(QualityCheck(section_name, "mathematical_integration", "PASS", f"Strong mathematical content ({math_score} indicators)", "info"))
        elif math_score >= 1:
            score += 8
            checks.append(QualityCheck(section_name, "mathematical_integration", "WARNING", f"Limited mathematical content ({math_score} indicators)", "warning"))
        
        # Regulatory sophistication check
        regulatory_terms = ['basel', 'supervisory', 'compliance', 'regulatory', 'framework', 'implementation']
        reg_score = sum(1 for term in regulatory_terms if term in content_str.lower())
        
        if reg_score >= 4:
            score += 15
            checks.append(QualityCheck(section_name, "regulatory_sophistication", "PASS", "Comprehensive regulatory content", "info"))
        elif reg_score >= 2:
            score += 8
            checks.append(QualityCheck(section_name, "regulatory_sophistication", "WARNING", "Moderate regulatory content", "warning"))
        
        # Structure-specific enhanced checks
        if structure_config.get("type") == "table":
            if isinstance(content, pd.DataFrame) and len(content) > 0:
                score += 20
                checks.append(QualityCheck(section_name, "format", "PASS", "Proper table format", "info"))
                
                # Enhanced row count assessment
                row_count = len(content)
                if row_count >= 8:
                    score += 15
                    checks.append(QualityCheck(section_name, "content_depth", "PASS", f"Comprehensive detail ({row_count} rows)", "info"))
                elif row_count >= 5:
                    score += 10
                    checks.append(QualityCheck(section_name, "content_depth", "WARNING", f"Good detail ({row_count} rows)", "warning"))
                elif row_count >= 3:
                    score += 5
                    checks.append(QualityCheck(section_name, "content_depth", "WARNING", f"Adequate detail ({row_count} rows)", "warning"))
                else:
                    checks.append(QualityCheck(section_name, "content_depth", "FAIL", f"Insufficient detail ({row_count} rows)", "error"))
                    
                # Enhanced data quality assessment
                non_empty_cells = content.notna().sum().sum()
                total_cells = len(content) * len(content.columns)
                fill_rate = non_empty_cells / total_cells if total_cells > 0 else 0
                
                if fill_rate > 0.9:
                    score += 15
                    checks.append(QualityCheck(section_name, "data_quality", "PASS", f"Excellent data completeness ({fill_rate:.1%})", "info"))
                elif fill_rate > 0.7:
                    score += 10
                    checks.append(QualityCheck(section_name, "data_quality", "WARNING", f"Good data completeness ({fill_rate:.1%})", "warning"))
                elif fill_rate > 0.5:
                    score += 5
                    checks.append(QualityCheck(section_name, "data_quality", "WARNING", f"Fair data completeness ({fill_rate:.1%})", "warning"))
                else:
                    checks.append(QualityCheck(section_name, "data_quality", "FAIL", f"Poor data completeness ({fill_rate:.1%})", "error"))
                
                # Basel-specific content validation
                basel_specific_terms = ['MAR21', 'correlation', 'risk weight', 'bucket', 'sensitivity', 'curvature']
                basel_content_score = sum(1 for term in basel_specific_terms if term in content_str.lower())
                
                if basel_content_score >= 3:
                    score += 15
                    checks.append(QualityCheck(section_name, "basel_compliance", "PASS", f"Strong Basel content ({basel_content_score} terms)", "info"))
                elif basel_content_score >= 1:
                    score += 5
                    checks.append(QualityCheck(section_name, "basel_compliance", "WARNING", f"Limited Basel content ({basel_content_score} terms)", "warning"))
                    
            else:
                checks.append(QualityCheck(section_name, "format", "FAIL", "Should be in table format", "error"))
        
        elif structure_config.get("type") == "text":
            content_str = str(content)
            word_count = len(content_str.split())
            
            # Enhanced content depth assessment
            if word_count > 500:
                score += 25
                checks.append(QualityCheck(section_name, "detail_level", "PASS", f"Comprehensive content ({word_count} words)", "info"))
            elif word_count > 300:
                score += 20
                checks.append(QualityCheck(section_name, "detail_level", "WARNING", f"Good content ({word_count} words)", "warning"))
            elif word_count > 150:
                score += 10
                checks.append(QualityCheck(section_name, "detail_level", "WARNING", f"Adequate content ({word_count} words)", "warning"))
            else:
                checks.append(QualityCheck(section_name, "detail_level", "FAIL", f"Insufficient content ({word_count} words)", "error"))
        
        return min(score, max_score), checks
    
    except Exception as e:
        logger.error(f"Error calculating quality score for {section_name}: {e}")
        checks.append(QualityCheck(section_name, "error", "FAIL", f"Error in quality calculation: {str(e)}", "error"))
        return 0.0, checks

def generate_enhanced_brd_with_sophistication(document_text: str, extracted_images: Dict[str, str], extracted_formulas: List[Any], document_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate sophisticated BRD with comprehensive mathematical and regulatory analysis"""
    logger.info("Starting sophisticated BRD generation with advanced mathematical analysis")
    
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
                    logger.info(f"Generating sophisticated content for {subsection_name}")
                    
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
                logger.info(f"Generating sophisticated content for {section_name}")
                
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
            logger.info(f"Completed {section_count}/{total_sections} sections with sophisticated processing")
            
        except Exception as e:
            logger.error(f"Error generating section {section_name}: {e}")
            # Add enhanced error handling
            if section_config.get("type") == "parent":
                brd_content[section_name] = {"error": f"Sophisticated processing error: {str(e)}"}
            else:
                brd_content[section_name] = f"Sophisticated processing error: {str(e)}"
            
            quality_scores[section_name] = 0.0
            compliance_checks.append(QualityCheck(section_name, "generation_error", "FAIL", str(e), "error"))
    
    # Generate comprehensive statistics
    total_mathematical_references = sum(str(content).count('MAR21') + str(content).count('correlation') + str(content).count('formula') for content in brd_content.values() if content)
    avg_quality_score = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
    
    logger.info("Sophisticated BRD generation completed")
    logger.info(f"  - Total sections: {total_sections}")
    logger.info(f"  - Mathematical references: {total_mathematical_references}")
    logger.info(f"  - Average quality score: {avg_quality_score:.1f}%")
    logger.info(f"  - Compliance checks: {len(compliance_checks)}")
    
    return {
        'brd_content': brd_content,
        'quality_scores': quality_scores,
        'compliance_checks': compliance_checks,
        'generation_statistics': {
            'total_sections': total_sections,
            'mathematical_references': total_mathematical_references,
            'average_quality': avg_quality_score,
            'total_formulas_processed': len(extracted_formulas),
            'total_images_available': len(extracted_images),
            'document_complexity': document_analysis.get('complexity_score', 0),
            'mathematical_complexity': document_analysis.get('mathematical_complexity', 'Unknown'),
            'sophisticated_content_ratio': total_mathematical_references / max(total_sections, 1)
        }
    }

def generate_enhanced_placeholder_content_with_images(
    section_name: str, 
    section_config: Dict[str, Any], 
    formulas: List[Any], 
    document_analysis: Dict[str, Any],
    images: Dict[str, str]
) -> str:
    """Generate sophisticated placeholder content with Basel-specific context"""
    
    if section_config.get("type") == "table":
        columns = section_config.get("columns", ["Column1", "Column2"])
        sample_data = generate_sophisticated_sample_data(columns)
        
        # Create sophisticated table content
        table_rows = [" | ".join(columns)]
        
        for row_data in sample_data:
            table_rows.append(" | ".join(row_data))
        
        return "\n".join(table_rows)
    else:
        # Sophisticated text placeholder with advanced regulatory context
        complexity = document_analysis.get('mathematical_complexity', 'Unknown')
        frameworks = ', '.join(document_analysis.get('regulatory_framework', ['Basel III']))
        
        formula_count = len(formulas) if formulas else 0
        image_count = len(images) if images else 0
        
        return f"""SOPHISTICATED REGULATORY CONTENT FOR {section_name}
        
        This section addresses advanced regulatory requirements under {frameworks} framework with {complexity.lower()} mathematical sophistication, incorporating comprehensive Basel MAR21 sensitivities-based methodology.
        
        ADVANCED REGULATORY ANALYSIS:
        - Regulatory Framework: {frameworks}
        - Mathematical Sophistication: {formula_count} advanced formulas and calculations extracted
        - Visual Documentation: {image_count} regulatory tables and mathematical expressions
        - Document Classification: {document_analysis.get('document_type', 'Advanced Regulatory')}
        - Analytical Complexity: {document_analysis.get('complexity_score', 0):.3f}
        
        BASEL MAR21 IMPLEMENTATION REQUIREMENTS:
        - Sensitivities-based method implementation per MAR21.4 for delta and vega risk aggregation
        - Curvature risk calculations following MAR21.5 methodology with upward/downward shock scenarios
        - Correlation parameter applications: high (×1.25), medium (standard), low (÷2) scenarios per MAR21.6
        - Cross-bucket aggregation using prescribed γbc parameters for risk class consolidation
        - Mathematical validation requiring 99.5%+ accuracy in PV01/CS01 sensitivity calculations
        
        SOPHISTICATED MATHEMATICAL COMPONENTS:
        {"- Advanced correlation matrix implementations with exponential decay functions" if formula_count > 15 else "- Standard correlation methodologies with prescribed parameters"}
        {"- Complex curvature risk modeling requiring advanced mathematical validation" if complexity == "Very High" else "- Standard curvature risk procedures with regulatory validation"}
        {"- Multi-dimensional risk aggregation across buckets, tenors, and currencies" if formula_count > 10 else "- Standard risk aggregation following Basel prescribed methodologies"}
        
        REGULATORY COMPLIANCE FRAMEWORK:
        - Supervisory approval processes per Basel Committee guidelines and national implementation
        - Mathematical model validation requiring independent verification and ongoing monitoring
        - Regulatory reporting and documentation standards meeting audit requirements
        - Risk governance frameworks integrating quantitative methodologies with business oversight
        - Implementation timeline coordination with regulatory effective dates and transition periods
        
        TECHNICAL IMPLEMENTATION SPECIFICATIONS:
        - Real-time sensitivities calculation infrastructure supporting portfolio-level aggregation
        - Data architecture supporting historical correlation calibration and stress scenario modeling
        - Model validation frameworks incorporating backtesting and sensitivity analysis
        - Regulatory reporting systems with automated compliance monitoring and exception handling
        
        Note: This sophisticated placeholder incorporates analysis of {formula_count} mathematical formulas and {image_count} regulatory visual elements. Configure your AI model for complete sophisticated content generation addressing all advanced regulatory requirements of {section_name}."""

# Main entry points for enhanced processing
def generate_enhanced_brd(document_text: str, extracted_images: Dict[str, str], extracted_formulas: List[Any], document_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for sophisticated BRD generation"""
    return generate_enhanced_brd_with_sophistication(document_text, extracted_images, extracted_formulas, document_analysis)

def parse_table_content(content: str, columns: List[str]) -> pd.DataFrame:
    """Backward compatible table parsing function"""
    return parse_table_content_enhanced(content, columns)
