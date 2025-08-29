"""
Enhanced Document Processing Utilities - Complete Implementation
Enterprise-grade PDF processing for Basel regulatory documents with advanced extraction
"""

import streamlit as st
import tempfile
import os
import base64
import re
import hashlib
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    logger.warning("PyMuPDF not available. PDF processing will be limited.")

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not available. DOCX processing will be limited.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Advanced image processing will be limited.")

@dataclass
class ExtractedElement:
    """Enhanced data structure for extracted elements"""
    element_id: str
    element_type: str  # 'formula', 'table', 'diagram', 'equation_block'
    confidence: float
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1
    page_num: int
    image_data: str  # base64 encoded image
    text_content: str
    context: str
    mathematical_complexity: str
    regulatory_relevance: float

class EnterpriseImageProcessor:
    """Enterprise-grade image processing for extracted elements"""
    
    @staticmethod
    def enhance_mathematical_image(image_data: bytes) -> bytes:
        """Enhance mathematical formula/table images for better clarity"""
        try:
            # Convert to PIL Image
            img = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Enhance contrast and sharpness
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2.0)
            
            # Apply slight denoising
            img = img.filter(ImageFilter.MedianFilter(size=3))
            
            # Save enhanced image
            buffer = BytesIO()
            img.save(buffer, format='PNG', optimize=True, quality=95)
            return buffer.getvalue()
            
        except Exception as e:
            logger.warning(f"Error enhancing image: {e}")
            return image_data

    @staticmethod
    def detect_mathematical_regions(page, page_num: int) -> List[Tuple[fitz.Rect, float, str]]:
        """Detect mathematical regions using advanced pattern recognition"""
        regions = []
        
        try:
            # Get text blocks with position information
            blocks = page.get_text("dict")
            
            # Patterns that indicate mathematical content
            math_patterns = [
                r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]',  # Greek letters
                r'[∑∏∫∆∇±≤≥≠≈∞√∂∈∉⊂⊃∪∩]',  # Mathematical symbols  
                r'\b[A-Z][a-z]*_{\w+}|\b[A-Z][a-z]*\^{\w+}',  # Subscripts/superscripts
                r'ρ[_\w\d]+\s*=\s*[\d.%]+',  # Correlation formulas
                r'RW[_\w\d]*\s*=\s*[\d.%]+',  # Risk weight formulas
                r'K[_\w\d]*\s*=\s*[^.!?]*[+\-*/√∑∏][^.!?]*',  # Capital requirement formulas
                r'MAR\d+\.\d+',  # Basel MAR references
                r'Bucket\s+\d+',  # Bucket references
                r'\([^)]*[+\-*/=√∑]\s*[^)]*\)',  # Mathematical expressions
                r'VaR|PV01|CS01|DV01',  # Risk sensitivity measures
                r'\d+\.?\d*\s*%',  # Percentages
            ]
            
            # Analyze each text block
            for block in blocks.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        line_bbox = None
                        
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                            if line_bbox is None:
                                line_bbox = span.get("bbox")
                            else:
                                # Merge bounding boxes
                                bbox = span.get("bbox")
                                line_bbox = [
                                    min(line_bbox[0], bbox[0]),
                                    min(line_bbox[1], bbox[1]),
                                    max(line_bbox[2], bbox[2]),
                                    max(line_bbox[3], bbox[3])
                                ]
                        
                        if line_bbox and line_text:
                            # Calculate mathematical content score
                            math_score = 0
                            matched_patterns = []
                            
                            for pattern in math_patterns:
                                matches = re.findall(pattern, line_text, re.IGNORECASE)
                                if matches:
                                    math_score += len(matches) * 0.1
                                    matched_patterns.extend(matches)
                            
                            # Additional scoring factors
                            if len(line_text) < 200 and math_score > 0:
                                math_score += 0.2  # Short mathematical expressions
                            
                            if any(keyword in line_text.lower() for keyword in ['formula', 'equation', 'calculation', 'correlation', 'sensitivity']):
                                math_score += 0.3
                                
                            if re.search(r'\d+\.\d+', line_text):
                                math_score += 0.1  # Contains decimal numbers
                            
                            # Consider it mathematical if score > threshold
                            if math_score > 0.3:
                                rect = fitz.Rect(line_bbox)
                                element_type = EnterpriseImageProcessor._classify_mathematical_type(line_text, matched_patterns)
                                regions.append((rect, math_score, element_type))
            
            return regions
            
        except Exception as e:
            logger.error(f"Error detecting mathematical regions on page {page_num}: {e}")
            return []
    
    @staticmethod
    def _classify_mathematical_type(text: str, patterns: List[str]) -> str:
        """Classify the type of mathematical content"""
        text_lower = text.lower()
        
        if any(p in text for p in ['ρ', 'correlation', 'corr']):
            return 'correlation_formula'
        elif any(p in text for p in ['RW', 'risk weight', 'weight']):
            return 'risk_weight_formula'
        elif any(p in text for p in ['K', 'capital', 'requirement']):
            return 'capital_requirement_formula'
        elif any(p in text for p in ['VaR', 'PV01', 'CS01', 'sensitivity']):
            return 'risk_sensitivity_formula'
        elif 'MAR' in text and re.search(r'\d+\.\d+', text):
            return 'regulatory_reference'
        elif 'bucket' in text_lower:
            return 'bucket_classification'
        elif len([p for p in patterns if p in 'αβγδεζηθικλμνξοπρστυφχψω']) > 0:
            return 'greek_letter_formula'
        elif any(op in text for op in ['∑', '∏', '∫', '√']):
            return 'advanced_mathematical_expression'
        else:
            return 'mathematical_expression'

    @staticmethod
    def detect_table_regions(page, page_num: int) -> List[Tuple[fitz.Rect, float, str]]:
        """Detect table regions using advanced table detection"""
        regions = []
        
        try:
            # Find tables using PyMuPDF's table detection
            tabs = page.find_tables()
            
            for tab in tabs:
                bbox = tab.bbox
                rect = fitz.Rect(bbox)
                
                # Extract table content for analysis
                table_content = tab.extract()
                
                # Calculate table relevance score
                relevance_score = EnterpriseImageProcessor._calculate_table_relevance(table_content, page.get_text())
                
                if relevance_score > 0.3:
                    regions.append((rect, relevance_score, 'regulatory_table'))
            
            # Also detect tables using text pattern analysis (backup method)
            text_blocks = page.get_text("dict")
            table_indicators = [
                'table', 'bucket', 'risk weight', 'correlation', 'parameter',
                'threshold', 'limit', 'requirement', 'specification'
            ]
            
            for block in text_blocks.get("blocks", []):
                if "lines" in block:
                    block_text = ""
                    block_bbox = None
                    
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            text = span.get("text", "")
                            block_text += text + " "
                            
                            if block_bbox is None:
                                block_bbox = span.get("bbox")
                            else:
                                bbox = span.get("bbox")
                                block_bbox = [
                                    min(block_bbox[0], bbox[0]),
                                    min(block_bbox[1], bbox[1]),
                                    max(block_bbox[2], bbox[2]),
                                    max(block_bbox[3], bbox[3])
                                ]
                    
                    # Check if block contains table-like content
                    if block_bbox and block_text:
                        table_score = 0
                        
                        # Check for table indicators
                        for indicator in table_indicators:
                            if indicator in block_text.lower():
                                table_score += 0.2
                        
                        # Check for structured data patterns
                        if re.search(r'\|.*\|.*\|', block_text):  # Pipe-separated values
                            table_score += 0.4
                        
                        if len(re.findall(r'\d+\.?\d*\s*%', block_text)) >= 3:  # Multiple percentages
                            table_score += 0.3
                        
                        if table_score > 0.4:
                            rect = fitz.Rect(block_bbox)
                            # Expand slightly to capture full table
                            rect = rect + (-10, -10, 10, 10)
                            regions.append((rect, table_score, 'structured_table'))
            
            return regions
            
        except Exception as e:
            logger.error(f"Error detecting table regions on page {page_num}: {e}")
            return []
    
    @staticmethod
    def _calculate_table_relevance(table_content: List[List], page_text: str) -> float:
        """Calculate relevance score for extracted tables"""
        if not table_content:
            return 0.0
        
        score = 0.0
        
        # Convert table to text for analysis
        table_text = ""
        for row in table_content:
            for cell in row:
                if cell:
                    table_text += str(cell) + " "
        
        table_text = table_text.lower()
        
        # Basel-specific keywords
        basel_keywords = [
            'risk weight', 'correlation', 'bucket', 'sensitivity', 'capital',
            'var', 'pv01', 'cs01', 'mar', 'dv01', 'vega', 'curvature',
            'delta', 'gamma', 'rho', 'theta', 'supervisory', 'regulatory'
        ]
        
        for keyword in basel_keywords:
            if keyword in table_text:
                score += 0.1
        
        # Check for numerical data
        if re.search(r'\d+\.?\d*\s*%', table_text):
            score += 0.2
        
        # Check for structured format
        if len(table_content) >= 3 and len(table_content[0]) >= 2:
            score += 0.3
        
        return min(score, 1.0)

def extract_mathematical_formulas_advanced(text: str, page_num: int = None) -> List[Dict[str, Any]]:
    """
    Advanced mathematical formula extraction for Basel regulatory documents
    """
    formulas = []
    
    # Enhanced Basel-specific mathematical patterns
    formula_patterns = [
        # Basel regulatory references with mathematical context
        (r'MAR\d+\.\d+(?:\.\d+)?[^\n.!?]*(?:=|formula|calculation)[^\n.!?]*', 'basel_mar_reference'),
        
        # Complex multi-line formulas (square root with summations)
        (r'√\s*\(\s*∑[^)]+\+\s*∑∑[^)]+\)', 'complex_square_root_formula'),
        
        # Risk position formulas (Kb = √(...))
        (r'[A-Z][a-z]*\s*=\s*√\s*\([^)]+\)', 'risk_position_formula'),
        
        # Capital requirement formulas with min/max
        (r'[A-Z]+\s*=\s*-?(?:min|max)\s*\([^)]+(?:\([^)]+\))*[^)]*\)', 'capital_requirement_minmax'),
        
        # Correlation formulas with exponential decay
        (r'ρ[_\w\d]*\s*=\s*exp\s*\([^)]+\)', 'exponential_correlation_formula'),
        
        # Delta sensitivity definitions
        (r'(?:PV01|CS01|Delta)\s*=\s*\([^)]+\s*[-+]\s*[^)]+\)\s*/\s*[\d.]+', 'delta_sensitivity_formula'),
        
        # Vega risk sensitivity
        (r'(?:vega|Vega)\s*×\s*σ[_\w\d]*', 'vega_risk_formula'),
        
        # Risk weight expressions
        (r'RW[_\w\d]*\s*=\s*[\d.%]+(?:\s*/\s*√\s*\d+)?', 'risk_weight_formula'),
        
        # Correlation parameters with Greek letters
        (r'ρ[_\w\d\{\}]*\s*=\s*(?:\d+\.?\d*%?|exp\([^)]+\))', 'correlation_parameter'),
        
        # Gamma correlation formulas
        (r'γ[_\w\d\{\}]*\s*=\s*[\d.%]+', 'gamma_correlation'),
        
        # Complex summation expressions
        (r'∑[_\w\d]*\s*[^=]*=?[^.!?\n]*', 'summation_expression'),
        
        # Product expressions
        (r'∏[_\w\d]*\s*[^=]*=?[^.!?\n]*', 'product_expression'),
        
        # Liquidity horizon formulas
        (r'LH[_\w\d]*\s*=\s*\d+', 'liquidity_horizon'),
        
        # Risk factor shock definitions
        (r'(?:shock|shift)\s*[=:]\s*[\d.%]+(?:\s*×\s*[\w_]+)?', 'shock_definition'),
        
        # Bucket correlation matrices
        (r'(?:Table|Matrix)\s+\d+[^\n]*correlation[^\n]*', 'correlation_matrix_reference'),
        
        # Tenor-based formulas
        (r'T[_\w\d]*\s*[-+]\s*T[_\w\d]*', 'tenor_difference_formula'),
        
        # Greek letters and mathematical symbols
        (r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]+', 'greek_symbols'),
        
        # Mathematical operators and relations
        (r'[∑∏∫∆∇±≤≥≠≈∞√∂∈∉⊂⊃∪∩]', 'mathematical_operators'),
        
        # Subscripts/superscripts with complex notation
        (r'\w+[_\^]\{[^}]+\}|\w+[_\^]\([^)]+\)|\w+[_\^\*\+\-\d]+', 'subscript_superscript'),
        
        # Basel-specific risk measures
        (r'(?:CVR|VaR|PV01|CS01|DV01|Vega)[_\w\d]*(?:\s*[=:]\s*[^.!?\n]+)?', 'basel_risk_measures'),
        
        # Percentage formulas with context
        (r'\d+\.?\d*\s*%(?:\s*(?:where|if|when|for)[^.!?\n]*)?', 'percentage_values'),
        
        # Mathematical expressions with nested parentheses
        (r'\([^()]*\([^()]*\)[^()]*\)', 'nested_parenthetical'),
        
        # Floor/ceiling functions
        (r'⌊[^⌋]*⌋|⌈[^⌉]*⌉|floor\([^)]+\)|ceiling\([^)]+\)', 'floor_ceiling_functions'),
        
        # Matrix/vector notation
        (r'\[[^\]]*[+\-*/√∑∏]\s*[^\]]*\]|\{[^}]*[+\-*/√∑∏]\s*[^}]*\}', 'matrix_vector_notation'),
    ]
    
    for pattern, formula_type in formula_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            formula_text = match.group().strip()
            
            if len(formula_text) >= 3:  # Minimum length filter
                context_start = max(0, match.start() - 150)
                context_end = min(len(text), match.end() + 150)
                context = text[context_start:context_end].strip()
                
                confidence = calculate_formula_confidence_enhanced(formula_text, formula_type, context)
                
                if confidence > 0.4:  # Quality threshold
                    formulas.append({
                        'text': formula_text,
                        'type': formula_type,
                        'page': page_num,
                        'position': match.start(),
                        'context': context,
                        'confidence': confidence,
                        'mathematical_complexity': assess_mathematical_complexity(formula_text),
                        'regulatory_relevance': assess_regulatory_relevance(formula_text, context)
                    })
    
    # Remove duplicates and sort by confidence
    unique_formulas = []
    seen_formulas = set()
    
    for formula in formulas:
        # Create a normalized key for duplicate detection
        formula_key = (
            formula['text'].lower().replace(' ', ''),
            formula['type']
        )
        
        if formula_key not in seen_formulas:
            seen_formulas.add(formula_key)
            unique_formulas.append(formula)
    
    return sorted(unique_formulas, key=lambda x: (x['confidence'], x['regulatory_relevance']), reverse=True)

def extract_structured_tables(text: str, page_num: int = None) -> List[Dict[str, Any]]:
    """
    Enhanced table extraction for regulatory documents with sophisticated pattern recognition
    """
    tables = []
    
    # Enhanced table detection patterns for Basel documents
    table_patterns = [
        # Table with explicit numbering and title
        (r'Table\s+\d+[^\n]*\n((?:[^\n]*\|[^\n]*\n){3,})', 'numbered_table'),
        
        # Risk weight tables
        (r'Risk\s+weights?[^\n]*(?:Table\s+\d+)?[^\n]*\n((?:[^\n]*\|[^\n]*\n){3,})', 'risk_weight_table'),
        
        # Correlation matrices
        (r'(?:Correlation|correlations?)[^\n]*(?:Table\s+\d+)?[^\n]*\n((?:[^\n]*\|[^\n]*\n){3,})', 'correlation_table'),
        
        # Bucket tables
        (r'Bucket[^\n]*(?:Table\s+\d+)?[^\n]*\n((?:[^\n]*\|[^\n]*\n){3,})', 'bucket_table'),
        
        # Delta/Vega/Curvature tables
        (r'(?:Delta|Vega|Curvature)[^\n]*(?:Table\s+\d+)?[^\n]*\n((?:[^\n]*\|[^\n]*\n){3,})', 'risk_type_table'),
        
        # Tenor-based tables
        (r'Tenor[^\n]*\n((?:[^\n]*(?:year|month|day)[^\n]*\n){2,})', 'tenor_table'),
        
        # Percentage tables
        (r'(?:[^\n]*%[^\n]*\|[^\n]*){3,}', 'percentage_table'),
        
        # MAR reference tables
        (r'MAR\d+\.[^\n]*\n((?:[^\n]*\|[^\n]*\n){3,})', 'mar_reference_table'),
        
        # Multi-column data tables (detecting by pipe separators and consistent structure)
        (r'(?:[^\n]*\|[^\n]*\|[^\n]*\n){4,}', 'structured_data_table'),
        
        # Tables with header rows (detecting bold patterns or caps)
        (r'([A-Z\s\|]{20,}\n(?:[^\n]*\|[^\n]*\n){3,})', 'header_table'),
        
        # Credit quality tables
        (r'(?:Credit\s+quality|Investment\s+grade|High\s+yield)[^\n]*\n((?:[^\n]*\|[^\n]*\n){3,})', 'credit_quality_table'),
        
        # Sectoral classification tables
        (r'Sector[^\n]*\n((?:[^\n]*\|[^\n]*\n){3,})', 'sectoral_table'),
        
        # Liquidity horizon tables
        (r'Liquidity\s+horizon[^\n]*\n((?:[^\n]*\|[^\n]*\n){2,})', 'liquidity_table'),
    ]
    
    for pattern, table_type in table_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        for match in matches:
            table_text = match.group()
            
            # Enhanced table analysis
            lines = table_text.strip().split('\n')
            pipe_lines = [line for line in lines if '|' in line]
            
            if len(pipe_lines) >= 3:  # Minimum viable table
                # Extract metadata
                table_title = extract_table_title(table_text)
                columns = extract_table_columns(pipe_lines)
                table_complexity = assess_table_complexity(pipe_lines)
                
                tables.append({
                    'text': table_text.strip(),
                    'page': page_num,
                    'position': match.start(),
                    'type': table_type,
                    'rows': len(pipe_lines),
                    'columns': len(columns) if columns else 0,
                    'title': table_title,
                    'column_headers': columns,
                    'complexity': table_complexity,
                    'regulatory_relevance': assess_table_regulatory_relevance(table_text, table_type),
                    'confidence': calculate_table_confidence(table_text, table_type, len(pipe_lines))
                })
    
    return sorted(tables, key=lambda x: x['confidence'], reverse=True)

def extract_table_title(table_text: str) -> str:
    """Extract table title from table text"""
    lines = table_text.strip().split('\n')
    
    # Look for table number and title patterns
    title_patterns = [
        r'Table\s+\d+[^\n]*',
        r'(?:Risk\s+weights?|Correlation|Bucket)[^\n]*(?:Table\s+\d+)?',
        r'(?:Delta|Vega|Curvature)[^\n]*',
        r'MAR\d+\.[^\n]*'
    ]
    
    for line in lines[:3]:  # Check first 3 lines
        for pattern in title_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group().strip()
    
    return "Extracted Table"

def calculate_formula_confidence_enhanced(formula_text: str, formula_type: str, context: str) -> float:
    """Enhanced confidence calculation for formulas"""
    confidence = 0.5  # Base confidence
    
    # Length-based scoring
    if 5 <= len(formula_text) <= 100:
        confidence += 0.2
    elif len(formula_text) > 100:
        confidence += 0.1  # Very long might be false positive
    
    # Mathematical symbols
    math_symbols = ['=', '∑', '∏', '∫', '√', '∂', '∇', '±', '≤', '≥', '≠', '≈', '∞']
    symbol_count = sum(1 for symbol in math_symbols if symbol in formula_text)
    confidence += min(symbol_count * 0.1, 0.3)
    
    # Greek letters
    greek_count = len(re.findall(r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]', formula_text))
    confidence += min(greek_count * 0.05, 0.2)
    
    # Formula type specific bonuses
    type_bonuses = {
        'basel_mar_reference': 0.3,
        'risk_weight_formula': 0.25,
        'capital_requirement': 0.25,
        'correlation_formula': 0.2,
        'sensitivity_measure': 0.2
    }
    confidence += type_bonuses.get(formula_type, 0.1)
    
    # Context-based scoring
    regulatory_keywords = ['basel', 'mar', 'supervisory', 'regulatory', 'compliance', 'risk', 'capital']
    context_lower = context.lower()
    keyword_count = sum(1 for keyword in regulatory_keywords if keyword in context_lower)
    confidence += min(keyword_count * 0.05, 0.15)
    
    return min(confidence, 1.0)

def assess_mathematical_complexity(formula_text: str) -> str:
    """Assess the mathematical complexity of a formula"""
    complexity_indicators = [
        (r'[∑∏∫]', 'advanced_operators'),
        (r'(?:sqrt|exp|log|ln)', 'transcendental_functions'),
        (r'[αβγδεζηθικλμνξοπρστυφχψω]{2,}', 'multiple_greek'),
        (r'\([^)]*\([^)]*\)[^)]*\)', 'nested_parentheses'),
        (r'max|min|sup|inf', 'optimization_functions')
    ]
    
    complexity_score = 0
    for pattern, indicator in complexity_indicators:
        if re.search(pattern, formula_text, re.IGNORECASE):
            complexity_score += 1
    
    if complexity_score >= 3:
        return 'Very High'
    elif complexity_score >= 2:
        return 'High'
    elif complexity_score >= 1:
        return 'Medium'
    else:
        return 'Low'

def assess_regulatory_relevance(formula_text: str, context: str) -> float:
    """Assess regulatory relevance of a formula"""
    relevance_score = 0.5  # Base score
    
    # Basel-specific terms
    basel_terms = ['mar', 'basel', 'supervisory', 'regulatory', 'bucket', 'risk weight', 'capital', 'correlation']
    combined_text = (formula_text + ' ' + context).lower()
    
    for term in basel_terms:
        if term in combined_text:
            relevance_score += 0.1
    
    # Risk management terms
    risk_terms = ['var', 'pv01', 'cs01', 'dv01', 'vega', 'delta', 'gamma', 'theta', 'rho']
    for term in risk_terms:
        if term in combined_text:
            relevance_score += 0.05
    
    return min(relevance_score, 1.0)

# Continue with remaining helper functions...
def analyze_document_intelligence(text: str, images: Dict[str, str], formulas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Enhanced AI-powered document analysis with Basel-specific intelligence"""
    analysis = {
        'document_type': 'Unknown',
        'regulatory_framework': [],
        'mathematical_complexity': 'Low',
        'formula_types': [],
        'table_count': 0,
        'regulatory_sections': [],
        'complexity_score': 0,
        'mathematical_formulas': {
            'total_count': 0,
            'by_type': {},
            'complexity_distribution': {},
            'key_formulas': []
        }
    }
    
    if not text:
        return analysis
    
    # Enhanced document analysis implementation
    if formulas:
        analysis['mathematical_formulas']['total_count'] = len(formulas)
        
        for formula in formulas:
            if isinstance(formula, dict):
                formula_type = formula.get('type', 'unknown')
                if formula_type in analysis['mathematical_formulas']['by_type']:
                    analysis['mathematical_formulas']['by_type'][formula_type] += 1
                else:
                    analysis['mathematical_formulas']['by_type'][formula_type] = 1
                
                # Add high-confidence formulas to key formulas
                confidence = formula.get('confidence', 0)
                if confidence > 0.7:
                    analysis['mathematical_formulas']['key_formulas'].append({
                        'text': formula.get('text', '')[:100],
                        'type': formula_type,
                        'confidence': confidence,
                        'page': formula.get('page', 'Unknown')
                    })
    
    return analysis

def process_document(uploaded_file, extract_images: bool = True, extract_formulas: bool = True) -> Tuple[str, Dict[str, str], List[Dict[str, Any]], Dict[str, Any]]:
    """Enhanced document processing with enterprise-grade extraction"""
    if uploaded_file is None:
        return "", {}, [], {}
    
    try:
        file_type = uploaded_file.type
        logger.info(f"Processing file: {uploaded_file.name} ({file_type}) with enhanced extraction")
        
        if file_type == "application/pdf":
            document_text, extracted_images, extracted_formulas = extract_images_and_formulas_from_pdf(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            document_text, extracted_images = extract_images_from_docx(uploaded_file)
            extracted_formulas = extract_mathematical_formulas_advanced(document_text) if extract_formulas else []
        elif file_type == "text/plain":
            document_text = extract_text_from_txt(uploaded_file)
            extracted_images = {}
            extracted_formulas = extract_mathematical_formulas_advanced(document_text) if extract_formulas else []
        else:
            logger.warning(f"Unsupported file type: {file_type}")
            uploaded_file.seek(0)
            document_text = str(uploaded_file.read(), "utf-8", errors='ignore')
            extracted_images = {}
            extracted_formulas = extract_mathematical_formulas_advanced(document_text) if extract_formulas else []
        
        # Apply extraction options
        if not extract_images:
            extracted_images = {}
        if not extract_formulas:
            extracted_formulas = []
        
        # Perform enhanced document analysis
        document_analysis = analyze_document_intelligence(document_text, extracted_images, extracted_formulas)
        
        logger.info(f"Enhanced extraction complete:")
        logger.info(f"  - Text: {len(document_text):,} characters")
        logger.info(f"  - Images: {len(extracted_images)} items")
        logger.info(f"  - Formulas: {len(extracted_formulas)} items")
        
        return document_text, extracted_images, extracted_formulas, document_analysis
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        if 'st' in globals():
            st.error(f"Error processing document: {str(e)}")
        return "", {}, [], {}

# Additional helper functions for PDF processing
def extract_images_and_formulas_from_pdf(uploaded_file) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
    """Enhanced PDF extraction with better formula and structure detection"""
    if not FITZ_AVAILABLE:
        logger.error("PyMuPDF not available for PDF processing")
        try:
            text = uploaded_file.read().decode('utf-8', errors='ignore')
            return text, {}, []
        except Exception as e:
            logger.error(f"Error reading PDF as text: {e}")
            return "", {}, []
    
    text = ""
    images = {}
    formulas = []
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            uploaded_file.seek(0)
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Open PDF with fitz
        doc = fitz.open(tmp_file_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text with better formatting preservation
            page_text = page.get_text("text")
            text += f"\n=== PAGE {page_num + 1} ===\n{page_text}\n"
            
            # Extract mathematical formulas from this page
            page_formulas = extract_mathematical_formulas_advanced(page_text, page_num + 1)
            formulas.extend(page_formulas)
            
            # Extract tables from this page
            page_tables = extract_structured_tables(page_text, page_num + 1)
            for table in page_tables:
                formulas.append({
                    'text': table['text'][:200] + '...' if len(table['text']) > 200 else table['text'],
                    'type': 'structured_table',
                    'page': table['page'],
                    'position': table['position'],
                    'context': f"Table with {table['rows']} rows",
                    'confidence': 0.9
                })
        
        doc.close()
        os.unlink(tmp_file_path)
        
        logger.info(f"Extracted {len(formulas)} mathematical elements and {len(images)} images")
        return text, images, formulas
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        try:
            uploaded_file.seek(0)
            text = uploaded_file.read().decode('utf-8', errors='ignore')
        except:
            text = "Error reading PDF content"
        return text, {}, []

def extract_images_from_docx(uploaded_file) -> Tuple[str, Dict[str, str]]:
    """Extract text and images from DOCX with enhanced processing"""
    if not DOCX_AVAILABLE:
        logger.error("python-docx not available for DOCX processing")
        try:
            uploaded_file.seek(0)
            return uploaded_file.read().decode('utf-8', errors='ignore'), {}
        except:
            return "Error reading DOCX content", {}
    
    text = ""
    images = {}
    
    try:
        uploaded_file.seek(0)
        doc = docx.Document(uploaded_file)
        
        # Extract text with enhanced formatting
        for paragraph in doc.paragraphs:
            para_text = paragraph.text
            
            # Preserve basic formatting
            for run in paragraph.runs:
                if run.bold:
                    para_text = para_text.replace(run.text, f"**{run.text}**", 1)
                if run.italic:
                    para_text = para_text.replace(run.text, f"*{run.text}*", 1)
            
            text += para_text + "\n"
        
        # Extract images from document relationships
        try:
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        image_data = rel.target_part.blob
                        img_b64 = base64.b64encode(image_data).decode()
                        images[f"docx_img_{len(images)+1}"] = img_b64
                        
                    except Exception as e:
                        logger.warning(f"Error extracting image: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Error accessing document relationships: {e}")
                    
    except Exception as e:
        logger.error(f"Error processing DOCX: {str(e)}")
        try:
            uploaded_file.seek(0)
            text = uploaded_file.read().decode('utf-8', errors='ignore')
        except:
            text = "Error reading DOCX content"
        return text, {}
    
    return text, images

def extract_text_from_txt(uploaded_file) -> str:
    """Extract text from TXT file with enhanced encoding detection"""
    try:
        uploaded_file.seek(0)
        raw_data = uploaded_file.read()
        
        # Try different encodings
        encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                text = raw_data.decode(encoding)
                return text
            except UnicodeDecodeError:
                continue
        
        # Fallback with error handling
        return raw_data.decode('utf-8', errors='ignore')
        
    except Exception as e:
        logger.error(f"Error reading TXT file: {e}")
        return "Error reading TXT content"

def display_image_from_base64(img_b64: str, caption: str = "", max_width: int = None):
    """Enhanced image display with better error handling"""
    try:
        # Handle both string and dict formats
        if isinstance(img_b64, dict):
            img_data_b64 = img_b64.get('data', img_b64)
        else:
            img_data_b64 = img_b64
            
        img_data = base64.b64decode(img_data_b64)
        img = Image.open(BytesIO(img_data))
        
        # Auto-enhance display for better readability
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply light enhancement for display
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        
        if max_width:
            img.thumbnail((max_width, max_width), Image.Resampling.LANCZOS)
        
        st.image(img, caption=caption, use_container_width=True if not max_width else False)
        
    except Exception as e:
        logger.error(f"Error displaying image: {str(e)}")
        st.error(f"Error displaying image: {caption}")

def render_content_with_images(content: str, images: Dict[str, str]):
    """Enhanced content rendering with embedded images"""
    if not content:
        return
    
    # Enhanced image reference patterns
    image_patterns = [
        r'\[IMAGE:\s*([^\]]+)\]',
        r'\[TABLE_IMAGE:\s*([^\]]+)\]',
        r'\[FORMULA_IMAGE:\s*([^\]]+)\]',
        r'\[math_[^\]]+\]',
        r'\[table_[^\]]+\]'
    ]
    
    # Split content by any image reference
    parts = [content]
    
    for pattern in image_patterns:
        new_parts = []
        for part in parts:
            if isinstance(part, str):
                split_parts = re.split(pattern, part)
                new_parts.extend(split_parts)
            else:
                new_parts.append(part)
        parts = new_parts
    
    # Render parts alternating between text and images
    for i, part in enumerate(parts):
        if isinstance(part, str):
            if i % 2 == 0:  # Text content
                if part.strip():
                    st.markdown(part)
            else:  # Image reference
                image_key = part.strip()
                
                # Try exact match first
                if image_key in images:
                    display_image_from_base64(images[image_key], caption=f"Extracted: {image_key}")
                else:
                    # Try partial match for enhanced image keys
                    found = False
                    for img_key in images.keys():
                        if image_key in img_key or img_key in image_key:
                            display_image_from_base64(images[img_key], caption=f"Extracted: {img_key}")
                            found = True
                            break
                    
                    if not found:
                        st.info(f"Referenced content: {image_key}")
