"""
Enhanced Document Processing Utilities with Table and Formula Extraction
"""
import streamlit as st
import tempfile
import os
import base64
import re
import json
from io import BytesIO
from PIL import Image
from typing import Dict, List, Tuple, Any
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

def extract_tables_from_text(text: str, page_num: int = None) -> List[Dict[str, Any]]:
    """
    Enhanced table extraction from regulatory documents
    """
    tables = []
    
    # Table detection patterns for regulatory documents
    table_patterns = [
        # Standard markdown-style tables with headers
        r'(\|[^|\n]+\|[^|\n]+\|\s*\n\s*\|[-\s|:]+\|\s*\n(?:\s*\|[^|\n]*\|\s*\n)+)',
        
        # Basel-specific tables with "Table X" headers
        r'(Table\s+\d+[^\n]*\n(?:[^\n]*\|[^\n]*\n){2,})',
        
        # Risk weight tables
        r'((?:Risk\s+weight|Bucket\s+number|Correlation)[^\n]*\n(?:[^\n]*\|[^\n]*\n){2,})',
        
        # Bucket definition tables
        r'(Bucket\s+number[^\n]*Credit\s+quality[^\n]*Sector[^\n]*\n(?:[^\n]*\|[^\n]*\n){2,})',
        
        # Simple tabular data with consistent column alignment
        r'(\n(?:\s*\w+[\s\d.%]+){3,}\s*\n(?:\s*[\w\d.%\-]+\s+){3,}\n)',
        
        # Correlation matrices
        r'((?:\d+\.\d+y|\d+\sy|\w+)\s+(?:\d+\.\d+%\s*){5,})',
    ]
    
    for i, pattern in enumerate(table_patterns):
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            table_text = match.group().strip()
            
            # Skip very small tables
            if len(table_text) < 100:
                continue
                
            # Count rows and columns
            lines = [line.strip() for line in table_text.split('\n') if line.strip()]
            
            # Detect table structure
            if '|' in table_text:
                # Pipe-separated table
                data_lines = [line for line in lines if '|' in line and not all(c in '|-: ' for c in line.replace('|', ''))]
                columns = len([cell.strip() for cell in data_lines[0].split('|') if cell.strip()]) if data_lines else 0
                rows = len(data_lines)
            else:
                # Space-separated or other format
                data_lines = lines
                avg_words_per_line = sum(len(line.split()) for line in data_lines) / len(data_lines) if data_lines else 0
                columns = int(avg_words_per_line)
                rows = len(data_lines)
            
            # Only process substantial tables
            if rows >= 2 and columns >= 2:
                table_info = {
                    'text': table_text,
                    'page': page_num,
                    'position': match.start(),
                    'type': 'extracted_table',
                    'rows': rows,
                    'columns': columns,
                    'pattern_type': f'pattern_{i+1}',
                    'confidence': min(0.9, 0.5 + (rows * columns) / 100),
                    'metadata': {
                        'has_pipes': '|' in table_text,
                        'has_headers': any(word in table_text.lower() for word in ['bucket', 'risk', 'weight', 'correlation', 'tenor']),
                        'is_numerical': len(re.findall(r'\d+\.?\d*%?', table_text)) > (rows * columns * 0.3)
                    }
                }
                tables.append(table_info)
    
    # Remove duplicate tables
    unique_tables = []
    seen_positions = set()
    
    for table in sorted(tables, key=lambda x: x['confidence'], reverse=True):
        position_range = range(table['position'], table['position'] + len(table['text']))
        if not any(pos in seen_positions for pos in position_range):
            unique_tables.append(table)
            seen_positions.update(position_range)
    
    return unique_tables[:20]  # Limit to top 20 tables

def extract_enhanced_mathematical_formulas(text: str, page_num: int = None) -> List[Dict[str, Any]]:
    """
    Enhanced mathematical formula extraction for regulatory documents
    """
    formulas = []
    
    # Enhanced mathematical patterns
    formula_patterns = [
        # Complex mathematical expressions with Greek letters
        (r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ][\w\d_]*\s*[=≈≤≥]\s*[^.!?;\n]{10,100}', 'greek_formula'),
        
        # Correlation formulas with ρ (rho)
        (r'ρ[\w\d_]*\s*=\s*[^.!?;\n]{5,80}', 'correlation_formula'),
        
        # Risk weight formulas with RW
        (r'RW[\w\d_]*\s*=\s*[^.!?;\n]{5,80}', 'risk_weight_formula'),
        
        # Capital requirement formulas with K
        (r'K[\w\d_]*\s*=\s*[^.!?;\n]{10,100}', 'capital_formula'),
        
        # Square root expressions
        (r'√\([^)]{10,100}\)|sqrt\([^)]{10,100}\)', 'square_root_formula'),
        
        # Summation formulas
        (r'∑[\w\d\s∈∩∪⊂⊃≠≤≥]{5,}[^.!?;\n]{10,100}', 'summation_formula'),
        
        # Sensitivity formulas (PV01, CS01, etc.)
        (r'(PV01|CS01|VaR|CVaR)[\w\d_]*\s*=\s*[^.!?;\n]{10,100}', 'sensitivity_formula'),
        
        # Complex fractions and ratios
        (r'[^.!?;\n]*\[[^]]{10,80}\]\s*/\s*[^.!?;\n]{5,50}', 'fraction_formula'),
        
        # Mathematical expressions with subscripts/superscripts in text
        (r'\w+_\{[^}]{2,20}\}[\s\w\d=+\-*/^()]{5,80}|\w+\^\{[^}]{2,20}\}[\s\w\d=+\-*/^()]{5,80}', 'subscript_superscript_formula'),
        
        # Exponential expressions
        (r'e\^[^.!?;\s]{2,30}|exp\([^)]{5,80}\)', 'exponential_formula'),
        
        # Basel-specific formulas with buckets
        (r'[Bb]ucket[\s\w\d]*[=:]\s*[^.!?;\n]{10,100}', 'bucket_formula'),
        
        # Percentage calculations
        (r'[\w\s]*[=]\s*[^.!?;\n]*\d+\.?\d*\s*%[^.!?;\n]{0,50}', 'percentage_formula'),
        
        # Matrix/vector notation
        (r'\[[^\]]{20,150}\]|\{[^}]{20,150}\}', 'matrix_formula'),
        
        # Regulatory formulas with "where" clauses
        (r'[^.!?;\n]{20,100}\s+where[\s:][^.!?]{30,200}', 'where_clause_formula'),
        
        # Floor/ceiling functions
        (r'⌊[^⌋]{5,50}⌋|⌈[^⌉]{5,50}⌉|floor\([^)]{5,50}\)|ceiling\([^)]{5,50}\)', 'floor_ceiling_formula'),
    ]
    
    for pattern, formula_type in formula_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            formula_text = match.group().strip()
            
            # Skip very short or very long matches
            if len(formula_text) < 5 or len(formula_text) > 500:
                continue
            
            # Extract context around formula
            context_start = max(0, match.start() - 150)
            context_end = min(len(text), match.end() + 150)
            context = text[context_start:context_end].strip()
            
            # Calculate confidence score
            confidence = calculate_formula_confidence_enhanced(formula_text, formula_type, context)
            
            if confidence > 0.3:  # Only keep formulas with reasonable confidence
                formulas.append({
                    'text': formula_text,
                    'type': formula_type,
                    'page': page_num,
                    'position': match.start(),
                    'context': context,
                    'confidence': confidence,
                    'metadata': {
                        'length': len(formula_text),
                        'has_equals': '=' in formula_text,
                        'has_greek': bool(re.search(r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]', formula_text)),
                        'has_subscript': bool(re.search(r'_\{[^}]+\}|_\d+', formula_text)),
                        'has_superscript': bool(re.search(r'\^\{[^}]+\}|\^\d+', formula_text)),
                        'numerical_density': len(re.findall(r'\d+\.?\d*', formula_text)) / len(formula_text.split())
                    }
                })
    
    # Remove overlapping formulas and sort by confidence
    unique_formulas = []
    positions_used = set()
    
    for formula in sorted(formulas, key=lambda x: x['confidence'], reverse=True):
        formula_range = range(formula['position'], formula['position'] + len(formula['text']))
        if not any(pos in positions_used for pos in formula_range):
            unique_formulas.append(formula)
            positions_used.update(formula_range)
    
    return unique_formulas[:50]  # Limit to top 50 formulas

def calculate_formula_confidence_enhanced(formula_text: str, formula_type: str, context: str) -> float:
    """Enhanced confidence calculation for mathematical formulas"""
    confidence = 0.4  # Base confidence
    
    # Length bonus (optimal length)
    length = len(formula_text)
    if 20 <= length <= 100:
        confidence += 0.3
    elif 10 <= length <= 200:
        confidence += 0.1
    
    # Mathematical symbols bonus
    math_symbols = ['=', '≈', '≤', '≥', '∑', '∫', '√', 'α', 'β', 'γ', 'δ', 'ε', 'ρ', 'σ', 'τ', 'φ', 'χ', 'ψ', 'ω']
    symbol_count = sum(1 for symbol in math_symbols if symbol in formula_text)
    confidence += min(0.3, symbol_count * 0.1)
    
    # Type-specific bonuses
    type_bonuses = {
        'correlation_formula': 0.2,
        'risk_weight_formula': 0.2,
        'capital_formula': 0.25,
        'sensitivity_formula': 0.2,
        'summation_formula': 0.15,
        'greek_formula': 0.2
    }
    confidence += type_bonuses.get(formula_type, 0.05)
    
    # Context relevance bonus
    regulatory_terms = ['basel', 'risk', 'capital', 'bucket', 'correlation', 'sensitivity', 'curvature', 'delta', 'vega']
    context_lower = context.lower()
    context_score = sum(1 for term in regulatory_terms if term in context_lower)
    confidence += min(0.2, context_score * 0.05)
    
    # Numerical content bonus
    numbers = re.findall(r'\d+\.?\d*', formula_text)
    if numbers:
        confidence += min(0.15, len(numbers) * 0.03)
    
    # Penalty for very common words that might be false positives
    common_words = ['the', 'and', 'for', 'with', 'this', 'that', 'are', 'was', 'were']
    word_count = len(formula_text.split())
    common_word_count = sum(1 for word in common_words if word.lower() in formula_text.lower())
    if word_count > 0:
        common_ratio = common_word_count / word_count
        if common_ratio > 0.3:
            confidence *= (1 - common_ratio * 0.5)
    
    return min(1.0, max(0.0, confidence))

def extract_images_and_formulas_from_pdf_enhanced(uploaded_file) -> Tuple[str, Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Enhanced PDF extraction with table and formula detection"""
    if not FITZ_AVAILABLE:
        logger.error("PyMuPDF not available for PDF processing")
        try:
            text = uploaded_file.read().decode('utf-8', errors='ignore')
            return text, {}, [], []
        except Exception as e:
            logger.error(f"Error reading PDF as text: {e}")
            return "", {}, [], []
    
    text = ""
    images = {}
    formulas = []
    tables = []
    
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
            
            # Extract enhanced mathematical formulas from this page
            page_formulas = extract_enhanced_mathematical_formulas(page_text, page_num + 1)
            formulas.extend(page_formulas)
            
            # Extract tables from this page
            page_tables = extract_tables_from_text(page_text, page_num + 1)
            tables.extend(page_tables)
            
            # Extract images with better metadata
            try:
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n < 5:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_b64 = base64.b64encode(img_data).decode()
                        
                        # Enhanced image metadata
                        img_key = f"page_{page_num+1}_img_{img_index+1}"
                        images[img_key] = {
                            'data': img_b64,
                            'page': page_num + 1,
                            'width': pix.width,
                            'height': pix.height,
                            'type': 'image',
                            'size_bytes': len(img_data)
                        }
                    pix = None
            except Exception as e:
                logger.warning(f"Error extracting images from page {page_num+1}: {e}")
        
        doc.close()
        os.unlink(tmp_file_path)
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        try:
            uploaded_file.seek(0)
            text = uploaded_file.read().decode('utf-8', errors='ignore')
        except:
            text = "Error reading PDF content"
        return text, {}, [], []
    
    # Convert image format for backward compatibility
    image_dict = {}
    for key, img_info in images.items():
        if isinstance(img_info, dict):
            image_dict[key] = img_info['data']
        else:
            image_dict[key] = img_info
    
    # Combine formulas and tables into the formulas list for backward compatibility
    combined_elements = formulas + tables
    
    logger.info(f"Enhanced extraction: {len(formulas)} formulas, {len(tables)} tables, {len(image_dict)} images")
    return text, image_dict, combined_elements, tables

def extract_images_from_docx_enhanced(uploaded_file) -> Tuple[str, Dict[str, str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Enhanced DOCX extraction with table and formula detection"""
    if not DOCX_AVAILABLE:
        logger.error("python-docx not available for DOCX processing")
        try:
            uploaded_file.seek(0)
            return uploaded_file.read().decode('utf-8', errors='ignore'), {}, [], []
        except:
            return "Error reading DOCX content", {}, [], []
    
    text = ""
    images = {}
    formulas = []
    tables = []
    
    try:
        uploaded_file.seek(0)
        doc = docx.Document(uploaded_file)
        
        # Extract text
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        # Extract tables from document
        for table in doc.tables:
            table_text = ""
            for row in table.rows:
                row_text = " | ".join([cell.text.strip() for cell in row.cells])
                table_text += row_text + "\n"
            
            if len(table_text.strip()) > 50:
                tables.append({
                    'text': table_text.strip(),
                    'type': 'docx_table',
                    'rows': len(table.rows),
                    'columns': len(table.rows[0].cells) if table.rows else 0,
                    'confidence': 0.9,
                    'metadata': {
                        'source': 'docx_native_table',
                        'has_headers': True
                    }
                })
        
        # Extract formulas from text
        formulas = extract_enhanced_mathematical_formulas(text)
        
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
        return text, {}, [], []
    
    # Combine formulas and tables
    combined_elements = formulas + tables
    
    return text, images, combined_elements, tables

def extract_text_from_txt_enhanced(uploaded_file) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Enhanced TXT extraction with table and formula detection"""
    try:
        uploaded_file.seek(0)
        text = uploaded_file.read().decode('utf-8', errors='ignore')
        
        # Extract formulas and tables
        formulas = extract_enhanced_mathematical_formulas(text)
        tables = extract_tables_from_text(text)
        
        # Combine for backward compatibility
        combined_elements = formulas + tables
        
        return text, combined_elements, tables
    except Exception as e:
        logger.error(f"Error reading TXT file: {e}")
        return "Error reading TXT content", [], []

def analyze_document_intelligence_enhanced(text: str, images: Dict[str, str], formulas: List[Dict[str, Any]], tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Enhanced AI-powered document analysis with table and formula insights"""
    analysis = {
        'document_type': 'Unknown',
        'regulatory_framework': [],
        'key_entities': [],
        'complexity_score': 0,
        'compliance_indicators': [],
        'stakeholder_mentions': [],
        'risk_indicators': [],
        'timeline_references': [],
        'mathematical_complexity': 'Low',
        'formula_types': [],
        'table_count': len(tables),
        'regulatory_sections': [],
        'extracted_tables': tables,
        'extracted_formulas': formulas,
        'table_analysis': {},
        'formula_analysis': {}
    }
    
    if not text:
        return analysis
    
    # Enhanced document type detection
    doc_type_indicators = {
        'basel_regulatory': ['basel', 'committee', 'banking', 'supervision', 'mar21', 'standardised', 'approach', 'capital', 'requirement'],
        'regulatory': ['regulation', 'compliance', 'requirement', 'shall', 'must', 'supervisory', 'framework'],
        'policy': ['policy', 'procedure', 'guideline', 'standard', 'governance'],
        'technical': ['specification', 'technical', 'architecture', 'design', 'implementation', 'algorithm'],
        'business': ['business', 'process', 'workflow', 'operation', 'strategy', 'stakeholder'],
        'financial': ['capital', 'risk', 'credit', 'market', 'liquidity', 'operational', 'trading']
    }
    
    text_lower = text.lower()
    max_score = 0
    detected_type = 'Unknown'
    
    for doc_type, indicators in doc_type_indicators.items():
        score = sum(text_lower.count(indicator) for indicator in indicators)
        if score > max_score:
            max_score = score
            detected_type = doc_type.replace('_', ' ').title()
    
    analysis['document_type'] = detected_type
    
    # Enhanced regulatory framework detection
    frameworks = [
        'basel iii', 'basel iv', 'basel committee', 'mar21', 'sensitivities-based method',
        'solvency ii', 'mifid ii', 'crd iv', 'brrd', 'sox', 'gdpr', 'dodd-frank',
        'pci-dss', 'iso 27001', 'coso', 'market risk', 'credit risk', 
        'operational risk', 'liquidity risk', 'standardised approach'
    ]
    analysis['regulatory_framework'] = [fw for fw in frameworks if fw in text_lower]
    
    # Analyze extracted formulas
    if formulas:
        formula_types = set()
        high_complexity_formulas = []
        
        for formula in formulas:
            if isinstance(formula, dict):
                formula_type = formula.get('type', 'unknown')
                formula_types.add(formula_type)
                
                if formula.get('confidence', 0) > 0.7:
                    high_complexity_formulas.append(formula)
        
        analysis['formula_types'] = list(formula_types)
        analysis['formula_analysis'] = {
            'total_formulas': len(formulas),
            'high_confidence_formulas': len(high_complexity_formulas),
            'formula_types': list(formula_types),
            'avg_confidence': sum(f.get('confidence', 0) for f in formulas) / len(formulas),
            'complexity_indicators': {
                'has_greek_letters': any('greek' in f.get('type', '') for f in formulas),
                'has_summations': any('summation' in f.get('type', '') for f in formulas),
                'has_correlations': any('correlation' in f.get('type', '') for f in formulas),
                'has_risk_formulas': any('risk' in f.get('type', '') for f in formulas)
            }
        }
        
        # Determine mathematical complexity
        if len(formulas) > 30:
            analysis['mathematical_complexity'] = 'Very High'
        elif len(formulas) > 20:
            analysis['mathematical_complexity'] = 'High'
        elif len(formulas) > 10:
            analysis['mathematical_complexity'] = 'Medium'
        elif len(formulas) > 0:
            analysis['mathematical_complexity'] = 'Low'
    
    # Analyze extracted tables
    if tables:
        table_types = set()
        large_tables = []
        
        for table in tables:
            if isinstance(table, dict):
                table_type = table.get('type', 'unknown')
                table_types.add(table_type)
                
                rows = table.get('rows', 0)
                cols = table.get('columns', 0)
                if rows * cols > 20:
                    large_tables.append(table)
        
        analysis['table_analysis'] = {
            'total_tables': len(tables),
            'large_tables': len(large_tables),
            'table_types': list(table_types),
            'avg_confidence': sum(t.get('confidence', 0) for t in tables) / len(tables) if tables else 0,
            'structural_info': {
                'has_correlation_matrices': any('correlation' in t.get('text', '').lower() for t in tables),
                'has_risk_weights': any('risk weight' in t.get('text', '').lower() for t in tables),
                'has_bucket_definitions': any('bucket' in t.get('text', '').lower() for t in tables),
                'total_data_points': sum(t.get('rows', 0) * t.get('columns', 0) for t in tables)
            }
        }
    
    # Enhanced complexity scoring
    complexity_factors = [
        len(text) > 100000,                        # Very large document
        len(images) > 20,                          # Many images
        len(formulas) > 15,                        # Complex formulas
        len(tables) > 10,                          # Many tables
        len(analysis['regulatory_framework']) > 3,  # Multiple regulations
        'basel' in text_lower and 'mar21' in text_lower,  # Basel-specific complexity
        len(analysis.get('formula_types', [])) > 5,  # Diverse formula types
        any(t.get('rows', 0) * t.get('columns', 0) > 50 for t in tables),  # Large tables
    ]
    analysis['complexity_score'] = sum(complexity_factors) / len(complexity_factors)
    
    return analysis

def process_document_enhanced(uploaded_file, extract_images: bool = True, extract_formulas: bool = True, extract_tables: bool = True) -> Tuple[str, Dict[str, str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Enhanced document processing with comprehensive table and formula extraction
    """
    if uploaded_file is None:
        return "", {}, [], {}
    
    try:
        file_type = uploaded_file.type
        logger.info(f"Processing file: {uploaded_file.name} ({file_type})")
        
        if file_type == "application/pdf":
            document_text, extracted_images, extracted_formulas, extracted_tables = extract_images_and_formulas_from_pdf_enhanced(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            document_text, extracted_images, extracted_formulas, extracted_tables = extract_images_from_docx_enhanced(uploaded_file)
        elif file_type == "text/plain":
            document_text, extracted_formulas, extracted_tables = extract_text_from_txt_enhanced(uploaded_file)
            extracted_images = {}
        else:
            logger.warning(f"Unsupported file type: {file_type}")
            uploaded_file.seek(0)
            document_text = str(uploaded_file.read(), "utf-8", errors='ignore')
            extracted_images = {}
            extracted_formulas = extract_enhanced_mathematical_formulas(document_text) if extract_formulas else []
            extracted_tables = extract_tables_from_text(document_text) if extract_tables else []
        
        # Apply extraction options
        if not extract_images:
            extracted_images = {}
        if not extract_formulas:
            # Remove formulas but keep tables if they were extracted
            extracted_formulas = [item for item in extracted_formulas if item.get('type') in ['extracted_table', 'docx_table']]
        if not extract_tables:
            # Remove tables but keep formulas
            extracted_formulas = [item for item in extracted_formulas if item.get('type') not in ['extracted_table', 'docx_table']]
            extracted_tables = []
        
        # Separate tables from formulas for analysis
        if not hasattr(extracted_tables, '__len__'):
            extracted_tables = [item for item in extracted_formulas if item.get('type') in ['extracted_table', 'docx_table']]
        
        # Enhanced document analysis with table and formula insights
        document_analysis = analyze_document_intelligence_enhanced(document_text, extracted_images, extracted_formulas, extracted_tables)
        
        logger.info(f"Enhanced extraction: {len(extracted_formulas)} elements ({document_analysis['formula_analysis'].get('total_formulas', 0)} formulas + {len(extracted_tables)} tables), {len(extracted_images)} images")
        
        return document_text, extracted_images, extracted_formulas, document_analysis
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        if 'st' in globals():
            st.error(f"Error processing document: {str(e)}")
        return "", {}, [], {}

def create_table_formula_visualization(tables: List[Dict[str, Any]], formulas: List[Dict[str, Any]]) -> str:
    """
    Create HTML visualization for extracted tables and formulas
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .extraction-container { 
                max-width: 1200px; margin: 0 auto; padding: 20px; 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .section { 
                margin-bottom: 30px; padding: 20px; 
                background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea;
            }
            .section h3 { 
                color: #2c3e50; margin-top: 0; display: flex; align-items: center;
            }
            .table-container { 
                background: white; padding: 15px; margin: 10px 0; 
                border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                overflow-x: auto;
            }
            .formula-container { 
                background: #f1f3f4; padding: 15px; margin: 10px 0; 
                border-radius: 6px; border: 2px solid #667eea;
                font-family: 'Courier New', monospace;
            }
            .metadata { 
                font-size: 0.85em; color: #666; margin-top: 10px; 
            }
            .confidence-score {
                display: inline-block; padding: 2px 8px; border-radius: 12px;
                font-size: 0.8em; font-weight: bold;
            }
            .high-confidence { background: #d4edda; color: #155724; }
            .medium-confidence { background: #fff3cd; color: #856404; }
            .low-confidence { background: #f8d7da; color: #721c24; }
            .extracted-table { border-collapse: collapse; width: 100%; margin: 10px 0; }
            .extracted-table th, .extracted-table td { 
                border: 1px solid #ddd; padding: 8px; text-align: left; 
            }
            .extracted-table th { background: #667eea; color: white; }
            .formula-text { 
                background: white; padding: 10px; border-radius: 4px; 
                margin: 8px 0; font-size: 1.1em; line-height: 1.6;
            }
            .icon { margin-right: 8px; }
        </style>
    </head>
    <body>
        <div class="extraction-container">
            <h2>Extracted Tables and Mathematical Formulas</h2>
    """
    
    # Add tables section
    if tables:
        html_content += """
            <div class="section">
                <h3><span class="icon">📊</span>Extracted Tables ({} found)</h3>
        """.format(len(tables))
        
        for i, table in enumerate(tables):
            confidence = table.get('confidence', 0)
            confidence_class = 'high-confidence' if confidence > 0.7 else 'medium-confidence' if confidence > 0.5 else 'low-confidence'
            
            html_content += f"""
                <div class="table-container">
                    <h4>Table {i+1} 
                        <span class="confidence-score {confidence_class}">
                            Confidence: {confidence:.1%}
                        </span>
                    </h4>
                    <div class="formula-text">{table.get('text', '').replace('|', ' | ')}</div>
                    <div class="metadata">
                        Rows: {table.get('rows', 0)} | Columns: {table.get('columns', 0)} | 
                        Type: {table.get('type', 'unknown')} | Page: {table.get('page', 'N/A')}
                    </div>
                </div>
            """
        
        html_content += "</div>"
    
    # Add formulas section
    if formulas:
        html_content += """
            <div class="section">
                <h3><span class="icon">∑</span>Mathematical Formulas ({} found)</h3>
        """.format(len(formulas))
        
        # Group formulas by type
        formula_groups = {}
        for formula in formulas:
            if formula.get('type') not in ['extracted_table', 'docx_table']:
                formula_type = formula.get('type', 'unknown')
                if formula_type not in formula_groups:
                    formula_groups[formula_type] = []
                formula_groups[formula_type].append(formula)
        
        for formula_type, formula_list in formula_groups.items():
            type_name = formula_type.replace('_', ' ').title()
            html_content += f"""
                <h4>{type_name} ({len(formula_list)} formulas)</h4>
            """
            
            for i, formula in enumerate(formula_list):
                confidence = formula.get('confidence', 0)
                confidence_class = 'high-confidence' if confidence > 0.7 else 'medium-confidence' if confidence > 0.5 else 'low-confidence'
                
                html_content += f"""
                    <div class="formula-container">
                        <div style="font-weight: bold; margin-bottom: 8px;">
                            Formula {i+1} 
                            <span class="confidence-score {confidence_class}">
                                Confidence: {confidence:.1%}
                            </span>
                        </div>
                        <div class="formula-text">{formula.get('text', '')}</div>
                        <div class="metadata">
                            Page: {formula.get('page', 'N/A')} | 
                            Length: {len(formula.get('text', ''))} chars |
                            Context: {formula.get('context', '')[:100]}...
                        </div>
                    </div>
                """
        
        html_content += "</div>"
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    return html_content

def render_enhanced_extraction_results(document_analysis: Dict[str, Any], extracted_images: Dict[str, str]):
    """
    Render enhanced extraction results in Streamlit
    """
    if not document_analysis:
        return
    
    # Display enhanced metrics
    st.subheader("Enhanced Extraction Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        formula_count = document_analysis.get('formula_analysis', {}).get('total_formulas', 0)
        st.metric("Mathematical Formulas", formula_count)
    
    with col2:
        table_count = document_analysis.get('table_count', 0)
        st.metric("Extracted Tables", table_count)
    
    with col3:
        image_count = len(extracted_images)
        st.metric("Images", image_count)
    
    with col4:
        complexity = document_analysis.get('mathematical_complexity', 'Unknown')
        st.metric("Math Complexity", complexity)
    
    # Display formula analysis
    formula_analysis = document_analysis.get('formula_analysis', {})
    if formula_analysis:
        st.subheader("Formula Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Formula Types Found:** {len(formula_analysis.get('formula_types', []))}")
            for formula_type in formula_analysis.get('formula_types', [])[:5]:
                st.write(f"• {formula_type.replace('_', ' ').title()}")
        
        with col2:
            avg_confidence = formula_analysis.get('avg_confidence', 0)
            st.success(f"**Average Confidence:** {avg_confidence:.1%}")
            high_conf = formula_analysis.get('high_confidence_formulas', 0)
            st.write(f"High confidence formulas: {high_conf}")
        
        with col3:
            complexity_indicators = formula_analysis.get('complexity_indicators', {})
            st.warning("**Complexity Indicators:**")
            for indicator, present in complexity_indicators.items():
                icon = "✅" if present else "❌"
                st.write(f"{icon} {indicator.replace('_', ' ').title()}")
    
    # Display table analysis
    table_analysis = document_analysis.get('table_analysis', {})
    if table_analysis:
        st.subheader("Table Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Total Tables:** {table_analysis.get('total_tables', 0)}")
            st.info(f"**Large Tables:** {table_analysis.get('large_tables', 0)}")
            st.info(f"**Average Confidence:** {table_analysis.get('avg_confidence', 0):.1%}")
        
        with col2:
            structural_info = table_analysis.get('structural_info', {})
            st.success("**Structural Elements Found:**")
            for element, present in structural_info.items():
                if isinstance(present, bool):
                    icon = "✅" if present else "❌"
                    st.write(f"{icon} {element.replace('_', ' ').title()}")
                else:
                    st.write(f"• {element.replace('_', ' ').title()}: {present}")
    
    # Display extraction visualization
    extracted_tables = document_analysis.get('extracted_tables', [])
    extracted_formulas = document_analysis.get('extracted_formulas', [])
    
    if extracted_tables or extracted_formulas:
        with st.expander("📊 View Extracted Tables and Formulas", expanded=False):
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Summary View", "Detailed Tables", "Detailed Formulas"])
            
            with tab1:
                # Create and display HTML visualization
                html_viz = create_table_formula_visualization(extracted_tables, extracted_formulas)
                st.components.v1.html(html_viz, height=600, scrolling=True)
            
            with tab2:
                if extracted_tables:
                    for i, table in enumerate(extracted_tables):
                        st.write(f"**Table {i+1}** (Confidence: {table.get('confidence', 0):.1%})")
                        st.code(table.get('text', ''), language='text')
                        st.caption(f"Rows: {table.get('rows', 0)}, Columns: {table.get('columns', 0)}, Page: {table.get('page', 'N/A')}")
                        st.divider()
                else:
                    st.info("No tables extracted from document")
            
            with tab3:
                formula_only = [f for f in extracted_formulas if f.get('type') not in ['extracted_table', 'docx_table']]
                if formula_only:
                    # Group by formula type
                    formula_groups = {}
                    for formula in formula_only:
                        formula_type = formula.get('type', 'unknown')
                        if formula_type not in formula_groups:
                            formula_groups[formula_type] = []
                        formula_groups[formula_type].append(formula)
                    
                    for formula_type, formulas in formula_groups.items():
                        st.subheader(f"{formula_type.replace('_', ' ').title()} ({len(formulas)})")
                        for i, formula in enumerate(formulas):
                            confidence = formula.get('confidence', 0)
                            confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
                            st.write(f"**Formula {i+1}** {confidence_color} {confidence:.1%}")
                            st.code(formula.get('text', ''), language='text')
                            if formula.get('context'):
                                st.caption(f"Context: {formula.get('context', '')[:200]}...")
                            st.divider()
                else:
                    st.info("No mathematical formulas extracted from document")

# Update the main process_document function to use enhanced version
def process_document(uploaded_file, extract_images: bool = True, extract_formulas: bool = True) -> Tuple[str, Dict[str, str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main document processing function - now with enhanced table and formula extraction
    """
    # Add table extraction option
    extract_tables = extract_formulas  # Link table extraction to formula extraction for UI compatibility
    
    return process_document_enhanced(uploaded_file, extract_images, extract_formulas, extract_tables)
