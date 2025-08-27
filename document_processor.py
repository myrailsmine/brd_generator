"""
Document Processing Utilities
"""

import streamlit as st
import tempfile
import os
import base64
import re
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

"""
Document Processing Utilities
Enhanced for regulatory documents with complex mathematical formulas and structured content
"""

import streamlit as st
import tempfile
import os
import base64
import re
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

def extract_mathematical_formulas_advanced(text: str, page_num: int = None) -> List[Dict[str, Any]]:
    """
    Enhanced mathematical formula extraction for regulatory documents
    """
    formulas = []
    
    # Enhanced mathematical patterns for Basel/regulatory documents
    formula_patterns = [
        # Greek letters and mathematical symbols
        (r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]+', 'greek_symbols'),
        
        # Mathematical operators and relations
        (r'[∑∏∫∆∇±≤≥≠≈∞√∂∈∉⊂⊃∪∩]', 'mathematical_operators'),
        
        # Complex formulas with subscripts/superscripts patterns
        (r'\w+_\{[^}]+\}|\w+\^\{[^}]+\}|\w+_\d+|\w+\^\d+', 'subscript_superscript'),
        
        # Correlation formulas (common in Basel documents)
        (r'ρ[_\w\d]+\s*=\s*[\d.%]+', 'correlation_formulas'),
        
        # Risk weight formulas
        (r'RW[_\w\d]*\s*=\s*[\d.%]+', 'risk_weight_formulas'),
        
        # Percentage formulas
        (r'\d+\.?\d*\s*%|\d+\s*percentage\s*points?', 'percentage_values'),
        
        # Mathematical expressions with parentheses
        (r'\([^)]*[+\-*/=√∑∏]\s*[^)]*\)', 'parenthetical_expressions'),
        
        # Capital requirement formulas
        (r'K[_\w\d]*\s*=\s*[^.!?]*[+\-*/√∑∏][^.!?]*', 'capital_requirement_formulas'),
        
        # Basel-specific formulas (sensitivity calculations)
        (r'[Ss]ensitivity\s*=\s*[^.!?]*', 'sensitivity_formulas'),
        
        # PV01, CS01 formulas
        (r'(PV01|CS01|VaR)\s*=\s*[^.!?]*', 'risk_sensitivity_formulas'),
        
        # Matrix/vector notation
        (r'\[[^\]]*[+\-*/]\s*[^\]]*\]|\{[^}]*[+\-*/]\s*[^}]*\}', 'matrix_vector_notation'),
        
        # Square root expressions
        (r'√\([^)]+\)|sqrt\([^)]+\)', 'square_root_expressions'),
        
        # Exponential expressions
        (r'e\^[^\s]+|exp\([^)]+\)', 'exponential_expressions'),
        
        # Floor/ceiling functions
        (r'⌊[^⌋]*⌋|⌈[^⌉]*⌉|floor\([^)]+\)|ceiling\([^)]+\)', 'floor_ceiling_functions'),
    ]
    
    for pattern, formula_type in formula_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            formula_text = match.group()
            # Skip very short matches that might be false positives
            if len(formula_text.strip()) >= 2:
                context_start = max(0, match.start() - 100)
                context_end = min(len(text), match.end() + 100)
                context = text[context_start:context_end]
                
                formulas.append({
                    'text': formula_text.strip(),
                    'type': formula_type,
                    'page': page_num,
                    'position': match.start(),
                    'context': context.strip(),
                    'confidence': calculate_formula_confidence(formula_text, formula_type)
                })
    
    # Remove duplicates and sort by confidence
    unique_formulas = []
    seen_formulas = set()
    for formula in formulas:
        formula_key = (formula['text'], formula['type'])
        if formula_key not in seen_formulas and formula['confidence'] > 0.3:
            seen_formulas.add(formula_key)
            unique_formulas.append(formula)
    
    return sorted(unique_formulas, key=lambda x: x['confidence'], reverse=True)

def calculate_formula_confidence(formula_text: str, formula_type: str) -> float:
    """Calculate confidence score for extracted formulas"""
    confidence = 0.5  # Base confidence
    
    # Increase confidence for certain characteristics
    if len(formula_text) > 10:
        confidence += 0.2
    
    if any(char in formula_text for char in ['=', '∑', '∏', '∫', '√', 'ρ']):
        confidence += 0.3
    
    if formula_type in ['capital_requirement_formulas', 'risk_weight_formulas', 'correlation_formulas']:
        confidence += 0.2
    
    if re.search(r'\d+\.?\d*\s*%', formula_text):
        confidence += 0.1
    
    return min(confidence, 1.0)

def extract_structured_tables(text: str, page_num: int = None) -> List[Dict[str, Any]]:
    """
    Extract structured table information from text
    """
    tables = []
    
    # Look for table patterns common in regulatory documents
    table_patterns = [
        # Table headers with multiple columns
        r'((?:[A-Z][a-z\s]+\s*\|\s*){2,}[A-Z][a-z\s]+)',
        
        # Numbered table references
        r'Table\s+\d+[^\n]*\n((?:[^\n]*\|[^\n]*\n){2,})',
        
        # Basel-specific table patterns
        r'(Bucket\s+number[^\n]*\n(?:[^\n]*\|[^\n]*\n){2,})',
        
        # Risk weight tables
        r'(Risk\s+weight[^\n]*\n(?:[^\n]*\|[^\n]*\n){2,})',
    ]
    
    for pattern in table_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            table_text = match.group()
            if len(table_text.strip()) > 50:  # Only consider substantial tables
                tables.append({
                    'text': table_text.strip(),
                    'page': page_num,
                    'position': match.start(),
                    'type': 'regulatory_table',
                    'rows': len([line for line in table_text.split('\n') if '|' in line])
                })
    
    return tables

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
                            'type': 'image'
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
        return text, {}, []
    
    # Convert image format for backward compatibility
    image_dict = {}
    for key, img_info in images.items():
        if isinstance(img_info, dict):
            image_dict[key] = img_info['data']
        else:
            image_dict[key] = img_info
    
    logger.info(f"Extracted {len(formulas)} mathematical elements and {len(image_dict)} images")
    return text, image_dict, formulas

def extract_images_from_docx(uploaded_file) -> Tuple[str, Dict[str, str]]:
    """Extract text and images from DOCX"""
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
        
        # Extract text
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
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
    """Extract text from TXT file"""
    try:
        uploaded_file.seek(0)
        text = uploaded_file.read().decode('utf-8', errors='ignore')
        return text
    except Exception as e:
        logger.error(f"Error reading TXT file: {e}")
        return "Error reading TXT content"

def display_image_from_base64(img_b64: str, caption: str = "", max_width: int = None):
    """Display image from base64 string"""
    try:
        img_data = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_data))
        
        if max_width:
            img.thumbnail((max_width, max_width))
        
        st.image(img, caption=caption, use_container_width=True if not max_width else False)
    except Exception as e:
        logger.error(f"Error displaying image: {str(e)}")
        st.error(f"Error displaying image: {caption}")

def render_content_with_images(content: str, images: Dict[str, str]):
    """Render text content and replace image references with actual images"""
    if not content:
        return
    
    # Split content by image references
    parts = re.split(r'\[IMAGE:\s*([^\]]+)\]', content)
    
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Text content
            if part.strip():
                st.markdown(part)
        else:  # Image reference
            image_key = part.strip()
            if image_key in images:
                display_image_from_base64(images[image_key], caption=image_key)
            else:
                st.info(f"Image reference: {image_key} (not found)")

def analyze_document_intelligence(text: str, images: Dict[str, str], formulas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Advanced AI-powered document analysis for regulatory documents"""
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
        'table_count': 0,
        'regulatory_sections': []
    }
    
    if not text:
        return analysis
    
    # Document type detection with enhanced patterns
    doc_type_indicators = {
        'regulatory': ['regulation', 'compliance', 'requirement', 'shall', 'must', 'basel', 'committee', 'supervisory'],
        'policy': ['policy', 'procedure', 'guideline', 'standard', 'framework'],
        'technical': ['specification', 'technical', 'architecture', 'design', 'implementation'],
        'business': ['business', 'process', 'workflow', 'operation', 'strategy'],
        'financial': ['capital', 'risk', 'credit', 'market', 'liquidity', 'operational']
    }
    
    text_lower = text.lower()
    max_score = 0
    detected_type = 'Unknown'
    
    for doc_type, indicators in doc_type_indicators.items():
        score = sum(text_lower.count(indicator) for indicator in indicators)
        if score > max_score:
            max_score = score
            detected_type = doc_type.title()
    
    analysis['document_type'] = detected_type
    
    # Enhanced regulatory framework detection
    frameworks = [
        'basel', 'basel iii', 'basel iv', 'solvency', 'mifid', 'crd', 'brrd',
        'sox', 'gdpr', 'dodd-frank', 'pci-dss', 'iso 27001', 'coso',
        'market risk', 'credit risk', 'operational risk', 'liquidity risk'
    ]
    analysis['regulatory_framework'] = [fw for fw in frameworks if fw in text_lower]
    
    # Mathematical complexity analysis based on extracted formulas
    if formulas:
        formula_types = set()
        high_complexity_types = ['capital_requirement_formulas', 'correlation_formulas', 'risk_weight_formulas']
        
        for formula in formulas:
            if isinstance(formula, dict):
                formula_types.add(formula.get('type', 'unknown'))
            else:
                formula_types.add('basic_formula')
        
        analysis['formula_types'] = list(formula_types)
        
        # Determine mathematical complexity
        if len(formulas) > 20:
            analysis['mathematical_complexity'] = 'Very High'
        elif len(formulas) > 10:
            analysis['mathematical_complexity'] = 'High'
        elif len(formulas) > 5:
            analysis['mathematical_complexity'] = 'Medium'
        elif len(formulas) > 0:
            analysis['mathematical_complexity'] = 'Low'
        
        # Check for high complexity formula types
        if any(ftype in high_complexity_types for ftype in formula_types):
            if analysis['mathematical_complexity'] in ['Low', 'Medium']:
                analysis['mathematical_complexity'] = 'High'
    
    # Extract regulatory sections
    section_patterns = [
        r'(\d+\.\d+(?:\.\d+)?\s+[A-Z][^.!?]*)',  # Numbered sections
        r'([A-Z][A-Z\s]+:?\s*[A-Z][^.!?]*)',      # All caps headings
        r'(MAR\d+\.\d+[^.!?]*)',                   # Basel MAR references
    ]
    
    for pattern in section_patterns:
        matches = re.findall(pattern, text)
        analysis['regulatory_sections'].extend(matches[:20])  # Limit to prevent overflow
    
    # Table counting
    analysis['table_count'] = text.count('Table') + text.count('table')
    
    # Enhanced entity extraction with Basel-specific terms
    entity_patterns = [
        r'\b[A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+\b',     # Names
        r'\b\d{2,4}[-/]\d{2}[-/]\d{2,4}\b',       # Dates
        r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',        # Money
        r'\b\d+\.?\d*\s*%\b',                      # Percentages
        r'\bMAR\d+\.\d+\b',                        # Basel references
        r'\bBucket\s+\d+\b',                       # Bucket references
    ]
    
    for pattern in entity_patterns:
        try:
            matches = re.findall(pattern, text)
            analysis['key_entities'].extend(matches[:15])  # Limit results
        except Exception as e:
            logger.warning(f"Error in entity extraction: {e}")
    
    # Enhanced complexity scoring
    complexity_factors = [
        len(text) > 50000,                         # Large document
        len(images) > 10,                          # Many images
        len(formulas) > 5,                         # Complex formulas
        len(analysis['regulatory_framework']) > 2,  # Multiple regulations
        analysis['table_count'] > 10,              # Many tables
        'correlation' in text_lower,               # Complex correlations
        'curvature' in text_lower,                 # Advanced risk concepts
        len(analysis['regulatory_sections']) > 10,  # Many sections
    ]
    analysis['complexity_score'] = sum(complexity_factors) / len(complexity_factors)
    
    return analysis

def process_document(uploaded_file, extract_images: bool = True, extract_formulas: bool = True) -> Tuple[str, Dict[str, str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Enhanced document processing with better formula and structure extraction
    """
    if uploaded_file is None:
        return "", {}, [], {}
    
    try:
        file_type = uploaded_file.type
        logger.info(f"Processing file: {uploaded_file.name} ({file_type})")
        
        if file_type == "application/pdf":
            document_text, extracted_images, extracted_formulas = extract_images_and_formulas_from_pdf(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            document_text, extracted_images = extract_images_from_docx(uploaded_file)
            # Extract formulas from DOCX text as well
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
        
        logger.info(f"Document processed successfully: {len(document_text)} chars, {len(extracted_images)} images, {len(extracted_formulas)} formulas")
        
        return document_text, extracted_images, extracted_formulas, document_analysis
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        if 'st' in globals():
            st.error(f"Error processing document: {str(e)}")
        return "", {}, [], {}
