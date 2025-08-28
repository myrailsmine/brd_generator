"""
Enhanced Document Processing Utilities
Improved for better extraction of images, tables, and mathematical formulas
"""
import streamlit as st
import tempfile
import os
import base64
import re
import numpy as np
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

def extract_mathematical_formulas_enhanced(text: str, page_num: int = None) -> List[Dict[str, Any]]:
    """
    Enhanced mathematical formula extraction with simpler, more reliable patterns
    """
    formulas = []
    
    # Simplified and more practical mathematical patterns
    formula_patterns = [
        # Basic mathematical expressions with equals
        (r'[A-Za-z_]\w*\s*=\s*[^.!?\n]{5,50}[+\-*/\^√∑∫]', 'equation_with_equals'),
        
        # Percentage formulas
        (r'\d+\.?\d*\s*%|\d+\s*percent', 'percentage_values'),
        
        # Fractions and ratios
        (r'\b\d+/\d+\b|\b\d+:\d+\b', 'fractions_ratios'),
        
        # Mathematical functions
        (r'\b(sin|cos|tan|log|ln|exp|sqrt|max|min|sum)\s*\([^)]+\)', 'mathematical_functions'),
        
        # Greek letters (common ones)
        (r'\b(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|phi|chi|omega)\b', 'greek_letters'),
        
        # Risk/Finance formulas (simplified)
        (r'\b(RW|VaR|PV01|CS01|CVA|EAD|LGD)\s*[=:]?\s*[^.!?\n]{3,30}', 'risk_formulas'),
        
        # Capital formulas
        (r'\bK\w*\s*=\s*[^.!?\n]{5,50}', 'capital_formulas'),
        
        # Numbers with operators
        (r'\b\d+\.?\d*\s*[+\-*/×÷]\s*\d+\.?\d*', 'arithmetic_operations'),
        
        # Squared/power notation
        (r'\w+\^2|\w+\^n|\w+²', 'power_notation'),
        
        # Square root expressions
        (r'√\w+|sqrt\(\w+\)', 'square_roots'),
        
        # Correlation patterns (simplified)
        (r'ρ\s*=\s*\d+\.?\d*|rho\s*=\s*\d+\.?\d*', 'correlation_simple'),
        
        # Summation patterns
        (r'∑|sum\s*of|total\s*of', 'summation_patterns'),
        
        # Matrix notation (simplified)
        (r'\[\s*\d+[,\s]+\d+\s*\]', 'matrix_simple'),
        
        # Exponential expressions
        (r'e\^\w+|exp\(\w+\)', 'exponential_simple'),
    ]
    
    for pattern, formula_type in formula_patterns:
        try:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                formula_text = match.group().strip()
                
                # Skip very short matches or common words
                if len(formula_text) >= 3 and not formula_text.lower() in ['the', 'and', 'for', 'are', 'not']:
                    context_start = max(0, match.start() - 80)
                    context_end = min(len(text), match.end() + 80)
                    context = text[context_start:context_end].strip()
                    
                    confidence = calculate_formula_confidence_simple(formula_text, formula_type, context)
                    
                    if confidence > 0.3:  # Only include reasonably confident matches
                        formulas.append({
                            'text': formula_text,
                            'type': formula_type,
                            'page': page_num,
                            'position': match.start(),
                            'context': context,
                            'confidence': confidence
                        })
        except Exception as e:
            logger.warning(f"Error processing pattern {formula_type}: {e}")
            continue
    
    # Remove duplicates and sort by confidence
    unique_formulas = []
    seen_formulas = set()
    
    for formula in formulas:
        formula_key = (formula['text'].lower(), formula['type'])
        if formula_key not in seen_formulas:
            seen_formulas.add(formula_key)
            unique_formulas.append(formula)
    
    return sorted(unique_formulas, key=lambda x: x['confidence'], reverse=True)

def calculate_formula_confidence_simple(formula_text: str, formula_type: str, context: str) -> float:
    """Calculate confidence score for extracted formulas with simpler logic"""
    confidence = 0.4  # Base confidence
    
    # Boost confidence based on formula characteristics
    if '=' in formula_text:
        confidence += 0.3
    
    if any(op in formula_text for op in ['+', '-', '*', '/', '^', '√', '∑']):
        confidence += 0.2
    
    if re.search(r'\d', formula_text):
        confidence += 0.1
    
    # Context-based confidence
    context_lower = context.lower()
    if any(word in context_lower for word in ['formula', 'equation', 'calculate', 'computation']):
        confidence += 0.2
    
    # Formula type specific boosts
    high_value_types = ['risk_formulas', 'capital_formulas', 'equation_with_equals']
    if formula_type in high_value_types:
        confidence += 0.2
    
    return min(confidence, 1.0)

def extract_tables_as_images_from_pdf(doc, page_num: int) -> List[Dict[str, Any]]:
    """Extract tables as images from PDF pages"""
    page = doc.load_page(page_num)
    tables = []
    
    try:
        # Find table-like structures using text blocks
        blocks = page.get_text("dict")["blocks"]
        
        # Look for rectangular regions that might contain tables
        potential_tables = []
        
        for block in blocks:
            if "lines" in block:
                # Check if this block has table-like characteristics
                lines = block["lines"]
                if len(lines) >= 3:  # At least 3 lines might be a table
                    # Check for alignment and spacing patterns
                    spans_per_line = [len(line.get("spans", [])) for line in lines]
                    if len(set(spans_per_line)) <= 2 and max(spans_per_line) >= 2:  # Consistent columns
                        bbox = block["bbox"]
                        potential_tables.append({
                            'bbox': bbox,
                            'lines': len(lines),
                            'confidence': 0.7 if max(spans_per_line) >= 3 else 0.5
                        })
        
        # Extract images of identified table regions
        for i, table_region in enumerate(potential_tables):
            try:
                bbox = table_region['bbox']
                # Expand bbox slightly for better capture
                expanded_bbox = [
                    max(0, bbox[0] - 10),
                    max(0, bbox[1] - 10),
                    min(page.rect.width, bbox[2] + 10),
                    min(page.rect.height, bbox[3] + 10)
                ]
                
                # Create a clip rectangle and render
                clip = fitz.Rect(expanded_bbox)
                mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat, clip=clip)
                
                if pix.width > 50 and pix.height > 50:  # Reasonable size check
                    img_data = pix.tobytes("png")
                    img_b64 = base64.b64encode(img_data).decode()
                    
                    tables.append({
                        'image_data': img_b64,
                        'bbox': bbox,
                        'page': page_num + 1,
                        'type': 'table_image',
                        'lines': table_region['lines'],
                        'confidence': table_region['confidence'],
                        'key': f"table_p{page_num+1}_t{i+1}"
                    })
                
                pix = None
                
            except Exception as e:
                logger.warning(f"Error extracting table image {i} from page {page_num+1}: {e}")
                continue
                
    except Exception as e:
        logger.warning(f"Error finding tables on page {page_num+1}: {e}")
    
    return tables

def extract_formulas_as_images_from_pdf(doc, page_num: int, text_formulas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract mathematical formulas as images from PDF pages"""
    page = doc.load_page(page_num)
    formula_images = []
    
    try:
        # Get all text instances on the page
        text_instances = page.get_text("dict")
        
        # For each detected text formula, try to find its location and extract as image
        for formula in text_formulas:
            if formula.get('page') == page_num + 1:
                try:
                    # Search for the formula text on the page
                    text_instances_list = page.search_for(formula['text'][:20])  # Search for first 20 chars
                    
                    if text_instances_list:
                        for rect in text_instances_list:
                            # Expand the rectangle to capture surrounding mathematical context
                            expanded_rect = fitz.Rect(
                                max(0, rect.x0 - 20),
                                max(0, rect.y0 - 10), 
                                min(page.rect.width, rect.x1 + 20),
                                min(page.rect.height, rect.y1 + 10)
                            )
                            
                            # Render the formula region as image
                            mat = fitz.Matrix(3, 3)  # 3x zoom for high quality
                            pix = page.get_pixmap(matrix=mat, clip=expanded_rect)
                            
                            if pix.width > 30 and pix.height > 15:  # Minimum size check
                                img_data = pix.tobytes("png")
                                img_b64 = base64.b64encode(img_data).decode()
                                
                                formula_images.append({
                                    'image_data': img_b64,
                                    'bbox': [rect.x0, rect.y0, rect.x1, rect.y1],
                                    'page': page_num + 1,
                                    'type': 'formula_image',
                                    'formula_text': formula['text'],
                                    'formula_type': formula['type'],
                                    'confidence': formula['confidence'],
                                    'key': f"formula_p{page_num+1}_f{len(formula_images)+1}"
                                })
                            
                            pix = None
                            break  # Take the first match
                            
                except Exception as e:
                    logger.warning(f"Error extracting formula image on page {page_num+1}: {e}")
                    continue
    
    except Exception as e:
        logger.warning(f"Error processing formulas as images on page {page_num+1}: {e}")
    
    return formula_images

def extract_diagrams_and_charts_from_pdf(doc, page_num: int) -> List[Dict[str, Any]]:
    """Extract diagrams, charts, and complex visual elements from PDF"""
    page = doc.load_page(page_num)
    diagrams = []
    
    try:
        # Get page drawings (vector graphics)
        drawings = page.get_drawings()
        
        if drawings:
            # Group nearby drawings that might form charts/diagrams
            drawing_groups = []
            current_group = []
            
            for drawing in drawings:
                if current_group:
                    # Check if this drawing is close to the current group
                    last_rect = current_group[-1]['rect']
                    current_rect = drawing['rect']
                    
                    # If drawings are close (within 100 units), group them
                    if (abs(current_rect.x0 - last_rect.x1) < 100 and 
                        abs(current_rect.y0 - last_rect.y0) < 50):
                        current_group.append(drawing)
                    else:
                        if len(current_group) >= 2:  # At least 2 drawings make a diagram
                            drawing_groups.append(current_group)
                        current_group = [drawing]
                else:
                    current_group = [drawing]
            
            # Add the last group
            if len(current_group) >= 2:
                drawing_groups.append(current_group)
            
            # Extract images of diagram groups
            for i, group in enumerate(drawing_groups):
                try:
                    # Calculate bounding box for the entire group
                    all_rects = [d['rect'] for d in group]
                    min_x0 = min(r.x0 for r in all_rects)
                    min_y0 = min(r.y0 for r in all_rects)
                    max_x1 = max(r.x1 for r in all_rects)
                    max_y1 = max(r.y1 for r in all_rects)
                    
                    # Expand for context
                    expanded_bbox = fitz.Rect(
                        max(0, min_x0 - 15),
                        max(0, min_y0 - 15),
                        min(page.rect.width, max_x1 + 15),
                        min(page.rect.height, max_y1 + 15)
                    )
                    
                    # Render as high-quality image
                    mat = fitz.Matrix(2.5, 2.5)  # High resolution
                    pix = page.get_pixmap(matrix=mat, clip=expanded_bbox)
                    
                    if pix.width > 100 and pix.height > 50:  # Reasonable diagram size
                        img_data = pix.tobytes("png")
                        img_b64 = base64.b64encode(img_data).decode()
                        
                        diagrams.append({
                            'image_data': img_b64,
                            'bbox': [min_x0, min_y0, max_x1, max_y1],
                            'page': page_num + 1,
                            'type': 'diagram_image',
                            'elements_count': len(group),
                            'confidence': min(0.9, 0.5 + len(group) * 0.1),
                            'key': f"diagram_p{page_num+1}_d{i+1}"
                        })
                    
                    pix = None
                    
                except Exception as e:
                    logger.warning(f"Error extracting diagram {i} from page {page_num+1}: {e}")
                    continue
    
    except Exception as e:
        logger.warning(f"Error extracting diagrams from page {page_num+1}: {e}")
    
    return diagrams

def extract_all_images_enhanced(doc, page_num: int) -> List[Dict[str, Any]]:
    """Enhanced image extraction with better metadata and error handling"""
    page = doc.load_page(page_num)
    images = []
    
    try:
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                # Get image reference
                xref = img[0]
                
                # Get the actual image
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Convert to base64
                img_b64 = base64.b64encode(image_bytes).decode()
                
                # Get image position on page
                image_rects = page.get_image_rects(xref)
                bbox = image_rects[0] if image_rects else [0, 0, 100, 100]
                
                # Get image dimensions
                try:
                    img_pil = Image.open(BytesIO(image_bytes))
                    width, height = img_pil.size
                except:
                    width, height = base_image.get("width", 0), base_image.get("height", 0)
                
                images.append({
                    'image_data': img_b64,
                    'bbox': bbox,
                    'page': page_num + 1,
                    'type': 'embedded_image',
                    'format': image_ext,
                    'width': width,
                    'height': height,
                    'confidence': 1.0,
                    'key': f"img_p{page_num+1}_i{img_index+1}"
                })
                
            except Exception as e:
                logger.warning(f"Error extracting image {img_index} from page {page_num+1}: {e}")
                continue
                
    except Exception as e:
        logger.warning(f"Error getting images from page {page_num+1}: {e}")
    
    return images

def extract_images_and_formulas_from_pdf_enhanced(uploaded_file) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
    """Enhanced PDF extraction with comprehensive image, table, and formula extraction"""
    if not FITZ_AVAILABLE:
        logger.error("PyMuPDF not available for PDF processing")
        try:
            text = uploaded_file.read().decode('utf-8', errors='ignore')
            return text, {}, []
        except Exception as e:
            logger.error(f"Error reading PDF as text: {e}")
            return "", {}, []
    
    text = ""
    all_images = {}
    all_extracted_elements = []
    
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
            
            # 1. Extract mathematical formulas from text
            text_formulas = extract_mathematical_formulas_enhanced(page_text, page_num + 1)
            all_extracted_elements.extend(text_formulas)
            
            # 2. Extract formulas as images
            formula_images = extract_formulas_as_images_from_pdf(doc, page_num, text_formulas)
            all_extracted_elements.extend(formula_images)
            
            # Add formula images to the images dict
            for formula_img in formula_images:
                all_images[formula_img['key']] = formula_img['image_data']
            
            # 3. Extract tables as images
            table_images = extract_tables_as_images_from_pdf(doc, page_num)
            all_extracted_elements.extend(table_images)
            
            # Add table images to the images dict
            for table_img in table_images:
                all_images[table_img['key']] = table_img['image_data']
            
            # 4. Extract regular embedded images
            embedded_images = extract_all_images_enhanced(doc, page_num)
            all_extracted_elements.extend(embedded_images)
            
            # Add embedded images to the images dict
            for emb_img in embedded_images:
                all_images[emb_img['key']] = emb_img['image_data']
            
            # 5. Extract diagrams and charts
            diagrams = extract_diagrams_and_charts_from_pdf(doc, page_num)
            all_extracted_elements.extend(diagrams)
            
            # Add diagram images to the images dict
            for diagram in diagrams:
                all_images[diagram['key']] = diagram['image_data']
        
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
    
    logger.info(f"Enhanced extraction complete: {len(all_extracted_elements)} total elements, {len(all_images)} images")
    return text, all_images, all_extracted_elements

def extract_images_from_docx_enhanced(uploaded_file) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
    """Enhanced DOCX extraction with formula and table detection"""
    if not DOCX_AVAILABLE:
        logger.error("python-docx not available for DOCX processing")
        try:
            uploaded_file.seek(0)
            return uploaded_file.read().decode('utf-8', errors='ignore'), {}, []
        except:
            return "Error reading DOCX content", {}, []
    
    text = ""
    images = {}
    extracted_elements = []
    
    try:
        uploaded_file.seek(0)
        doc = docx.Document(uploaded_file)
        
        # Extract text and detect formulas
        for paragraph in doc.paragraphs:
            para_text = paragraph.text
            text += para_text + "\n"
            
            # Extract formulas from this paragraph
            para_formulas = extract_mathematical_formulas_enhanced(para_text)
            extracted_elements.extend(para_formulas)
        
        # Extract images from document relationships
        try:
            for i, rel in enumerate(doc.part.rels.values()):
                if "image" in rel.target_ref:
                    try:
                        image_data = rel.target_part.blob
                        img_b64 = base64.b64encode(image_data).decode()
                        key = f"docx_img_{i+1}"
                        images[key] = img_b64
                        
                        # Add to extracted elements
                        extracted_elements.append({
                            'image_data': img_b64,
                            'page': 1,
                            'type': 'embedded_image',
                            'confidence': 1.0,
                            'key': key
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error extracting image {i}: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Error accessing document relationships: {e}")
        
        # Extract tables information
        for i, table in enumerate(doc.tables):
            try:
                table_text = ""
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells])
                    table_text += row_text + "\n"
                
                if table_text.strip():
                    extracted_elements.append({
                        'text': table_text,
                        'type': 'docx_table',
                        'page': 1,
                        'confidence': 0.9,
                        'rows': len(table.rows),
                        'cols': len(table.columns) if table.rows else 0
                    })
                    
            except Exception as e:
                logger.warning(f"Error extracting table {i}: {e}")
                continue
                    
    except Exception as e:
        logger.error(f"Error processing DOCX: {str(e)}")
        try:
            uploaded_file.seek(0)
            text = uploaded_file.read().decode('utf-8', errors='ignore')
        except:
            text = "Error reading DOCX content"
        return text, {}, []
    
    return text, images, extracted_elements

def process_document_enhanced(uploaded_file, extract_images: bool = True, extract_formulas: bool = True) -> Tuple[str, Dict[str, str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Enhanced document processing with comprehensive extraction capabilities
    """
    if uploaded_file is None:
        return "", {}, [], {}
    
    try:
        file_type = uploaded_file.type
        logger.info(f"Processing file: {uploaded_file.name} ({file_type})")
        
        if file_type == "application/pdf":
            document_text, extracted_images, extracted_elements = extract_images_and_formulas_from_pdf_enhanced(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            document_text, extracted_images, extracted_elements = extract_images_from_docx_enhanced(uploaded_file)
        elif file_type == "text/plain":
            document_text = extract_text_from_txt(uploaded_file)
            extracted_images = {}
            extracted_elements = extract_mathematical_formulas_enhanced(document_text) if extract_formulas else []
        else:
            logger.warning(f"Unsupported file type: {file_type}")
            uploaded_file.seek(0)
            document_text = str(uploaded_file.read(), "utf-8", errors='ignore')
            extracted_images = {}
            extracted_elements = extract_mathematical_formulas_enhanced(document_text) if extract_formulas else []
        
        # Apply extraction options
        if not extract_images:
            # Remove image data but keep metadata
            extracted_images = {}
            extracted_elements = [e for e in extracted_elements if 'image_data' not in e]
            
        if not extract_formulas:
            # Remove formula elements
            extracted_elements = [e for e in extracted_elements if not e.get('type', '').endswith('formula')]
        
        # Enhanced document analysis
        document_analysis = analyze_document_intelligence_enhanced(document_text, extracted_images, extracted_elements)
        
        logger.info(f"Enhanced processing complete: {len(document_text)} chars, {len(extracted_images)} images, {len(extracted_elements)} elements")
        
        return document_text, extracted_images, extracted_elements, document_analysis
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        if 'st' in globals():
            st.error(f"Error processing document: {str(e)}")
        return "", {}, [], {}

def analyze_document_intelligence_enhanced(text: str, images: Dict[str, str], elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Enhanced document analysis with better categorization of extracted elements"""
    analysis = {
        'document_type': 'Unknown',
        'regulatory_framework': [],
        'key_entities': [],
        'complexity_score': 0,
        'mathematical_complexity': 'Low',
        'formula_types': [],
        'image_types': [],
        'table_count': 0,
        'formula_count': 0,
        'diagram_count': 0,
        'regulatory_sections': [],
        'extraction_summary': {}
    }
    
    if not text and not elements:
        return analysis
    
    # Categorize extracted elements
    formulas = [e for e in elements if 'formula' in e.get('type', '')]
    tables = [e for e in elements if 'table' in e.get('type', '')]
    diagrams = [e for e in elements if 'diagram' in e.get('type', '')]
    embedded_images = [e for e in elements if e.get('type') == 'embedded_image']
    
    analysis['formula_count'] = len(formulas)
    analysis['table_count'] = len(tables)
    analysis['diagram_count'] = len(diagrams)
    
    # Formula type analysis
    formula_types = set()
    for formula in formulas:
        if formula.get('type'):
            formula_types.add(formula['type'])
    analysis['formula_types'] = list(formula_types)
    
    # Image type analysis
    image_types = set()
    for element in elements:
        if 'image_data' in element:
            image_types.add(element.get('type', 'unknown'))
    analysis['image_types'] = list(image_types)
    
    # Mathematical complexity based on extracted elements
    if len(formulas) > 15:
        analysis['mathematical_complexity'] = 'Very High'
    elif len(formulas) > 8:
        analysis['mathematical_complexity'] = 'High'
    elif len(formulas) > 3:
        analysis['mathematical_complexity'] = 'Medium'
    elif len(formulas) > 0:
        analysis['mathematical_complexity'] = 'Low'
    
    # Document type detection (enhanced with element analysis)
    if text:
        text_lower = text.lower()
        doc_type_scores = {
            'regulatory': text_lower.count('regulation') + text_lower.count('compliance') + text_lower.count('basel'),
            'financial': text_lower.count('risk') + text_lower.count('capital') + len(formulas) * 0.5,
            'technical': text_lower.count('specification') + text_lower.count('technical') + len(diagrams),
            'academic': text_lower.count('study') + text_lower.count('research') + len(formulas) * 0.3
        }
        analysis['document_type'] = max(doc_type_scores, key=doc_type_scores.get).title()
    
    # Extraction summary
    analysis['extraction_summary'] = {
        'total_elements': len(elements),
        'text_formulas': len([e for e in formulas if 'image_data' not in e]),
        'formula_images': len([e for e in formulas if 'image_data' in e]),
        'table_images': len(tables),
        'diagrams': len(diagrams),
        'embedded_images': len(embedded_images),
        'pages_processed': len(set(e.get('page', 1) for e in elements))
    }
    
    # Complexity scoring
    complexity_factors = [
        len(text) > 50000,
        len(images) > 10,
        len(formulas) > 5,
        len(tables) > 3,
        len(diagrams) > 2,
        analysis['formula_count'] > 10
    ]
    analysis['complexity_score'] = sum(complexity_factors) / len(complexity_factors)
    
    return analysis

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
    """Display image from base64 string with better error handling"""
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
