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

def extract_images_and_formulas_from_pdf(uploaded_file) -> Tuple[str, Dict[str, str], List[str]]:
    """Extract text, images, and formulas from PDF"""
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
            
            # Extract text
            page_text = page.get_text()
            text += page_text + "\n"
            
            # Extract images
            try:
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n < 5:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_b64 = base64.b64encode(img_data).decode()
                        images[f"page_{page_num+1}_img_{img_index+1}"] = img_b64
                    pix = None
            except Exception as e:
                logger.warning(f"Error extracting images from page {page_num+1}: {e}")
            
            # Extract potential formulas (text patterns that might be mathematical)
            formula_patterns = [
                r'[=∑∏∫∆∇±≤≥≠≈∞]',
                r'\b\w+\s*[+\-*/=]\s*\w+\b',
                r'\([^)]*[+\-*/]\s*[^)]*\)',
                r'\b\d+\.\d+\b',
                r'[xy]\^?\d*',
            ]
            
            for pattern in formula_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                formulas.extend(matches)
        
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
    
    # Remove duplicates from formulas
    formulas = list(set(formulas))
    return text, images, formulas

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
        
        st.image(img, caption=caption, use_column_width=True if not max_width else False)
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

def analyze_document_intelligence(text: str, images: Dict[str, str], formulas: List[str]) -> Dict[str, Any]:
    """Advanced AI-powered document analysis"""
    analysis = {
        'document_type': 'Unknown',
        'regulatory_framework': [],
        'key_entities': [],
        'complexity_score': 0,
        'compliance_indicators': [],
        'stakeholder_mentions': [],
        'risk_indicators': [],
        'timeline_references': []
    }
    
    if not text:
        return analysis
    
    # Document type detection
    doc_type_indicators = {
        'regulatory': ['regulation', 'compliance', 'requirement', 'shall', 'must'],
        'policy': ['policy', 'procedure', 'guideline', 'standard'],
        'technical': ['specification', 'technical', 'architecture', 'design'],
        'business': ['business', 'process', 'workflow', 'operation']
    }
    
    text_lower = text.lower()
    max_score = 0
    detected_type = 'Unknown'
    
    for doc_type, indicators in doc_type_indicators.items():
        score = sum(1 for indicator in indicators if indicator in text_lower)
        if score > max_score:
            max_score = score
            detected_type = doc_type.title()
    
    analysis['document_type'] = detected_type
    
    # Regulatory framework detection
    frameworks = ['sox', 'gdpr', 'basel', 'mifid', 'dodd-frank', 'pci-dss', 'iso 27001']
    analysis['regulatory_framework'] = [fw for fw in frameworks if fw in text_lower]
    
    # Key entity extraction (simplified)
    entity_patterns = [
        r'\b[A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+\b',  # Names
        r'\b\d{2,4}[-/]\d{2}[-/]\d{2,4}\b',    # Dates
        r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',     # Money
    ]
    
    for pattern in entity_patterns:
        try:
            matches = re.findall(pattern, text)
            analysis['key_entities'].extend(matches[:10])  # Limit results
        except Exception as e:
            logger.warning(f"Error in entity extraction: {e}")
    
    # Complexity scoring
    complexity_factors = [
        len(text) > 50000,  # Large document
        len(images) > 10,   # Many images
        len(formulas) > 5,  # Complex formulas
        len(analysis['regulatory_framework']) > 2,  # Multiple regulations
    ]
    analysis['complexity_score'] = sum(complexity_factors) / len(complexity_factors)
    
    return analysis

def process_document(uploaded_file, extract_images: bool = True, extract_formulas: bool = True) -> Tuple[str, Dict[str, str], List[str], Dict[str, Any]]:
    """
    Process uploaded document and return text, images, formulas, and analysis
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
            extracted_formulas = []
        elif file_type == "text/plain":
            document_text = extract_text_from_txt(uploaded_file)
            extracted_images = {}
            extracted_formulas = []
        else:
            logger.warning(f"Unsupported file type: {file_type}")
            uploaded_file.seek(0)
            document_text = str(uploaded_file.read(), "utf-8", errors='ignore')
            extracted_images = {}
            extracted_formulas = []
        
        # Apply extraction options
        if not extract_images:
            extracted_images = {}
        if not extract_formulas:
            extracted_formulas = []
        
        # Perform document analysis
        document_analysis = analyze_document_intelligence(document_text, extracted_images, extracted_formulas)
        
        logger.info(f"Document processed successfully: {len(document_text)} chars, {len(extracted_images)} images, {len(extracted_formulas)} formulas")
        
        return document_text, extracted_images, extracted_formulas, document_analysis
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        if 'st' in globals():
            st.error(f"Error processing document: {str(e)}")
        return "", {}, [], {}
