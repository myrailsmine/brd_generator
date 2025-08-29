BRD generator /

Analyze the code..all the options which are available for BRD generation are good don't change those. 

document processor for basel pdf documens unable to extract mathamatical formulas and tables. So modify the code to extract those as image first ..may be store in cache and then modify the extact utils to embed those images while writing the BRD.

Use the best in class enterprise level pdf extractor code to generate the business requirements and ensure that each section context is setting properly. 

Also add ability to modify each section.

I'll analyze the code and enhance the document processor for Basel PDF documents to better extract mathematical formulas and tables as images, then modify the extraction utilities to embed those images in the BRD. Here's my comprehensive solution:

Don';t change title of pages but core functionality needs to be enhanced .. make it super great and using the top class pdf processors and ensure BRD generated can embed images tables as image etc

I'll enhance the core functionality with enterprise-grade PDF processing while keeping all existing titles and options intact. Here's the enhanced document processor:

Continue

Now let me create the enhanced AI processor that can properly embed the extracted images into the BRD:

Continue

Continue

Now let me create the enhanced export utilities that properly handle embedded images:

Continue

Finally, let me update the main tabs.py to use the enhanced functionality:

Perfect! I've created a comprehensive enhancement to your BRD generation system. Here's a summary of the major improvements:
🚀 Enhanced Core Functionality
1. Enterprise-Grade PDF Processing

    Advanced Mathematical Formula Detection: Uses sophisticated pattern recognition to identify Basel-specific formulas, correlations, risk weights, and regulatory references
    Table Region Extraction: Automatically detects and extracts regulatory tables as high-resolution images
    Smart Caching: Implements intelligent caching to avoid re-processing the same documents
    Enhanced Image Processing: Applies image enhancement techniques for better clarity of extracted mathematical content

2. Intelligent Image Embedding System

    Contextual Image Placement: AI determines the best sections to embed relevant images based on content analysis
    Smart Image References: Creates structured references like [IMAGE_EMBED:id|type|description] that are preserved during editing
    Multi-Format Support: Images are properly embedded in Word, PDF, Excel, and JSON exports
    Image Classification: Automatically categorizes extracted content (formulas, tables, diagrams, etc.)

3. Advanced BRD Generation

    Enhanced AI Prompting: Sophisticated prompts that specifically address Basel regulatory requirements
    Section-Specific Intelligence: Each BRD section gets tailored content based on its purpose and extracted mathematical content
    Quality Scoring: Comprehensive quality assessment with specific checks for regulatory compliance
    Image Integration: Automatically embeds relevant mathematical formulas and tables in appropriate sections

4. Professional Export System

    Word Export: Full image embedding with proper formatting and captions
    PDF Export: High-quality images with professional layout and regulatory-appropriate styling
    Excel Export: Data tables plus image reference summary sheet
    JSON Export: Complete metadata including image references and document analysis
    Export Validation: Pre-export validation with recommendations for improvement

🎯 Key Features for Basel Documents
Mathematical Content Extraction

    Basel MAR references (MAR21.88, etc.)
    Risk weight formulas
    Capital requirement calculations
    Correlation parameters
    Sensitivity measures (VaR, PV01, CS01)
    Greek letter notations
    Supervisory parameters

Regulatory Intelligence

    Document type detection (Basel, Solvency II, MiFID, etc.)
    Regulatory framework identification
    Compliance requirement mapping
    Supervisory parameter recognition
    Bucket classification handling

Enhanced User Experience

    Real-time Image Previews: See extracted formulas and tables while editing
    Quality Insights: Specific recommendations for each section
    Export Validation: Check content completeness before export
    Bulk Export: Generate all formats simultaneously
    Enhanced Metrics: Detailed statistics on extraction quality and content completeness

🔧 Usage

All existing functionality is preserved - just use the system as before, but now it will:

    Automatically detect and extract mathematical formulas and regulatory tables as images
    Intelligently embed relevant visual content into appropriate BRD sections
    Generate higher quality content with proper regulatory terminology and references
    Export professional documents with embedded images and enhanced formatting
    Provide detailed insights on document complexity and extraction quality

The system now operates at an enterprise level, specifically optimized for Basel regulatory documents with complex mathematical content, while maintaining backward compatibility with all existing features.


"""
Enhanced Document Processing Utilities - Core Functionality
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

class EnhancedPDFExtractor:
    """Enterprise-grade PDF extractor with advanced caching and processing"""
    
    def __init__(self):
        self.cache = {}
        self.extracted_elements = []
        self.image_processor = EnterpriseImageProcessor()
    
    @st.cache_data
    def _get_cache_key(_self, file_content: bytes, extract_images: bool, extract_formulas: bool) -> str:
        """Generate cache key for processed document"""
        return hashlib.md5(f"{len(file_content)}_{extract_images}_{extract_formulas}".encode()).hexdigest()
    
    def extract_images_and_formulas_from_pdf_enhanced(self, uploaded_file) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
        """Enhanced PDF extraction with enterprise-grade processing"""
        if not FITZ_AVAILABLE:
            logger.error("PyMuPDF not available for PDF processing")
            return self._fallback_extraction(uploaded_file)
        
        try:
            # Create cache key
            uploaded_file.seek(0)
            file_content = uploaded_file.read()
            cache_key = self._get_cache_key(file_content, True, True)
            
            # Check cache
            if cache_key in self.cache:
                logger.info("Using cached extraction results")
                return self.cache[cache_key]
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            # Initialize results
            text = ""
            images = {}
            formulas = []
            
            # Open PDF with enhanced settings
            doc = fitz.open(tmp_file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text with better formatting
                page_text = self._extract_enhanced_text(page, page_num)
                text += f"\n=== PAGE {page_num + 1} ===\n{page_text}\n"
                
                # Extract mathematical regions as images
                math_regions = self.image_processor.detect_mathematical_regions(page, page_num + 1)
                
                for i, (rect, confidence, element_type) in enumerate(math_regions):
                    try:
                        # Extract region as image
                        mat = fitz.Matrix(3.0, 3.0)  # High resolution
                        pix = page.get_pixmap(matrix=mat, clip=rect)
                        img_data = pix.tobytes("png")
                        
                        # Enhance image
                        enhanced_img_data = self.image_processor.enhance_mathematical_image(img_data)
                        img_b64 = base64.b64encode(enhanced_img_data).decode()
                        
                        # Store enhanced image
                        element_id = f"math_p{page_num+1}_r{i+1}"
                        images[element_id] = {
                            'data': img_b64,
                            'page': page_num + 1,
                            'type': 'mathematical_formula',
                            'element_type': element_type,
                            'confidence': confidence,
                            'bbox': list(rect)
                        }
                        
                        # Create formula entry
                        context_text = self._extract_context(page, rect)
                        formulas.append({
                            'text': context_text[:200],
                            'type': element_type,
                            'page': page_num + 1,
                            'confidence': confidence,
                            'context': context_text,
                            'image_id': element_id,
                            'bbox': list(rect)
                        })
                        
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Error extracting mathematical region: {e}")
                
                # Extract table regions as images
                table_regions = self.image_processor.detect_table_regions(page, page_num + 1)
                
                for i, (rect, confidence, element_type) in enumerate(table_regions):
                    try:
                        # Extract table as high-resolution image
                        mat = fitz.Matrix(2.5, 2.5)  # High resolution for tables
                        pix = page.get_pixmap(matrix=mat, clip=rect)
                        img_data = pix.tobytes("png")
                        
                        # Enhance table image
                        enhanced_img_data = self.image_processor.enhance_mathematical_image(img_data)
                        img_b64 = base64.b64encode(enhanced_img_data).decode()
                        
                        # Store enhanced table image
                        element_id = f"table_p{page_num+1}_t{i+1}"
                        images[element_id] = {
                            'data': img_b64,
                            'page': page_num + 1,
                            'type': 'regulatory_table',
                            'element_type': element_type,
                            'confidence': confidence,
                            'bbox': list(rect)
                        }
                        
                        # Create table entry for formulas list
                        table_context = self._extract_context(page, rect)
                        formulas.append({
                            'text': f"[TABLE IMAGE: {element_id}]",
                            'type': element_type,
                            'page': page_num + 1,
                            'confidence': confidence,
                            'context': table_context,
                            'image_id': element_id,
                            'bbox': list(rect)
                        })
                        
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Error extracting table region: {e}")
                
                # Extract regular images
                try:
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n < 5:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_b64 = base64.b64encode(img_data).decode()
                            
                            img_key = f"image_p{page_num+1}_i{img_index+1}"
                            images[img_key] = {
                                'data': img_b64,
                                'page': page_num + 1,
                                'type': 'document_image',
                                'width': pix.width,
                                'height': pix.height
                            }
                        pix = None
                        
                except Exception as e:
                    logger.warning(f"Error extracting regular images from page {page_num+1}: {e}")
            
            doc.close()
            os.unlink(tmp_file_path)
            
            # Convert image format for backward compatibility
            image_dict = {}
            for key, img_info in images.items():
                if isinstance(img_info, dict):
                    image_dict[key] = img_info['data']
                else:
                    image_dict[key] = img_info
            
            # Cache results
            result = (text, image_dict, formulas)
            self.cache[cache_key] = result
            
            logger.info(f"Enhanced extraction complete: {len(formulas)} elements, {len(image_dict)} images")
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced PDF processing: {str(e)}")
            return self._fallback_extraction(uploaded_file)
    
    def _extract_enhanced_text(self, page, page_num: int) -> str:
        """Extract text with enhanced formatting preservation"""
        try:
            # Get text with layout information
            blocks = page.get_text("dict")
            text_content = []
            
            for block in blocks.get("blocks", []):
                if "lines" in block:
                    block_text = []
                    for line in block["lines"]:
                        line_text = ""
                        for span in line.get("spans", []):
                            span_text = span.get("text", "")
                            # Preserve formatting indicators
                            if span.get("flags", 0) & 2**4:  # Bold
                                span_text = f"**{span_text}**"
                            if span.get("flags", 0) & 2**1:  # Italic
                                span_text = f"*{span_text}*"
                            line_text += span_text
                        if line_text.strip():
                            block_text.append(line_text.strip())
                    
                    if block_text:
                        text_content.append(" ".join(block_text))
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.warning(f"Error in enhanced text extraction: {e}")
            return page.get_text()
    
    def _extract_context(self, page, rect: fitz.Rect, expand: int = 100) -> str:
        """Extract context around a specific region"""
        try:
            # Expand rectangle to get surrounding context
            context_rect = rect + (-expand, -expand, expand, expand)
            context_rect = context_rect & page.rect  # Clip to page bounds
            
            context_text = page.get_textbox(context_rect)
            return context_text.strip()
            
        except Exception as e:
            logger.warning(f"Error extracting context: {e}")
            return ""
    
    def _fallback_extraction(self, uploaded_file) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
        """Fallback extraction method"""
        try:
            uploaded_file.seek(0)
            text = uploaded_file.read().decode('utf-8', errors='ignore')
            return text, {}, []
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return "", {}, []

# Global extractor instance
pdf_extractor = EnhancedPDFExtractor()

def extract_mathematical_formulas_advanced(text: str, page_num: int = None) -> List[Dict[str, Any]]:
    """Enhanced mathematical formula extraction with Basel-specific patterns"""
    formulas = []
    
    # Enhanced Basel-specific patterns
    formula_patterns = [
        # Basel regulatory references
        (r'MAR\d+\.\d+(?:\.\d+)?[^\n.!?]*', 'basel_mar_reference'),
        
        # Risk weight formulas  
        (r'RW[_\w\d]*\s*=\s*[\d.%\s\+\-\*/\(\)]+', 'risk_weight_formula'),
        
        # Capital requirement formulas
        (r'K[_\w\d]*\s*=\s*[^.!?]*(?:[+\-*/√∑∏]|max|min)[^.!?]*', 'capital_requirement'),
        
        # Correlation formulas
        (r'ρ[_\w\d]*(?:\([^)]+\))?\s*=\s*[\d.%\s\+\-\*/\(\)]+', 'correlation_formula'),
        
        # Sensitivity measures
        (r'(?:VaR|PV01|CS01|DV01|Vega)[_\w\d]*\s*=\s*[^.!?]*', 'sensitivity_measure'),
        
        # Greek letters in formulas
        (r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]+[_\w\d\s=+\-*/\(\)]*', 'greek_formula'),
        
        # Mathematical operators
        (r'(?:sqrt|exp|log|ln|sin|cos|tan)\s*\([^)]+\)', 'mathematical_function'),
        
        # Complex mathematical expressions
        (r'\([^)]*[+\-*/=√∑∏]\s*[^)]*\)', 'mathematical_expression'),
        
        # Bucket classifications
        (r'Bucket\s+(?:number\s+)?\d+[^\n.!?]*', 'bucket_classification'),
        
        # Supervisory parameters
        (r'(?:gamma|phi|alpha|beta|delta|lambda)[_\w\d]*\s*=\s*[\d.%]+', 'supervisory_parameter'),
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
    
    # Numerical content
    if re.search(r'\d+\.?\d*\s*%', formula_text):
        confidence += 0.1
    
    if re.search(r'\d+\.?\d*', formula_text):
        confidence += 0.05
    
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

# Update main processing functions to use enhanced extractor
def extract_images_and_formulas_from_pdf(uploaded_file) -> Tuple[str, Dict[str, str], List[Dict[str, Any]]]:
    """Enhanced PDF extraction - main entry point"""
    return pdf_extractor.extract_images_and_formulas_from_pdf_enhanced(uploaded_file)

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
                        
                        # Enhance image if it appears to be mathematical/tabular
                        if len(image_data) < 500000:  # Only enhance smaller images
                            try:
                                enhanced_data = EnterpriseImageProcessor.enhance_mathematical_image(image_data)
                                img_b64 = base64.b64encode(enhanced_data).decode()
                            except:
                                img_b64 = base64.b64encode(image_data).decode()
                        else:
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
                        st.info(f"📊 Referenced content: {image_key}")

def analyze_document_intelligence(text: str, images: Dict[str, str], formulas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Enhanced AI-powered document analysis with enterprise-grade intelligence"""
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
        'regulatory_sections': [],
        'image_analysis': {},
        'extraction_quality': {},
        'document_structure': {}
    }
    
    if not text:
        return analysis
    
    text_lower = text.lower()
    
    # Enhanced document type detection
    doc_type_patterns = {
        'basel_regulatory': ['basel', 'mar21', 'mar22', 'mar23', 'supervisory', 'committee on banking supervision'],
        'risk_management': ['var', 'risk management', 'pv01', 'cs01', 'market risk', 'credit risk', 'operational risk'],
        'compliance': ['compliance', 'regulatory', 'sox', 'gdpr', 'regulation', 'supervisory'],
        'financial_reporting': ['financial', 'accounting', 'ifrs', 'gaap', 'reporting', 'statement'],
        'technical_specification': ['api', 'technical', 'specification', 'system', 'architecture', 'implementation']
    }
    
    max_score = 0
    detected_type = 'Unknown'
    
    for doc_type, patterns in doc_type_patterns.items():
        score = sum(text_lower.count(pattern.lower()) for pattern in patterns)
        if score > max_score:
            max_score = score
            detected_type = doc_type.replace('_', ' ').title()
    
    analysis['document_type'] = detected_type
    
    # Enhanced regulatory framework detection
    frameworks = {
        'Basel III/IV': ['basel', 'basel iii', 'basel iv', 'mar21', 'mar22', 'mar23'],
        'Solvency II': ['solvency', 'solvency ii', 'eiopa'],
        'MiFID II': ['mifid', 'mifid ii', 'esma'],
        'GDPR': ['gdpr', 'data protection', 'privacy regulation'],
        'SOX': ['sox', 'sarbanes oxley', 'sarbanes-oxley'],
        'IFRS': ['ifrs', 'international financial reporting'],
        'Dodd-Frank': ['dodd frank', 'dodd-frank', 'volcker']
    }
    
    detected_frameworks = []
    for framework, keywords in frameworks.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_frameworks.append(framework)
    
    analysis['regulatory_framework'] = detected_frameworks
    
    # Enhanced mathematical complexity analysis
    if formulas:
        formula_types = set()
        complexity_scores = []
        
        for formula in formulas:
            if isinstance(formula, dict):
                formula_type = formula.get('type', 'unknown')
                formula_types.add(formula_type)
                
                # Get mathematical complexity
                math_complexity = formula.get('mathematical_complexity', 'Low')
                complexity_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
                complexity_scores.append(complexity_mapping.get(math_complexity, 1))
        
        analysis['formula_types'] = list(formula_types)
        
        # Determine overall mathematical complexity
        if complexity_scores:
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            if avg_complexity >= 3.5:
                analysis['mathematical_complexity'] = 'Very High'
            elif avg_complexity >= 2.5:
                analysis['mathematical_complexity'] = 'High'
            elif avg_complexity >= 1.5:
                analysis['mathematical_complexity'] = 'Medium'
            else:
                analysis['mathematical_complexity'] = 'Low'
        
        # High complexity indicators
        high_complexity_types = [
            'basel_mar_reference', 'capital_requirement', 'correlation_formula',
            'sensitivity_measure', 'advanced_mathematical_expression'
        ]
        
        if any(ftype in high_complexity_types for ftype in formula_types):
            if analysis['mathematical_complexity'] in ['Low', 'Medium']:
                analysis['mathematical_complexity'] = 'High'
    
    # Enhanced image analysis
    image_analysis = {
        'total_images': len(images),
        'mathematical_images': 0,
        'table_images': 0,
        'diagram_images': 0,
        'document_images': 0
    }
    
    for img_key, img_data in images.items():
        if isinstance(img_data, dict):
            img_type = img_data.get('type', 'unknown')
            element_type = img_data.get('element_type', 'unknown')
        else:
            # Infer from key name
            if 'math' in img_key.lower() or 'formula' in img_key.lower():
                img_type = 'mathematical_formula'
            elif 'table' in img_key.lower():
                img_type = 'regulatory_table'
            else:
                img_type = 'document_image'
            element_type = img_type
        
        if img_type == 'mathematical_formula':
            image_analysis['mathematical_images'] += 1
        elif img_type == 'regulatory_table':
            image_analysis['table_images'] += 1
        elif 'diagram' in img_type:
            image_analysis['diagram_images'] += 1
        else:
            image_analysis['document_images'] += 1
    
    analysis['image_analysis'] = image_analysis
    
    # Enhanced regulatory section detection
    section_patterns = [
        r'(MAR\d+\.\d+(?:\.\d+)?[^\n.!?]*)',  # Basel MAR sections
        r'(\d+\.\d+(?:\.\d+)?\s+[A-Z][^\n.!?]*)',  # Numbered sections
        r'([A-Z][A-Z\s]{3,}:?\s*[A-Z][^\n.!?]*)',  # All caps headings
        r'(Article\s+\d+[^\n.!?]*)',  # Article references
        r'(Section\s+\d+[^\n.!?]*)',  # Section references
        r'(Chapter\s+\d+[^\n.!?]*)',  # Chapter references
    ]
    
    regulatory_sections = []
    for pattern in section_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        regulatory_sections.extend(matches[:15])  # Limit to prevent overflow
    
    analysis['regulatory_sections'] = regulatory_sections
    
    # Enhanced entity extraction
    entity_patterns = [
        r'\b[A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\b',  # Names/Organizations
        r'\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b',  # Dates
        r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',  # Money amounts
        r'\b\d+\.?\d*\s*%\b',  # Percentages
        r'\bMAR\d+\.\d+\b',  # Basel MAR references
        r'\bBucket\s+\d+\b',  # Bucket classifications
        r'\b[A-Z]{2,5}\d+\b',  # Regulatory codes
    ]
    
    key_entities = []
    for pattern in entity_patterns:
        try:
            matches = re.findall(pattern, text)
            # Filter and clean matches
            filtered_matches = [m for m in matches if len(m.strip()) > 2]
            key_entities.extend(filtered_matches[:20])  # Limit results
        except Exception as e:
            logger.warning(f"Error in entity extraction: {e}")
    
    analysis['key_entities'] = list(set(key_entities))  # Remove duplicates
    
    # Enhanced complexity scoring
    complexity_factors = [
        len(text) > 100000,  # Very large document
        len(images) > 20,  # Many images
        len(formulas) > 15,  # Many formulas
        len(detected_frameworks) > 2,  # Multiple regulations
        image_analysis['mathematical_images'] > 5,  # Many math images
        image_analysis['table_images'] > 5,  # Many table images
        'correlation' in text_lower,  # Complex correlations
        'curvature' in text_lower,  # Advanced risk concepts
        len(regulatory_sections) > 20,  # Many sections
        any('very high' in str(f.get('mathematical_complexity', '')).lower() for f in formulas),  # Very high math complexity
    ]
    
    analysis['complexity_score'] = sum(complexity_factors) / len(complexity_factors)
    
    # Document structure analysis
    structure_analysis = {
        'has_table_of_contents': 'contents' in text_lower or 'table of contents' in text_lower,
        'has_executive_summary': 'executive summary' in text_lower or 'summary' in text_lower,
        'has_appendices': 'appendix' in text_lower or 'annexe' in text_lower,
        'section_count': len(regulatory_sections),
        'average_section_length': len(text) / max(len(regulatory_sections), 1),
        'mathematical_content_ratio': len(formulas) / max(len(text.split()), 1) * 1000,  # Formulas per 1000 words
    }
    
    analysis['document_structure'] = structure_analysis
    
    # Extraction quality metrics
    extraction_quality = {
        'text_extraction_quality': 'High' if len(text) > 1000 else 'Low',
        'image_extraction_success': len(images) > 0,
        'formula_extraction_success': len(formulas) > 0,
        'mathematical_content_detected': analysis['mathematical_complexity'] != 'Low',
        'regulatory_content_detected': len(detected_frameworks) > 0,
        'overall_extraction_score': 0
    }
    
    # Calculate overall extraction score
    score_factors = [
        extraction_quality['text_extraction_quality'] == 'High',
        extraction_quality['image_extraction_success'],
        extraction_quality['formula_extraction_success'],
        extraction_quality['mathematical_content_detected'],
        extraction_quality['regulatory_content_detected']
    ]
    
    extraction_quality['overall_extraction_score'] = sum(score_factors) / len(score_factors)
    analysis['extraction_quality'] = extraction_quality
    
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
        
        # Log extraction results
        logger.info(f"Enhanced extraction complete:")
        logger.info(f"  - Text: {len(document_text):,} characters")
        logger.info(f"  - Images: {len(extracted_images)} items")
        logger.info(f"  - Formulas: {len(extracted_formulas)} items")
        logger.info(f"  - Document type: {document_analysis.get('document_type', 'Unknown')}")
        logger.info(f"  - Mathematical complexity: {document_analysis.get('mathematical_complexity', 'Unknown')}")
        logger.info(f"  - Regulatory frameworks: {len(document_analysis.get('regulatory_framework', []))}")
        
        return document_text, extracted_images, extracted_formulas, document_analysis
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        if 'st' in globals():
            st.error(f"Error processing document: {str(e)}")
        return "", {}, [], {}
