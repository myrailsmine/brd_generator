"""
Enhanced Export Utilities with Media Embedding Support
"""

import pandas as pd
import re
import base64
from io import BytesIO
from datetime import datetime
from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)

# Optional imports with fallbacks
try:
    import docx
    from docx.shared import Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.shared import OxmlElement, qn
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not available. Word export will be disabled.")

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus.flowables import KeepTogether
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available. PDF export will be disabled.")

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def export_to_word_docx_enhanced(brd_content: Dict[str, Any], document_analysis: Dict[str, Any] = None) -> BytesIO:
    """Enhanced Word export with embedded images, formulas, and tables"""
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx not available. Please install it to use Word export.")
    
    try:
        doc = docx.Document()
        
        # Add title
        title = doc.add_heading('Business Requirements Document', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add generation date and metadata
        doc.add_paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}")
        
        if document_analysis:
            # Add document intelligence summary
            doc.add_heading('Document Analysis Summary', level=1)
            
            summary_table = doc.add_table(rows=1, cols=2)
            summary_table.style = 'Light Grid Accent 1'
            
            # Add metadata rows
            metadata_items = [
                ('Document Type', document_analysis.get('document_type', 'Unknown')),
                ('Mathematical Complexity', document_analysis.get('mathematical_complexity', 'Unknown')),
                ('Regulatory Framework', ', '.join(document_analysis.get('regulatory_framework', []))),
                ('Formula Count', str(document_analysis.get('formula_analysis', {}).get('total_formulas', 0))),
                ('Table Count', str(document_analysis.get('table_count', 0))),
                ('Complexity Score', f"{document_analysis.get('complexity_score', 0):.2f}")
            ]
            
            for label, value in metadata_items:
                if value and value != 'Unknown':
                    row_cells = summary_table.add_row().cells
                    row_cells[0].text = label
                    row_cells[1].text = value
        
        doc.add_page_break()
        
        # Process enhanced document if available
        enhanced_doc = document_analysis.get('enhanced_document') if document_analysis else None
        if enhanced_doc:
            add_enhanced_content_to_word(doc, enhanced_doc, document_analysis)
        else:
            # Fallback to standard content processing
            add_standard_content_to_word(doc, brd_content)
        
        # Save to bytes
        doc_bytes = BytesIO()
        doc.save(doc_bytes)
        doc_bytes.seek(0)
        return doc_bytes
    
    except Exception as e:
        logger.error(f"Error exporting to Word: {e}")
        raise

def add_enhanced_content_to_word(doc, enhanced_text: str, document_analysis: Dict[str, Any]):
    """Add enhanced content with embedded elements to Word document"""
    
    # Split content by embedded elements
    sections = re.split(r'(\*\*\[(?:MATHEMATICAL FORMULA|EXTRACTED TABLE|EMBEDDED IMAGE)[^\]]*\]\*\*)', enhanced_text)
    
    for section in sections:
        if not section.strip():
            continue
            
        if section.startswith('**[MATHEMATICAL FORMULA'):
            add_formula_to_word(doc, section)
        elif section.startswith('**[EXTRACTED TABLE'):
            add_table_to_word(doc, section)
        elif section.startswith('**[EMBEDDED IMAGE'):
            add_image_to_word(doc, section, document_analysis)
        else:
            # Regular text content
            add_text_content_to_word(doc, section)

def add_formula_to_word(doc, formula_section: str):
    """Add mathematical formula to Word document with formatting"""
    
    # Extract formula type and confidence
    header_match = re.search(r'\*\*\[MATHEMATICAL FORMULA - ([^\]]+)\]\*\*', formula_section)
    formula_type = header_match.group(1) if header_match else "Mathematical Formula"
    
    confidence_match = re.search(r'Confidence: ([\d.]+%)', formula_section)
    confidence = confidence_match.group(1) if confidence_match else "N/A"
    
    # Add formula heading
    formula_heading = doc.add_heading(formula_type, level=3)
    
    # Extract formula from code block
    formula_match = re.search(r'```\n(.*?)\n```', formula_section, re.DOTALL)
    if formula_match:
        formula_text = formula_match.group(1)
        
        # Add formula in a bordered paragraph
        formula_para = doc.add_paragraph()
        formula_run = formula_para.add_run(formula_text)
        formula_run.font.name = 'Courier New'
        formula_run.font.size = docx.shared.Pt(11)
        
        # Add border around formula
        para_format = formula_para.paragraph_format
        para_format.left_indent = Inches(0.5)
        para_format.right_indent = Inches(0.5)
        para_format.space_before = docx.shared.Pt(6)
        para_format.space_after = docx.shared.Pt(6)
    
    # Add confidence info
    confidence_para = doc.add_paragraph()
    confidence_run = confidence_para.add_run(f"Confidence: {confidence}")
    confidence_run.font.italic = True
    confidence_run.font.size = docx.shared.Pt(9)

def add_table_to_word(doc, table_section: str):
    """Add extracted table to Word document with proper formatting"""
    
    # Extract table metadata
    metadata_match = re.search(r'\*(\d+) rows × (\d+) columns \(Confidence: ([\d.]+%)\)\*', table_section)
    if metadata_match:
        rows, cols, confidence = metadata_match.groups()
        table_heading = doc.add_heading(f'Data Table ({rows}×{cols})', level=3)
    
    # Extract table content
    table_lines = []
    lines = table_section.split('\n')
    
    for line in lines:
        if line.strip().startswith('|') and '---' not in line:
            table_lines.append(line.strip())
    
    if table_lines and len(table_lines) > 1:
        try:
            # Parse table structure
            headers = [cell.strip() for cell in table_lines[0].split('|')[1:-1]]
            data_rows = []
            
            for line in table_lines[1:]:
                if line.strip():
                    row = [cell.strip() for cell in line.split('|')[1:-1]]
                    if len(row) == len(headers):
                        data_rows.append(row)
            
            if data_rows and headers:
                # Create Word table
                word_table = doc.add_table(rows=1, cols=len(headers))
                word_table.style = 'Light Grid Accent 1'
                
                # Add headers
                header_cells = word_table.rows[0].cells
                for i, header in enumerate(headers):
                    header_cells[i].text = header
                    header_cells[i].paragraphs[0].runs[0].font.bold = True
                
                # Add data rows
                for row_data in data_rows:
                    row_cells = word_table.add_row().cells
                    for i, cell_data in enumerate(row_data):
                        if i < len(row_cells):
                            row_cells[i].text = str(cell_data)
            
        except Exception as e:
            logger.warning(f"Error creating Word table: {e}")
            # Fallback to preformatted text
            fallback_para = doc.add_paragraph()
            fallback_run = fallback_para.add_run('\n'.join(table_lines))
            fallback_run.font.name = 'Courier New'
    
    # Add confidence information
    if metadata_match:
        confidence_para = doc.add_paragraph()
        confidence_run = confidence_para.add_run(f"Table Confidence: {confidence}")
        confidence_run.font.italic = True
        confidence_run.font.size = docx.shared.Pt(9)

def add_image_to_word(doc, image_section: str, document_analysis: Dict[str, Any]):
    """Add embedded image to Word document"""
    
    # Extract image reference
    img_match = re.search(r'\[EMBEDDED IMAGE: ([^\]]+)\]', image_section)
    if not img_match:
        return
    
    img_key = img_match.group(1)
    
    # Extract base64 data
    b64_match = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)', image_section)
    if not b64_match:
        # Add placeholder text
        doc.add_paragraph(f"[Image: {img_key}]")
        return
    
    try:
        if PIL_AVAILABLE:
            # Decode and process image
            b64_data = b64_match.group(1)
            img_bytes = base64.b64decode(b64_data)
            
            # Create image stream
            img_stream = BytesIO(img_bytes)
            
            # Add image heading
            img_heading = doc.add_heading(f'Image: {img_key}', level=3)
            
            # Add image to document
            img_paragraph = doc.add_paragraph()
            img_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Calculate appropriate size (max 6 inches width)
            run = img_paragraph.runs[0] if img_paragraph.runs else img_paragraph.add_run()
            doc_img = doc.add_picture(img_stream, width=Inches(6))
            
        else:
            # Fallback without PIL
            doc.add_paragraph(f"[Image: {img_key} - Image processing not available]")
            
    except Exception as e:
        logger.warning(f"Error adding image to Word: {e}")
        doc.add_paragraph(f"[Image: {img_key} - Error loading image]")

def add_text_content_to_word(doc, text_content: str):
    """Add regular text content to Word document with markdown-like formatting"""
    
    lines = text_content.split('\n')
    current_para = None
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_para:
                current_para = None
            continue
        
        # Handle headers
        if line.startswith('# '):
            doc.add_heading(line[2:], level=1)
            current_para = None
        elif line.startswith('## '):
            doc.add_heading(line[3:], level=2)
            current_para = None
        elif line.startswith('### '):
            doc.add_heading(line[4:], level=3)
            current_para = None
        elif line.startswith('---'):
            doc.add_page_break()
            current_para = None
        elif line.startswith('- ') or line.startswith('* '):
            # Bullet point
            para = doc.add_paragraph(line[2:], style='List Bullet')
            current_para = None
        else:
            # Regular paragraph
            if not current_para:
                current_para = doc.add_paragraph()
            
            # Handle bold text
            if '**' in line:
                parts = re.split(r'\*\*(.*?)\*\*', line)
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        current_para.add_run(part)
                    else:
                        run = current_para.add_run(part)
                        run.font.bold = True
            else:
                current_para.add_run(line + ' ')

def add_standard_content_to_word(doc, brd_content: Dict[str, Any]):
    """Add standard BRD content to Word document (original functionality)"""
    
    for section_name, content in brd_content.items():
        # Add section heading
        doc.add_heading(section_name, level=1)
        
        if isinstance(content, dict):
            # Handle parent sections with subsections
            for subsection_name, subcontent in content.items():
                doc.add_heading(subsection_name, level=2)
                
                if isinstance(subcontent, pd.DataFrame) and not subcontent.empty:
                    # Add table
                    table = doc.add_table(rows=1, cols=len(subcontent.columns))
                    table.style = 'Light Grid Accent 1'
                    
                    # Add header row
                    hdr_cells = table.rows[0].cells
                    for i, column in enumerate(subcontent.columns):
                        hdr_cells[i].text = str(column)
                    
                    # Add data rows
                    for _, row in subcontent.iterrows():
                        row_cells = table.add_row().cells
                        for i, value in enumerate(row):
                            row_cells[i].text = str(value) if pd.notna(value) else ""
                else:
                    # Add text content
                    clean_text = re.sub(r'\[IMAGE:\s*[^\]]+\]', '[Image Reference]', str(subcontent))
                    doc.add_paragraph(clean_text)
                
                doc.add_paragraph()  # Add spacing
        
        elif isinstance(content, pd.DataFrame) and not content.empty:
            # Add table
            table = doc.add_table(rows=1, cols=len(content.columns))
            table.style = 'Light Grid Accent 1'
            
            # Add header row
            hdr_cells = table.rows[0].cells
            for i, column in enumerate(content.columns):
                hdr_cells[i].text = str(column)
            
            # Add data rows
            for _, row in content.iterrows():
                row_cells = table.add_row().cells
                for i, value in enumerate(row):
                    row_cells[i].text = str(value) if pd.notna(value) else ""
        
        else:
            # Add text content
            clean_text = re.sub(r'\[IMAGE:\s*[^\]]+\]', '[Image Reference]', str(content))
            doc.add_paragraph(clean_text)
        
        doc.add_page_break()

def export_to_pdf_enhanced(brd_content: Dict[str, Any], document_analysis: Dict[str, Any] = None) -> BytesIO:
    """Enhanced PDF export with embedded media support"""
    if not REPORTLAB_AVAILABLE:
        raise ImportError("ReportLab not available. Please install it to use PDF export.")
    
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.darkblue,
            alignment=1,  # Center alignment
            spaceAfter=30
        )
        
        formula_style = ParagraphStyle(
            'FormulaStyle',
            parent=styles['Code'],
            fontSize=11,
            fontName='Courier',
            backColor=colors.lightgrey,
            borderColor=colors.blue,
            borderWidth=1,
            leftIndent=20,
            rightIndent=20,
            spaceBefore=12,
            spaceAfter=12
        )
        
        # Add title
        story.append(Paragraph("Business Requirements Document", title_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Add document analysis if available
        if document_analysis:
            add_analysis_to_pdf(story, document_analysis, styles)
        
        # Process enhanced document if available
        enhanced_doc = document_analysis.get('enhanced_document') if document_analysis else None
        if enhanced_doc:
            add_enhanced_content_to_pdf(story, enhanced_doc, styles, formula_style, document_analysis)
        else:
            # Fallback to standard content
            add_standard_content_to_pdf(story, brd_content, styles)
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    except Exception as e:
        logger.error(f"Error exporting to PDF: {e}")
        raise

def add_analysis_to_pdf(story, document_analysis: Dict[str, Any], styles):
    """Add document analysis summary to PDF"""
    
    story.append(Paragraph("Document Analysis Summary", styles['Heading1']))
    
    # Create analysis table
    analysis_data = [['Attribute', 'Value']]
    
    analysis_items = [
        ('Document Type', document_analysis.get('document_type', 'Unknown')),
        ('Mathematical Complexity', document_analysis.get('mathematical_complexity', 'Unknown')),
        ('Formula Count', str(document_analysis.get('formula_analysis', {}).get('total_formulas', 0))),
        ('Table Count', str(document_analysis.get('table_count', 0))),
        ('Complexity Score', f"{document_analysis.get('complexity_score', 0):.2f}")
    ]
    
    for label, value in analysis_items:
        if value and value != 'Unknown':
            analysis_data.append([label, value])
    
    if len(analysis_data) > 1:
        analysis_table = Table(analysis_data)
        analysis_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(analysis_table)
    
    story.append(Spacer(1, 20))

def add_enhanced_content_to_pdf(story, enhanced_text: str, styles, formula_style, document_analysis: Dict[str, Any]):
    """Add enhanced content with embedded elements to PDF"""
    
    # Split content by embedded elements
    sections = re.split(r'(\*\*\[(?:MATHEMATICAL FORMULA|EXTRACTED TABLE|EMBEDDED IMAGE)[^\]]*\]\*\*)', enhanced_text)
    
    for section in sections:
        if not section.strip():
            continue
            
        if section.startswith('**[MATHEMATICAL FORMULA'):
            add_formula_to_pdf(story, section, styles, formula_style)
        elif section.startswith('**[EXTRACTED TABLE'):
            add_table_to_pdf(story, section, styles)
        elif section.startswith('**[EMBEDDED IMAGE'):
            add_image_to_pdf(story, section, styles, document_analysis)
        else:
            # Regular text content
            add_text_content_to_pdf(story, section, styles)

def add_formula_to_pdf(story, formula_section: str, styles, formula_style):
    """Add mathematical formula to PDF"""
    
    # Extract formula type
    header_match = re.search(r'\*\*\[MATHEMATICAL FORMULA - ([^\]]+)\]\*\*', formula_section)
    formula_type = header_match.group(1) if header_match else "Mathematical Formula"
    
    story.append(Paragraph(formula_type, styles['Heading3']))
    
    # Extract formula
    formula_match = re.search(r'```\n(.*?)\n```', formula_section, re.DOTALL)
    if formula_match:
        formula_text = formula_match.group(1)
        story.append(Paragraph(formula_text, formula_style))
    
    # Add confidence
    confidence_match = re.search(r'Confidence: ([\d.]+%)', formula_section)
    if confidence_match:
        confidence = confidence_match.group(1)
        story.append(Paragraph(f"<i>Confidence: {confidence}</i>", styles['Normal']))
    
    story.append(Spacer(1, 12))

def add_table_to_pdf(story, table_section: str, styles):
    """Add extracted table to PDF"""
    
    # Extract table metadata
    metadata_match = re.search(r'\*(\d+) rows × (\d+) columns \(Confidence: ([\d.]+%)\)\*', table_section)
    if metadata_match:
        rows, cols, confidence = metadata_match.groups()
        story.append(Paragraph(f'Data Table ({rows}×{cols})', styles['Heading3']))
    
    # Extract and create table
    table_lines = []
    lines = table_section.split('\n')
    
    for line in lines:
        if line.strip().startswith('|') and '---' not in line:
            table_lines.append(line.strip())
    
    if table_lines and len(table_lines) > 1:
        try:
            # Parse table structure
            headers = [cell.strip() for cell in table_lines[0].split('|')[1:-1]]
            data_rows = [headers]
            
            for line in table_lines[1:]:
                if line.strip():
                    row = [cell.strip() for cell in line.split('|')[1:-1]]
                    if len(row) == len(headers):
                        data_rows.append(row)
            
            if len(data_rows) > 1:
                pdf_table = Table(data_rows)
                pdf_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(pdf_table)
            
        except Exception as e:
            logger.warning(f"Error creating PDF table: {e}")
            # Fallback to preformatted text
            story.append(Paragraph('<pre>' + '\n'.join(table_lines) + '</pre>', styles['Code']))
    
    story.append(Spacer(1, 12))

def add_image_to_pdf(story, image_section: str, styles, document_analysis: Dict[str, Any]):
    """Add embedded image to PDF"""
    
    # Extract image reference
    img_match = re.search(r'\[EMBEDDED IMAGE: ([^\]]+)\]', image_section)
    if not img_match:
        return
    
    img_key = img_match.group(1)
    story.append(Paragraph(f'Image: {img_key}', styles['Heading3']))
    
    # Extract base64 data
    b64_match = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)', image_section)
    if b64_match and PIL_AVAILABLE:
        try:
            b64_data = b64_match.group(1)
            img_bytes = base64.b64decode(b64_data)
            img_stream = BytesIO(img_bytes)
            
            # Add image to PDF (max width 4 inches)
            pdf_image = Image(img_stream, width=4*inch, height=3*inch)
            story.append(pdf_image)
            
        except Exception as e:
            logger.warning(f"Error adding image to PDF: {e}")
            story.append(Paragraph(f"[Image: {img_key} - Error loading image]", styles['Normal']))
    else:
        story.append(Paragraph(f"[Image: {img_key}]", styles['Normal']))
    
    story.append(Spacer(1, 12))

def add_text_content_to_pdf(story, text_content: str, styles):
    """Add regular text content to PDF"""
    
    lines = text_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Handle headers
        if line.startswith('# '):
            story.append(Paragraph(line[2:], styles['Heading1']))
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], styles['Heading2']))
        elif line.startswith('### '):
            story.append(Paragraph(line[4:], styles['Heading3']))
        elif line.startswith('---'):
            story.append(Spacer(1, 20))
        elif line.startswith('- ') or line.startswith('* '):
            # Bullet point
            story.append(Paragraph(f"• {line[2:]}", styles['Normal']))
        else:
            # Regular paragraph
            # Handle bold text
            if '**' in line:
                # Simple bold conversion for PDF
                line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            
            story.append(Paragraph(line, styles['Normal']))

def add_standard_content_to_pdf(story, brd_content: Dict[str, Any], styles):
    """Add standard BRD content to PDF (original functionality)"""
    
    for section_name, content in brd_content.items():
        # Add section heading
        story.append(Paragraph(section_name, styles['Heading1']))
        
        if isinstance(content, dict):
            for subsection_name, subcontent in content.items():
                story.append(Paragraph(subsection_name, styles['Heading2']))
                
                if isinstance(subcontent, pd.DataFrame) and not subcontent.empty:
                    # Create table data
                    table_data = [subcontent.columns.tolist()]
                    for _, row in subcontent.iterrows():
                        table_data.append([str(val) if pd.notna(val) else "" for val in row])
                    
                    # Create table
                    table = Table(table_data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(table)
                else:
                    clean_text = re.sub(r'\[IMAGE:\s*[^\]]+\]', '[Image Reference]', str(subcontent))
                    story.append(Paragraph(clean_text, styles['Normal']))
                
                story.append(Spacer(1, 12))
        
        elif isinstance(content, pd.DataFrame) and not content.empty:
            # Create table data
            table_data = [content.columns.tolist()]
            for _, row in content.iterrows():
                table_data.append([str(val) if pd.notna(val) else "" for val in row])
            
            # Create table
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        else:
            clean_text = re.sub(r'\[IMAGE:\s*[^\]]+\]', '[Image Reference]', str(content))
            story.append(Paragraph(clean_text, styles['Normal']))
        
        story.append(
