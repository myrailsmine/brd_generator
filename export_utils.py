"""
Export Utilities for BRD Generation
"""

import pandas as pd
import re
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
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not available. Word export will be disabled.")

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available. PDF export will be disabled.")

def export_to_word_docx(brd_content: Dict[str, Any]) -> BytesIO:
    """Export BRD content to Word document"""
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx not available. Please install it to use Word export.")
    
    try:
        doc = docx.Document()
        
        # Add title
        title = doc.add_heading('Business Requirements Document', 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add generation date
        doc.add_paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}")
        doc.add_page_break()
        
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
                        # Add text content (removing image references for Word)
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
        
        # Save to bytes
        doc_bytes = BytesIO()
        doc.save(doc_bytes)
        doc_bytes.seek(0)
        return doc_bytes
    
    except Exception as e:
        logger.error(f"Error exporting to Word: {e}")
        raise

def export_to_pdf(brd_content: Dict[str, Any]) -> BytesIO:
    """Export BRD content to PDF"""
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
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.darkblue,
            spaceAfter=12
        )
        
        # Add title
        story.append(Paragraph("Business Requirements Document", title_style))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        for section_name, content in brd_content.items():
            # Add section heading
            story.append(Paragraph(section_name, heading_style))
            
            if isinstance(content, dict):
                # Handle subsections
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
            
            story.append(Spacer(1, 20))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    except Exception as e:
        logger.error(f"Error exporting to PDF: {e}")
        raise

def export_to_excel(brd_content: Dict[str, Any]) -> BytesIO:
    """Export BRD content to Excel"""
    try:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            sheet_count = 1
            
            for section_name, content in brd_content.items():
                if isinstance(content, dict):
                    for subsection_name, subcontent in content.items():
                        if isinstance(subcontent, pd.DataFrame) and not subcontent.empty:
                            # Create safe sheet name
                            sheet_name = f"Sheet{sheet_count}_{subsection_name.split('.')[0]}"[:31]
                            sheet_name = re.sub(r'[^\w\s-]', '', sheet_name).strip()[:31]
                            
                            try:
                                subcontent.to_excel(writer, sheet_name=sheet_name, index=False)
                                sheet_count += 1
                            except Exception as e:
                                logger.warning(f"Error writing sheet {sheet_name}: {e}")
                elif isinstance(content, pd.DataFrame) and not content.empty:
                    # Create safe sheet name  
                    sheet_name = f"Sheet{sheet_count}_{section_name.split('.')[0]}"[:31]
                    sheet_name = re.sub(r'[^\w\s-]', '', sheet_name).strip()[:31]
                    
                    try:
                        content.to_excel(writer, sheet_name=sheet_name, index=False)
                        sheet_count += 1
                    except Exception as e:
                        logger.warning(f"Error writing sheet {sheet_name}: {e}")
        
        buffer.seek(0)
        return buffer
    
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        raise

def export_to_json(brd_content: Dict[str, Any]) -> str:
    """Export BRD content to JSON format"""
    try:
        import json
        
        # Convert DataFrames to dictionaries for JSON serialization
        json_content = {}
        for section_name, content in brd_content.items():
            if isinstance(content, dict):
                json_content[section_name] = {}
                for subsection_name, subcontent in content.items():
                    if isinstance(subcontent, pd.DataFrame):
                        json_content[section_name][subsection_name] = subcontent.to_dict('records')
                    else:
                        json_content[section_name][subsection_name] = str(subcontent)
            elif isinstance(content, pd.DataFrame):
                json_content[section_name] = content.to_dict('records')
            else:
                json_content[section_name] = str(content)
        
        return json.dumps(json_content, indent=2, ensure_ascii=False)
    
    except Exception as e:
        logger.error(f"Error exporting to JSON: {e}")
        raise
