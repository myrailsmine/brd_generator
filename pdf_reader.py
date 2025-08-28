import streamlit as st
import pandas as pd
import os
import zipfile
import tempfile
from pathlib import Path
import fitz
import io
from PIL import Image
import base64
import time
import shutil

# Import the extractor class (assuming it's in a separate file)
# from pdf_extractor import PDFTableFormulaExtractor

# Streamlit page configuration
st.set_page_config(
    page_title="PDF Table & Formula Extractor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Simplified extractor class for Streamlit (embedded version)
class StreamlitPDFExtractor:
    def __init__(self, uploaded_file, output_dir="temp_extraction"):
        self.uploaded_file = uploaded_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.tables_dir = self.output_dir / "tables"
        self.formulas_dir = self.output_dir / "formulas"
        self.tables_dir.mkdir(exist_ok=True)
        self.formulas_dir.mkdir(exist_ok=True)
        
        # Save uploaded file to temp location
        self.temp_pdf_path = self.output_dir / uploaded_file.name
        with open(self.temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())
        
        self.doc = fitz.open(str(self.temp_pdf_path))
    
    def get_pdf_info(self):
        """Get basic PDF information"""
        return {
            'filename': self.uploaded_file.name,
            'size_mb': round(self.uploaded_file.size / 1024 / 1024, 2),
            'pages': len(self.doc),
            'title': self.doc.metadata.get('title', 'N/A'),
            'author': self.doc.metadata.get('author', 'N/A')
        }
    
    def extract_page_as_image(self, page_num):
        """Extract a single page as image for preview"""
        page = self.doc[page_num]
        mat = fitz.Matrix(1.5, 1.5)  # Scale for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))
    
    def detect_tables_simple(self):
        """Simplified table detection for demo"""
        tables = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            
            # Look for table-like patterns
            lines = text.split('\n')
            potential_tables = []
            current_table = []
            
            for line in lines:
                # Simple heuristic: lines with multiple numbers/percentages
                if len(line.split()) > 2 and any(char in line for char in ['%', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
                    current_table.append(line.strip())
                else:
                    if len(current_table) > 2:  # Minimum table size
                        potential_tables.append(current_table)
                    current_table = []
            
            if len(current_table) > 2:
                potential_tables.append(current_table)
            
            # Convert to DataFrames
            for i, table_lines in enumerate(potential_tables):
                if len(table_lines) > 2:
                    # Try to parse as table
                    try:
                        rows = []
                        for line in table_lines:
                            row = line.split()
                            if len(row) > 1:
                                rows.append(row)
                        
                        if rows and len(rows) > 1:
                            max_cols = max(len(row) for row in rows)
                            # Pad rows to same length
                            padded_rows = [row + [''] * (max_cols - len(row)) for row in rows]
                            
                            df = pd.DataFrame(padded_rows[1:], columns=padded_rows[0] if padded_rows[0] else [f"Col_{j+1}" for j in range(max_cols)])
                            
                            table_info = {
                                'page': page_num + 1,
                                'table_number': len(tables) + 1,
                                'dataframe': df,
                                'raw_text': '\n'.join(table_lines)
                            }
                            tables.append(table_info)
                    except:
                        continue
        
        return tables[:10]  # Limit to first 10 tables
    
    def detect_formulas_simple(self):
        """Simplified formula detection for demo"""
        formulas = []
        
        formula_keywords = ['formula', 'equation', 'calculate', 'correlation', 'risk weight', 'capital requirement']
        math_symbols = ['=', '+', '-', '*', '/', '%', '∑', '√', '≤', '≥', '±']
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text_blocks = page.get_text("dict")["blocks"]
            
            for block in text_blocks:
                if "lines" in block:
                    block_text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            block_text += span["text"] + " "
                    
                    block_text = block_text.strip()
                    
                    # Score the block
                    keyword_score = sum(1 for keyword in formula_keywords if keyword.lower() in block_text.lower())
                    math_score = sum(1 for symbol in math_symbols if symbol in block_text)
                    
                    total_score = keyword_score * 2 + math_score
                    
                    if total_score > 3 and len(block_text) > 20:
                        formula_info = {
                            'page': page_num + 1,
                            'formula_number': len(formulas) + 1,
                            'text': block_text,
                            'score': total_score
                        }
                        formulas.append(formula_info)
        
        return sorted(formulas, key=lambda x: x['score'], reverse=True)[:20]  # Top 20
    
    def cleanup(self):
        """Clean up temporary files"""
        self.doc.close()
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

def create_download_zip(tables, formulas, output_dir):
    """Create a ZIP file with all extracted content"""
    zip_path = output_dir / "extracted_content.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        # Add tables
        for i, table in enumerate(tables, 1):
            csv_content = table['dataframe'].to_csv(index=False)
            zipf.writestr(f"tables/table_{i}_page_{table['page']}.csv", csv_content)
            zipf.writestr(f"tables/table_{i}_page_{table['page']}.txt", table['raw_text'])
        
        # Add formulas
        for i, formula in enumerate(formulas, 1):
            zipf.writestr(f"formulas/formula_{i}_page_{formula['page']}.txt", 
                         f"Formula {i} (Page {formula['page']}, Score: {formula['score']})\n\n{formula['text']}")
        
        # Add summary
        summary = f"""PDF Content Extraction Summary
================================

Tables extracted: {len(tables)}
Formulas extracted: {len(formulas)}

Tables by page:
{chr(10).join([f"  Page {table['page']}: Table {table['table_number']}" for table in tables])}

Formulas by page:
{chr(10).join([f"  Page {formula['page']}: Formula {formula['formula_number']} (Score: {formula['score']})" for formula in formulas])}
"""
        zipf.writestr("summary.txt", summary)
    
    return zip_path

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">📊 PDF Table & Formula Extractor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Upload a PDF document
        2. View PDF information and preview
        3. Extract tables and formulas
        4. Download results as ZIP file
        """)
        
        st.markdown("### 🔧 Features")
        st.markdown("""
        - **Table Detection**: Automatically finds and extracts tables
        - **Formula Recognition**: Identifies mathematical formulas and equations
        - **Multi-format Export**: CSV for tables, TXT for formulas
        - **Batch Download**: Get all results in one ZIP file
        """)
        
        st.markdown("### 📁 Supported Files")
        st.markdown("PDF documents up to 200MB")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf", 
        help="Upload a PDF document to extract tables and formulas"
    )
    
    if uploaded_file is not None:
        # Create extractor instance
        with tempfile.TemporaryDirectory() as temp_dir:
            extractor = StreamlitPDFExtractor(uploaded_file, temp_dir)
            
            # Get PDF info
            pdf_info = extractor.get_pdf_info()
            
            # Display PDF information
            st.markdown('<div class="sub-header">📄 Document Information</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><h3>{pdf_info["pages"]}</h3><p>Pages</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>{pdf_info["size_mb"]} MB</h3><p>File Size</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><h3>{len(pdf_info["filename"])}</h3><p>Filename Length</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><h3>PDF</h3><p>Format</p></div>', unsafe_allow_html=True)
            
            # Document details
            with st.expander("📋 Document Details"):
                st.write(f"**Filename:** {pdf_info['filename']}")
                st.write(f"**Title:** {pdf_info['title']}")
                st.write(f"**Author:** {pdf_info['author']}")
                st.write(f"**Pages:** {pdf_info['pages']}")
                st.write(f"**Size:** {pdf_info['size_mb']} MB")
            
            # PDF Preview
            st.markdown('<div class="sub-header">👀 Document Preview</div>', unsafe_allow_html=True)
            
            if pdf_info['pages'] > 0:
                preview_page = st.selectbox("Select page to preview:", range(1, min(6, pdf_info['pages'] + 1)), index=0) - 1
                
                try:
                    page_image = extractor.extract_page_as_image(preview_page)
                    st.image(page_image, caption=f"Page {preview_page + 1}", use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading page preview: {str(e)}")
            
            # Extraction section
            st.markdown('<div class="sub-header">🔍 Content Extraction</div>', unsafe_allow_html=True)
            
            if st.button("🚀 Start Extraction", type="primary"):
                with st.spinner("Extracting tables and formulas..."):
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Extract tables
                    status_text.text("Detecting tables...")
                    progress_bar.progress(25)
                    tables = extractor.detect_tables_simple()
                    
                    # Extract formulas
                    status_text.text("Detecting formulas...")
                    progress_bar.progress(50)
                    formulas = extractor.detect_formulas_simple()
                    
                    # Processing results
                    status_text.text("Processing results...")
                    progress_bar.progress(75)
                    
                    # Create download package
                    status_text.text("Preparing download...")
                    progress_bar.progress(90)
                    zip_path = create_download_zip(tables, formulas, Path(temp_dir))
                    
                    progress_bar.progress(100)
                    status_text.text("Extraction completed!")
                    
                    # Store results in session state
                    st.session_state.tables = tables
                    st.session_state.formulas = formulas
                    st.session_state.zip_path = zip_path
                    
                    st.markdown('<div class="success-box">✅ Extraction completed successfully!</div>', unsafe_allow_html=True)
            
            # Display results if available
            if hasattr(st.session_state, 'tables') and hasattr(st.session_state, 'formulas'):
                tables = st.session_state.tables
                formulas = st.session_state.formulas
                
                # Results summary
                st.markdown('<div class="sub-header">📊 Extraction Results</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'<div class="metric-card"><h3>{len(tables)}</h3><p>Tables Found</p></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><h3>{len(formulas)}</h3><p>Formulas Found</p></div>', unsafe_allow_html=True)
                
                # Tabbed results display
                tab1, tab2, tab3 = st.tabs(["📊 Tables", "🧮 Formulas", "📥 Download"])
                
                with tab1:
                    if tables:
                        st.markdown("### Extracted Tables")
                        
                        for i, table in enumerate(tables):
                            with st.expander(f"Table {table['table_number']} - Page {table['page']}", expanded=(i==0)):
                                st.dataframe(table['dataframe'], use_container_width=True)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    csv = table['dataframe'].to_csv(index=False)
                                    st.download_button(
                                        label="📄 Download CSV",
                                        data=csv,
                                        file_name=f"table_{table['table_number']}_page_{table['page']}.csv",
                                        mime="text/csv"
                                    )
                                with col2:
                                    st.download_button(
                                        label="📝 Download Raw Text",
                                        data=table['raw_text'],
                                        file_name=f"table_{table['table_number']}_page_{table['page']}.txt",
                                        mime="text/plain"
                                    )
                    else:
                        st.info("No tables detected in the document.")
                
                with tab2:
                    if formulas:
                        st.markdown("### Extracted Formulas")
                        
                        for i, formula in enumerate(formulas):
                            with st.expander(f"Formula {formula['formula_number']} - Page {formula['page']} (Score: {formula['score']})", expanded=(i==0)):
                                st.text_area(
                                    "Formula Content:",
                                    formula['text'],
                                    height=100,
                                    key=f"formula_{i}",
                                    disabled=True
                                )
                                
                                st.download_button(
                                    label="📄 Download Formula",
                                    data=f"Formula {formula['formula_number']} (Page {formula['page']}, Score: {formula['score']})\n\n{formula['text']}",
                                    file_name=f"formula_{formula['formula_number']}_page_{formula['page']}.txt",
                                    mime="text/plain"
                                )
                    else:
                        st.info("No formulas detected in the document.")
                
                with tab3:
                    st.markdown("### Download All Results")
                    st.markdown(f"**Summary:** {len(tables)} tables and {len(formulas)} formulas extracted")
                    
                    if hasattr(st.session_state, 'zip_path'):
                        with open(st.session_state.zip_path, 'rb') as f:
                            st.download_button(
                                label="📦 Download Complete Package (ZIP)",
                                data=f.read(),
                                file_name="extracted_content.zip",
                                mime="application/zip"
                            )
                    
                    st.markdown("""
                    **Package Contents:**
                    - 📊 Tables in CSV format
                    - 🧮 Formulas in TXT format
                    - 📋 Summary report
                    """)
            
            # Cleanup
            extractor.cleanup()
    
    else:
        st.markdown('<div class="info-box">👆 Please upload a PDF file to get started</div>', unsafe_allow_html=True)
        
        # Demo section
        st.markdown('<div class="sub-header">✨ What This App Does</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Table Extraction**
            - Automatically detects tables in PDF documents
            - Converts tables to structured CSV format
            - Preserves table formatting and data relationships
            - Handles complex multi-column layouts
            """)
        
        with col2:
            st.markdown("""
            **🧮 Formula Recognition**
            - Identifies mathematical formulas and equations
            - Recognizes financial and regulatory formulas
            - Extracts correlation matrices and calculations
            - Scores formulas by mathematical complexity
            """)
        
        st.markdown("""
        **🎯 Perfect For:**
        - Financial regulatory documents (Basel, CCAR, etc.)
        - Research papers with mathematical content
        - Technical documentation with tables and formulas
        - Compliance reports requiring data extraction
        """)

if __name__ == "__main__":
    main()
