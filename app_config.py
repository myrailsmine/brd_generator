"""
Application Configuration
"""

import streamlit as st
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class User:
    id: str
    name: str
    email: str
    role: str

@dataclass
class BRDSection:
    name: str
    content: Any
    quality_score: float
    compliance_status: str
    last_modified: datetime
    modified_by: str
    comments: List[str]
    
@dataclass
class QualityCheck:
    section: str
    check_type: str
    status: str
    message: str
    severity: str

# Enhanced BRD Structure with Quality Metrics
ENHANCED_BRD_STRUCTURE = {
    "1. Executive Summary": {
        "type": "text",
        "description": "High-level summary of business requirements and expected outcomes",
        "quality_criteria": ["completeness", "clarity", "business_value_alignment"],
        "required_elements": ["business_objective", "scope_summary", "success_metrics"]
    },
    "2. Background": {
        "type": "text",
        "description": "Detailed context and background information",
        "quality_criteria": ["regulatory_compliance", "stakeholder_coverage", "historical_context"],
        "required_elements": ["current_state", "drivers_for_change", "regulatory_context"]
    },
    "3. Scope": {
        "type": "parent",
        "subsections": {
            "3.1. In Scope": {
                "type": "table",
                "columns": ["ID", "Description", "Priority", "Owner", "Success Criteria"]
            },
            "3.2. Out of Scope": {
                "type": "table", 
                "columns": ["ID", "Description", "Rationale", "Future Consideration"]
            }
        }
    },
    "4. Stakeholder Analysis": {
        "type": "table",
        "columns": ["Stakeholder", "Role", "Interest Level", "Influence Level", "Communication Strategy", "Approval Required"]
    },
    "5. Assumptions and Dependencies": {
        "type": "parent",
        "subsections": {
            "5.1. Assumptions": {
                "type": "table",
                "columns": ["ID", "Description", "Impact", "Risk Level", "Validation Required"]
            },
            "5.2. Dependencies": {
                "type": "table",
                "columns": ["ID", "Description", "Impact", "Owner", "Target Date", "Status"]
            }
        }
    },
    "6. Business Requirements": {
        "type": "table",
        "columns": ["Unique Rule Ref", "BR ID", "BR Name", "BR Description", "BR Owner", "BR Type", "Priority", "Success Criteria", "Acceptance Criteria"]
    },
    "7. Functional Requirements": {
        "type": "table",
        "columns": ["FR ID", "FR Name", "Description", "Related BR", "Priority", "Complexity", "Owner", "Status"]
    },
    "8. Non-Functional Requirements": {
        "type": "table",
        "columns": ["NFR ID", "Category", "Description", "Metric", "Target Value", "Priority"]
    },
    "9. Risk Assessment": {
        "type": "table",
        "columns": ["Risk ID", "Description", "Probability", "Impact", "Risk Score", "Mitigation Strategy", "Owner"]
    },
    "10. Applicable Regulations": {
        "type": "table",
        "columns": ["Unique Rule Ref", "Regulation", "Section", "Regulatory Text", "Compliance Requirement", "Impact Assessment"]
    },
    "11. Implementation Timeline": {
        "type": "table",
        "columns": ["Phase", "Milestone", "Description", "Start Date", "End Date", "Dependencies", "Owner"]
    },
    "12. Success Metrics and KPIs": {
        "type": "table",
        "columns": ["Metric ID", "Metric Name", "Description", "Baseline", "Target", "Measurement Method", "Frequency"]
    },
    "13. Approval Matrix": {
        "type": "table",
        "columns": ["Role", "Name", "Responsibility", "Approval Level", "Date Required", "Status"]
    },
    "14. Appendix": {
        "type": "table",
        "columns": ["ID", "Name", "Description", "Type", "Location"]
    }
}

def configure_app():
    """Configure Streamlit application"""
    st.set_page_config(
        page_title="AI-Powered BRD Generator Pro",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced UI
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        .quality-score {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        .compliance-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            margin: 0.25rem;
        }
        .badge-excellent { background-color: #10B981; color: white; }
        .badge-good { background-color: #F59E0B; color: white; }
        .badge-needs-attention { background-color: #EF4444; color: white; }
        .timeline-item {
            border-left: 3px solid #667eea;
            padding-left: 1rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """Initialize enhanced session state"""
    defaults = {
        'brd_content': {},
        'extracted_images': {},
        'extracted_formulas': [],
        'generated': False,
        'edited_tables': {},
        'quality_scores': {},
        'compliance_checks': [],
        'document_analysis': {},
        'users': [],
        'current_user': User('user1', 'Current User', 'user@company.com', 'Business Analyst'),
        'comments': [],
        'version_history': [],
        'workflow_status': 'Draft',
        'stakeholders': [],
        'approval_chain': [],
        'document_text': '',
        'llm_config': {
            'base_url': "http://lwnde002xdgpu.sdi.corp.bankofamerica.com:8123/v1",
            'api_key': "dummy",
            'model': "/phoenix/workspaces/nbkm74lv/llama3.3-4bit-awq",
            'temperature': 0.3,
            'max_tokens': 4000
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
