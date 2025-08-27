"""
Analytics Dashboard Components
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from config.app_config import ENHANCED_BRD_STRUCTURE
from utils.logger import get_logger

logger = get_logger(__name__)

# Optional imports for visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Some visualizations will be disabled.")

def create_compliance_dashboard():
    """Create an interactive compliance dashboard"""
    st.subheader("Compliance Dashboard")
    
    # Calculate overall metrics
    total_sections = len(ENHANCED_BRD_STRUCTURE)
    completed_sections = len(st.session_state.get('brd_content', {}))
    
    # Quality scoring
    quality_scores = list(st.session_state.get('quality_scores', {}).values())
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="quality-score">{avg_quality:.0f}%</div>
            <div>Overall Quality</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        completion_rate = (completed_sections / total_sections) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="quality-score">{completion_rate:.0f}%</div>
            <div>Completion Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        compliance_checks = st.session_state.get('compliance_checks', [])
        risk_count = len([c for c in compliance_checks if c.severity == 'error'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="quality-score">{risk_count}</div>
            <div>High Risk Items</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        approval_chain = st.session_state.get('approval_chain', [])
        pending_approvals = len([a for a in approval_chain if a.get('status') == 'pending'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="quality-score">{pending_approvals}</div>
            <div>Pending Approvals</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quality trend chart
    if quality_scores and PLOTLY_AVAILABLE:
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(quality_scores) + 1)),
                y=quality_scores,
                mode='lines+markers',
                name='Quality Score',
                line=dict(color='#667eea', width=3)
            ))
            fig.update_layout(
                title="Quality Score by Section",
                xaxis_title="Section Number",
                yaxis_title="Quality Score (%)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating quality chart: {e}")
            st.error("Error creating quality visualization")
    elif quality_scores:
        # Fallback to simple bar chart
        df_quality = pd.DataFrame({
            'Section': range(1, len(quality_scores) + 1),
            'Quality Score': quality_scores
        })
        st.bar_chart(df_quality.set_index('Section'))

def create_stakeholder_matrix():
    """Create interactive stakeholder influence/interest matrix"""
    st.subheader("Stakeholder Analysis Matrix")
    
    # Sample stakeholder data (in real app, this would come from the BRD)
    stakeholders_data = [
        {"name": "Business Sponsor", "interest": 9, "influence": 9, "category": "Champion"},
        {"name": "Compliance Officer", "interest": 8, "influence": 7, "category": "Key Player"},
        {"name": "IT Team", "interest": 6, "influence": 8, "category": "Key Player"},
        {"name": "End Users", "interest": 7, "influence": 4, "category": "Subject"},
        {"name": "Legal Team", "interest": 8, "influence": 6, "category": "Key Player"},
    ]
    
    if PLOTLY_AVAILABLE:
        try:
            df_stakeholders = pd.DataFrame(stakeholders_data)
            
            fig = px.scatter(
                df_stakeholders, 
                x="interest", 
                y="influence",
                text="name",
                color="category",
                size_max=60,
                title="Stakeholder Interest vs Influence Matrix"
            )
            
            fig.update_traces(textposition="top center")
            fig.update_layout(
                xaxis_title="Interest Level (1-10)",
                yaxis_title="Influence Level (1-10)",
                height=400
            )
            
            # Add quadrant lines
            fig.add_hline(y=5, line_dash="dash", line_color="gray")
            fig.add_vline(x=5, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating stakeholder matrix: {e}")
            # Fallback to table
            df_stakeholders = pd.DataFrame(stakeholders_data)
            st.dataframe(df_stakeholders)
    else:
        # Fallback to table display
        df_stakeholders = pd.DataFrame(stakeholders_data)
        st.dataframe(df_stakeholders)

def create_risk_heatmap():
    """Create risk assessment heatmap"""
    st.subheader("Risk Heat Map")
    
    # Sample risk data
    risks = [
        {"risk": "Regulatory Changes", "probability": 7, "impact": 9},
        {"risk": "Timeline Delays", "probability": 6, "impact": 6},
        {"risk": "Budget Overrun", "probability": 5, "impact": 7},
        {"risk": "Stakeholder Conflicts", "probability": 4, "impact": 5},
        {"risk": "Technical Complexity", "probability": 8, "impact": 6},
    ]
    
    if PLOTLY_AVAILABLE:
        try:
            df_risks = pd.DataFrame(risks)
            df_risks["risk_score"] = df_risks["probability"] * df_risks["impact"]
            
            fig = px.scatter(
                df_risks,
                x="probability",
                y="impact",
                size="risk_score",
                text="risk",
                color="risk_score",
                color_continuous_scale="Reds",
                title="Risk Probability vs Impact Analysis"
            )
            
            fig.update_traces(textposition="top center")
            fig.update_layout(
                xaxis_title="Probability (1-10)",
                yaxis_title="Impact (1-10)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating risk heatmap: {e}")
            # Fallback to table
            df_risks = pd.DataFrame(risks)
            df_risks["risk_score"] = df_risks["probability"] * df_risks["impact"]
            st.dataframe(df_risks)
    else:
        # Fallback to table display
        df_risks = pd.DataFrame(risks)
        df_risks["risk_score"] = df_risks["probability"] * df_risks["impact"]
        st.dataframe(df_risks)
