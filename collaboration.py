"""
Collaboration Hub Component
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

def render_collaboration_hub():
    """Render collaboration hub"""
    st.subheader("Collaboration Hub")
    
    # User management section
    st.write("**Team Members**")
    team_members = [
        {"name": "Alice Johnson", "role": "Business Analyst", "status": "Active", "last_seen": "5 min ago"},
        {"name": "Bob Smith", "role": "Compliance Officer", "status": "Active", "last_seen": "2 hours ago"},
        {"name": "Carol Davis", "role": "Technical Lead", "status": "Away", "last_seen": "1 day ago"},
    ]
    
    for member in team_members:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**{member['name']}**")
        with col2:
            st.write(member['role'])
        with col3:
            if member['status'] == 'Active':
                st.success(member['status'])
            else:
                st.warning(member['status'])
        with col4:
            st.write(member['last_seen'])
    
    st.markdown("---")
    
    # Comments and feedback section
    st.write("**Comments & Feedback**")
    
    # Comment input
    new_comment = st.text_area("Add a comment", placeholder="Share your feedback or ask questions...")
    if st.button("Post Comment"):
        if new_comment:
            if 'comments' not in st.session_state:
                st.session_state.comments = []
            st.session_state.comments.append({
                'user': st.session_state.current_user.name,
                'comment': new_comment,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("Comment added!")
            st.rerun()
    
    # Display comments
    if st.session_state.get('comments'):
        for comment in reversed(st.session_state.comments[-10:]):  # Show last 10 comments
            with st.container():
                st.write(f"**{comment['user']}** - {comment['timestamp']}")
                st.write(comment['comment'])
                st.markdown("---")
    else:
        st.info("No comments yet. Start a discussion!")
    
    # Approval workflow
    st.markdown("---")
    st.write("**Approval Workflow**")
    
    approval_stages = [
        {"stage": "Business Analysis Review", "assignee": "Alice Johnson", "status": "Completed", "date": "2024-01-15"},
        {"stage": "Compliance Review", "assignee": "Bob Smith", "status": "In Progress", "date": "2024-01-16"},
        {"stage": "Technical Review", "assignee": "Carol Davis", "status": "Pending", "date": "2024-01-17"},
        {"stage": "Final Approval", "assignee": "Management", "status": "Pending", "date": "2024-01-18"},
    ]
    
    for stage in approval_stages:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**{stage['stage']}**")
        with col2:
            st.write(stage['assignee'])
        with col3:
            if stage['status'] == 'Completed':
                st.success(stage['status'])
            elif stage['status'] == 'In Progress':
                st.warning(stage['status'])
            else:
                st.info(stage['status'])
        with col4:
            st.write(stage['date'])
    
    # Version history
    st.markdown("---")
    st.write("**Version History**")
    
    versions = [
        {"version": "v1.0", "author": "AI Generator", "date": "2024-01-15 10:30", "changes": "Initial BRD generation"},
        {"version": "v1.1", "author": "Alice Johnson", "date": "2024-01-15 14:20", "changes": "Updated executive summary"},
        {"version": "v1.2", "author": "Bob Smith", "date": "2024-01-16 09:15", "changes": "Added compliance requirements"},
    ]
    
    for version in versions:
        with st.expander(f"{version['version']} - {version['date']}"):
            st.write(f"**Author:** {version['author']}")
            st.write(f"**Changes:** {version['changes']}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Restore {version['version']}", key=f"restore_{version['version']}"):
                    st.info("Version restore functionality coming soon!")
            with col2:
                if st.button(f"Compare {version['version']}", key=f"compare_{version['version']}"):
                    st.info("Version comparison coming soon!")
