"""
Enhanced Streamlit UI - Complete Workflow with Query/Location Display
"""
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from agents import (
    FilterAgent, GroupingAgent, GroupReviewAgent,
    LabelingAgent, LabelReviewAgent, RegroupAgent,
    RelabelAgent, SuperiorAgent
)

st.set_page_config(
    page_title="Data Lableing Agent",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .query-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .location-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .workflow-step {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border-left: 6px solid #1f77b4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .filtered-doc {
        background-color: gray;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #f44336;
    }
    .group-card {
        background-color: gray;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 5px solid #ffc107;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .review-box {
        background-color: gray;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #4caf50;
    }
    .reject-box {
        background-color: gray;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #f44336;
    }
    .relabel-box {
        background-color: gray;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #ff9800;
    }
    .doc-card {
        background-color: gray;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 2px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .label-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        margin: 0.2rem;
    }
    .label-relevant { background-color: #d4edda; color: #155724; }
    .label-somewhat { background-color: #fff3cd; color: #856404; }
    .label-acceptable { background-color: #d1ecf1; color: #0c5460; }
    .label-notsure { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)


def load_data(input_id: int) -> dict:
    """Load data from JSON file"""
    try:
        with open("input_data.json", 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        for item in data_list:
            if item.get("id") == input_id:
                return item
        raise ValueError(f"No data found with id={input_id}")
    except FileNotFoundError:
        raise FileNotFoundError("input_data.json not found")


def display_query_location(query: str, location: str):
    """Display Query and Location prominently"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='query-box'>
            🔍 QUERY: {query}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='location-box'>
            📍 LOCATION: {location if location else 'Not Specified'}
        </div>
        """, unsafe_allow_html=True)


def display_complete_workflow(output):
    """Display COMPLETE workflow with ALL details"""
    workflow_steps = output.get("workflow_steps", [])
    
    if not workflow_steps:
        st.warning("⚠️ No workflow steps captured")
        return
    
    st.markdown("---")
    st.header("📊 Complete Workflow Execution Timeline")
    st.info(f"**{len(workflow_steps)} workflow steps captured**")
    
    for i, step in enumerate(workflow_steps, 1):
        step_name = step.get("step_name", "Unknown Step")
        agent_name = step.get("agent_name", "Unknown Agent")
        details = step.get("details", {})
        
        with st.expander(f"**Step {i}: {step_name}** - `{agent_name}`", expanded=(i<=5)):
            
            # === FILTERING STEP ===
            if "Filtering" in step_name:
                st.markdown("### 🔍 Document Filtering")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Documents", details.get('total_docs', 0))
                col2.metric("✅ Kept", details.get('kept', 0))
                col3.metric("🚫 Filtered", details.get('filtered', 0))
                
                filtered_docs = details.get("filtered_docs", [])
                if filtered_docs:
                    st.markdown("#### 🚫 Filtered Out Documents:")
                    for doc in filtered_docs:
                        st.markdown(f"""
                        <div class='filtered-doc'>
                            <h5>{doc['title']}</h5>
                            <small><b>ID:</b> <code>{doc['doc_id']}</code></small><br>
                            <b>❌ Reason:</b> {doc['reason']}
                        </div>
                        """, unsafe_allow_html=True)
            
            # === GROUPING STEP ===
            elif step_name == "Grouping":
                st.markdown("### 📦 Document Grouping")
                groups = details.get("groups", [])
                st.success(f"**{len(groups)} groups created**")
                
                for idx, group in enumerate(groups, 1):
                    st.markdown(f"""
                    <div class='group-card'>
                        <h4>📁 Group {idx}: {group['name']}</h4>
                        <p><b>📊 Document Count:</b> {group['document_count']}</p>
                        <p><b>💡 Theme:</b> {group['theme']}</p>
                        <p><b>📄 Documents:</b></p>
                        <ul>
                            {''.join([f"<li><code>{doc_id}</code>: {title}</li>" for doc_id, title in zip(group['document_ids'], group['document_titles'])])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # === GROUP REVIEW STEP ===
            elif "Group Review" in step_name:
                st.markdown("### 🔎 Group Review")
                approved = details.get("approved", False)
                feedback = details.get("feedback", "")
                attempt = details.get("attempt", 0)
                
                if approved:
                    st.markdown(f"""
                    <div class='review-box'>
                        <h4>✅ APPROVED (Attempt {attempt})</h4>
                        <p><b>Feedback:</b> {feedback}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='reject-box'>
                        <h4>❌ REJECTED (Attempt {attempt})</h4>
                        <p><b>Feedback:</b> {feedback}</p>
                        <p><i>Regrouping required...</i></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # === REGROUPING STEP ===
            elif "Regrouping" in step_name:
                st.markdown("### 🔄 Regrouping")
                st.warning(f"**Groups reorganized based on reviewer feedback**")
                
                groups = details.get("groups", [])
                st.info(f"**{len(groups)} new groups created after regrouping**")
                
                for idx, group in enumerate(groups, 1):
                    st.markdown(f"""
                    <div class='group-card'>
                        <h4>📁 NEW Group {idx}: {group['name']}</h4>
                        <p><b>📊 Document Count:</b> {group['document_count']}</p>
                        <p><b>💡 Theme:</b> {group['theme']}</p>
                        <p><b>📄 Documents:</b></p>
                        <ul>
                            {''.join([f"<li>{title}</li>" for title in group['document_titles']])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # === LABELING STEP ===
            elif step_name == "Labeling":
                st.markdown("### 🏷️ Document Labeling (Group-Based)")
                
                labels_assigned = details.get("labels_assigned", {})
                cols = st.columns(4)
                labels = ["relevant", "somewhat_relevant", "acceptable", "not_sure"]
                emojis = ["✅", "⚠️", "ℹ️", "❓"]
                
                for col, label, emoji in zip(cols, labels, emojis):
                    col.metric(f"{emoji} {label.replace('_', ' ').title()}", labels_assigned.get(label, 0))
                
                # Show which groups got which labels
                groups_labeled = details.get("groups_labeled", [])
                if groups_labeled:
                    st.markdown("#### 📦 Groups with Their Labels:")
                    for group_info in groups_labeled:
                        label = group_info['label']
                        label_class = label.replace('_', '')
                        st.markdown(f"""
                        <div class='group-card'>
                            <h5>📁 {group_info['group_name']}</h5>
                            <p><b>Label:</b> <span class='label-badge label-{label_class}'>{label.upper()}</span></p>
                            <p><b>Documents ({group_info['document_count']}):</b></p>
                            <ul>
                                {''.join([f"<li>{title}</li>" for title in group_info['document_titles']])}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
            
            # === LABEL REVIEW STEP ===
            elif "Label Review" in step_name:
                st.markdown("### 🔎 Label Review")
                approved = details.get("approved", False)
                feedback = details.get("feedback", "")
                rejected_docs = details.get("rejected_docs", [])
                attempt = details.get("attempt", 0)
                
                if approved:
                    st.markdown(f"""
                    <div class='review-box'>
                        <h4>✅ APPROVED (Attempt {attempt})</h4>
                        <p><b>Feedback:</b> {feedback}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='reject-box'>
                        <h4>❌ REJECTED (Attempt {attempt})</h4>
                        <p><b>Feedback:</b> {feedback}</p>
                        <p><b>Documents requiring relabeling:</b> {len(rejected_docs)}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # === RELABELING STEP ===
            elif "Relabeling" in step_name:
                st.markdown("### 🔄 Relabeling")
                relabeling_details = details.get("relabeling_details", [])
                
                if relabeling_details:
                    st.warning(f"**{len(relabeling_details)} documents relabeled**")
                    
                    for doc_info in relabeling_details:
                        old_label_class = doc_info['old_label'].replace('_', '')
                        new_label_class = doc_info['new_label'].replace('_', '')
                        
                        st.markdown(f"""
                        <div class='relabel-box'>
                            <h5>{doc_info['title']}</h5>
                            <p><b>Doc ID:</b> <code>{doc_info['doc_id']}</code></p>
                            <p>
                                <b>OLD Label:</b> <span class='label-badge label-{old_label_class}'>{doc_info['old_label'].upper()}</span>
                                ➡️
                                <b>NEW Label:</b> <span class='label-badge label-{new_label_class}'>{doc_info['new_label'].upper()}</span>
                            </p>
                            <p><b>Old Reason:</b> {doc_info['old_reason']}</p>
                            <p><b>New Reason:</b> {doc_info['new_reason']}</p>
                            <p><b>Confidence:</b> {doc_info['confidence'].upper()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No detailed relabeling information available")
            
            st.markdown("---")


def display_final_results(output):
    """Display final labeling results"""
    report = output.get("detailed_report", {})
    labeling_details = report.get("labeling_details", {})
    filtered_docs = report.get("filtered_documents", [])
    
    st.markdown("---")
    st.header("📋 Final Labeling Results")
    
    tabs = st.tabs(["✅ Relevant", "⚠️ Somewhat", "ℹ️ Acceptable", "❓ Not Sure", "🚫 Filtered"])
    
    label_keys = ["relevant", "somewhat_relevant", "acceptable", "not_sure"]
    
    for tab, label_key in zip(tabs[:4], label_keys):
        with tab:
            docs = labeling_details.get(label_key, [])
            if docs:
                st.markdown(f"**{len(docs)} documents**")
                for doc in docs:
                    st.markdown(f"""
                    <div class='doc-card'>
                        <h4>{doc['title']}</h4>
                        <p><b>ID:</b> <code>{doc['doc_id']}</code></p>
                        <p><b>Confidence:</b> {doc['confidence'].upper()}</p>
                        <p><b>Reasoning:</b> {doc['reason']}</p>
                        <p><b>Labeled by:</b> {doc['labeled_by']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No documents in this category")
    
    with tabs[4]:
        if filtered_docs:
            st.markdown(f"**{len(filtered_docs)} documents filtered**")
            for doc in filtered_docs:
                st.markdown(f"""
                <div class='filtered-doc'>
                    <h4>{doc['title']}</h4>
                    <p><b>ID:</b> <code>{doc['doc_id']}</code></p>
                    <p><b>Reason:</b> {doc['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No documents were filtered out")


def main():
    """Main Streamlit app"""
    
    st.markdown("<h1 class='main-header'>🏷️ Data Labeling Agent</h1>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        input_id = st.number_input(
            "Dataset ID",
            min_value=1,
            max_value=100,
            value=1,
            step=1
        )
        
        st.markdown("---")
        run_button = st.button("🚀 Run Labeling", type="primary", use_container_width=True)
        st.markdown("---")
        
        st.info("""
        **Complete Workflow:**
        1. 🔍 **Filtering**
        2. 📦 **Grouping**
        3. 🔎 **Group Review**
        4. 🔄 **Regrouping** (if needed)
        5. 🏷️ **Labeling** (group-based)
        6. 🔎 **Label Review**
        7. 🔄 **Relabeling** (if needed)
        8. ✅ **Final Results**
        """)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success("✅ API Key Loaded")
        else:
            st.error("❌ No API Key")
    
    if run_button:
        try:
            with st.spinner("📂 Loading data..."):
                data = load_data(input_id)
            
            st.success(f"✓ Loaded dataset {input_id}")
            
            # DISPLAY QUERY AND LOCATION PROMINENTLY
            query = data.get("data", {}).get("text", "")
            location = data.get("data", {}).get("location", "")
            display_query_location(query, location)
            
            with st.spinner("🤖 Initializing agents..."):
                filter_agent = FilterAgent()
                grouping_agent = GroupingAgent()
                group_review_agent = GroupReviewAgent()
                labeling_agent = LabelingAgent()
                label_review_agent = LabelReviewAgent()
                regroup_agent = RegroupAgent()
                relabel_agent = RelabelAgent()
                superior_agent = SuperiorAgent(
                    filter_agent, grouping_agent, group_review_agent,
                    labeling_agent, label_review_agent, regroup_agent, relabel_agent
                )
            
            st.success("✓ Agents initialized")
            
            with st.spinner("🔄 Processing documents..."):
                output = superior_agent.process_documents(data.get("data", {}))
            
            # Display complete workflow
            display_complete_workflow(output)
            
            # Display final results
            display_final_results(output)
            
            # Download section
            st.markdown("---")
            st.header("💾 Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "📥 Download Full Output",
                    json.dumps(output, indent=2),
                    f"output_id_{input_id}.json",
                    "application/json"
                )
            
            with col2:
                st.download_button(
                    "📥 Download Report",
                    json.dumps(output.get("detailed_report", {}), indent=2),
                    f"report_id_{input_id}.json",
                    "application/json"
                )
            
            st.balloons()
            st.success("✅ Processing Complete!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    else:
        st.info("👈 Enter a dataset ID and click 'Run Labeling' to start")


if __name__ == "__main__":
    main()
