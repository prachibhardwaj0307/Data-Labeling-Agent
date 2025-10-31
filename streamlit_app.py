"""
Enhanced Streamlit UI - Complete Workflow with Label Studio API Integration
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
from utils.label_studio_client import LabelStudioClient


st.set_page_config(
    page_title="Data Labeling Agent",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state
if 'label_overrides' not in st.session_state:
    st.session_state.label_overrides = {}

if 'current_data' not in st.session_state:
    st.session_state.current_data = None

if 'current_output' not in st.session_state:
    st.session_state.current_output = None

if 'current_task_id' not in st.session_state:
    st.session_state.current_task_id = None

if 'current_annotation_id' not in st.session_state:
    st.session_state.current_annotation_id = None


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
    .existing-label-header {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
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
        background-color: #44444E;
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
        background-color: #44444E;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 2px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .existing-doc-card {
        background-color: #44444E;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 2px solid ;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .content-preview {
        background-color: black;
        padding: 10px;
        margin-top: 10px;
        border-radius: 5px;
        border: 1px solid #dee2e6;
        max-height: 300px;
        overflow-y: auto;
    }
    .move-to-container {
        background-color: #fff8e1;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border: 2px solid #ffc107;
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
    .label-irrelevant { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)


def get_label_studio_client():
    """Get Label Studio client from environment variables"""
    base_url = os.getenv("LABEL_STUDIO_URL")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    
    if not base_url or not api_key:
        raise ValueError("LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY must be set in .env file")
    
    return LabelStudioClient(base_url, api_key)


def load_data_from_api(task_id: int) -> dict:
    """Load data from Label Studio API (only ground_truth annotation)"""
    try:
        client = get_label_studio_client()
        data = client.get_task(task_id)
        
        if data.get("annotations") and len(data["annotations"]) > 0:
            st.session_state.current_annotation_id = data["annotations"][0]["id"]
        
        return data
        
    except Exception as e:
        raise Exception(f"Failed to load task {task_id}: {str(e)}")


def generate_updated_ranker(output, data):
    """Generate updated ranker dict for Label Studio (includes previously labeled docs)"""
    final_output = apply_label_overrides(output)
    labeling_details = final_output["detailed_report"]["labeling_details"]
    
    # Build ranker dict with newly labeled documents
    ranker = {
        "relevant": [],
        "somewhat_relevant": [],
        "acceptable": [],
        "not_sure": [],
        "irrelevant": [],
        "New Doc": []
    }
    
    # Add all newly labeled documents
    for label_key in ["relevant", "somewhat_relevant", "acceptable", "not_sure", "irrelevant"]:
        docs = labeling_details.get(label_key, [])
        for doc in docs:
            ranker[label_key].append(doc["doc_id"])
    
    # Filtered documents go to irrelevant
    filtered_docs = final_output["detailed_report"].get("filtered_documents", [])
    for doc in filtered_docs:
        if doc["doc_id"] not in ranker["irrelevant"]:
            ranker["irrelevant"].append(doc["doc_id"])
    
    # ✅ ADD: Merge previously labeled documents (not re-processed)
    annotations_list = data.get("annotations", [])
    if annotations_list:
        try:
            existing_annotations = annotations_list[0]["result"][0]["value"]["ranker"]
            new_doc_ids = set(existing_annotations.get("New Doc", []))
            
            # Add existing labeled docs (those NOT in "New Doc")
            for label_key in ["relevant", "somewhat_relevant", "acceptable", "not_sure", "irrelevant"]:
                existing_doc_ids = existing_annotations.get(label_key, [])
                for doc_id in existing_doc_ids:
                    # Only add if NOT in new_doc_ids AND not already in ranker
                    if doc_id not in new_doc_ids and doc_id not in ranker[label_key]:
                        ranker[label_key].append(doc_id)
        except:
            pass  # If no existing annotations, just continue with new labels
    
    return ranker


def save_results_to_label_studio(task_id: int, annotation_id: int, updated_ranker: dict):
    """Save updated labels back to Label Studio (PATCH)"""
    try:
        client = get_label_studio_client()
        success = client.update_task_annotation(task_id, annotation_id, updated_ranker)
        return success
    except Exception as e:
        st.error(f"Failed to save to Label Studio: {str(e)}")
        return False


def create_new_annotation(task_id: int, ranker: dict, ground_truth: bool = False):
    """Create a new annotation in Label Studio (POST)"""
    try:
        client = get_label_studio_client()
        result = client.create_new_annotation(task_id, ranker, ground_truth)
        return result
    except Exception as e:
        st.error(f"Failed to create annotation: {str(e)}")
        return None


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
            
            if "Filtering" in step_name:
                st.markdown("### 🔍 Document Filtering")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Documents", details.get('total_new_docs', 0))
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
            
            elif step_name == "Labeling":
                st.markdown("### 🏷️ Document Labeling (Group-Based)")
                
                labels_assigned = details.get("labels_assigned", {})
                cols = st.columns(4)
                labels = ["relevant", "somewhat_relevant", "acceptable", "not_sure"]
                emojis = ["✅", "⚠️", "ℹ️", "❓"]
                
                for col, label, emoji in zip(cols, labels, emojis):
                    col.metric(f"{emoji} {label.replace('_', ' ').title()}", labels_assigned.get(label, 0))
                
                examples_used = details.get("examples_used", {})
                if any(examples_used.values()):
                    st.info(f"📚 Used {sum(examples_used.values())} reference examples: "
                           f"RELEVANT: {examples_used.get('relevant', 0)}, "
                           f"SOMEWHAT: {examples_used.get('somewhat_relevant', 0)}, "
                           f"ACCEPTABLE: {examples_used.get('acceptable', 0)}")
                
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


def get_available_labels(current_label):
    """Get available labels for moving (excluding current label)"""
    all_labels = ["relevant", "somewhat_relevant", "acceptable", "not_sure", "irrelevant"]
    return [label for label in all_labels if label != current_label]


def apply_label_overrides(output):
    """Apply user's label overrides to output data (includes filtered docs)"""
    if not st.session_state.label_overrides:
        return output
    
    import copy
    output = copy.deepcopy(output)
    labeling_details = output["detailed_report"]["labeling_details"]
    filtered_docs = output["detailed_report"].get("filtered_documents", [])
    
    moved_docs = {}
    
    # Check docs that need to be moved
    for doc_id, new_label in st.session_state.label_overrides.items():
        # Check in current labeled docs
        found = False
        for old_label_key in list(labeling_details.keys()):
            for doc in labeling_details[old_label_key]:
                if doc['doc_id'] == doc_id:
                    moved_docs[doc_id] = {
                        'doc': doc,
                        'old_label': old_label_key,
                        'new_label': new_label,
                        'from_filtered': False
                    }
                    found = True
                    break
            if found:
                break
        
        # ✅ NEW: Check in filtered docs
        if not found:
            for doc in filtered_docs:
                if doc['doc_id'] == doc_id:
                    moved_docs[doc_id] = {
                        'doc': doc,
                        'old_label': 'filtered',
                        'new_label': new_label,
                        'from_filtered': True
                    }
                    break
    
    # Remove docs from old labels
    for doc_id, move_info in moved_docs.items():
        if move_info['from_filtered']:
            # Remove from filtered_docs list
            output["detailed_report"]["filtered_documents"] = [
                doc for doc in filtered_docs if doc['doc_id'] != doc_id
            ]
        else:
            # Remove from old label category
            old_label = move_info['old_label']
            labeling_details[old_label] = [
                doc for doc in labeling_details[old_label] if doc['doc_id'] != doc_id
            ]
    
    # Add docs to new labels
    for doc_id, move_info in moved_docs.items():
        new_label = move_info['new_label']
        doc = move_info['doc'].copy()
        
        if move_info['from_filtered']:
            doc['reason'] = f"[MANUAL OVERRIDE] Moved from FILTERED by user. Original: {doc['reason']}"
        else:
            doc['reason'] = f"[MANUAL OVERRIDE] Moved from {move_info['old_label']} by user. Original: {doc['reason']}"
        
        doc['labeled_by'] = "User (Manual Override)"
        
        if new_label not in labeling_details:
            labeling_details[new_label] = []
        labeling_details[new_label].append(doc)
    
    return output

def display_final_results(output, data):
    """Display final labeling results WITH MOVE TO DROPDOWN"""
    
    output = apply_label_overrides(output)
    
    report = output.get("detailed_report", {})
    labeling_details = report.get("labeling_details", {})
    filtered_docs = report.get("filtered_documents", [])
    
    items = data.get("data", {}).get("items", [])
    items_map = {item["id"]: item for item in items}
    
    st.markdown("---")
    st.header("📋 Final Labeling Results (With Manual Override)")
    
    stats = report.get("statistics", {})
    if stats:
        st.markdown("### 📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Docs", stats.get("total_documents", 0))
        col2.metric("Already Labeled", stats.get("existing_labeled", 0))
        col3.metric("New Processed", stats.get("new_documents_processed", 0))
        col4.metric("Newly Labeled", stats.get("newly_labeled", 0))
    
    current_counts = {
        "relevant": len(labeling_details.get("relevant", [])),
        "somewhat_relevant": len(labeling_details.get("somewhat_relevant", [])),
        "acceptable": len(labeling_details.get("acceptable", [])),
        "not_sure": len(labeling_details.get("not_sure", [])),
        "irrelevant": len(labeling_details.get("irrelevant", []))
    }
    
    if st.session_state.label_overrides:
        st.warning(f"⚠️ **{len(st.session_state.label_overrides)} manual overrides applied**")
        if st.button("🔄 Reset All Overrides"):
            st.session_state.label_overrides = {}
            st.rerun()
    
    st.info(f"📊 **Current Distribution:** "
           f"RELEVANT: {current_counts['relevant']} | "
           f"SOMEWHAT: {current_counts['somewhat_relevant']} | "
           f"ACCEPTABLE: {current_counts['acceptable']} | "
           f"NOT SURE: {current_counts['not_sure']} | "
           f"IRRELEVANT: {current_counts['irrelevant']}")
    
    tabs = st.tabs(["✅ Relevant", "⚠️ Somewhat", "ℹ️ Acceptable", "❓ Not Sure", "🚫 Irrelevant", "🗑️ Filtered"])
    
    label_keys = ["relevant", "somewhat_relevant", "acceptable", "not_sure", "irrelevant"]
    
    label_display_map = {
        "relevant": "✅ Relevant",
        "somewhat_relevant": "⚠️ Somewhat Relevant",
        "acceptable": "ℹ️ Acceptable",
        "not_sure": "❓ Not Sure",
        "irrelevant": "🚫 Irrelevant"
    }
    
    for tab, label_key in zip(tabs[:5], label_keys):
        with tab:
            docs = labeling_details.get(label_key, [])
            if docs:
                st.markdown(f"**{len(docs)} documents in '{label_key.replace('_', ' ').title()}'**")
                
                for idx, doc in enumerate(docs):
                    doc_id = doc['doc_id']
                    
                    doc_content = ""
                    if doc_id in items_map:
                        doc_content = items_map[doc_id].get('html', 'No content')[:500]
                    
                    st.markdown(f"""
                    <div class='doc-card'>
                        <h4>{doc['title']}</h4>
                        <p><b>ID:</b> <code>{doc_id}</code></p>
                        <p><b>Confidence:</b> {doc['confidence'].upper()}</p>
                        <p><b>Reasoning:</b> {doc['reason']}</p>
                        <p><b>Labeled by:</b> {doc['labeled_by']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📄 View Content Preview"):
                        st.markdown(f"""
                        <div class='content-preview'>
                        {doc_content}...
                        </div>
                        """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        available_labels = get_available_labels(label_key)
                        options = ["-- Select to Move --"] + [label_display_map[lbl] for lbl in available_labels]
                        
                        select_key = f"move_{label_key}_{doc_id}_{idx}"
                        
                        selected = st.selectbox(
                            "Select new label:",
                            options,
                            key=select_key,
                            label_visibility="collapsed"
                        )
                        
                        if selected != "-- Select to Move --":
                            for key, display in label_display_map.items():
                                if display == selected:
                                    st.session_state.label_overrides[doc_id] = key
                                    st.success(f"✓ Moving to: {selected}")
                                    st.rerun()
                                    break
                    
                    with col2:
                        if doc_id in st.session_state.label_overrides:
                            target_label = st.session_state.label_overrides[doc_id]
                            st.info(f"➡️ Will move to: {label_display_map[target_label]}")
                    
                    st.markdown("---")
            else:
                st.info(f"No documents in '{label_key.replace('_', ' ').title()}' category")
    
    with tabs[5]:
        if filtered_docs:
            st.markdown(f"**{len(filtered_docs)} documents filtered**")
            
            for idx, doc in enumerate(filtered_docs):
                doc_id = doc['doc_id']
                
                # Get content from items_map
                doc_content = ""
                if doc_id in items_map:
                    doc_content = items_map[doc_id].get('html', 'No content')[:500]
                
                # Display filtered document card
                st.markdown(f"""
                <div class='filtered-doc'>
                    <h4>{doc['title']}</h4>
                    <p><b>ID:</b> <code>{doc_id}</code></p>
                    <p><b>Reason:</b> {doc['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Content preview
                with st.expander("📄 View Content Preview"):
                    st.markdown(f"""
                    <div class='content-preview'>
                    {doc_content}...
                    </div>
                    """, unsafe_allow_html=True)
                
                # MOVE TO section (from filtered to a label category)
                st.markdown("#### 🔄 Move Document To:")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # All labels available for filtered docs
                    all_labels = ["relevant", "somewhat_relevant", "acceptable", "not_sure", "irrelevant"]
                    options = ["-- Select to Move --"] + [label_display_map[lbl] for lbl in all_labels]
                    
                    select_key = f"move_filtered_{doc_id}_{idx}"
                    
                    selected = st.selectbox(
                        "Select label:",
                        options,
                        key=select_key,
                        label_visibility="collapsed"
                    )
                    
                    if selected != "-- Select to Move --":
                        # Move filtered doc to selected label
                        for key, display in label_display_map.items():
                            if display == selected:
                                st.session_state.label_overrides[doc_id] = key
                                st.success(f"✓ Moving to: {selected}")
                                st.rerun()
                                break
                
                with col2:
                    if doc_id in st.session_state.label_overrides:
                        target_label = st.session_state.label_overrides[doc_id]
                        st.info(f"➡️ Will move to: {label_display_map[target_label]}")
                
                st.markdown("---")
        else:
            st.info("No documents were filtered out")


def display_existing_labels(data, output):
    """Display labels that were already in the dataset"""
    
    annotations_list = data.get("annotations", [])
    if not annotations_list:
        return
    
    try:
        existing_annotations = annotations_list[0]["result"][0]["value"]["ranker"]
    except:
        return
    
    new_doc_ids = set(existing_annotations.get("New Doc", []))
    
    existing_labeled_count = 0
    for label_key in ["relevant", "somewhat_relevant", "acceptable", "not_sure", "irrelevant"]:
        doc_ids = existing_annotations.get(label_key, [])
        for doc_id in doc_ids:
            if doc_id not in new_doc_ids:
                existing_labeled_count += 1
    
    if existing_labeled_count == 0:
        return
    
    st.markdown("---")
    st.markdown("""
    <div class='existing-label-header'>
        📦 PREVIOUSLY LABELED DOCUMENTS (Not Re-processed by Agent)
    </div>
    """, unsafe_allow_html=True)
    
    st.info(f"**{existing_labeled_count} documents** were already labeled and preserved (not re-processed by agent)")
    
    items = data.get("data", {}).get("items", [])
    items_map = {item["id"]: item for item in items}
    
    tabs = st.tabs([
        "✅ Relevant (Existing)", 
        "⚠️ Somewhat (Existing)", 
        "ℹ️ Acceptable (Existing)",
        "❓ Not Sure (Existing)",
        "🚫 Irrelevant (Existing)"
    ])
    
    label_keys = ["relevant", "somewhat_relevant", "acceptable", "not_sure", "irrelevant"]
    
    for tab, label_key in zip(tabs, label_keys):
        with tab:
            doc_ids = existing_annotations.get(label_key, [])
            
            existing_docs = []
            for doc_id in doc_ids:
                if doc_id not in new_doc_ids:
                    existing_docs.append(doc_id)
            
            if existing_docs:
                st.markdown(f"**{len(existing_docs)} documents with existing '{label_key}' label**")
                
                for doc_id in existing_docs:
                    if doc_id in items_map:
                        item = items_map[doc_id]
                        st.markdown(f"""
                        <div class='existing-doc-card'>
                            <h4>📄 {item['title']}</h4>
                            <p><b>ID:</b> <code>{doc_id}</code></p>
                            <p><b>Label:</b> <span class='label-badge label-{label_key.replace('_', '')}'>{label_key.upper().replace('_', ' ')}</span></p>
                            <p><b>Status:</b> <span style='color: blue; font-weight: bold;'>✓ Preserved from previous labeling</span></p>
                            <details>
                                <summary><b>View Content Preview</b></summary>
                                <div class='content-preview'>
                                {item.get('html', 'No content')[:500]}...
                                </div>
                            </details>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info(f"No existing documents with '{label_key}' label")


def main():
    """Main Streamlit app with Label Studio API integration"""
    
    st.markdown("<h1 class='main-header'>🏷️ Data Labeling Agent</h1>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        task_id = st.number_input(
            "Label Studio Task ID",
            min_value=1,
            max_value=999999,
            value=35851,
            step=1,
            help="Enter the Task ID from Label Studio"
        )
        
        st.markdown("---")
        run_button = st.button("🚀 Run Labeling", type="primary", use_container_width=True)
        
        if st.session_state.current_output is not None:
            st.markdown("---")
            if st.button("💾 Save to Label Studio (Update)", type="secondary", use_container_width=True):
                if st.session_state.current_task_id and st.session_state.current_annotation_id:
                    updated_ranker = generate_updated_ranker(st.session_state.current_output, st.session_state.current_data)
                    
                    with st.spinner("Updating Label Studio annotation..."):
                        success = save_results_to_label_studio(
                            st.session_state.current_task_id,
                            st.session_state.current_annotation_id,
                            updated_ranker
                        )
                    
                    if success:
                        st.success("✅ Successfully updated Label Studio!")
                        st.balloons()
                    else:
                        st.error("❌ Failed to update Label Studio")
        
        st.markdown("---")
        
        st.info("""
        **Complete Workflow:**
        1. 🔍 **Filtering** (New Docs Only)
        2. 📦 **Grouping** (By Topic + Year)
        3. 🔎 **Group Review**
        4. 🔄 **Regrouping** (if needed)
        5. 🏷️ **Labeling** (Year-Based)
        6. 🔎 **Label Review** (Max 10 RELEVANT)
        7. 🔄 **Relabeling** (if needed)
        8. ✅ **Final Results + Manual Override**
        9. 💾 **Save to Label Studio**
        """)
        
        if st.session_state.label_overrides:
            st.warning(f"⚠️ **{len(st.session_state.label_overrides)} manual changes**")
        
        openai_key = os.getenv("OPENAI_API_KEY")
        ls_url = os.getenv("LABEL_STUDIO_URL")
        ls_key = os.getenv("LABEL_STUDIO_API_KEY")
        
        if openai_key:
            st.success("✅ OpenAI API Key Loaded")
        else:
            st.error("❌ No OpenAI API Key")
        
        if ls_url and ls_key:
            st.success("✅ Label Studio Connected")
        else:
            st.error("❌ Label Studio Not Configured")
    
    if run_button:
        st.session_state.label_overrides = {}
        st.session_state.current_task_id = task_id
        
        try:
            with st.spinner(f"📂 Loading task {task_id} from Label Studio..."):
                data = load_data_from_api(task_id)
                st.session_state.current_data = data
            
            st.success(f"✓ Loaded task {task_id} from Label Studio")
            
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
                output = superior_agent.process_documents(data)
                st.session_state.current_output = output
            
            display_complete_workflow(output)
            display_final_results(output, data)
            display_existing_labels(data, output)
            
            st.markdown("---")
            st.header("💾 Download & Create New Annotation")
            
            final_output = apply_label_overrides(output)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    "📥 Download Full Output (with manual changes)",
                    json.dumps(final_output, indent=2),
                    f"output_task_{task_id}_final.json",
                    "application/json"
                )
            
            with col2:
                st.download_button(
                    "📥 Download Report",
                    json.dumps(final_output.get("detailed_report", {}), indent=2),
                    f"report_task_{task_id}_final.json",
                    "application/json"
                )
            
            with col3:
                if st.button("➕ Create New Annotation", use_container_width=True):
                    updated_ranker = generate_updated_ranker(st.session_state.current_output, st.session_state.current_data)
                    
                    with st.spinner("Creating new annotation in Label Studio..."):
                        result = create_new_annotation(task_id, updated_ranker, ground_truth=False)
                    
                    if result:
                        st.success(f"✅ Created new annotation! ID: {result.get('id')}")
                        st.balloons()
                    else:
                        st.error("❌ Failed to create annotation")
            
            st.success("✅ Processing Complete!")
            st.info("💡 Use 'Save to Label Studio (Update)' to update existing annotation or 'Create New Annotation' to add a new one")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    elif st.session_state.current_output is not None and st.session_state.current_data is not None:
        data = st.session_state.current_data
        output = st.session_state.current_output
        
        query = data.get("data", {}).get("text", "")
        location = data.get("data", {}).get("location", "")
        
        display_query_location(query, location)
        display_complete_workflow(output)
        display_final_results(output, data)
        display_existing_labels(data, output)
        
        st.markdown("---")
        st.header("💾 Download & Create New Annotation")
        
        final_output = apply_label_overrides(output)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📥 Download Full Output (with manual changes)",
                json.dumps(final_output, indent=2),
                f"output_task_{st.session_state.current_task_id}_final.json",
                "application/json"
            )
        
        with col2:
            st.download_button(
                "📥 Download Report",
                json.dumps(final_output.get("detailed_report", {}), indent=2),
                f"report_task_{st.session_state.current_task_id}_final.json",
                "application/json"
            )
        
        with col3:
            if st.button("➕ Create New Annotation", use_container_width=True):
                updated_ranker = generate_updated_ranker(st.session_state.current_output, st.session_state.current_data)
                
                with st.spinner("Creating new annotation in Label Studio..."):
                    result = create_new_annotation(st.session_state.current_task_id, updated_ranker, ground_truth=False)
                
                if result:
                    st.success(f"✅ Created new annotation! ID: {result.get('id')}")
                    st.balloons()
                else:
                    st.error("❌ Failed to create annotation")
    
    else:
        st.info("👈 Enter a Label Studio Task ID and click 'Run Labeling' to start")


if __name__ == "__main__":
    main()
