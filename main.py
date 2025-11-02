"""
Main script to process documents from Label Studio
"""
import json
import sys
import os
from dotenv import load_dotenv
from agents.superior_agent import SuperiorAgent
from agents.filter_agent import FilterAgent
from agents.grouping_agent import GroupingAgent
from agents.group_review_agent import GroupReviewAgent
from agents.labeling_agent import LabelingAgent
from agents.label_review_agent import LabelReviewAgent
from agents.regroup_agent import RegroupAgent
from agents.relabel_agent import RelabelAgent
from utils.label_studio_client import LabelStudioClient

load_dotenv()

def get_label_studio_client():
    """Get Label Studio client from environment variables"""
    base_url = os.getenv("LABEL_STUDIO_URL")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    
    if not base_url or not api_key:
        raise ValueError("LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY must be set in .env file")
    
    return LabelStudioClient(base_url, api_key)

def load_data_from_api(task_id: int) -> dict:
    """Load data from Label Studio API"""
    try:
        client = get_label_studio_client()
        data = client.get_task(task_id)
        return data
    except Exception as e:
        raise Exception(f"Failed to load task {task_id}: {str(e)}")

def main(task_id: int):
    """Process documents for a given Label Studio task ID"""
    
    try:
        print(f"📂 Loading task {task_id} from Label Studio...")
        dataset = load_data_from_api(task_id)
        print(f"✓ Loaded task {task_id} from Label Studio")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return

    # Initialize agents
    filter_agent = FilterAgent()
    grouping_agent = GroupingAgent()
    group_review_agent = GroupReviewAgent()
    labeling_agent = LabelingAgent()
    label_review_agent = LabelReviewAgent()
    regroup_agent = RegroupAgent()
    relabel_agent = RelabelAgent()
    
    superior_agent = SuperiorAgent(
        filter_agent,
        grouping_agent,
        group_review_agent,
        labeling_agent,
        label_review_agent,
        regroup_agent,
        relabel_agent
    )
    
    # CRITICAL: Pass the ENTIRE dataset item (includes id, data, annotations)
    result = superior_agent.process_documents(dataset)
    
    # Save results
    output_file = f"output_id_{task_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result["updated_annotations"], f, indent=2)
    
    report_file = f"report_id_{task_id}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(result["detailed_report"], f, indent=2)
    
    workflow_file = f"workflow_id_{task_id}.json"
    with open(workflow_file, 'w', encoding='utf-8') as f:
        json.dump(result["workflow_steps"], f, indent=2)
    
    print(f"\n✅ Results saved:")
    print(f"   - {output_file}")
    print(f"   - {report_file}")
    print(f"   - {workflow_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        task_id = int(sys.argv[1])
    else:
        # Default task ID for testing
        task_id = 35851
    
    main(task_id)