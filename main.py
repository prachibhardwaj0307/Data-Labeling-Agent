"""
Main script to process documents
"""
import json
import sys
from agents.superior_agent import SuperiorAgent
from agents.filter_agent import FilterAgent
from agents.grouping_agent import GroupingAgent
from agents.group_review_agent import GroupReviewAgent
from agents.labeling_agent import LabelingAgent
from agents.label_review_agent import LabelReviewAgent
from agents.regroup_agent import RegroupAgent
from agents.relabel_agent import RelabelAgent

def main(dataset_id: int):
    """Process documents for given dataset ID"""
    
    # Load input data
    with open('input_data.json', 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # Find dataset by ID
    dataset = None
    for item in all_data:
        if item.get("id") == dataset_id:
            dataset = item
            break
    
    if not dataset:
        print(f"❌ Dataset with ID {dataset_id} not found")
        return
    
    print(f"✓ Found dataset {dataset_id}")
    
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
    # NOT just dataset["data"]
    result = superior_agent.process_documents(dataset)
    
    # Save results
    output_file = f"output_id_{dataset_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result["updated_annotations"], f, indent=2)
    
    report_file = f"report_id_{dataset_id}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(result["detailed_report"], f, indent=2)
    
    workflow_file = f"workflow_id_{dataset_id}.json"
    with open(workflow_file, 'w', encoding='utf-8') as f:
        json.dump(result["workflow_steps"], f, indent=2)
    
    print(f"\n✅ Results saved:")
    print(f"   - {output_file}")
    print(f"   - {report_file}")
    print(f"   - {workflow_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_id = int(sys.argv[1])
    else:
        dataset_id = 1
    
    main(dataset_id)
