"""
Main orchestrator for the document labeling system
"""
from dotenv import load_dotenv
load_dotenv()

import json
import sys
import os
from agents import (
    FilterAgent,
    GroupingAgent,
    GroupReviewAgent,
    LabelingAgent,
    LabelReviewAgent,
    RegroupAgent,
    RelabelAgent,
    SuperiorAgent
)

def load_data(input_id: int, data_file: str = "input_data.json") -> dict:
    """Load data by ID from JSON file"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    for item in data_list:
        if item.get("id") == input_id:
            return item
    
    raise ValueError(f"No data found with id={input_id}")

def save_output(output: dict, output_file: str = "output.json"):
    """Save output to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Output saved to {output_file}")

def print_summary(output: dict):
    """Print a summary of results"""
    report = output.get("detailed_report", {})
    stats = report.get("statistics", {})
    
    print("\n" + "="*80)
    print("LABELING SUMMARY")
    print("="*80)
    print(f"Query: {report.get('query', 'N/A')}")
    print(f"Location: {report.get('location', 'N/A')}")
    print(f"\nTotal Documents: {stats.get('total_documents', 0)}")
    print(f"Filtered Out: {stats.get('filtered_out', 0)}")
    print(f"Labeled: {stats.get('successfully_labeled', 0)}")
    print(f"\nLabel Distribution:")
    dist = stats.get('label_distribution', {})
    print(f"  - Relevant: {dist.get('relevant', 0)}")
    print(f"  - Somewhat Relevant: {dist.get('somewhat_relevant', 0)}")
    print(f"  - Acceptable: {dist.get('acceptable', 0)}")
    print(f"  - Not Sure: {dist.get('not_sure', 0)}")
    print(f"  - Irrelevant (Filtered): {dist.get('irrelevant', 0)}")
    print(f"\nReview Cycles:")
    print(f"  - Group Review Attempts: {stats.get('group_review_attempts', 0)}")
    print(f"  - Label Review Attempts: {stats.get('label_review_attempts', 0)}")
    print("="*80)

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <id>")
        print("Example: python main.py 1")
        sys.exit(1)
    
    input_id = int(sys.argv[1])
    
    print(f"\n{'='*80}")
    print(f"DOCUMENT LABELING SYSTEM - Processing ID: {input_id}")
    print(f"{'='*80}\n")
    
    # Check for API keys
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No API key found!")
        print("Please set OPENAI_API_KEY environment variable:")
        print("  export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    print(f"✓ API key found")
    print(f"Loading data for ID: {input_id}...\n")
    
    try:
        data = load_data(input_id)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)
    
    # Initialize all agents
    print("Initializing agents...")
    filter_agent = FilterAgent()
    grouping_agent = GroupingAgent()
    group_review_agent = GroupReviewAgent()
    labeling_agent = LabelingAgent()
    label_review_agent = LabelReviewAgent()
    regroup_agent = RegroupAgent()
    relabel_agent = RelabelAgent()
    
    # Initialize superior agent with all other agents
    superior_agent = SuperiorAgent(
        filter_agent=filter_agent,
        grouping_agent=grouping_agent,
        group_review_agent=group_review_agent,
        labeling_agent=labeling_agent,
        label_review_agent=label_review_agent,
        regroup_agent=regroup_agent,
        relabel_agent=relabel_agent
    )
    
    print("✓ All agents initialized\n")
    
    # Process documents
    try:
        output = superior_agent.process_documents(data.get("data", {}))
    except Exception as e:
        print(f"\nERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save results
    output_file = f"output_id_{input_id}.json"
    save_output(output, output_file)
    
    # Also save detailed report separately
    report_file = f"report_id_{input_id}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(output.get("detailed_report", {}), f, indent=2, ensure_ascii=False)
    print(f"✓ Detailed report saved to {report_file}")
    
    # Print summary
    print_summary(output)

if __name__ == "__main__":
    main()
