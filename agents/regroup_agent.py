"""
Regroup Agent - Reorganizes document groups based on reviewer feedback using LLM
"""
from typing import List
import json
from models.data_models import Document, DocumentGroup, GroupReviewDecision
from utils.helpers import Logger, extract_text_from_html
from utils.llm_client import LLMClient

class RegroupAgent:
    """
    Agent responsible for regrouping documents when initial grouping is rejected
    Uses LLM to address reviewer feedback and create improved groups
    """
    
    def __init__(self):
        self.name = "RegroupAgent"
        self.logger = Logger()
        self.llm = LLMClient()
        
        # Define the system prompt
        self.system_prompt = """You are a Regrouping Agent specialized in reorganizing document groups based on reviewer feedback.

Your role is to:
1. Carefully analyze the reviewer's feedback about current groupings
2. Identify specific problems with existing groups
3. Create IMPROVED groupings that directly address all issues
4. Ensure new groups are better organized, sized, and themed than before

COMMON ISSUES TO ADDRESS:
- Generic group names (e.g., "Miscellaneous", "Other") → Create specific, descriptive names
- Oversized groups (>10 documents) → Split into smaller, focused groups
- Undersized groups with generic names → Either merge OR create specific single-doc group names
- Unclear themes → Create more specific, descriptive themes
- Poor grouping logic → Regroup by clearer criteria
- Mixed themes → Separate into distinct thematic groups

REGROUPING STRATEGIES:
- Split large groups by sub-topics or document types
- Single-document groups MUST have specific, meaningful names
- Replace "Miscellaneous" with specific names like:
  * "Unique_Tax_Document_W8BEN"
  * "Single_Global_Policy"
  * "Standalone_Presentation"
- Improve group names to be more descriptive
- Refine themes to be more specific and clear
- Ensure each group has a coherent, single focus

QUALITY REQUIREMENTS:
- All groups must be 1-10 documents (single-doc groups OK!)
- Clear, descriptive group names (NO "Miscellaneous", "Other", "Various")
- Specific, meaningful themes
- Address ALL points from reviewer feedback
- Explain what improvements were made

**CRITICAL: Never use generic names like "Miscellaneous" or "Remaining" - always be specific!**

You must show how you've addressed the reviewer's specific concerns."""


    def regroup_documents(self, groups: List[DocumentGroup], review: GroupReviewDecision) -> List[DocumentGroup]:
        """
        Regroup documents based on reviewer feedback using LLM
        """
        self.logger.log(self.name, f"Regrouping based on feedback: {review.feedback}")
        
        # Flatten all documents from all groups
        all_docs = []
        current_groups_info = []
        
        for group in groups:
            all_docs.extend(group.documents)
            current_groups_info.append({
                "name": group.name,
                "theme": group.theme,
                "document_count": len(group.documents),
                "document_ids": [doc.id for doc in group.documents],
                "document_titles": [doc.title for doc in group.documents]  # Show all titles
            })
        
        # Prepare document data
        docs_data = []
        for doc in all_docs:
            docs_data.append({
                "id": doc.id,
                "title": doc.title,
                "content_preview": extract_text_from_html(doc.html)[:500]
            })
        
        # THIS IS THE UPDATED USER PROMPT - REPLACE THE OLD ONE:
        user_prompt = f"""Reviewer Feedback: {review.feedback}
Attempt Number: {review.attempt_number + 1}

Current Groups (PROBLEMATIC):
{json.dumps(current_groups_info, indent=2)}

All Documents to Regroup:
{json.dumps(docs_data, indent=2)}

Based on the reviewer's feedback, create IMPROVED groupings that address all issues.

**KEY RULES:**
- Single-document groups are ACCEPTABLE but MUST have specific, descriptive names
- NEVER use generic names like "Miscellaneous", "Other", "Remaining", "Various"
- For single-document groups, use descriptive names based on the document's content
- Examples of GOOD single-doc group names:
  * "Tax_Form_W8BEN_Entity"
  * "Global_Benefits_Overview_2025"
  * "Standalone_India_Holiday_Calendar"

Respond in JSON format:
{{
    "analysis_of_feedback": "What specific issues did the reviewer identify?",
    "improvements_made": "What changes are you making to address the feedback?",
    "groups": [
        {{
            "group_name": "Improved, specific, descriptive name (NOT generic!)",
            "theme": "Clearer, more specific theme",
            "document_ids": ["doc_id1", "doc_id2", ...],
            "improvements_from_previous": "How is this group better than before?"
        }}
    ]
}}

IMPORTANT:
- Directly address ALL points in the reviewer feedback
- Keep groups between 1-10 documents
- Create specific names for ALL groups (especially single-document ones)
- Ensure every document is assigned to exactly ONE group
- Explain what you improved"""


        try:
            # Call LLM and get structured response
            response = self.llm.call_with_json_response(self.system_prompt, user_prompt)
            
            # Log analysis
            analysis = response.get("analysis_of_feedback", "No analysis provided")
            improvements = response.get("improvements_made", "No improvements specified")
            
            self.logger.log(self.name, f"Analysis: {analysis}")
            self.logger.log(self.name, f"Improvements: {improvements}")
            
            new_groups = []
            doc_map = {doc.id: doc for doc in all_docs}
            assigned_docs = set()
            
            # Process each group from LLM response
            for group_data in response.get("groups", []):
                group_docs = []
                for doc_id in group_data.get("document_ids", []):
                    if doc_id in doc_map and doc_id not in assigned_docs:
                        group_docs.append(doc_map[doc_id])
                        assigned_docs.add(doc_id)
                
                if group_docs:
                    group = DocumentGroup(
                        name=group_data.get("group_name", f"Regrouped_{len(new_groups)+1}"),
                        documents=group_docs,
                        theme=group_data.get("theme", ""),
                        reasons=[
                            f"Regrouped based on feedback: {review.feedback}",
                            group_data.get("improvements_from_previous", "")
                        ],
                        attempt=review.attempt_number + 1
                    )
                    new_groups.append(group)
                    
                    self.logger.log(self.name, 
                        f"✓ Created improved group '{group.name}' with {len(group_docs)} docs")
            
            # Handle unassigned documents
            unassigned = [doc for doc in all_docs if doc.id not in assigned_docs]
            if unassigned:
                self.logger.log(self.name, 
                    f"⚠️ Found {len(unassigned)} unassigned documents")
                group = DocumentGroup(
                    name="Remaining_Documents",
                    documents=unassigned,
                    theme="Documents not assigned during regrouping",
                    reasons=["Documents that didn't fit into primary groups"],
                    attempt=review.attempt_number + 1
                )
                new_groups.append(group)
            
            self.logger.log(self.name, 
                f"Regrouping complete: Created {len(new_groups)} improved groups")
            return new_groups
            
        except Exception as e:
            # Fallback: return original groups with incremented attempt
            self.logger.log(self.name, 
                          f"⚠️ LLM regrouping failed: {e}. Keeping original groups.", 
                          "WARNING")
            for group in groups:
                group.attempt += 1
                group.reasons.append(f"Regrouping failed, keeping original structure")
            return groups
