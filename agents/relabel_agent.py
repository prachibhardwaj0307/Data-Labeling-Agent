"""
Relabel Agent - Relabels documents based on reviewer feedback using LLM
"""
from typing import Dict, List
import json
from models.data_models import LabelingDecision, LabelReviewDecision
from utils.helpers import Logger, extract_text_from_html
from utils.llm_client import LLMClient

class RelabelAgent:
    """
    Agent responsible for relabeling documents when initial labels are rejected
    Uses LLM to address reviewer feedback and create improved labels
    """
    
    def __init__(self):
        self.name = "RelabelAgent"
        self.logger = Logger()
        self.llm = LLMClient()
        
        # Define the system prompt
        self.system_prompt = """You are a Relabeling Agent specialized in improving document labels based on reviewer feedback.

Your role is to:
1. Analyze the reviewer's feedback about current labeling decisions
2. Identify specific problems with labels and reasoning
3. Create IMPROVED labels with better, more detailed reasoning
4. Ensure labels are more accurate and well-justified than before

LABEL CATEGORIES:
- **RELEVANT**: Directly answers the query with current, comprehensive information
  - Contains exactly what the user asked for
  - Information is up-to-date (2024-2025)
  - Comprehensive and complete

- **SOMEWHAT_RELEVANT**: Partially addresses query OR good content but older/incomplete
  - Provides some useful information but not everything
  - May be from 2021-2023 (older but still valuable)
  - Related but doesn't fully answer the query

- **ACCEPTABLE**: Related but limited value OR older documents providing context
  - Tangentially related to the query
  - Background or contextual information only
  - Older documents (pre-2021) that provide historical context

- **NOT_SURE**: Missing title, no link, unclear content, or ambiguous relevance
  - Document has structural issues (no title, broken link)
  - Cannot determine relevance with confidence
  - Content is too vague or unclear

COMMON ISSUES TO FIX:
- Generic reasoning → Make it specific with details
- Wrong confidence level → Adjust based on evidence
- Unclear label choice → Choose more appropriate label
- Missing details → Add document year, specific content mentions
- Vague descriptions → Be explicit about why the label fits

IMPROVEMENT STRATEGIES:
- Mention specific year/date if found
- Quote or reference specific content from the document
- Explain exactly why it does/doesn't answer the query
- Be explicit about document quality and completeness
- Justify confidence level with evidence

You must show substantial improvement over the previous label."""

    def relabel_documents(self, labeling_results: Dict[str, List[LabelingDecision]], 
                         review: LabelReviewDecision,
                         query: str,
                         location: str = "") -> Dict[str, List[LabelingDecision]]:
        """
        Relabel documents based on reviewer feedback using LLM
        
        Args:
            labeling_results: Current labeling results that were rejected
            review: Review decision with feedback
            query: User's search query
            location: User's location (optional)
            
        Returns:
            Improved labeling results dictionary
        """
        self.logger.log(self.name, 
            f"Relabeling attempt #{review.attempt_number + 1} based on feedback")
        self.logger.log(self.name, f"Reviewer feedback: {review.feedback}")
        self.logger.log(self.name, 
            f"Documents to relabel: {len(review.rejected_docs)}")
        
        # Get all documents that need relabeling
        docs_to_relabel = []
        doc_current_info = {}
        
        for label, decisions in labeling_results.items():
            for decision in decisions:
                if decision.doc_id in review.rejected_docs:
                    docs_to_relabel.append({
                        "doc_id": decision.doc_id,
                        "current_label": label,
                        "current_reason": decision.reason,
                        "current_confidence": decision.confidence
                    })
                    doc_current_info[decision.doc_id] = (label, decision)
        
        if not docs_to_relabel:
            self.logger.log(self.name, "No documents to relabel")
            return labeling_results
        
        # Create the user prompt
        user_prompt = f"""RELABELING TASK:

Query: "{query}"
Location: {location if location else "Not specified"}

REVIEWER FEEDBACK (MUST ADDRESS): {review.feedback}
Current Attempt: {review.attempt_number + 1} of 3

DOCUMENTS TO RELABEL (problematic labels):
{json.dumps(docs_to_relabel, indent=2)}

TASK:
Based on the reviewer's feedback, provide IMPROVED labels for these documents.

For each document:
1. Analyze why the previous label was problematic
2. Choose a better label
3. Provide detailed, specific reasoning (not generic)
4. Justify your confidence level

OUTPUT FORMAT (JSON):
{{
    "analysis_of_feedback": "What issues did the reviewer find with these labels?",
    "relabeled": [
        {{
            "doc_id": "document_id",
            "new_label": "RELEVANT|SOMEWHAT_RELEVANT|ACCEPTABLE|NOT_SURE",
            "new_reason": "Detailed, specific reason - mention document year, specific content, why it matches the query or doesn't",
            "confidence": "high|medium|low",
            "what_was_improved": "Explain what's better about this label vs. the previous one"
        }}
    ]
}}

REQUIREMENTS:
- Address the reviewer's specific concerns
- Provide MORE DETAILED reasoning than before
- Be specific (mention years, content details, etc.)
- Explain why the document does/doesn't answer "{query}"
- Avoid generic phrases like "relates to query" - be explicit!"""

        try:
            # Call LLM and get structured response
            response = self.llm.call_with_json_response(self.system_prompt, user_prompt)
            
            # Log analysis
            analysis = response.get("analysis_of_feedback", "No analysis provided")
            self.logger.log(self.name, f"Analysis: {analysis}")
            
            # Create new results dict (copy existing)
            new_results = {
                "relevant": labeling_results.get("relevant", []).copy(),
                "somewhat_relevant": labeling_results.get("somewhat_relevant", []).copy(),
                "acceptable": labeling_results.get("acceptable", []).copy(),
                "not_sure": labeling_results.get("not_sure", []).copy()
            }
            
            # Process relabeled documents
            for relabeled in response.get("relabeled", []):
                doc_id = relabeled.get("doc_id")
                new_label = relabeled.get("new_label", "NOT_SURE").lower()
                
                # Validate label
                if new_label not in ["relevant", "somewhat_relevant", "acceptable", "not_sure"]:
                    new_label = "not_sure"
                
                # Remove from old label category
                if doc_id in doc_current_info:
                    old_label, old_decision = doc_current_info[doc_id]
                    new_results[old_label] = [
                        d for d in new_results[old_label] if d.doc_id != doc_id
                    ]
                
                # Add to new label category
                new_decision = LabelingDecision(
                    doc_id=doc_id,
                    label=new_label,
                    reason=relabeled.get("new_reason", "Relabeled based on feedback"),
                    confidence=relabeled.get("confidence", "medium"),
                    agent_name=self.name
                )
                
                new_results[new_label].append(new_decision)
                
                improvement = relabeled.get("what_was_improved", "")
                self.logger.log_decision(
                    self.name,
                    doc_id,
                    f"{new_label.upper()} (relabeled from {doc_current_info[doc_id][0]})",
                    f"{new_decision.reason}\n  Improvement: {improvement}"
                )
            
            self.logger.log(self.name, 
                f"Relabeling complete: {len(response.get('relabeled', []))} documents relabeled")
            return new_results
            
        except Exception as e:
            # Fallback: return original labels
            self.logger.log(self.name, 
                          f"⚠️ LLM relabeling failed: {e}. Keeping original labels.", 
                          "WARNING")
            return labeling_results
