"""
Label Review Agent - Reviews labeling decisions using LLM
"""
from typing import Dict, List
import json
from models.data_models import LabelingDecision, LabelReviewDecision
from utils.helpers import Logger
from utils.llm_client import LLMClient
import config

class LabelReviewAgent:
    def __init__(self):
        self.name = "LabelReviewAgent"
        self.logger = Logger()
        self.llm = LLMClient()
        
        self.system_prompt = """You are a Label Review Agent responsible for quality-checking document labeling decisions.

Your role is to evaluate whether labels are:
1. Appropriate for the document content
2. Well-justified with clear reasoning
3. Consistent across similar documents
4. Confident and reliable

REVIEW CRITERIA:
- Labels should match the document's relevance to the query
- Reasoning should be specific and detailed (not generic)
- Confidence levels should be appropriate
- Too many "NOT_SURE" labels indicates poor labeling

DECISION RULES:
- APPROVE if labels are well-justified and appropriate
- REJECT if labels need revision (identify specific problematic documents)
- After 3 attempts, APPROVE with feedback (forced approval)

Provide specific feedback on which documents need relabeling."""

    def review_labels(self, labeling_results: Dict[str, List[LabelingDecision]], attempt: int) -> LabelReviewDecision:
        """Review labeling decisions using LLM"""
        total_docs = sum(len(decisions) for decisions in labeling_results.values())
        self.logger.log(self.name, f"Reviewing {total_docs} labeled documents (Attempt {attempt}/{config.MAX_LABEL_REVIEW_ATTEMPTS})")
        
        # Prepare label data for review
        review_data = {}
        for label, decisions in labeling_results.items():
            review_data[label] = [{
                "doc_id": d.doc_id,
                "reason": d.reason,
                "confidence": d.confidence
            } for d in decisions[:10]]  # Sample first 10 of each category
        
        stats = {
            "total": total_docs,
            "relevant": len(labeling_results.get("relevant", [])),
            "somewhat_relevant": len(labeling_results.get("somewhat_relevant", [])),
            "acceptable": len(labeling_results.get("acceptable", [])),
            "not_sure": len(labeling_results.get("not_sure", []))
        }
        
        user_prompt = f"""Review Attempt: {attempt} of {config.MAX_LABEL_REVIEW_ATTEMPTS}

Label Statistics:
{json.dumps(stats, indent=2)}

Sample Labeled Documents:
{json.dumps(review_data, indent=2)}

Evaluate these labeling decisions and decide whether to APPROVE or REJECT them.

Check for:
1. Appropriate label assignments
2. Clear, specific reasoning (not generic)
3. Appropriate confidence levels
4. Balanced distribution (not too many NOT_SURE)

Respond in JSON format:
{{
    "decision": "APPROVE" or "REJECT",
    "feedback": "Detailed explanation of your decision",
    "problematic_docs": ["doc_id1", "doc_id2", ...],
    "issues": ["issue1", "issue2", ...],
    "suggestions": "How to improve labeling if rejected"
}}

IMPORTANT: If this is attempt {config.MAX_LABEL_REVIEW_ATTEMPTS}, you MUST approve regardless of issues."""

        try:
            response = self.llm.call_with_json_response(self.system_prompt, user_prompt)
            
            decision = response.get("decision", "REJECT").upper()
            feedback = response.get("feedback", "No feedback provided")
            rejected_docs = response.get("problematic_docs", [])
            
            # Force approval on final attempt
            if attempt >= config.MAX_LABEL_REVIEW_ATTEMPTS:
                decision = "APPROVE"
                feedback = f"FORCED APPROVAL after {attempt} attempts. {feedback}"
                rejected_docs = []
            
            approved = decision == "APPROVE"
            
            self.logger.log(self.name, f"{'APPROVED' if approved else 'REJECTED'}: {feedback}")
            if rejected_docs:
                self.logger.log(self.name, f"Problematic documents: {', '.join(rejected_docs[:5])}")
            
            return LabelReviewDecision(
                approved=approved,
                feedback=feedback,
                attempt_number=attempt,
                rejected_docs=rejected_docs
            )
            
        except Exception as e:
            self.logger.log(self.name, f"LLM review failed: {e}. Defaulting to approval.", "WARNING")
            return LabelReviewDecision(
                approved=True,
                feedback=f"Approved due to LLM error: {e}",
                attempt_number=attempt,
                rejected_docs=[]
            )
