"""
Label Review Agent - Reviews document labels with RELEVANT document limit
"""
from typing import List, Dict
import json
from models.data_models import LabelingDecision, LabelReviewDecision
from utils.helpers import Logger
from utils.llm_client import LLMClient
import config

class LabelReviewAgent:
    """
    Agent responsible for reviewing document labels before finalization
    Enforces maximum 10 RELEVANT documents rule
    """
    
    def __init__(self):
        self.name = "LabelReviewAgent"
        self.logger = Logger()
        self.llm = LLMClient()
        
        self.system_prompt = """You are a Label Review Agent specialized in quality-checking document labels.

Your role is to evaluate whether document labels are:
1. Accurate and well-justified
2. Consistent with label definitions
3. Properly reasoned
4. Within acceptable quantity limits

**CRITICAL RULE: MAXIMUM 10 RELEVANT DOCUMENTS**
- If more than 10 documents are labeled as RELEVANT, you MUST REJECT
- Identify the TOP 10 most relevant documents (those that most directly answer the query)
- The remaining "relevant" documents should be downgraded to SOMEWHAT_RELEVANT

LABEL QUALITY CRITERIA:

**RELEVANT** (Maximum 10 documents):
- Directly answers the query with current, comprehensive information
- For the correct location
- Most valuable and complete documents
- If >10 documents are labeled RELEVANT, downgrade excess to SOMEWHAT_RELEVANT

**SOMEWHAT_RELEVANT**:
- Partially answers query OR older documents
- Still valuable but not the primary answer
- Excess "relevant" documents go here

**ACCEPTABLE**:
- Correct topic but wrong location
- Provides context but not primary answer

**NOT_SURE**:
- Missing information or unclear relevance
- Documents that truly cannot be confidently labeled

REVIEW DECISION RULES:
- REJECT if more than 10 documents are labeled RELEVANT
- REJECT if labels are clearly incorrect
- REJECT if reasoning is insufficient
- APPROVE if labels are accurate and RELEVANT count ≤ 10
- After 3 attempts, APPROVE with feedback (forced approval)

Provide specific, actionable feedback for improvements."""

    def review_labels(self, labeling_results: Dict[str, List[LabelingDecision]], 
                     attempt: int) -> LabelReviewDecision:
        """
        Review document labels with RELEVANT limit enforcement
        
        Args:
            labeling_results: Dictionary mapping labels to list of LabelingDecision objects
            attempt: Current attempt number
            
        Returns:
            LabelReviewDecision with approval status and feedback
        """
        self.logger.log(self.name, 
            f"Reviewing labels (Attempt {attempt}/{config.MAX_LABEL_REVIEW_ATTEMPTS})")
        
        # Count documents by label
        label_counts = {
            "relevant": len(labeling_results.get("relevant", [])),
            "somewhat_relevant": len(labeling_results.get("somewhat_relevant", [])),
            "acceptable": len(labeling_results.get("acceptable", [])),
            "not_sure": len(labeling_results.get("not_sure", []))
        }
        
        self.logger.log(self.name, f"Label counts: {label_counts}")
        
        # **ENFORCE RELEVANT LIMIT**
        relevant_count = label_counts["relevant"]
        if relevant_count > 10:
            self.logger.log(self.name, 
                f"⚠️ TOO MANY RELEVANT DOCS: {relevant_count} (max is 10)")
            
            # Identify all relevant docs for review
            relevant_docs = labeling_results.get("relevant", [])
            rejected_doc_ids = [d.doc_id for d in relevant_docs]
            
            feedback = (
                f"REJECTED: {relevant_count} documents labeled as RELEVANT (maximum is 10). "
                f"Please review all {relevant_count} relevant documents and keep only the TOP 10 that "
                f"most directly and comprehensively answer the query. "
                f"Downgrade the remaining {relevant_count - 10} documents to SOMEWHAT_RELEVANT. "
                f"Consider: recency, completeness, direct query match, and information quality."
            )
            
            return LabelReviewDecision(
                approved=False,
                feedback=feedback,
                rejected_docs=rejected_doc_ids,
                attempt_number=attempt
            )
        
        # Prepare label data for LLM review
        labels_data = {}
        for label, decisions in labeling_results.items():
            labels_data[label] = [{
                "doc_id": d.doc_id,
                "label": label,
                "reason": d.reason,
                "confidence": d.confidence
            } for d in decisions]
        
        user_prompt = f"""Review Attempt: {attempt} of {config.MAX_LABEL_REVIEW_ATTEMPTS}

Label Distribution:
- Relevant: {label_counts['relevant']} (MAX ALLOWED: 10)
- Somewhat Relevant: {label_counts['somewhat_relevant']}
- Acceptable: {label_counts['acceptable']}
- Not Sure: {label_counts['not_sure']}

Labeled Documents:
{json.dumps(labels_data, indent=2)}

**CRITICAL CHECK:**
✅ Are there 10 or fewer RELEVANT documents? (Currently: {label_counts['relevant']})
❌ If more than 10 RELEVANT, you MUST REJECT and specify which should be downgraded

Evaluate these labels and decide whether to APPROVE or REJECT.

Check for:
1. **RELEVANT COUNT ≤ 10** (If >10, automatically REJECT)
2. Are labels accurate based on reasons provided?
3. Is reasoning sufficient and clear?
4. Are labels consistent with definitions?
5. Are there any obvious misclassifications?

Respond in JSON format:
{{
    "decision": "APPROVE" or "REJECT",
    "feedback": "Detailed explanation of your decision. If rejecting due to >10 relevant, specify which documents should be downgraded.",
    "issues_found": ["issue1", "issue2", ...],
    "relevant_count_ok": "true/false - Is RELEVANT count ≤ 10?",
    "suggested_downgrades": ["doc_id1", "doc_id2", ...] (if RELEVANT > 10, list docs to downgrade to SOMEWHAT_RELEVANT)
}}

**REMEMBER:**
- If RELEVANT > 10 → MUST REJECT
- If attempt {config.MAX_LABEL_REVIEW_ATTEMPTS} → MUST APPROVE regardless
- Provide specific, actionable feedback"""

        try:
            response = self.llm.call_with_json_response(self.system_prompt, user_prompt)
            
            decision = response.get("decision", "REJECT").upper()
            feedback = response.get("feedback", "No feedback provided")
            relevant_count_ok = response.get("relevant_count_ok", "false").lower() == "true"
            suggested_downgrades = response.get("suggested_downgrades", [])
            
            # Force approval on final attempt
            if attempt >= config.MAX_LABEL_REVIEW_ATTEMPTS:
                decision = "APPROVE"
                feedback = f"FORCED APPROVAL after {attempt} attempts. {feedback}"
            
            # If LLM says relevant count is not OK, force rejection
            if not relevant_count_ok and decision == "APPROVE":
                decision = "REJECT"
                feedback = f"Cannot approve: RELEVANT count exceeds 10. {feedback}"
            
            approved = decision == "APPROVE"
            
            # Determine which docs need relabeling
            rejected_docs = []
            if not approved and relevant_count > 10:
                # If specific downgrades suggested, use those
                if suggested_downgrades:
                    rejected_docs = suggested_downgrades
                else:
                    # Otherwise, mark all relevant docs for review
                    rejected_docs = [d.doc_id for d in labeling_results.get("relevant", [])]
            
            self.logger.log(self.name, 
                f"{'✅ APPROVED' if approved else '❌ REJECTED'}: {feedback}")
            
            if rejected_docs:
                self.logger.log(self.name, 
                    f"Documents to relabel: {len(rejected_docs)}")
            
            return LabelReviewDecision(
                approved=approved,
                feedback=feedback,
                rejected_docs=rejected_docs,
                attempt_number=attempt
            )
            
        except Exception as e:
            self.logger.log(self.name, 
                f"LLM review failed: {e}. Defaulting to approval.", "WARNING")
            return LabelReviewDecision(
                approved=True,
                feedback=f"Approved due to LLM error: {e}",
                rejected_docs=[],
                attempt_number=attempt
            )
