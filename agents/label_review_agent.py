"""
Label Review Agent - Reviews labeling decisions with MAX 10 RELEVANT enforcement
"""
from typing import Dict, List
import json
from models.data_models import LabelingDecision, LabelReviewDecision
from utils.helpers import Logger
from utils.llm_client import LLMClient

class LabelReviewAgent:
    """
    Agent responsible for reviewing labeling decisions
    ENFORCES: Maximum 10 documents in RELEVANT category
    """
    
    def __init__(self):
        self.name = "LabelReviewAgent"
        self.logger = Logger()
        self.llm = LLMClient()
        
        self.system_prompt = """You are a Label Review Agent specialized in quality control of document labeling.

**CRITICAL RULE: MAXIMUM 10 RELEVANT DOCUMENTS**

Your role:
1. Review labeling decisions for accuracy and consistency
2. **ENFORCE: No more than 10 documents can be labeled as RELEVANT**
3. If >10 RELEVANT, REJECT and request relabeling

**REVIEW CRITERIA:**

1. **RELEVANT Count Check (CRITICAL)**:
   - If RELEVANT count > 10 → **AUTOMATIC REJECTION**
   - Feedback: "Too many RELEVANT documents ({count}). Maximum is 10. Relabel to keep only the TOP 10 most relevant based on year, quality, and completeness."

2. **Label Consistency**:
   - Are labels appropriate for content?
   - Year-based prioritization (if year is available): 2025 > 2024 > 2023
   - If year is unknown, evaluate by topic and location match
   - Are location rules followed? (wrong location = ACCEPTABLE, not RELEVANT)

3. **Quality Check**:
   - Are RELEVANT documents truly the best answers?
   - Are SOMEWHAT_RELEVANT documents appropriately downgraded from RELEVANT?
   - Are NOT_SURE labels justified?

**APPROVAL CONDITIONS:**
- RELEVANT count ≤ 10
- Labels seem reasonable and consistent
- No obvious misclassifications

**REJECTION REASONS:**
- RELEVANT count > 10 (MUST reject)
- Obvious misclassifications
- Inconsistent labeling patterns

**NOTE:** If document year is unknown, evaluate based on content quality and topic/location match, not year.

Be strict but fair. Quality is important."""

    def review_labels(self, labeling_results: Dict[str, List[LabelingDecision]], 
                     attempt: int) -> LabelReviewDecision:
        """
        Review labeling decisions with MAX 10 RELEVANT enforcement
        
        Args:
            labeling_results: Dictionary of labels to decisions
            attempt: Current review attempt number
        """
        self.logger.log(self.name, f"Reviewing labels (Attempt {attempt}/3)")
        
        # Count documents by label
        label_counts = {
            label: len(decisions) 
            for label, decisions in labeling_results.items()
        }
        
        self.logger.log(self.name, f"Label counts: {label_counts}")
        
        relevant_count = label_counts.get("relevant", 0)
        
        # CRITICAL: Check if RELEVANT > 10
        if relevant_count > 10:
            self.logger.log(self.name, 
                f"❌ AUTOMATIC REJECTION: {relevant_count} RELEVANT (max is 10)")
            
            # Return all RELEVANT docs for relabeling
            relevant_doc_ids = [d.doc_id for d in labeling_results.get("relevant", [])]
            
            return LabelReviewDecision(
                approved=False,
                feedback=f"TOO MANY RELEVANT DOCUMENTS: {relevant_count} documents labeled as RELEVANT. "
                        f"Maximum is 10. Please relabel to select only the TOP 10 most relevant documents "
                        f"based on: 1) Year (if available, 2025 first), 2) Completeness, 3) Direct query match, "
                        f"4) Location match. Downgrade the remaining {relevant_count - 10} to SOMEWHAT_RELEVANT.",
                rejected_docs=relevant_doc_ids,
                attempt_number=attempt
            )
        
        # If RELEVANT ≤ 10, ask LLM for quality review
        try:
            # Prepare label summary for LLM
            label_summary = []
            for label, decisions in labeling_results.items():
                if decisions:
                    label_summary.append({
                        "label": label,
                        "count": len(decisions),
                        "examples": [
                            {
                                "doc_id": d.doc_id,
                                "reason": d.reason[:200],  # Truncate long reasons
                                "confidence": d.confidence
                            }
                            for d in decisions[:3]  # Show first 3 examples
                        ]
                    })
            
            user_prompt = f"""LABEL REVIEW TASK (Attempt {attempt}/3):

**Label Distribution:**
{json.dumps(label_counts, indent=2)}

**RELEVANT Count: {relevant_count} / 10 (MAX)** ✓

**Label Details:**
{json.dumps(label_summary, indent=2)}

**Review Questions:**
1. Is the RELEVANT count appropriate (≤10)? ✓ YES (checked already)
2. Do the RELEVANT documents appear to be the best answers?
3. Are SOMEWHAT_RELEVANT and ACCEPTABLE appropriately distinguished?
4. Are NOT_SURE labels justified or should they be re-evaluated based on content?
5. Is there consistency in the labeling?

**NOTE:** If documents have unknown years, they should be evaluated based on:
- Topic relevance
- Location match
- Content quality
Not automatically marked as NOT_SURE just because year is missing.

**Decision:**
- If labels look reasonable and consistent → APPROVE
- If there are obvious issues (besides RELEVANT count) → REJECT with specific feedback

Respond in JSON (use strings, not booleans):
{{
    "approved": "yes or no (string)",
    "feedback": "Detailed feedback explaining your decision",
    "issues_found": ["list of specific issues if rejected"],
    "rejected_doc_ids": ["list of doc IDs that need relabeling, empty if approved"]
}}

**Remember:** Use strings for all fields, NOT boolean values."""

            response = self.llm.call_with_json_response(self.system_prompt, user_prompt)
            
            # Handle approved field - can be boolean or string
            approved_value = response.get("approved", "no")
            if isinstance(approved_value, bool):
                approved = approved_value
            elif isinstance(approved_value, str):
                approved = approved_value.lower() in ["yes", "true", "approved"]
            else:
                approved = str(approved_value).lower() in ["yes", "true", "approved"]
            
            feedback = response.get("feedback", "No feedback provided")
            rejected_docs = response.get("rejected_doc_ids", [])
            
            # Ensure rejected_docs is a list
            if not isinstance(rejected_docs, list):
                rejected_docs = []
            
            if approved:
                self.logger.log(self.name, f"✅ APPROVED: {feedback}")
            else:
                self.logger.log(self.name, f"❌ REJECTED: {feedback}")
                self.logger.log(self.name, f"Documents to relabel: {len(rejected_docs)}")
            
            return LabelReviewDecision(
                approved=approved,
                feedback=feedback,
                rejected_docs=rejected_docs,
                attempt_number=attempt
            )
            
        except Exception as e:
            self.logger.log(self.name, 
                f"LLM review failed: {e}. Defaulting to approval.", "WARNING")
            
            # On error, approve if RELEVANT ≤ 10
            return LabelReviewDecision(
                approved=True,
                feedback=f"Approved due to LLM error: {e}",
                rejected_docs=[],
                attempt_number=attempt
            )
