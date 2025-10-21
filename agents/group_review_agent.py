"""
Group Review Agent - Reviews document groups using LLM
"""
from typing import List
import json
from models.data_models import DocumentGroup, GroupReviewDecision
from utils.helpers import Logger
from utils.llm_client import LLMClient
import config

class GroupReviewAgent:
    """
    Agent responsible for reviewing document groups before labeling
    Uses LLM to assess group quality and coherence
    """
    
    def __init__(self):
        self.name = "GroupReviewAgent"
        self.logger = Logger()
        self.llm = LLMClient()
        
        # UPDATED SYSTEM PROMPT - NOW ACCEPTS SINGLE-DOCUMENT GROUPS
        self.system_prompt = """You are a Group Review Agent responsible for quality-checking document groupings.

Your role is to evaluate whether document groups are:
1. Well-formed (appropriate size: 1-10 documents)
2. Coherent (documents share a common theme OR document is unique)
3. Clearly described (group name and theme are meaningful)
4. Ready for batch labeling

REVIEW CRITERIA:
- Groups can have 1-10 documents (ideal: 3-7, but 1 is ACCEPTABLE)
- **SINGLE-DOCUMENT groups are PERFECTLY ACCEPTABLE** if:
  * The document is unique and doesn't fit with others
  * It has a clear, descriptive group name
  * The theme explains why it's alone
- Each multi-document group must have a clear, unifying theme
- Group names should be descriptive and meaningful
- Documents within multi-document groups should be related

WHAT TO APPROVE:
✅ Groups with 1 document if they're unique (e.g., "Unique_Tax_Form", "Single_Policy_Document")
✅ Groups with 2-10 documents that share a theme
✅ Clear, descriptive names for all groups
✅ Themes that explain the grouping logic

WHAT TO REJECT:
❌ Groups with unclear or generic names (e.g., "Miscellaneous", "Other", "Random")
❌ Multi-document groups where documents don't actually relate
❌ Groups with misleading themes that don't match the documents
❌ Poor organization that could be improved

DECISION RULES:
- APPROVE if all groups meet quality standards (including single-doc groups with good names)
- REJECT if groups need better names or reorganization
- After 3 attempts, APPROVE with feedback (forced approval)

**IMPORTANT: Single-document groups are NOT a reason to reject!**

Provide detailed reasoning for your decision."""

    def review_groups(self, groups: List[DocumentGroup], attempt: int) -> GroupReviewDecision:
        """
        Review document groups using LLM
        
        Args:
            groups: List of document groups to review
            attempt: Current attempt number
            
        Returns:
            GroupReviewDecision with approval status and feedback
        """
        self.logger.log(self.name, f"Reviewing {len(groups)} groups (Attempt {attempt}/{config.MAX_GROUP_REVIEW_ATTEMPTS})")
        
        # Prepare group data for review
        groups_data = []
        for group in groups:
            groups_data.append({
                "name": group.name,
                "theme": group.theme,
                "document_count": len(group.documents),
                "document_ids": [doc.id for doc in group.documents],
                "document_titles": [doc.title for doc in group.documents],  # Show ALL titles
                "attempt": group.attempt
            })
        
        user_prompt = f"""Review Attempt: {attempt} of {config.MAX_GROUP_REVIEW_ATTEMPTS}

Document Groups to Review:
{json.dumps(groups_data, indent=2)}

Evaluate these groups and decide whether to APPROVE or REJECT them for labeling.

Check for:
1. Appropriate group sizes (1-10 documents - **SINGLE-DOC GROUPS ARE OK!**)
2. Clear themes and coherence
3. Meaningful group names (not generic like "Miscellaneous")
4. Overall quality for batch labeling

**IMPORTANT RULES:**
- Single-document groups are ACCEPTABLE if they have descriptive names
- Only reject if group names are too generic or themes don't match documents
- Don't reject just because a group has 1 document

Respond in JSON format:
{{
    "decision": "APPROVE" or "REJECT",
    "feedback": "Detailed explanation of your decision",
    "issues_found": ["issue1", "issue2", ...] or [],
    "suggestions": "Specific suggestions for improvement if rejected",
    "single_doc_groups_ok": "true/false - are single-document groups acceptable?"
}}

REMEMBER: 
- Single-document groups with good names → APPROVE ✅
- Generic names like "Miscellaneous" → REJECT ❌
- If this is attempt {config.MAX_GROUP_REVIEW_ATTEMPTS}, you MUST approve regardless of issues."""

        try:
            response = self.llm.call_with_json_response(self.system_prompt, user_prompt)
            
            decision = response.get("decision", "REJECT").upper()
            feedback = response.get("feedback", "No feedback provided")
            
            # Force approval on final attempt
            if attempt >= config.MAX_GROUP_REVIEW_ATTEMPTS:
                decision = "APPROVE"
                feedback = f"FORCED APPROVAL after {attempt} attempts. {feedback}"
            
            approved = decision == "APPROVE"
            
            self.logger.log(self.name, f"{'✅ APPROVED' if approved else '❌ REJECTED'}: {feedback}")
            
            return GroupReviewDecision(
                approved=approved,
                feedback=feedback,
                attempt_number=attempt
            )
            
        except Exception as e:
            self.logger.log(self.name, f"LLM review failed: {e}. Defaulting to approval.", "WARNING")
            return GroupReviewDecision(
                approved=True,
                feedback=f"Approved due to LLM error: {e}",
                attempt_number=attempt
            )
