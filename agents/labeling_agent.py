"""
Labeling Agent - Labels ENTIRE GROUPS using LLM with detailed reasoning
"""
from typing import List, Dict
import json
from models.data_models import Document, DocumentGroup, LabelingDecision
from utils.helpers import Logger, extract_text_from_html
from utils.llm_client import LLMClient

class LabelingAgent:
    """
    Agent responsible for labeling ENTIRE GROUPS of documents
    All documents in a group receive the same label
    """
    
    def __init__(self):
        self.name = "LabelingAgent"
        self.logger = Logger()
        self.llm = LLMClient()
        
        # System prompt for GROUP-BASED labeling
        self.system_prompt = """You are a Document Labeling Agent specialized in categorizing GROUPS of documents.

**CRITICAL: You label ENTIRE GROUPS, not individual documents!**

Your role is to assign ONE label to an entire group that will apply to ALL documents in that group.

LABEL DEFINITIONS:

**RELEVANT**: 
- Group directly answers query with current, comprehensive information
- FOR THE USER'S LOCATION
- Content is up-to-date and complete
- All documents in group serve the query purpose

**SOMEWHAT_RELEVANT**: 
- Group answers query for user's location
- BUT older (2021-2023) OR incomplete
- Still provides valuable information
- Documents are related but not perfect

**ACCEPTABLE**: 
- Group answers the query correctly
- BUT for a DIFFERENT location than user specified
- OR global/general information that provides context
- Example: Query="benefits", User="India", Group contains "US Benefits docs" → ACCEPTABLE

**NOT_SURE**: 
- Group has unclear or insufficient content
- Missing important information
- Cannot determine relevance confidently

**IMPORTANT RULES:**
✅ One label for the ENTIRE group
✅ Consider group theme and all documents together
✅ Wrong location + right topic = ACCEPTABLE (not irrelevant)
✅ Analyze both titles and content
✅ Provide detailed reasoning

You must justify why this label applies to ALL documents in the group."""

    def label_documents(self, groups: List[DocumentGroup], query: str, location: str = "") -> Dict[str, List[LabelingDecision]]:
        """
        Label ENTIRE GROUPS using LLM (not individual documents)
        
        Args:
            groups: List of document groups
            query: User's search query
            location: User's location
            
        Returns:
            Dictionary mapping labels to list of LabelingDecision objects
        """
        self.logger.log(self.name, f"Labeling {len(groups)} GROUPS (group-based labeling)")
        
        results = {
            "relevant": [],
            "somewhat_relevant": [],
            "acceptable": [],
            "not_sure": []
        }
        
        for group in groups:
            self.logger.log(self.name, f"Processing GROUP '{group.name}' with {len(group.documents)} documents")
            
            # Label the ENTIRE GROUP
            group_label = self._label_group(group, query, location)
            
            # Apply the same label to ALL documents in the group
            for doc in group.documents:
                decision = LabelingDecision(
                    doc_id=doc.id,
                    label=group_label["label"],
                    reason=f"[GROUP: {group.name}] {group_label['reason']}",
                    confidence=group_label["confidence"],
                    agent_name=self.name
                )
                results[decision.label].append(decision)
            
            self.logger.log(self.name, 
                f"✓ GROUP '{group.name}' labeled as {group_label['label'].upper()} "
                f"(applied to {len(group.documents)} documents)")
        
        return results

    def _label_group(self, group: DocumentGroup, query: str, location: str) -> dict:
        """
        Label an ENTIRE GROUP of documents with ONE label
        
        Args:
            group: Document group to label
            query: User's search query
            location: User's location
            
        Returns:
            Dictionary with label, reason, and confidence for the entire group
        """
        # Prepare group information with document details
        docs_summary = []
        for doc in group.documents:
            content = extract_text_from_html(doc.html)[:500]
            docs_summary.append({
                "id": doc.id,
                "title": doc.title,
                "content_preview": content,
                "has_valid_title": bool(doc.title and doc.title.strip() and doc.title.lower() not in ['no title', 'untitled', ''])
            })
        
        user_prompt = f"""GROUP LABELING TASK:

Query: "{query}"
User's Location: {location if location else "Not specified (any location acceptable)"}

GROUP TO LABEL:
- **Group Name:** {group.name}
- **Group Theme:** {group.theme}
- **Document Count:** {len(group.documents)}
- **Documents in Group:**
{json.dumps(docs_summary, indent=2)}

**TASK:** Assign ONE label to this ENTIRE GROUP that will apply to ALL {len(group.documents)} documents.

**CRITICAL INSTRUCTIONS:**

1. **READ CONTENT, NOT JUST TITLES** - Analyze document content carefully
2. **LOCATION LOGIC:**
   - Correct topic + Correct location = RELEVANT
   - Correct topic + WRONG location = **ACCEPTABLE** (NOT irrelevant!)
   - Wrong topic entirely = NOT_SURE (or would be filtered)
3. **GROUP ANALYSIS:**
   - Consider the group theme
   - Look at what all documents have in common
   - Decide if ALL documents fit one label

Choose ONE label for the ENTIRE GROUP:

**RELEVANT:** Group answers query "{query}" for location "{location}" with current info
**SOMEWHAT_RELEVANT:** Group answers query for "{location}" but older/incomplete
**ACCEPTABLE:** Group answers query BUT for different location, or provides general context
**NOT_SURE:** Group has unclear content, missing info, or ambiguous relevance

Respond in JSON format:
{{
    "label": "RELEVANT|SOMEWHAT_RELEVANT|ACCEPTABLE|NOT_SURE",
    "reason": "Detailed explanation: Why does this label apply to this ENTIRE GROUP? Mention: group theme, document analysis (not just titles!), location match/mismatch, query relevance, recency",
    "confidence": "high|medium|low",
    "group_location": "India/US/UK/Global/Mixed/Unknown - What location is this group for?",
    "applies_to_all_docs": "Brief explanation why this label fits ALL {len(group.documents)} documents in the group",
    "content_analysis": "What did you learn from reading the document CONTENT (not just titles)?"
}}

**REMEMBER:**
✅ One label for entire group
✅ Read content, not just titles
✅ Wrong location but right topic = ACCEPTABLE
✅ Explain why label applies to ALL documents"""

        try:
            response = self.llm.call_with_json_response(self.system_prompt, user_prompt)
            
            label = response.get("label", "NOT_SURE").lower()
            if label not in ["relevant", "somewhat_relevant", "acceptable", "not_sure"]:
                label = "not_sure"
            
            reason = response.get("reason", "No reason provided")
            applies_to_all = response.get("applies_to_all_docs", "")
            content_analysis = response.get("content_analysis", "")
            group_location = response.get("group_location", "Unknown")
            
            # Build comprehensive reason
            full_reason = (
                f"{reason} | "
                f"Group Location: {group_location}. "
                f"Applies to all: {applies_to_all} | "
                f"Content Analysis: {content_analysis}"
            )
            
            self.logger.log_decision(
                self.name, 
                f"GROUP:{group.name}", 
                label.upper(), 
                full_reason
            )
            
            return {
                "label": label,
                "reason": full_reason,
                "confidence": response.get("confidence", "medium")
            }
            
        except Exception as e:
            self.logger.log(self.name, f"LLM group labeling failed: {e}. Labeling as NOT_SURE.", "WARNING")
            return {
                "label": "not_sure",
                "reason": f"Failed to label group '{group.name}': {e}",
                "confidence": "low"
            }
