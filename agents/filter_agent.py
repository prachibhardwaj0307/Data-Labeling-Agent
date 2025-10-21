"""
Filter Agent - Removes clearly irrelevant documents using LLM
"""
from typing import List, Tuple, Dict
import json
from models.data_models import Document
from utils.helpers import Logger, extract_text_from_html
from utils.llm_client import LLMClient

class FilterAgent:
    """
    Agent responsible for filtering out clearly irrelevant documents
    Uses LLM to make intelligent filtering decisions
    """
    
    def __init__(self):
        self.name = "FilterAgent"
        self.logger = Logger()
        self.llm = LLMClient()
        
        # Define the system prompt that guides the LLM's behavior
        self.system_prompt = """You are a Document Filter Agent specialized in identifying clearly irrelevant documents.

Your role is to analyze documents and filter out those that are:
1. Missing valid titles (title is empty, "No Title", "Holiday Balance", or clearly placeholder text)
2. Missing proper links or meaningful content
3. Completely unrelated to the user's query
4. Technical/system documents with no user-facing value
5. Broken or malformed documents

CRITICAL FILTERING RULES:
- Only filter out documents that are CLEARLY and OBVIOUSLY irrelevant
- When in doubt about relevance, KEEP the document (let other agents decide)
- Check if the title provides meaningful information
- Verify that content exists and is substantive
- Consider query keywords and document topic alignment
- Location-specific documents should only be filtered if they're for completely different locations

IMPORTANT: You are the first line of defense. Be conservative - it's better to keep a marginally relevant document than to filter out something potentially useful.

You must provide clear, specific reasoning for EVERY filtering decision."""

    def filter_documents(self, documents: List[Document], query: str, location: str = "") -> Tuple[List[Document], List[Document], Dict[str, str]]:
        """
        Filter out clearly irrelevant documents using LLM analysis
        
        Args:
            documents: List of Document objects to filter
            query: User's search query
            location: User's location (optional)
            
        Returns:
            Tuple of (kept_documents, removed_documents, reasons_dict)
        """
        self.logger.log(self.name, f"Starting LLM-based filtering for {len(documents)} documents")
        self.logger.log(self.name, f"Query: '{query}', Location: '{location or 'Not specified'}'")
        
        if not documents:
            return [], [], {}
        
        # Prepare document data for LLM (keeping it concise)
        docs_data = []
        for doc in documents:
            content_preview = extract_text_from_html(doc.html)[:400]  # First 400 chars
            docs_data.append({
                "id": doc.id,
                "title": doc.title,
                "content_preview": content_preview,
                "has_link": bool("href=" in doc.html)
            })
        
        # Create the user prompt with context
        # In the user_prompt, update the filtering criteria:

        user_prompt = f"""FILTERING TASK:

Query: "{query}"
Location: {location if location else "Not specified (global)"}

Documents to Analyze ({len(documents)} total):
{json.dumps(docs_data, indent=2)}

TASK:
Analyze each document and decide which ones should be FILTERED OUT (removed from processing).

**CRITICAL RULES:**

Filter a document ONLY if:
1. Completely unrelated to the query "{query}" (different topic entirely)
2. System/technical document with no user value (e.g., raw HTML, error pages)
3. Clearly for a VERY different location (if user specified location)
4. Broken or malformed HTML with no content

**DO NOT FILTER:**
❌ Documents with "No Title" or missing titles → KEEP THESE (mark them NOT_SURE later)
❌ Documents about same topic but different location → KEEP (mark them ACCEPTABLE later)
❌ Documents with partial relevance → KEEP
❌ Older documents about the same topic → KEEP

**LOCATION FILTERING:**
- Query="{query}", User Location="{location}"
- Document about "{query}" for US/UK/Canada → **KEEP** (mark ACCEPTABLE later)
- Document about "{query}" for {location} → **KEEP** (mark RELEVANT/SOMEWHAT_RELEVANT later)
- Only filter if TOPIC is completely different, not just location

OUTPUT FORMAT (JSON):
{{
    "filtered_out": [
        {{
            "doc_id": "document_id",
            "reason": "Specific reason: [Why is this COMPLETELY UNRELATED to '{query}'? Be explicit]"
        }}
    ],
    "kept_documents": ["doc_id1", "doc_id2", ...],
    "summary": "Brief summary of filtering decisions"
}}

**REMEMBER:**
✅ No title → KEEP (don't filter)
✅ Wrong location but right topic → KEEP (don't filter)
❌ Only filter if topic is COMPLETELY different"""


        try:
            # Call LLM and get structured response
            response = self.llm.call_with_json_response(self.system_prompt, user_prompt)
            
            # Process the response
            filtered_out_info = {item["doc_id"]: item["reason"] 
                               for item in response.get("filtered_out", [])}
            kept_ids = set(response.get("kept_documents", []))
            summary = response.get("summary", "No summary provided")
            
            filtered = []
            removed = []
            removal_reasons = {}  # Store reasons for each removed document
            
            # Classify each document based on LLM decision
            for doc in documents:
                if doc.id in filtered_out_info:
                    reason = filtered_out_info[doc.id]
                    self.logger.log(self.name, f"❌ FILTERING OUT {doc.id}: {reason}")
                    doc.current_label = "irrelevant"  # Mark as irrelevant
                    removed.append(doc)
                    removal_reasons[doc.id] = reason  # Store the reason
                else:
                    self.logger.log(self.name, f"✓ KEEPING {doc.id}: '{doc.title[:50]}'")
                    filtered.append(doc)
            
            # Log summary
            self.logger.log(self.name, f"Filtering complete: {len(filtered)} kept, {len(removed)} removed")
            self.logger.log(self.name, f"Summary: {summary}")
            
            return filtered, removed, removal_reasons
            
        except Exception as e:
            # Fallback: if LLM fails, keep all documents (safe default)
            self.logger.log(self.name, 
                          f"⚠️ LLM filtering failed: {e}. KEEPING ALL DOCUMENTS as safety measure.", 
                          "WARNING")
            return documents, [], {}
