"""
Filter Agent - Removes irrelevant documents from NEW documents only
"""
from typing import List, Tuple, Dict
import json
from models.data_models import Document
from utils.helpers import Logger, extract_text_from_html
from utils.llm_client import LLMClient

class FilterAgent:
    """
    Agent responsible for filtering out clearly irrelevant NEW documents
    Preserves already-labeled documents
    """
    
    def __init__(self):
        self.name = "FilterAgent"
        self.logger = Logger()
        self.llm = LLMClient()
        
        self.system_prompt = """You are a Document Filtering Agent specialized in identifying irrelevant documents.

**YOUR ROLE:** Remove ONLY clearly irrelevant documents from NEW unlabeled documents.

**IMPORTANT:** 
- You are filtering NEW documents only
- These documents will be compared to already-labeled examples later
- Be conservative - when in doubt, KEEP the document

**FILTERING CRITERIA:**

FILTER OUT (Irrelevant):
- Completely unrelated to query topic
- Wrong domain entirely (e.g., technical docs when query is about benefits)
- Spam or invalid content
- Empty or "No Title" with no content

KEEP:
- Documents answering query (even if different location)
- Documents with partial relevance
- Documents that might be context-relevant
- Older versions of relevant documents
- When uncertain, KEEP

**SPECIAL RULES:**
- Wrong location but correct topic → KEEP (will be labeled ACCEPTABLE later)
- Older year but relevant topic → KEEP (will be labeled SOMEWHAT_RELEVANT later)
- Missing title but has content → KEEP (will be labeled NOT_SURE later)

Provide clear reasoning for each filtered document."""

    def filter_documents(self, documents: List[Document], query: str, 
                        location: str = "") -> Tuple[List[Document], List[Document], Dict[str, str]]:
        """
        Filter NEW documents, removing only clearly irrelevant ones
        
        Returns:
            - kept_documents: Documents that passed filtering
            - removed_documents: Documents that were filtered out
            - filter_reasons: Dictionary mapping doc_id to reason for filtering
        """
        self.logger.log(self.name, 
            f"Filtering {len(documents)} NEW documents for query: '{query}'")
        
        if not documents:
            return [], [], {}
        
        # Prepare document data for LLM
        docs_data = []
        for doc in documents:
            content = extract_text_from_html(doc.html)[:2000]
            docs_data.append({
                "id": doc.id,
                "title": doc.title,
                "content_preview": content[:500]  # First 500 chars for preview
            })
        
        user_prompt = f"""FILTERING TASK:

Query: "{query}"
User Location: "{location if location else "Not specified"}"

NEW Documents to Filter ({len(documents)} total):
{json.dumps(docs_data, indent=2)}

**TASK:** Identify ONLY clearly irrelevant documents to filter out.

**REMEMBER:**
- These are NEW unlabeled documents
- Be conservative - when in doubt, KEEP
- Wrong location but correct topic → KEEP
- Older documents but relevant → KEEP
- Partial relevance → KEEP

**DECISION CRITERIA:**
1. Is document completely unrelated to "{query}"?
2. Is it spam, invalid, or empty content?
3. Is it wrong domain entirely?

If YES to any above → FILTER OUT
Otherwise → KEEP

Respond in JSON format:
{{
    "keep": [
        {{
            "doc_id": "document_id",
            "reason": "Why keeping this document"
        }}
    ],
    "filter": [
        {{
            "doc_id": "document_id",
            "reason": "Why filtering this document (must be clearly irrelevant)"
        }}
    ],
    "filtering_summary": "Brief summary of filtering decisions"
}}

**BE CONSERVATIVE:** When uncertain, KEEP the document."""

        try:
            response = self.llm.call_with_json_response(self.system_prompt, user_prompt)
            
            keep_list = response.get("keep", [])
            filter_list = response.get("filter", [])
            
            # Create sets for quick lookup
            keep_ids = {item.get("doc_id") for item in keep_list}
            filter_ids = {item.get("doc_id") for item in filter_list}
            
            # Create reason mapping
            filter_reasons = {
                item.get("doc_id"): item.get("reason", "No reason provided")
                for item in filter_list
            }
            
            # Separate documents
            kept_documents = []
            removed_documents = []
            
            for doc in documents:
                if doc.id in filter_ids:
                    removed_documents.append(doc)
                    self.logger.log(self.name, 
                        f"❌ Filtered: {doc.title[:50]}... | Reason: {filter_reasons.get(doc.id, 'N/A')}")
                else:
                    kept_documents.append(doc)
            
            self.logger.log(self.name, 
                f"✓ Filtering complete: Kept {len(kept_documents)}, Removed {len(removed_documents)}")
            
            return kept_documents, removed_documents, filter_reasons
            
        except Exception as e:
            self.logger.log(self.name, 
                f"LLM filtering failed: {e}. Keeping all documents as fallback.", "ERROR")
            
            # Fallback: Keep all documents
            return documents, [], {}
