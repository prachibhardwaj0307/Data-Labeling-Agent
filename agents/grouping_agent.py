"""
Grouping Agent - Clusters similar documents using LLM
"""
from typing import List
import json
from models.data_models import Document, DocumentGroup
from utils.helpers import Logger, extract_text_from_html
from utils.llm_client import LLMClient

class GroupingAgent:
    """
    Agent responsible for grouping similar documents for efficient batch labeling
    Uses LLM to identify themes and create meaningful groups
    """
    
    def __init__(self):
        self.name = "GroupingAgent"
        self.logger = Logger()
        self.llm = LLMClient()
        
        # Define the system prompt
        # self.system_prompt = """You are a Document Grouping Agent specialized in clustering similar documents for efficient batch processing.
        self.system_prompt = """You are a Document Grouping Agent specialized in clustering similar documents for efficient batch processing.

Your role is to:
1. Analyze document titles and content to identify themes
2. Identify common topics, document types, or purposes
3. Group similar documents together for batch labeling
4. Create clear, descriptive group names and themes
5. Ensure groups are appropriately sized for efficient processing

GROUPING GUIDELINES:
- Group documents with similar topics, themes, or document types
- Ideal group size: 3-7 documents (minimum 1, maximum 10)
- Single-document groups are ALLOWED if the document is unique
- Documents about the same topic but different locations CAN be grouped together
- Documents of the same type (handbooks, policies, presentations) CAN be grouped
- Create clear, meaningful group names (e.g., "Employee Benefits Documents", "India-specific Policies")
- Identify the common theme that connects documents in each group

GROUPING STRATEGIES:
- By topic: Benefits, holidays, policies, handbooks, etc.
- By document type: Presentations, handbooks, forms, etc.
- By region: India-specific, US-specific, UK-specific, Global
- By recency: 2025 documents, 2024 documents, older documents
- By audience: Employees, managers, HR, etc.

QUALITY REQUIREMENTS:
- Every document MUST be assigned to exactly ONE group
- Single-document groups are acceptable for unique documents
- No empty groups allowed
- Group names must be descriptive and meaningful
- Themes must clearly explain what connects the documents

Provide detailed reasoning for your grouping decisions."""


    def group_documents(self, documents: List[Document], query: str) -> List[DocumentGroup]:
        """
        Group similar documents using LLM analysis
        
        Args:
            documents: List of Document objects to group
            query: User's search query for context
            
        Returns:
            List of DocumentGroup objects
        """
        self.logger.log(self.name, f"Grouping {len(documents)} documents using LLM")
        self.logger.log(self.name, f"Query context: '{query}'")
        
        if not documents:
            return []
        
        # If only 1-2 documents, create a single group
        if len(documents) <= 2:
            self.logger.log(self.name, "Only 1-2 documents, creating single group")
            return [DocumentGroup(
                name="All_Documents",
                documents=documents,
                theme=f"Documents related to '{query}'",
                reasons=["Small document set - grouped together"]
            )]
        
        # Prepare document data for LLM
        docs_data = []
        for doc in documents:
            content_preview = extract_text_from_html(doc.html)[:600]  # First 600 chars
            docs_data.append({
                "id": doc.id,
                "title": doc.title,
                "content_preview": content_preview
            })
        
        # Create the user prompt
        user_prompt = f"""GROUPING TASK:

Query: "{query}"

Documents to Group ({len(documents)} total):
{json.dumps(docs_data, indent=2)}

TASK:
Analyze these documents and create meaningful groups based on similarity, theme, or topic.

Consider:
- Document topics and content
- Document types (handbook, policy, presentation, etc.)
- Geographic regions mentioned (India, US, UK, Global)
- Document recency (year mentioned in title/content)
- Audience or purpose

OUTPUT FORMAT (JSON):
{{
    "groups": [
        {{
            "group_name": "Clear, descriptive name (e.g., 'India Employee Benefits', 'US Handbooks 2025')",
            "theme": "Detailed explanation of what connects these documents",
            "document_ids": ["doc_id1", "doc_id2", "doc_id3"],
            "reasoning": "Why these specific documents belong together"
        }}
    ],
    "grouping_strategy": "Explain your overall grouping approach"
}}

REQUIREMENTS:
- Create 1-10 documents per group (ideal: 3-7)
- SINGLE-DOCUMENT groups are ACCEPTABLE for unique documents
- Every document must be assigned to exactly ONE group
- Group names must be descriptive and unique
- Aim for {max(1, len(documents) // 5)} to {len(documents) // 2} groups total"""

        try:
            # Call LLM and get structured response
            response = self.llm.call_with_json_response(self.system_prompt, user_prompt)
            
            groups = []
            doc_map = {doc.id: doc for doc in documents}
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
                        name=group_data.get("group_name", f"Group_{len(groups)+1}"),
                        documents=group_docs,
                        theme=group_data.get("theme", "No theme provided"),
                        reasons=[group_data.get("reasoning", "No reasoning provided")]
                    )
                    groups.append(group)
                    
                    self.logger.log(self.name, 
                        f"✓ Created group '{group.name}' with {len(group_docs)} documents")
                    self.logger.log(self.name, f"  Theme: {group.theme}")
            
            # Handle any unassigned documents
            unassigned = [doc for doc in documents if doc.id not in assigned_docs]
            if unassigned:
                self.logger.log(self.name, 
                    f"⚠️ Found {len(unassigned)} unassigned documents, creating additional group")
                group = DocumentGroup(
                    name="Additional_Documents",
                    documents=unassigned,
                    theme="Documents not assigned to primary groups",
                    reasons=["Documents that didn't fit into main thematic groups"]
                )
                groups.append(group)
            
            # Log summary
            strategy = response.get("grouping_strategy", "No strategy provided")
            self.logger.log(self.name, f"Grouping complete: Created {len(groups)} groups")
            self.logger.log(self.name, f"Strategy: {strategy}")
            
            return groups
            
        except Exception as e:
            # Fallback: create a single group with all documents
            self.logger.log(self.name, 
                          f"⚠️ LLM grouping failed: {e}. Creating single group as fallback.", 
                          "WARNING")
            return [DocumentGroup(
                name="All_Documents",
                documents=documents,
                theme=f"All documents related to '{query}'",
                reasons=[f"Fallback grouping due to LLM error: {str(e)[:100]}"]
            )]
