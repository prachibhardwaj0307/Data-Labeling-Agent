"""
Grouping Agent - Groups documents by similarity AND year
"""
from typing import List
import json
import re
from models.data_models import Document, DocumentGroup
from utils.helpers import Logger, extract_text_from_html
from utils.llm_client import LLMClient

class GroupingAgent:
    """
    Agent responsible for grouping similar documents
    Groups by topic AND year for recency-aware organization
    """
    
    def __init__(self):
        self.name = "GroupingAgent"
        self.logger = Logger()
        self.llm = LLMClient()
        
        self.system_prompt = """You are a Document Grouping Agent specialized in clustering similar documents.

**CRITICAL: GROUP BY TOPIC AND YEAR - SEPARATE GROUPS FOR DIFFERENT YEARS**

Your role is to:
1. Identify document topics and themes
2. Extract year/date information from titles and content
3. **CREATE SEPARATE GROUPS FOR EACH YEAR** within the same topic
4. Prioritize recent documents (2024-2025)

GROUPING STRATEGY:
- Primary: Group by TOPIC (benefits, holidays, policies, etc.)
- Secondary: **SEPARATE BY YEAR** within topics (THIS IS CRITICAL)
- Example: If you have 5 "Benefits" documents:
  - 3 from 2025 → "Employee Benefits 2025" (separate group)
  - 2 from 2024 → "Employee Benefits 2024" (separate group)
  - NOT: "Employee Benefits" (mixed years - WRONG)

YEAR DETECTION:
- Look for years in titles: "2025", "2024", "2023", etc.
- Look for dates: "Jan 2025", "December 2024", etc.
- If no year found, mark as "Undated" but try to infer from content

GROUP NAMING CONVENTION (MANDATORY):
- **ALWAYS use format: "[Topic] [Year]"**
- Examples: "India Benefits 2025", "US Holidays 2024", "UK Policies 2023"
- Never mix years in one group
- If multiple years exist for a topic → Create multiple groups

QUALITY REQUIREMENTS:
- Groups can have 1-10 documents
- Each group MUST have clear year identification in name
- **DO NOT mix 2025 docs with 2024 docs**
- Provide reasoning for year-based grouping

This year-based grouping is CRITICAL for downstream labeling where 2025 = RELEVANT and older = SOMEWHAT_RELEVANT."""

    def group_documents(self, documents: List[Document], query: str) -> List[DocumentGroup]:
        """
        Group documents by topic and year
        """
        self.logger.log(self.name, f"Grouping {len(documents)} documents by topic AND YEAR")
        
        if not documents:
            return []
        
        # Prepare document data with year extraction
        docs_data = []
        for doc in documents:
            content = extract_text_from_html(doc.html)[:500]
            
            # Extract year from title or content
            year = self._extract_year(doc.title, content)
            
            docs_data.append({
                "id": doc.id,
                "title": doc.title,
                "content_preview": content,
                "detected_year": year
            })
        
        user_prompt = f"""GROUPING TASK WITH YEAR SEPARATION:

Query: "{query}"

Documents to Group ({len(documents)} total):
{json.dumps(docs_data, indent=2)}

**CRITICAL INSTRUCTION: GROUP BY TOPIC AND YEAR**

For each document:
1. Check "detected_year" field
2. Identify the topic
3. Create SEPARATE groups for EACH YEAR of the same topic

Example: If you have "Benefits" docs from 2025, 2024, 2023:
- Create 3 groups:
  - "India Benefits 2025" (most recent)
  - "India Benefits 2024" (previous year)
  - "India Benefits 2023" (older)

GROUP NAMING RULES:
- Use format: "[Topic] [Year]"
- Examples: "Employee Benefits 2025", "Holiday Calendar 2024", "Policies 2023"
- For undated docs: "[Topic] (Undated)" or "[Topic] (Unknown Year)"
- NEVER mix years in one group

OUTPUT FORMAT (JSON):
{{
    "groups": [
        {{
            "group_name": "Clear name with YEAR (e.g., 'India Benefits 2025')",
            "theme": "Description including year/recency information",
            "document_ids": ["doc_id1", "doc_id2"],
            "year": "2025/2024/2023/Unknown",
            "reasoning": "Why these docs grouped together, emphasizing year separation"
        }}
    ],
    "grouping_strategy": "Explain your year-based grouping approach"
}}

REQUIREMENTS:
- **Separate groups by year when possible**
- Prioritize identifying recent (2024-2025) documents
- Clear year labels in ALL group names
- 1-10 documents per group"""

        try:
            response = self.llm.call_with_json_response(self.system_prompt, user_prompt)
            
            groups_data = response.get("groups", [])
            groups = []
            
            doc_map = {doc.id: doc for doc in documents}
            
            for group_data in groups_data:
                group_docs = []
                for doc_id in group_data.get("document_ids", []):
                    if doc_id in doc_map:
                        group_docs.append(doc_map[doc_id])
                
                if group_docs:
                    group = DocumentGroup(
                        name=group_data.get("group_name", "Unnamed Group"),
                        documents=group_docs,
                        theme=group_data.get("theme", "No theme"),
                        reasons=[group_data.get("reasoning", "No reasoning")]
                    )
                    groups.append(group)
                    
                    year = group_data.get("year", "Unknown")
                    self.logger.log(self.name, 
                        f"Created group '{group.name}' with {len(group_docs)} docs (Year: {year})")
            
            self.logger.log(self.name, f"Created {len(groups)} year-aware groups")
            return groups
            
        except Exception as e:
            self.logger.log(self.name, f"LLM grouping failed: {e}", "ERROR")
            
            # Fallback: Create single group
            return [DocumentGroup(
                name="All Documents",
                documents=documents,
                theme="Fallback grouping due to error",
                reasons=["Grouped due to LLM error"]
            )]
    
    def _extract_year(self, title: str, content: str) -> str:
        """Extract year from title or content"""
        text = f"{title} {content}"
        
        # Look for 4-digit years (2020-2030)
        years = re.findall(r'\b(202[0-9]|203[0])\b', text)
        
        if years:
            # Return most recent year found
            return max(years)
        
        return "Unknown"
