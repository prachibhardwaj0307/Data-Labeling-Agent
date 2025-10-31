"""
Labeling Agent - Labels GROUPS with hybrid approach and RICH EXAMPLES
"""
from typing import List, Dict
import json
import re
from datetime import datetime
from models.data_models import Document, DocumentGroup, LabelingDecision
from utils.helpers import Logger, extract_text_from_html
from utils.llm_client import LLMClient

class LabelingAgent:
    """Agent responsible for labeling ENTIRE GROUPS with rich examples"""
    
    def __init__(self):
        self.name = "LabelingAgent"
        self.logger = Logger()
        self.llm = LLMClient()
        self.current_year = datetime.now().year
        
        self.system_prompt = f"""You are a Document Labeling Agent specialized in categorizing GROUPS of documents.

**CRITICAL: You label ENTIRE GROUPS, not individual documents!**

Current Year: {self.current_year}

**LABELING RULES (DEPENDS ON YEAR AVAILABILITY):**

**IF YEAR IS KNOWN (has 2025, 2024, 2023, etc.):**
1. **RELEVANT** - When:
   - FROM CURRENT YEAR ({self.current_year})
   - Answers query for user's location
   - Maximum 10 documents total

2. **SOMEWHAT_RELEVANT** - When:
   - From PREVIOUS YEARS (2024, 2023, etc.)
   - Answers query for user's location
   - Still useful but not current

3. **ACCEPTABLE** - When:
   - Any year but WRONG location
   - OR very old (pre-2022)

**IF YEAR IS UNKNOWN (no year detected):**
**IGNORE YEAR RULES - Use traditional evaluation:**

1. **RELEVANT** - When:
   - Directly answers the query
   - Matches user's location
   - High quality and complete information
   - **Similar to RELEVANT examples you'll see**

2. **SOMEWHAT_RELEVANT** - When:
   - Partially answers the query
   - Matches location but incomplete
   - Related but not perfect match
   - **Similar to SOMEWHAT_RELEVANT examples**

3. **ACCEPTABLE** - When:
   - Correct topic but WRONG location
   - Tangentially related
   - Provides context but not direct answer
   - **Similar to ACCEPTABLE examples**

4. **NOT_SURE** - ONLY when:
   - Cannot determine relevance
   - Completely unclear content

**CRITICAL DISTINCTION:**
- **Year Known** → Year is PRIMARY factor (2025 > 2024 > 2023)
- **Year Unknown** → Compare to examples and evaluate by topic + location + quality

**LOCATION RULES (always apply):**
- Correct location = RELEVANT or SOMEWHAT_RELEVANT
- Wrong location = ACCEPTABLE (regardless of other factors)

**USE EXAMPLES TO GUIDE YOUR DECISIONS**

Label ENTIRE groups consistently."""

    def label_documents(self, groups: List[DocumentGroup], query: str, 
                       location: str = "", label_examples: Dict = None) -> Dict[str, List[LabelingDecision]]:
        """Label ENTIRE GROUPS with rich examples"""
        
        self.logger.log(self.name, f"Labeling {len(groups)} GROUPS with RICH EXAMPLES")
        
        self.examples = label_examples or {"relevant": [], "somewhat_relevant": [], "acceptable": []}
        
        if label_examples:
            total_examples = sum(len(v) for v in label_examples.values())
            self.logger.log(self.name, f"Using {total_examples} RICH examples (with content) as reference")
        
        results = {
            "relevant": [],
            "somewhat_relevant": [],
            "acceptable": [],
            "not_sure": []
        }
        
        for group in groups:
            group_label = self._label_group(group, query, location)
            
            for doc in group.documents:
                decision = LabelingDecision(
                    doc_id=doc.id,
                    label=group_label["label"],
                    reason=f"[GROUP: {group.name}] {group_label['reason']}",
                    confidence=group_label["confidence"],
                    agent_name=self.name
                )
                results[decision.label].append(decision)
            
            self.logger.log(self.name, f"✓ GROUP '{group.name}' → {group_label['label'].upper()}")
        
        return results

    def _label_group(self, group: DocumentGroup, query: str, location: str) -> dict:
        """Label entire group with rich examples"""
        
        group_year = self._extract_year_from_group(group)
        
        docs_summary = []
        for doc in group.documents:
            content = extract_text_from_html(doc.html)[:400]
            docs_summary.append({
                "id": doc.id,
                "title": doc.title,
                "content_preview": content
            })
        
        # Prepare examples section with FULL CONTENT
        examples_section = ""
        if self.examples and any(self.examples.values()):
            
            # Format examples with content previews
            def format_examples(examples, max_show=3):
                formatted = []
                for ex in examples[:max_show]:
                    formatted.append({
                        "id": ex.get("id"),
                        "title": ex.get("title"),
                        "content_preview": ex.get("content_preview", "No content"),  # ✅ Show content
                        "label": ex.get("label")
                    })
                return formatted
            
            relevant_examples = format_examples(self.examples.get('relevant', []))
            somewhat_examples = format_examples(self.examples.get('somewhat_relevant', []))
            acceptable_examples = format_examples(self.examples.get('acceptable', []))
            
            examples_section = f"""
**📚 REFERENCE EXAMPLES (Already Labeled Documents WITH CONTENT):**

These documents were previously labeled. Use them to understand what type of content belongs in each category.

✅ RELEVANT Examples ({len(self.examples.get('relevant', []))} total, showing first 3):
{json.dumps(relevant_examples, indent=2)}

⚠️ SOMEWHAT_RELEVANT Examples ({len(self.examples.get('somewhat_relevant', []))} total, showing first 3):
{json.dumps(somewhat_examples, indent=2)}

ℹ️ ACCEPTABLE Examples ({len(self.examples.get('acceptable', []))} total, showing first 3):
{json.dumps(acceptable_examples, indent=2)}

**CRITICAL:** Compare the NEW documents you're labeling to these examples. Documents similar to RELEVANT examples should be labeled RELEVANT (unless year rules override).
"""
        
        user_prompt = f"""GROUP LABELING WITH RICH EXAMPLES:

Query: "{query}"
User Location: "{location}"
Current Year: {self.current_year}

{examples_section}

**GROUP TO LABEL:**
- Name: {group.name}
- Theme: {group.theme}
- Detected Year: {group_year}
- Document Count: {len(group.documents)}

Documents:
{json.dumps(docs_summary, indent=2)}

**LABELING APPROACH:**

**CASE 1: Year is KNOWN ({group_year} is a valid year)**
Use YEAR-BASED RULES:
- If year = {self.current_year} + correct location "{location}" + answers query → RELEVANT
- If year = 2024 or older + correct location + answers query → SOMEWHAT_RELEVANT  
- If wrong location (regardless of year) → ACCEPTABLE

**CASE 2: Year is UNKNOWN ({group_year} = "Unknown")**
Use TRADITIONAL EVALUATION + COMPARE TO EXAMPLES:
- If directly answers query + correct location + **similar to RELEVANT examples** → RELEVANT
- If partially answers query + correct location + **similar to SOMEWHAT examples** → SOMEWHAT_RELEVANT
- If correct topic but wrong location + **similar to ACCEPTABLE examples** → ACCEPTABLE
- If unclear/irrelevant → NOT_SURE

**IMPORTANT FOR THIS GROUP:**
- Detected Year: {group_year}
- If {group_year} = "Unknown" → Compare to examples above, evaluate by content similarity
- If {group_year} = actual year → Use year-based rules

Respond in JSON (use strings, not booleans):
{{
    "label": "RELEVANT|SOMEWHAT_RELEVANT|ACCEPTABLE|NOT_SURE",
    "reason": "Explain: 1) Year known/unknown? 2) Which examples is this similar to? 3) Why this label?",
    "confidence": "high|medium|low",
    "evaluation_method": "year-based or example-comparison",
    "similar_to_examples": "Which example category (RELEVANT/SOMEWHAT/ACCEPTABLE) is this most similar to?",
    "topic_match": "yes or no (string)",
    "location_match": "yes or no (string)"
}}

**REMEMBER:** 
- Use examples to guide decisions when year is unknown
- If similar to RELEVANT examples AND correct location → likely RELEVANT
- Return strings for all fields, NOT booleans"""

        try:
            response = self.llm.call_with_json_response(self.system_prompt, user_prompt)
            
            # Handle label field
            label = response.get("label", "NOT_SURE")
            if isinstance(label, str):
                label = label.lower()
            else:
                label = str(label).lower()
            
            if label not in ["relevant", "somewhat_relevant", "acceptable", "not_sure"]:
                label = "not_sure"
            
            reason = response.get("reason", "No reason")
            evaluation_method = response.get("evaluation_method", "unknown")
            similar_to = response.get("similar_to_examples", "none")
            
            # Handle topic_match
            topic_match = response.get("topic_match", "unknown")
            if isinstance(topic_match, bool):
                topic_match = "yes" if topic_match else "no"
            elif not isinstance(topic_match, str):
                topic_match = str(topic_match)
            
            # Handle location_match
            location_match = response.get("location_match", "unknown")
            if isinstance(location_match, bool):
                location_match = "yes" if location_match else "no"
            elif not isinstance(location_match, str):
                location_match = str(location_match)
            
            full_reason = (f"{reason} | Year: {group_year} ({evaluation_method}). "
                          f"Similar to: {similar_to} | Topic: {topic_match}, Location: {location_match}")
            
            return {
                "label": label,
                "reason": full_reason,
                "confidence": response.get("confidence", "medium")
            }
            
        except Exception as e:
            self.logger.log(self.name, f"Labeling failed: {e}", "ERROR")
            return {"label": "not_sure", "reason": f"Failed: {e}", "confidence": "low"}
    
    def _extract_year_from_group(self, group: DocumentGroup) -> str:
        """Extract year from group name"""
        text = f"{group.name} {group.theme}"
        years = re.findall(r'\b(202[0-9]|203[0])\b', text)
        return max(years) if years else "Unknown"
