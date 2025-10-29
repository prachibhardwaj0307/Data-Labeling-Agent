"""
Relabel Agent - Relabels problematic documents with TOP 10 RELEVANT enforcement, 
year prioritization, and example-based learning
"""
from typing import Dict, List
import json
import re
from datetime import datetime
from models.data_models import LabelingDecision, LabelReviewDecision
from utils.helpers import Logger
from utils.llm_client import LLMClient

class RelabelAgent:
    """
    Agent responsible for relabeling documents that failed review
    Enforces maximum 10 RELEVANT documents rule with year-based ranking and examples
    """
    
    def __init__(self):
        self.name = "RelabelAgent"
        self.logger = Logger()
        self.llm = LLMClient()
        
        # Get current year
        self.current_year = datetime.now().year
        
        self.system_prompt = f"""You are a Relabeling Agent specialized in correcting document labels based on reviewer feedback.

**CRITICAL RULE: MAXIMUM 10 RELEVANT DOCUMENTS**

Current Year: {self.current_year}

When you have more than 10 RELEVANT documents, you must:
1. Rank ALL relevant documents by these criteria (MOST IMPORTANT FIRST):
   - **YEAR/RECENCY** ({self.current_year} documents rank HIGHEST, then 2024, 2023, etc.)
   - **Completeness**: Comprehensive, detailed coverage
   - **Direct Match**: Directly answers the query (not tangential)
   - **Location Match**: Exact location match preferred
   - **Quality**: Authoritative, well-written information
   - **Similarity to Examples**: Match patterns from already-labeled documents

2. Select TOP 10 as RELEVANT (prioritizing current year)
3. Downgrade remaining to SOMEWHAT_RELEVANT

**YEAR PRIORITY IS CRITICAL:**
- Documents from {self.current_year} should be in positions 1-5 if they exist
- Documents from 2024 should be in positions 6-8 if they exist
- Older documents (2023 and earlier) should be positions 9-10 or downgraded

**USE PROVIDED EXAMPLES:**
- Compare documents to already-labeled examples
- Maintain consistency with existing labeling patterns
- Consider both year AND similarity to examples

Be objective and justify your rankings clearly with emphasis on year and examples."""

    def relabel_documents(self, current_labels: Dict[str, List[LabelingDecision]], 
                         review: LabelReviewDecision, query: str, location: str,
                         label_examples: Dict = None) -> Dict[str, List[LabelingDecision]]:
        """
        Relabel documents based on review feedback with TOP 10 enforcement and examples
        
        Args:
            current_labels: Current labeling results
            review: Review decision with feedback
            query: User's search query
            location: User's location
            label_examples: Examples from already-labeled documents
        """
        self.logger.log(self.name, 
            f"Relabeling based on review: {review.feedback[:100]}...")
        
        # Store examples
        self.examples = label_examples or {"relevant": [], "somewhat_relevant": [], "acceptable": []}
        
        if label_examples:
            total_examples = sum(len(v) for v in label_examples.values())
            self.logger.log(self.name, f"Using {total_examples} examples for relabeling")
        
        # Get relevant documents count
        relevant_docs = current_labels.get("relevant", [])
        relevant_count = len(relevant_docs)
        
        self.logger.log(self.name, f"Current RELEVANT count: {relevant_count}")
        
        # Check if this is a "too many relevant" issue
        if relevant_count > 10:
            self.logger.log(self.name, 
                f"⚠️ RELEVANT OVERFLOW DETECTED: {relevant_count} > 10")
            self.logger.log(self.name, "Initiating TOP 10 selection with year prioritization...")
            
            return self._select_top_10_relevant(current_labels, query, location, relevant_docs)
        else:
            self.logger.log(self.name, "No overflow issue, returning current labels")
            return current_labels
    
    def _select_top_10_relevant(self, current_labels: Dict[str, List[LabelingDecision]], 
                               query: str, location: str, 
                               relevant_docs: List[LabelingDecision]) -> Dict[str, List[LabelingDecision]]:
        """
        Select TOP 10 RELEVANT documents with year prioritization, examples, and downgrade the rest
        """
        self.logger.log(self.name, f"Selecting TOP 10 from {len(relevant_docs)} RELEVANT documents")
        
        # Extract year from each document's reasoning
        docs_with_years = []
        for decision in relevant_docs:
            # Try to extract year from reasoning
            year = self._extract_year_from_reason(decision.reason)
            docs_with_years.append({
                "doc_id": decision.doc_id,
                "current_reason": decision.reason,
                "confidence": decision.confidence,
                "detected_year": year
            })
        
        # Prepare examples section for prompt
        examples_section = ""
        if self.examples and any(self.examples.values()):
            examples_section = f"""
**📚 REFERENCE EXAMPLES (Use for consistency):**

✅ RELEVANT Examples ({len(self.examples.get('relevant', []))})
{json.dumps(self.examples.get('relevant', [])[:3], indent=2)}

⚠️ SOMEWHAT_RELEVANT Examples ({len(self.examples.get('somewhat_relevant', []))})
{json.dumps(self.examples.get('somewhat_relevant', [])[:3], indent=2)}

**IMPORTANT:** Prioritize documents similar to RELEVANT examples when selecting TOP 10.
"""
        
        # Prepare data for ranking
        user_prompt = f"""URGENT TASK: Select TOP 10 RELEVANT documents from {len(relevant_docs)} candidates.

**CURRENT YEAR: {self.current_year}**

Query: "{query}"
User Location: "{location}"

{examples_section}

Current RELEVANT documents (MUST select only TOP 10):
{json.dumps(docs_with_years, indent=2)}

**RANKING CRITERIA (PRIORITY ORDER - YEAR IS MOST IMPORTANT):**

1. **YEAR/RECENCY** ⭐ HIGHEST PRIORITY ⭐
   - {self.current_year} documents → Rank 1-5 (MUST prioritize)
   - 2024 documents → Rank 6-8 (second priority)
   - 2023 documents → Rank 9-10 (lower priority)
   - 2022 and older → Should be downgraded to SOMEWHAT_RELEVANT

2. **Similarity to Examples** - Match patterns from provided examples

3. **Completeness** - Comprehensive coverage of "{query}"

4. **Direct Query Match** - Directly answers "{query}"

5. **Location Match** - Matches "{location}" exactly

6. **Information Quality** - Detailed, authoritative content

**CRITICAL INSTRUCTIONS:**
- Look at "detected_year" for each document
- Prioritize {self.current_year} documents in TOP 5 positions
- Consider similarity to RELEVANT examples
- Only include older documents if no current year alternatives exist
- Explain year-based ranking and example similarity in your reasoning

Respond in JSON format:
{{
    "top_10_relevant": [
        {{
            "doc_id": "id1",
            "rank": 1,
            "selection_reason": "Why this is #1: [MUST mention YEAR first, example similarity, then other factors]"
        }},
        {{
            "doc_id": "id2",
            "rank": 2,
            "selection_reason": "Why this is #2: [MUST mention YEAR first, example similarity, then other factors]"
        }},
        ... (exactly 10 documents, ranked by YEAR first)
    ],
    "downgraded_to_somewhat": [
        {{
            "doc_id": "id11",
            "downgrade_reason": "Why NOT in top 10: [e.g., older year, less similar to examples, less comprehensive, etc.]"
        }},
        ... (remaining {len(relevant_docs) - 10} documents)
    ],
    "ranking_methodology": "Explain your overall selection criteria with EMPHASIS on year prioritization and example similarity"
}}

**CRITICAL: Prioritize CURRENT YEAR ({self.current_year}) documents first! You MUST return exactly 10 for top_10_relevant.**"""

        try:
            self.logger.log(self.name, "Calling LLM for TOP 10 selection with year prioritization and examples...")
            response = self.llm.call_with_json_response(self.system_prompt, user_prompt)
            
            top_10_list = response.get("top_10_relevant", [])
            downgrade_list = response.get("downgraded_to_somewhat", [])
            methodology = response.get("ranking_methodology", "No methodology provided")
            
            self.logger.log(self.name, f"LLM selected {len(top_10_list)} for TOP 10")
            self.logger.log(self.name, f"LLM selected {len(downgrade_list)} to downgrade")
            self.logger.log(self.name, f"Methodology: {methodology}")
            
            # Validate we got the right counts
            if len(top_10_list) != 10:
                self.logger.log(self.name, 
                    f"⚠️ WARNING: Expected 10, got {len(top_10_list)}. Adjusting...", "WARNING")
                top_10_list = top_10_list[:10]  # Take first 10
            
            # Create document map
            doc_map = {d.doc_id: d for d in relevant_docs}
            
            # Build new labels dictionary
            new_labels = {
                "relevant": [],
                "somewhat_relevant": list(current_labels.get("somewhat_relevant", [])),
                "acceptable": list(current_labels.get("acceptable", [])),
                "not_sure": list(current_labels.get("not_sure", []))
            }
            
            # Add TOP 10 to RELEVANT
            for item in top_10_list:
                doc_id = item.get("doc_id")
                rank = item.get("rank", 0)
                reason = item.get("selection_reason", "Selected as top 10")
                
                if doc_id in doc_map:
                    original = doc_map[doc_id]
                    new_decision = LabelingDecision(
                        doc_id=doc_id,
                        label="relevant",
                        reason=f"[🏆 TOP 10 - Rank #{rank}] {reason}",
                        confidence="high",
                        agent_name=self.name
                    )
                    new_labels["relevant"].append(new_decision)
                    self.logger.log(self.name, f"  ✅ Rank {rank}: {doc_id}")
            
            # Downgrade rest to SOMEWHAT_RELEVANT
            for item in downgrade_list:
                doc_id = item.get("doc_id")
                reason = item.get("downgrade_reason", "Not in top 10")
                
                if doc_id in doc_map:
                    new_decision = LabelingDecision(
                        doc_id=doc_id,
                        label="somewhat_relevant",
                        reason=f"[⬇️ DOWNGRADED FROM RELEVANT] {reason}",
                        confidence="medium",
                        agent_name=self.name
                    )
                    new_labels["somewhat_relevant"].append(new_decision)
                    self.logger.log(self.name, f"  ⬇️ Downgraded: {doc_id}")
            
            self.logger.log(self.name, 
                f"✅ Relabeling complete: {len(new_labels['relevant'])} RELEVANT, "
                f"{len(new_labels['somewhat_relevant'])} SOMEWHAT_RELEVANT")
            
            return new_labels
            
        except Exception as e:
            self.logger.log(self.name, 
                f"❌ LLM relabeling failed: {e}. Using fallback.", "ERROR")
            
            # FALLBACK: Sort by year first, then confidence
            self.logger.log(self.name, "Using fallback: prioritizing by year then confidence")
            
            # Add year to each document
            docs_with_year = []
            for doc in relevant_docs:
                year = self._extract_year_from_reason(doc.reason)
                year_int = int(year) if year != "Unknown" else 0
                docs_with_year.append((doc, year_int, year))
            
            # Sort by year (descending), then confidence
            confidence_order = {"high": 3, "medium": 2, "low": 1}
            sorted_docs = sorted(
                docs_with_year,
                key=lambda x: (x[1], confidence_order.get(x[0].confidence, 0)),
                reverse=True
            )
            
            new_labels = {
                "relevant": [],
                "somewhat_relevant": list(current_labels.get("somewhat_relevant", [])),
                "acceptable": list(current_labels.get("acceptable", [])),
                "not_sure": list(current_labels.get("not_sure", []))
            }
            
            # Keep top 10
            for i, (doc, year_int, year_str) in enumerate(sorted_docs[:10], 1):
                new_decision = LabelingDecision(
                    doc_id=doc.doc_id,
                    label="relevant",
                    reason=f"[TOP 10 by year ({year_str}) and confidence - #{i}] {doc.reason}",
                    confidence=doc.confidence,
                    agent_name=self.name
                )
                new_labels["relevant"].append(new_decision)
            
            # Downgrade rest
            for doc, year_int, year_str in sorted_docs[10:]:
                new_decision = LabelingDecision(
                    doc_id=doc.doc_id,
                    label="somewhat_relevant",
                    reason=f"[DOWNGRADED - not in top 10 by year ({year_str})] {doc.reason}",
                    confidence="medium",
                    agent_name=self.name
                )
                new_labels["somewhat_relevant"].append(new_decision)
            
            self.logger.log(self.name, 
                f"Fallback complete: {len(new_labels['relevant'])} RELEVANT")
            
            return new_labels
    
    def _extract_year_from_reason(self, reason: str) -> str:
        """Extract year from reasoning text"""
        # Look for 4-digit years (2020-2030)
        years = re.findall(r'\b(202[0-9]|203[0])\b', reason)
        
        if years:
            # Return most recent year found
            return max(years)
        
        return "Unknown"
