"""
Relabel Agent - Relabels problematic documents with TOP 10 RELEVANT enforcement
"""
from typing import Dict, List
import json
from models.data_models import LabelingDecision, LabelReviewDecision
from utils.helpers import Logger, extract_text_from_html
from utils.llm_client import LLMClient

class RelabelAgent:
    """
    Agent responsible for relabeling documents that failed review
    Enforces maximum 10 RELEVANT documents rule
    """
    
    def __init__(self):
        self.name = "RelabelAgent"
        self.logger = Logger()
        self.llm = LLMClient()
        
        self.system_prompt = """You are a Relabeling Agent specialized in correcting document labels.

**CRITICAL RULE: MAXIMUM 10 RELEVANT DOCUMENTS**

When you have more than 10 RELEVANT documents, you must:
1. Rank ALL relevant documents by these criteria (most important first):
   - **Recency**: 2024-2025 documents rank highest
   - **Completeness**: Comprehensive, detailed coverage
   - **Direct Match**: Directly answers the query (not tangential)
   - **Quality**: Authoritative, well-written information
   - **Location Match**: Exact location match preferred

2. Select TOP 10 as RELEVANT
3. Downgrade remaining to SOMEWHAT_RELEVANT

Be objective and justify your rankings clearly."""

    def relabel_documents(self, current_labels: Dict[str, List[LabelingDecision]], 
                         review: LabelReviewDecision, query: str, location: str) -> Dict[str, List[LabelingDecision]]:
        """
        Relabel documents based on review feedback with TOP 10 enforcement
        """
        self.logger.log(self.name, 
            f"Relabeling based on review: {review.feedback[:100]}...")
        
        # Get relevant documents count
        relevant_docs = current_labels.get("relevant", [])
        relevant_count = len(relevant_docs)
        
        self.logger.log(self.name, f"Current RELEVANT count: {relevant_count}")
        
        # Check if this is a "too many relevant" issue
        if relevant_count > 10:
            self.logger.log(self.name, 
                f"⚠️ RELEVANT OVERFLOW DETECTED: {relevant_count} > 10")
            self.logger.log(self.name, "Initiating TOP 10 selection process...")
            
            return self._select_top_10_relevant(current_labels, query, location, relevant_docs)
        else:
            self.logger.log(self.name, "No overflow issue, returning current labels")
            return current_labels
    
    def _select_top_10_relevant(self, current_labels: Dict[str, List[LabelingDecision]], 
                               query: str, location: str, 
                               relevant_docs: List[LabelingDecision]) -> Dict[str, List[LabelingDecision]]:
        """
        Select TOP 10 RELEVANT documents and downgrade the rest
        """
        self.logger.log(self.name, f"Selecting TOP 10 from {len(relevant_docs)} RELEVANT documents")
        
        # Prepare data for ranking
        docs_for_ranking = []
        for idx, decision in enumerate(relevant_docs, 1):
            docs_for_ranking.append({
                "doc_id": decision.doc_id,
                "current_reason": decision.reason,
                "confidence": decision.confidence,
                "index": idx
            })
        
        user_prompt = f"""URGENT TASK: Select TOP 10 RELEVANT documents from {len(relevant_docs)} candidates.

Query: "{query}"
User Location: "{location}"

Current RELEVANT documents (MUST select only TOP 10):
{json.dumps(docs_for_ranking, indent=2)}

RANKING CRITERIA (Priority Order):
1. **Recency** - Newer is better (2024-2025 highest priority)
2. **Completeness** - Comprehensive coverage of "{query}"
3. **Direct Query Match** - Directly answers "{query}" (not tangential)
4. **Information Quality** - Detailed, authoritative content
5. **Location Match** - Matches "{location}" exactly

TASK: Carefully analyze the reasons and select the BEST 10 documents.

Respond in JSON format:
{{
    "top_10_relevant": [
        {{
            "doc_id": "id1",
            "rank": 1,
            "selection_reason": "Why this is #1: [mention recency, completeness, direct match]"
        }},
        {{
            "doc_id": "id2",
            "rank": 2,
            "selection_reason": "Why this is #2: [specific reasons]"
        }},
        ... (exactly 10 documents)
    ],
    "downgraded_to_somewhat": [
        {{
            "doc_id": "id11",
            "downgrade_reason": "Why NOT in top 10: [e.g., older, less comprehensive, tangential]"
        }},
        ... (remaining {len(relevant_docs) - 10} documents)
    ],
    "ranking_methodology": "Brief explanation of your overall selection criteria"
}}

CRITICAL: You MUST return exactly 10 documents in top_10_relevant and {len(relevant_docs) - 10} in downgraded_to_somewhat."""

        try:
            self.logger.log(self.name, "Calling LLM for TOP 10 selection...")
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
            
            # FALLBACK: Keep first 10 by confidence, downgrade rest
            self.logger.log(self.name, "Using fallback: keeping first 10 by confidence")
            
            # Sort by confidence (high > medium > low)
            confidence_order = {"high": 3, "medium": 2, "low": 1}
            sorted_docs = sorted(relevant_docs, 
                               key=lambda x: confidence_order.get(x.confidence, 0), 
                               reverse=True)
            
            new_labels = {
                "relevant": [],
                "somewhat_relevant": list(current_labels.get("somewhat_relevant", [])),
                "acceptable": list(current_labels.get("acceptable", [])),
                "not_sure": list(current_labels.get("not_sure", []))
            }
            
            # Keep top 10
            for i, doc in enumerate(sorted_docs[:10], 1):
                new_decision = LabelingDecision(
                    doc_id=doc.doc_id,
                    label="relevant",
                    reason=f"[TOP 10 by confidence - #{i}] {doc.reason}",
                    confidence=doc.confidence,
                    agent_name=self.name
                )
                new_labels["relevant"].append(new_decision)
            
            # Downgrade rest
            for doc in sorted_docs[10:]:
                new_decision = LabelingDecision(
                    doc_id=doc.doc_id,
                    label="somewhat_relevant",
                    reason=f"[DOWNGRADED - not in top 10 by confidence] {doc.reason}",
                    confidence="medium",
                    agent_name=self.name
                )
                new_labels["somewhat_relevant"].append(new_decision)
            
            self.logger.log(self.name, 
                f"Fallback complete: {len(new_labels['relevant'])} RELEVANT")
            
            return new_labels
