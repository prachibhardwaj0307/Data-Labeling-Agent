"""
Superior Agent - Orchestrates the entire document labeling workflow with complete tracking
"""
from typing import Dict, List, Any
from models.data_models import Document, ProcessingStats
from utils.helpers import Logger
import config

class SuperiorAgent:
    """
    Master agent that coordinates all other agents in the labeling workflow
    Implements complete workflow with full tracking for UI display
    """
    
    def __init__(self, filter_agent, grouping_agent, group_review_agent, 
                 labeling_agent, label_review_agent, regroup_agent, relabel_agent):
        self.name = "SuperiorAgent"
        self.logger = Logger()
        
        # Initialize all sub-agents
        self.filter_agent = filter_agent
        self.grouping_agent = grouping_agent
        self.group_review_agent = group_review_agent
        self.labeling_agent = labeling_agent
        self.label_review_agent = label_review_agent
        self.regroup_agent = regroup_agent
        self.relabel_agent = relabel_agent
        
        # Initialize statistics tracker
        self.stats = ProcessingStats()
        
        # Store filtered documents info
        self.removed_docs_info = []
        
        # Workflow tracking for UI display
        self.workflow_steps = []
    
    def _add_workflow_step(self, step_name, agent_name, details):
        """Track workflow step for UI display"""
        step = {
            "step_name": step_name,
            "agent_name": agent_name,
            "details": details
        }
        self.workflow_steps.append(step)
        print(f"[WORKFLOW] Added step: {step_name}")
    
    def process_documents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main workflow coordinator - orchestrates all agents through the complete pipeline
        """
        self.logger.log(self.name, "=" * 80)
        self.logger.log(self.name, "🚀 STARTING DOCUMENT LABELING WORKFLOW")
        self.logger.log(self.name, "=" * 80)
        
        # Extract input data
        query = data.get("text", "")
        location = data.get("location", "")
        items = data.get("items", [])
        existing_annotations = data.get("annotations", [{}])[0].get("result", [{}])[0].get("value", {}).get("ranker", {})
        
        self.logger.log(self.name, f"📋 Query: '{query}'")
        self.logger.log(self.name, f"📍 Location: '{location or 'Not specified'}'")
        self.logger.log(self.name, f"📄 Total documents: {len(items)}")
        
        # Create Document objects
        documents = []
        for item in items:
            doc = Document(
                id=item.get("id", ""),
                title=item.get("title", ""),
                html=item.get("html", ""),
                current_label="New Doc"
            )
            documents.append(doc)
        
        self.stats.total_documents = len(documents)
        
        # Create document map for later use
        doc_map = {doc.id: doc for doc in documents}
        
        # Learn from existing labeled documents
        self._learn_from_existing_labels(documents, existing_annotations)
        
        # Get documents that need labeling
        new_docs = [doc for doc in documents if doc.current_label == "New Doc"]
        self.logger.log(self.name, f"🆕 New documents to label: {len(new_docs)}")
        
        if not new_docs:
            self.logger.log(self.name, "✓ No new documents to label. Ending workflow.")
            return self._generate_output(documents, {}, query, location)
        
        # ============================================================
        # STEP 1: FILTERING
        # ============================================================
        self.logger.log(self.name, "\n" + "="*80)
        self.logger.log(self.name, "🔍 STEP 1: FILTERING DOCUMENTS")
        self.logger.log(self.name, "="*80)
        
        filtered_docs, removed_docs, filter_reasons = self.filter_agent.filter_documents(
            new_docs, query, location
        )
        self.stats.filtered_documents = len(removed_docs)
        
        # STORE REMOVED DOCS WITH REASONS
        self.removed_docs_info = []
        for doc in removed_docs:
            self.removed_docs_info.append({
                "doc_id": doc.id,
                "title": doc.title,
                "reason": filter_reasons.get(doc.id, "No reason provided"),
                "confidence": "high",
                "labeled_by": "FilterAgent"
            })
        
        # TRACK FILTERING STEP
        self._add_workflow_step(
            "Filtering",
            "FilterAgent",
            {
                "total_docs": len(new_docs),
                "kept": len(filtered_docs),
                "filtered": len(removed_docs),
                "filtered_docs": self.removed_docs_info
            }
        )
        
        self.logger.log(self.name, 
            f"✓ Filtering complete: {len(filtered_docs)} kept, {len(removed_docs)} removed")
        
        if not filtered_docs:
            self.logger.log(self.name, 
                "⚠️ No documents passed filtering. Ending workflow.")
            return self._generate_output(documents, {}, query, location)
        
        # ============================================================
        # STEP 2: GROUPING + REVIEW LOOP
        # ============================================================
        self.logger.log(self.name, "\n" + "="*80)
        self.logger.log(self.name, "📦 STEP 2: GROUPING DOCUMENTS")
        self.logger.log(self.name, "="*80)
        
        groups = self.grouping_agent.group_documents(filtered_docs, query)
        
        # TRACK INITIAL GROUPING
        groups_info = self._get_groups_info(groups)
        self._add_workflow_step(
            "Grouping",
            "GroupingAgent",
            {
                "groups_created": len(groups),
                "groups": groups_info
            }
        )
        
        group_attempt = 1
        
        # Group review loop (max 3 attempts)
        while group_attempt <= config.MAX_GROUP_REVIEW_ATTEMPTS:
            self.logger.log(self.name, 
                f"\n--- 🔎 Group Review Attempt {group_attempt}/{config.MAX_GROUP_REVIEW_ATTEMPTS} ---")
            
            review = self.group_review_agent.review_groups(groups, group_attempt)
            self.stats.group_review_attempts = group_attempt
            
            # TRACK GROUP REVIEW
            self._add_workflow_step(
                f"Group Review Attempt {group_attempt}",
                "GroupReviewAgent",
                {
                    "approved": review.approved,
                    "feedback": review.feedback,
                    "attempt": group_attempt,
                    "groups_reviewed": groups_info
                }
            )
            
            if review.approved:
                self.logger.log(self.name, 
                    f"✅ Groups APPROVED on attempt {group_attempt}")
                self.logger.log(self.name, f"Feedback: {review.feedback}")
                break
            else:
                self.logger.log(self.name, 
                    f"❌ Groups REJECTED on attempt {group_attempt}")
                self.logger.log(self.name, f"Feedback: {review.feedback}")
                
                if group_attempt < config.MAX_GROUP_REVIEW_ATTEMPTS:
                    self.logger.log(self.name, "🔄 Initiating regrouping...")
                    groups = self.regroup_agent.regroup_documents(groups, review)
                    
                    # TRACK REGROUPING
                    groups_info = self._get_groups_info(groups)
                    self._add_workflow_step(
                        f"Regrouping Attempt {group_attempt}",
                        "RegroupAgent",
                        {
                            "groups_created": len(groups),
                            "groups": groups_info,
                            "changes_made": f"Regrouped based on feedback: {review.feedback}"
                        }
                    )
                else:
                    self.logger.log(self.name, 
                        "⚠️ Max attempts reached. Proceeding with current groups.")
                
                group_attempt += 1
        
        # ============================================================
        # STEP 3: LABELING + REVIEW LOOP
        # ============================================================
        self.logger.log(self.name, "\n" + "="*80)
        self.logger.log(self.name, "🏷️  STEP 3: LABELING DOCUMENTS")
        self.logger.log(self.name, "="*80)
        
        labeling_results = self.labeling_agent.label_documents(
            groups, query, location
        )
        
        # TRACK LABELING
        groups_with_labels = self._get_groups_with_labels(groups, labeling_results)
        self._add_workflow_step(
            "Labeling",
            "LabelingAgent",
            {
                "labels_assigned": {
                    "relevant": len(labeling_results.get("relevant", [])),
                    "somewhat_relevant": len(labeling_results.get("somewhat_relevant", [])),
                    "acceptable": len(labeling_results.get("acceptable", [])),
                    "not_sure": len(labeling_results.get("not_sure", []))
                },
                "groups_labeled": groups_with_labels
            }
        )
        
        label_attempt = 1
        
        # Label review loop (max 3 attempts)
        while label_attempt <= config.MAX_LABEL_REVIEW_ATTEMPTS:
            self.logger.log(self.name, 
                f"\n--- 🔎 Label Review Attempt {label_attempt}/{config.MAX_LABEL_REVIEW_ATTEMPTS} ---")
            
            review = self.label_review_agent.review_labels(
                labeling_results, label_attempt
            )
            self.stats.label_review_attempts = label_attempt
            
            # TRACK LABEL REVIEW
            self._add_workflow_step(
                f"Label Review Attempt {label_attempt}",
                "LabelReviewAgent",
                {
                    "approved": review.approved,
                    "feedback": review.feedback,
                    "rejected_docs": review.rejected_docs,
                    "attempt": label_attempt
                }
            )
            
            if review.approved:
                self.logger.log(self.name, 
                    f"✅ Labels APPROVED on attempt {label_attempt}")
                self.logger.log(self.name, f"Feedback: {review.feedback}")
                break
            else:
                self.logger.log(self.name, 
                    f"❌ Labels REJECTED on attempt {label_attempt}")
                self.logger.log(self.name, f"Feedback: {review.feedback}")
                self.logger.log(self.name, 
                    f"Problematic documents: {len(review.rejected_docs)}")
                
                if label_attempt < config.MAX_LABEL_REVIEW_ATTEMPTS:
                    self.logger.log(self.name, "🔄 Initiating relabeling...")
                    
                    # CAPTURE OLD LABELS BEFORE RELABELING
                    old_labels = {}
                    for label_type, decisions in labeling_results.items():
                        for decision in decisions:
                            if decision.doc_id in review.rejected_docs:
                                old_labels[decision.doc_id] = {
                                    "old_label": label_type,
                                    "title": doc_map.get(decision.doc_id).title if decision.doc_id in doc_map else "Unknown",
                                    "old_reason": decision.reason,
                                    "old_confidence": decision.confidence
                                }
                    
                    # RELABEL
                    labeling_results = self.relabel_agent.relabel_documents(
                        labeling_results, review, query, location
                    )
                    
                    # CAPTURE NEW LABELS AFTER RELABELING
                    new_labels = {}
                    for label_type, decisions in labeling_results.items():
                        for decision in decisions:
                            if decision.doc_id in review.rejected_docs:
                                new_labels[decision.doc_id] = {
                                    "new_label": label_type,
                                    "new_reason": decision.reason,
                                    "confidence": decision.confidence
                                }
                    
                    # MERGE OLD AND NEW FOR DISPLAY
                    relabeling_details = []
                    for doc_id in review.rejected_docs:
                        old_info = old_labels.get(doc_id, {})
                        new_info = new_labels.get(doc_id, {})
                        
                        relabeling_details.append({
                            "doc_id": doc_id,
                            "title": old_info.get("title", "Unknown"),
                            "old_label": old_info.get("old_label", "unknown"),
                            "new_label": new_info.get("new_label", "unknown"),
                            "old_reason": old_info.get("old_reason", "N/A"),
                            "new_reason": new_info.get("new_reason", "N/A"),
                            "confidence": new_info.get("confidence", "low")
                        })
                    
                    # TRACK RELABELING WITH DETAILS
                    self._add_workflow_step(
                        f"Relabeling Attempt {label_attempt}",
                        "RelabelAgent",
                        {
                            "relabeled_count": len(review.rejected_docs),
                            "relabeled_docs": review.rejected_docs,
                            "relabeling_details": relabeling_details
                        }
                    )
                else:
                    self.logger.log(self.name, 
                        "⚠️ Max attempts reached. Proceeding with current labels.")
                
                label_attempt += 1
        
        # ============================================================
        # STEP 4: FINALIZE AND GENERATE OUTPUT
        # ============================================================
        self.logger.log(self.name, "\n" + "="*80)
        self.logger.log(self.name, "📊 FINALIZING RESULTS")
        self.logger.log(self.name, "="*80)
        
        # Update statistics
        for label, decisions in labeling_results.items():
            count = len(decisions)
            if label == "relevant":
                self.stats.relevant_count = count
            elif label == "somewhat_relevant":
                self.stats.somewhat_relevant_count = count
            elif label == "acceptable":
                self.stats.acceptable_count = count
            elif label == "not_sure":
                self.stats.not_sure_count = count
        
        self.stats.labeled_documents = sum(
            len(d) for d in labeling_results.values()
        )
        
        # Generate final output
        output = self._generate_output(documents, labeling_results, query, location)
        
        self.logger.log(self.name, "\n" + "="*80)
        self.logger.log(self.name, "✅ WORKFLOW COMPLETE")
        self.logger.log(self.name, "="*80)
        self._print_stats()
        
        print(f"\n[DEBUG] Total workflow steps captured: {len(self.workflow_steps)}")
        
        return output
    
    def _get_groups_info(self, groups):
        """Extract group information for tracking"""
        return [{
            "name": g.name,
            "theme": g.theme,
            "document_count": len(g.documents),
            "document_titles": [d.title for d in g.documents],
            "document_ids": [d.id for d in g.documents],
            "reasoning": getattr(g, 'reasons', ['No reasoning provided'])
        } for g in groups]
    
    def _get_groups_with_labels(self, groups, labeling_results):
        """Map which groups got which labels"""
        # Create a reverse map: doc_id -> label
        doc_to_label = {}
        for label, decisions in labeling_results.items():
            for decision in decisions:
                doc_to_label[decision.doc_id] = label
        
        # Map groups to their labels
        groups_with_labels = []
        for group in groups:
            doc_ids = [d.id for d in group.documents]
            # Get the label of the first document (all in group should have same label)
            group_label = doc_to_label.get(doc_ids[0], "unknown") if doc_ids else "unknown"
            
            groups_with_labels.append({
                "group_name": group.name,
                "label": group_label,
                "document_count": len(group.documents),
                "document_titles": [d.title for d in group.documents]
            })
        
        return groups_with_labels
    
    def _learn_from_existing_labels(self, documents: List[Document], 
                                    existing_annotations: Dict):
        """Learn from existing labeled documents"""
        labeled_count = 0
        
        for label_type, doc_ids in existing_annotations.items():
            if label_type == "New Doc":
                continue
            
            for doc_id in doc_ids:
                for doc in documents:
                    if doc.id == doc_id:
                        doc.current_label = label_type
                        labeled_count += 1
        
        self.logger.log(self.name, 
            f"📚 Learned from {labeled_count} existing labeled documents")
    
    def _generate_output(self, all_documents: List[Document], 
                        labeling_results: Dict[str, List[Any]], 
                        query: str, location: str) -> Dict[str, Any]:
        """Generate final output with both updated annotations and detailed report"""
        
        # Create document ID to title mapping
        doc_map = {doc.id: doc for doc in all_documents}
        
        # Create updated annotations structure
        updated_ranker = {
            "relevant": [],
            "somewhat_relevant": [],
            "acceptable": [],
            "not_sure": [],
            "irrelevant": []
        }
        
        # Add existing labels
        for doc in all_documents:
            if doc.current_label != "New Doc":
                if doc.current_label in updated_ranker:
                    updated_ranker[doc.current_label].append(doc.id)
                else:
                    updated_ranker["not_sure"].append(doc.id)
        
        # Add new labels
        for label, decisions in labeling_results.items():
            for decision in decisions:
                if decision.doc_id not in updated_ranker[label]:
                    updated_ranker[label].append(decision.doc_id)
        
        # Create detailed report with TITLE and FILTERED DOCUMENTS
        report = {
            "query": query,
            "location": location,
            "statistics": {
                "total_documents": self.stats.total_documents,
                "existing_labeled": self.stats.total_documents - sum(
                    1 for d in all_documents if d.current_label == "New Doc"
                ),
                "newly_processed": sum(
                    1 for d in all_documents if d.current_label == "New Doc"
                ),
                "filtered_out": self.stats.filtered_documents,
                "successfully_labeled": self.stats.labeled_documents,
                "group_review_attempts": self.stats.group_review_attempts,
                "label_review_attempts": self.stats.label_review_attempts,
                "label_distribution": {
                    "relevant": self.stats.relevant_count,
                    "somewhat_relevant": self.stats.somewhat_relevant_count,
                    "acceptable": self.stats.acceptable_count,
                    "not_sure": self.stats.not_sure_count,
                    "irrelevant": len(self.removed_docs_info)
                }
            },
            "labeling_details": {},
            "filtered_documents": self.removed_docs_info
        }
        
        # Add detailed reasoning for each label WITH TITLE
        for label, decisions in labeling_results.items():
            report["labeling_details"][label] = [
                {
                    "doc_id": d.doc_id,
                    "title": doc_map.get(d.doc_id).title if d.doc_id in doc_map else "Unknown",
                    "reason": d.reason,
                    "confidence": d.confidence,
                    "labeled_by": d.agent_name
                }
                for d in decisions
            ]
        
        print(f"\n[DEBUG] Returning output with {len(self.workflow_steps)} workflow steps")
        
        return {
            "updated_annotations": updated_ranker,
            "detailed_report": report,
            "workflow_steps": self.workflow_steps  # Include workflow tracking
        }
    
    def _print_stats(self):
        """Print workflow statistics"""
        self.logger.log(self.name, "\n📊 Workflow Statistics:")
        self.logger.log(self.name, f"  Total documents: {self.stats.total_documents}")
        self.logger.log(self.name, f"  Filtered out: {self.stats.filtered_documents}")
        self.logger.log(self.name, f"  Successfully labeled: {self.stats.labeled_documents}")
        self.logger.log(self.name, f"\n🏷️  Label Distribution:")
        self.logger.log(self.name, f"  ✓ Relevant: {self.stats.relevant_count}")
        self.logger.log(self.name, f"  ~ Somewhat Relevant: {self.stats.somewhat_relevant_count}")
        self.logger.log(self.name, f"  ≈ Acceptable: {self.stats.acceptable_count}")
        self.logger.log(self.name, f"  ? Not Sure: {self.stats.not_sure_count}")
        self.logger.log(self.name, f"  ✗ Irrelevant (Filtered): {len(self.removed_docs_info)}")
        self.logger.log(self.name, f"\n🔄 Review Cycles:")
        self.logger.log(self.name, f"  Group reviews: {self.stats.group_review_attempts}")
        self.logger.log(self.name, f"  Label reviews: {self.stats.label_review_attempts}")
