"""
Superior Agent - Orchestrates workflow with learning from existing labels
Processes ONLY "New Doc" documents, uses others as reference examples WITH FULL CONTENT
"""
from typing import Dict, List, Any
from models.data_models import Document, ProcessingStats
from utils.helpers import Logger, extract_text_from_html
import config

class SuperiorAgent:
    """Master agent that coordinates all other agents with example-based learning"""
    
    def __init__(self, filter_agent, grouping_agent, group_review_agent, 
                 labeling_agent, label_review_agent, regroup_agent, relabel_agent):
        self.name = "SuperiorAgent"
        self.logger = Logger()
        
        self.filter_agent = filter_agent
        self.grouping_agent = grouping_agent
        self.group_review_agent = group_review_agent
        self.labeling_agent = labeling_agent
        self.label_review_agent = label_review_agent
        self.regroup_agent = regroup_agent
        self.relabel_agent = relabel_agent
        
        self.stats = ProcessingStats()
        self.removed_docs_info = []
        self.workflow_steps = []
        self.label_examples = {}
    
    def _add_workflow_step(self, step_name, agent_name, details):
        """Track workflow step"""
        step = {
            "step_name": step_name,
            "agent_name": agent_name,
            "details": details
        }
        self.workflow_steps.append(step)
    
    def process_documents(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main workflow - processes ONLY 'New Doc' documents"""
        
        self.logger.log(self.name, "="*80)
        self.logger.log(self.name, "🚀 STARTING WORKFLOW WITH EXAMPLE LEARNING")
        self.logger.log(self.name, "="*80)
        
        # Determine data structure and extract accordingly
        if "data" in data:
            # Full structure with id, data, annotations
            query = data.get("data", {}).get("text", "")
            location = data.get("data", {}).get("location", "")
            items = data.get("data", {}).get("items", [])
            annotations_list = data.get("annotations", [])
        else:
            # Just the data object
            query = data.get("text", "")
            location = data.get("location", "")
            items = data.get("items", [])
            annotations_list = data.get("annotations", [])
        
        # Extract existing annotations
        existing_annotations = {}
        
        try:
            if annotations_list and isinstance(annotations_list, list) and len(annotations_list) > 0:
                first_annotation = annotations_list[0]
                
                if "result" in first_annotation:
                    result_list = first_annotation["result"]
                    
                    if result_list and isinstance(result_list, list) and len(result_list) > 0:
                        first_result = result_list[0]
                        
                        if "value" in first_result and "ranker" in first_result["value"]:
                            existing_annotations = first_result["value"]["ranker"]
                            self.logger.log(self.name, 
                                f"✓ Extracted annotations: {list(existing_annotations.keys())}")
                            self.logger.log(self.name, 
                                f"✓ Counts: " + 
                                ", ".join([f"{k}: {len(v)}" for k, v in existing_annotations.items()]))
            
            if not existing_annotations:
                self.logger.log(self.name, 
                    "⚠️ No existing annotations - treating all as 'New Doc'")
        
        except Exception as e:
            self.logger.log(self.name, f"⚠️ Error extracting annotations: {e}", "WARNING")
            existing_annotations = {}
        
        self.logger.log(self.name, 
            f"Query: '{query}', Location: '{location}', Total Docs: {len(items)}")
        
        # Create ALL documents with default "New Doc" label
        all_documents = [Document(
            id=item.get("id", ""),
            title=item.get("title", ""),
            html=item.get("html", ""),
            current_label="New Doc"
        ) for item in items]
        
        self.stats.total_documents = len(all_documents)
        doc_map = {doc.id: doc for doc in all_documents}
        
        # Learn from existing labels WITH FULL CONTENT
        self._learn_from_existing_labels(all_documents, existing_annotations)
        
        # Get ONLY "New Doc" documents
        new_docs_to_label = [doc for doc in all_documents if doc.current_label == "New Doc"]
        
        self.logger.log(self.name, 
            f"📚 Reference examples: {sum(len(v) for v in self.label_examples.values())}")
        self.logger.log(self.name, 
            f"🆕 New documents to label: {len(new_docs_to_label)}")
        self.logger.log(self.name, 
            f"✅ Already labeled (preserved): {len(all_documents) - len(new_docs_to_label)}")
        
        if not new_docs_to_label:
            self.logger.log(self.name, 
                "✓ No new documents to label")
            return self._generate_output(all_documents, {}, query, location)
        
        # FILTERING
        self.logger.log(self.name, "\n🔍 STEP 1: FILTERING")
        
        filtered_docs, removed_docs, filter_reasons = self.filter_agent.filter_documents(
            new_docs_to_label, query, location
        )
        
        self.stats.filtered_documents = len(removed_docs)
        
        for doc in removed_docs:
            doc.current_label = "irrelevant"
        
        self.removed_docs_info = [{
            "doc_id": doc.id,
            "title": doc.title,
            "reason": filter_reasons.get(doc.id, "No reason"),
            "confidence": "high",
            "labeled_by": "FilterAgent"
        } for doc in removed_docs]
        
        self._add_workflow_step("Filtering", "FilterAgent", {
            "total_new_docs": len(new_docs_to_label),
            "kept": len(filtered_docs),
            "filtered": len(removed_docs),
            "filtered_docs": self.removed_docs_info
        })
        
        if not filtered_docs:
            self.logger.log(self.name, "All new documents filtered out")
            return self._generate_output(all_documents, {}, query, location)
        
        # GROUPING
        self.logger.log(self.name, "\n📦 STEP 2: GROUPING BY TOPIC AND YEAR")
        
        groups = self.grouping_agent.group_documents(filtered_docs, query)
        
        groups_info = [{
            "name": g.name,
            "theme": g.theme,
            "document_count": len(g.documents),
            "document_titles": [d.title for d in g.documents],
            "document_ids": [d.id for d in g.documents],
            "reasoning": g.theme
        } for g in groups]
        
        self._add_workflow_step("Grouping", "GroupingAgent", {
            "groups_created": len(groups),
            "groups": groups_info
        })
        
        # GROUP REVIEW LOOP
        group_attempt = 1
        while group_attempt <= config.MAX_GROUP_REVIEW_ATTEMPTS:
            self.logger.log(self.name, f"\n🔎 Group Review Attempt {group_attempt}")
            
            review = self.group_review_agent.review_groups(groups, group_attempt)
            self.stats.group_review_attempts = group_attempt
            
            self._add_workflow_step(f"Group Review Attempt {group_attempt}", "GroupReviewAgent", {
                "approved": review.approved,
                "feedback": review.feedback,
                "attempt": group_attempt,
                "groups_reviewed": groups_info
            })
            
            if review.approved:
                self.logger.log(self.name, f"✅ APPROVED: {review.feedback}")
                break
            else:
                self.logger.log(self.name, f"❌ REJECTED: {review.feedback}")
                
                if group_attempt < config.MAX_GROUP_REVIEW_ATTEMPTS:
                    groups = self.regroup_agent.regroup_documents(groups, review)
                    
                    groups_info = [{
                        "name": g.name,
                        "theme": g.theme,
                        "document_count": len(g.documents),
                        "document_titles": [d.title for d in g.documents],
                        "document_ids": [d.id for d in g.documents],
                        "reasoning": g.theme
                    } for g in groups]
                    
                    self._add_workflow_step(f"Regrouping Attempt {group_attempt}", "RegroupAgent", {
                        "groups_created": len(groups),
                        "groups": groups_info,
                        "changes_made": review.feedback
                    })
                
                group_attempt += 1
        
        # LABELING WITH YEAR-BASED PRIORITIZATION AND RICH EXAMPLES
        self.logger.log(self.name, "\n🏷️ STEP 3: LABELING WITH RICH EXAMPLES")
        
        labeling_results = self.labeling_agent.label_documents(
            groups, query, location, label_examples=self.label_examples
        )
        
        groups_with_labels = self._get_groups_with_labels(groups, labeling_results)
        self._add_workflow_step("Labeling", "LabelingAgent", {
            "labels_assigned": {
                "relevant": len(labeling_results.get("relevant", [])),
                "somewhat_relevant": len(labeling_results.get("somewhat_relevant", [])),
                "acceptable": len(labeling_results.get("acceptable", [])),
                "not_sure": len(labeling_results.get("not_sure", []))
            },
            "groups_labeled": groups_with_labels,
            "examples_used": {
                "relevant": len(self.label_examples.get("relevant", [])),
                "somewhat_relevant": len(self.label_examples.get("somewhat_relevant", [])),
                "acceptable": len(self.label_examples.get("acceptable", []))
            }
        })
        
        # LABEL REVIEW LOOP
        label_attempt = 1
        while label_attempt <= config.MAX_LABEL_REVIEW_ATTEMPTS:
            self.logger.log(self.name, f"\n🔎 Label Review Attempt {label_attempt}")
            
            review = self.label_review_agent.review_labels(labeling_results, label_attempt)
            self.stats.label_review_attempts = label_attempt
            
            self._add_workflow_step(f"Label Review Attempt {label_attempt}", "LabelReviewAgent", {
                "approved": review.approved,
                "feedback": review.feedback,
                "rejected_docs": review.rejected_docs,
                "attempt": label_attempt
            })
            
            if review.approved:
                self.logger.log(self.name, f"✅ APPROVED: {review.feedback}")
                break
            else:
                self.logger.log(self.name, f"❌ REJECTED: {review.feedback}")
                
                if label_attempt < config.MAX_LABEL_REVIEW_ATTEMPTS:
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
                    
                    labeling_results = self.relabel_agent.relabel_documents(
                        labeling_results, review, query, location, 
                        label_examples=self.label_examples
                    )
                    
                    new_labels = {}
                    for label_type, decisions in labeling_results.items():
                        for decision in decisions:
                            if decision.doc_id in review.rejected_docs:
                                new_labels[decision.doc_id] = {
                                    "new_label": label_type,
                                    "new_reason": decision.reason,
                                    "confidence": decision.confidence
                                }
                    
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
                    
                    self._add_workflow_step(f"Relabeling Attempt {label_attempt}", "RelabelAgent", {
                        "relabeled_count": len(review.rejected_docs),
                        "relabeled_docs": review.rejected_docs,
                        "relabeling_details": relabeling_details
                    })
                
                label_attempt += 1
        
        # FINALIZE
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
        
        self.stats.labeled_documents = sum(len(d) for d in labeling_results.values())
        
        output = self._generate_output(all_documents, labeling_results, query, location)
        
        self.logger.log(self.name, "\n✅ WORKFLOW COMPLETE")
        self._print_stats()
        
        return output
    
    def _learn_from_existing_labels(self, documents: List[Document], 
                                    existing_annotations: Dict):
        """
        Learn from existing labeled documents with FULL document details
        Mark documents with their existing labels
        """
        labeled_count = 0
        new_doc_count = 0
        
        # Store examples with FULL document info for better context
        self.label_examples = {
            "relevant": [],
            "somewhat_relevant": [],
            "acceptable": []
        }
        
        self.logger.log(self.name, f"Processing annotations with keys: {list(existing_annotations.keys())}")
        
        # Create a map for quick lookup
        doc_map = {doc.id: doc for doc in documents}
        
        # Process each label category
        for label_type, doc_ids in existing_annotations.items():
            self.logger.log(self.name, f"Processing '{label_type}': {len(doc_ids)} documents")
            
            for doc_id in doc_ids:
                if doc_id in doc_map:
                    doc = doc_map[doc_id]
                    doc.current_label = label_type
                    
                    # Count based on label type
                    if label_type == "New Doc":
                        new_doc_count += 1
                    else:
                        labeled_count += 1
                    
                    # Store FULL DOCUMENT DETAILS as examples (not just title)
                    if label_type in ["relevant", "somewhat_relevant", "acceptable"]:
                        # Extract content preview
                        content_preview = extract_text_from_html(doc.html)[:300]  # First 300 chars
                        
                        self.label_examples[label_type].append({
                            "id": doc.id,
                            "title": doc.title,
                            "content_preview": content_preview,  # ✅ Added full content
                            "label": label_type,
                            "html_snippet": doc.html[:200]  # First 200 chars of HTML
                        })
        
        self.logger.log(self.name, 
            f"📚 Learned from {labeled_count} existing labeled documents WITH FULL CONTENT")
        self.logger.log(self.name, 
            f"📝 Examples breakdown:")
        self.logger.log(self.name, 
            f"   - RELEVANT: {len(self.label_examples['relevant'])} (with content)")
        self.logger.log(self.name, 
            f"   - SOMEWHAT_RELEVANT: {len(self.label_examples['somewhat_relevant'])} (with content)")
        self.logger.log(self.name, 
            f"   - ACCEPTABLE: {len(self.label_examples['acceptable'])} (with content)")
        self.logger.log(self.name, 
            f"   - NEW DOC (to be labeled): {new_doc_count}")
    
    def _get_groups_with_labels(self, groups, labeling_results):
        """Map groups to labels"""
        doc_to_label = {}
        for label, decisions in labeling_results.items():
            for decision in decisions:
                doc_to_label[decision.doc_id] = label
        
        groups_with_labels = []
        for group in groups:
            doc_ids = [d.id for d in group.documents]
            group_label = doc_to_label.get(doc_ids[0], "unknown") if doc_ids else "unknown"
            
            groups_with_labels.append({
                "group_name": group.name,
                "label": group_label,
                "document_count": len(group.documents),
                "document_titles": [d.title for d in group.documents]
            })
        
        return groups_with_labels
    
    def _generate_output(self, all_documents, labeling_results, query, location):
        """Generate final output"""
        doc_map = {doc.id: doc for doc in all_documents}
        
        updated_ranker = {
            "relevant": [],
            "somewhat_relevant": [],
            "acceptable": [],
            "not_sure": [],
            "irrelevant": [],
            "New Doc": []
        }
        
        for doc in all_documents:
            if doc.current_label != "New Doc" and doc.current_label != "unknown":
                if doc.current_label in updated_ranker:
                    updated_ranker[doc.current_label].append(doc.id)
        
        for label, decisions in labeling_results.items():
            for decision in decisions:
                if decision.doc_id not in updated_ranker[label]:
                    updated_ranker[label].append(decision.doc_id)
        
        report = {
            "query": query,
            "location": location,
            "statistics": {
                "total_documents": self.stats.total_documents,
                "existing_labeled": self.stats.total_documents - len([d for d in all_documents if d.current_label == "New Doc"]),
                "new_documents_processed": len([d for d in all_documents if d.current_label == "New Doc"]),
                "filtered_out": self.stats.filtered_documents,
                "newly_labeled": self.stats.labeled_documents,
                "group_review_attempts": self.stats.group_review_attempts,
                "label_review_attempts": self.stats.label_review_attempts,
                "label_distribution": {
                    "relevant": len(updated_ranker["relevant"]),
                    "somewhat_relevant": len(updated_ranker["somewhat_relevant"]),
                    "acceptable": len(updated_ranker["acceptable"]),
                    "not_sure": len(updated_ranker["not_sure"]),
                    "irrelevant": len(updated_ranker["irrelevant"])
                }
            },
            "labeling_details": {},
            "filtered_documents": self.removed_docs_info
        }
        
        for label, decisions in labeling_results.items():
            report["labeling_details"][label] = [{
                "doc_id": d.doc_id,
                "title": doc_map.get(d.doc_id).title if d.doc_id in doc_map else "Unknown",
                "reason": d.reason,
                "confidence": d.confidence,
                "labeled_by": d.agent_name
            } for d in decisions]
        
        return {
            "updated_annotations": updated_ranker,
            "detailed_report": report,
            "workflow_steps": self.workflow_steps
        }
    
    def _print_stats(self):
        """Print statistics"""
        self.logger.log(self.name, f"\n📊 Statistics:")
        self.logger.log(self.name, f"  Total documents: {self.stats.total_documents}")
        self.logger.log(self.name, f"  Reference examples: {sum(len(v) for v in self.label_examples.values())}")
        self.logger.log(self.name, f"  New docs filtered: {self.stats.filtered_documents}")
        self.logger.log(self.name, f"  New docs labeled: {self.stats.labeled_documents}")
