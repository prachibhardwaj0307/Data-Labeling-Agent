"""
Grouping Agent - Groups documents by semantic similarity using sentence transformers and clustering.
"""
from typing import List, Dict, Tuple
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from models.data_models import Document, DocumentGroup
from utils.helpers import Logger, extract_text_from_html
from utils.llm_client import LLMClient

class GroupingAgent:
    """
    Agent responsible for grouping similar documents based on semantic content.
    """
    
    def __init__(self):
        self.name = "GroupingAgent"
        self.logger = Logger()
        self.llm = LLMClient()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.system_prompt = """You are a Document Group Naming Agent. Your task is to analyze a group of documents and create a concise, descriptive name, theme, and reason for the group."""

    def group_documents(self, documents: List[Document], query: str) -> List[DocumentGroup]:
        """
        Group documents by semantic similarity using sentence embeddings and clustering.
        """
        self.logger.log(self.name, f"Grouping {len(documents)} documents by semantic similarity.")
        
        if not documents:
            return []

        # Generate embeddings for document content
        doc_contents = [doc.title + " " + extract_text_from_html(doc.html) for doc in documents]
        embeddings = self.model.encode(doc_contents, convert_to_tensor=False)

        # Determine the optimal number of clusters
        num_docs = len(documents)
        if num_docs < 3:
            num_clusters = 1
        else:
            # Heuristic for determining the number of clusters
            num_clusters = int(np.sqrt(num_docs)) + 1
            if num_clusters < 2:
                num_clusters = 2
            if num_clusters > 10:
                num_clusters = 10

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)

        # Create document groups based on clusters
        doc_groups: Dict[int, List[Document]] = {i: [] for i in range(num_clusters)}
        for i, doc in enumerate(documents):
            doc_groups[clusters[i]].append(doc)

        # Create DocumentGroup objects
        groups = []
        for cluster_id, docs_in_cluster in doc_groups.items():
            if not docs_in_cluster:
                continue

            group_name, group_theme, group_reason = self._get_group_details(docs_in_cluster, query)
            
            group = DocumentGroup(
                name=group_name,
                documents=docs_in_cluster,
                theme=group_theme,
                reasons=[group_reason]
            )
            groups.append(group)
            self.logger.log(self.name, f"Created group '{group.name}' with {len(docs_in_cluster)} docs.")

        self.logger.log(self.name, f"Created {len(groups)} groups.")
        return groups

    def _get_group_details(self, documents: List[Document], query: str) -> Tuple[str, str, str]:
        """Generate a name, theme, and reason for a group of documents using an LLM."""
        doc_previews = []
        for doc in documents:
            doc_previews.append({
                "title": doc.title,
                "content_preview": extract_text_from_html(doc.html)[:200]
            })

        user_prompt = f"""Given the following documents and the original query, generate a concise and descriptive name, theme, and reason for the group.

Original Query: "{query}"

Documents:
{json.dumps(doc_previews, indent=2)}

Respond in JSON format:
{{
    "group_name": "A concise name for the group (e.g., 'Indian Employee Handbooks 2025')",
    "theme": "A one-sentence theme that summarizes the content of the group.",
    "reason": "A brief explanation of why these documents are grouped together."
}}
"""
        try:
            response = self.llm.call_with_json_response(self.system_prompt, user_prompt)
            return response.get("group_name", "Unnamed Group"), response.get("theme", "No theme provided"), response.get("reason", "No reason provided")
        except Exception as e:
            self.logger.log(self.name, f"LLM group naming failed: {e}", "ERROR")
            return "Unnamed Group", "Could not generate theme due to an error.", "Could not generate reason due to an error."

    def _extract_year(self, title: str, content: str) -> str:
        """Extract year from title or content"""
        text = f"{title} {content}"
        
        # Look for 4-digit years (2020-2030)
        years = re.findall(r'\b(202[0-9]|203[0])\b', text)
        
        if years:
            # Return most recent year found
            return max(years)
        
        return "Unknown"