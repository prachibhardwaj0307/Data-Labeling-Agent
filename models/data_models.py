"""
Data models for the document labeling system
"""
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
import re


class LabelType(Enum):
    """Document label types"""
    RELEVANT = "relevant"
    SOMEWHAT_RELEVANT = "somewhat_relevant"
    SEMANTICALLY_ACCEPTABLE = "acceptable"
    NOT_SURE = "not_sure"
    IRRELEVANT = "irrelevant"
    NEW_DOC = "New Doc"


@dataclass
class Document:
    """
    Represents a document to be labeled
    """
    id: str
    title: str
    html: str
    current_label: str = "New Doc"
    
    def has_valid_content(self) -> bool:
        """
        Check if document has valid title and content
        
        Returns:
            True if document is valid
        """
        return bool(
            self.title and 
            self.title.strip() and 
            self.title.lower() not in ['no title', 'untitled', '', 'holiday balance'] and
            self.html and 
            self.html.strip() and
            len(self.html.strip()) > 10  # At least some content
        )
    
    def extract_text_content(self) -> str:
        """
        Extract text from HTML
        
        Returns:
            Plain text content
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', self.html)
        # Remove multiple whitespaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def has_link(self) -> bool:
        """
        Check if document has a valid link
        
        Returns:
            True if document contains href
        """
        return bool(self.html and 'href=' in self.html)
    
    def get_year(self) -> Optional[int]:
        """
        Extract year from title or content
        
        Returns:
            Year as integer or None
        """
        text = f"{self.title} {self.extract_text_content()}"
        years = re.findall(r'20[2-3][0-9]', text)
        if years:
            return max(int(year) for year in years)
        return None
    
    def __repr__(self) -> str:
        """String representation"""
        return f"Document(id='{self.id}', title='{self.title[:50]}...', label='{self.current_label}')"


@dataclass
class LabelingDecision:
    """
    Represents a labeling decision for a document
    """
    doc_id: str
    label: str
    reason: str
    confidence: str  # "high", "medium", "low"
    agent_name: str = ""
    
    def __post_init__(self):
        """Validate fields after initialization"""
        # Validate label
        valid_labels = ["relevant", "somewhat_relevant", "acceptable", "not_sure", "irrelevant"]
        if self.label not in valid_labels:
            raise ValueError(f"Invalid label: {self.label}. Must be one of {valid_labels}")
        
        # Validate confidence
        valid_confidence = ["high", "medium", "low"]
        if self.confidence not in valid_confidence:
            raise ValueError(f"Invalid confidence: {self.confidence}. Must be one of {valid_confidence}")
    
    def __repr__(self) -> str:
        """String representation"""
        return f"LabelingDecision(doc='{self.doc_id}', label='{self.label}', conf='{self.confidence}')"


@dataclass
class GroupReviewDecision:
    """
    Review decision for a document group
    """
    approved: bool
    feedback: str
    attempt_number: int
    reviewer: str = "GroupReviewAgent"
    issues: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        """String representation"""
        status = "APPROVED" if self.approved else "REJECTED"
        return f"GroupReviewDecision({status}, attempt={self.attempt_number})"


@dataclass
class LabelReviewDecision:
    """
    Review decision for labeled documents
    """
    approved: bool
    feedback: str
    attempt_number: int
    reviewer: str = "LabelReviewAgent"
    rejected_docs: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        """String representation"""
        status = "APPROVED" if self.approved else "REJECTED"
        return f"LabelReviewDecision({status}, attempt={self.attempt_number}, rejected={len(self.rejected_docs)})"


@dataclass
class DocumentGroup:
    """
    Represents a group of similar documents
    """
    name: str
    documents: List[Document]
    theme: str
    reasons: List[str] = field(default_factory=list)
    attempt: int = 1
    
    def __post_init__(self):
            """Validate after initialization"""
            if not self.documents:
                raise ValueError("DocumentGroup must contain at least one document")
            if not self.name:
                raise ValueError("DocumentGroup must have a name")
            
            # Log if single-document group
            if len(self.documents) == 1:
                # Single-document groups are now allowed
                pass
    
    def get_document_ids(self) -> List[str]:
        """
        Get list of document IDs in this group
        
        Returns:
            List of document IDs
        """
        return [doc.id for doc in self.documents]
    
    def get_document_titles(self) -> List[str]:
        """
        Get list of document titles in this group
        
        Returns:
            List of document titles
        """
        return [doc.title for doc in self.documents]
    
    def size(self) -> int:
        """
        Get number of documents in group
        
        Returns:
            Number of documents
        """
        return len(self.documents)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"DocumentGroup(name='{self.name}', docs={len(self.documents)}, attempt={self.attempt})"


@dataclass
class ProcessingStats:
    """
    Statistics for tracking the labeling process
    """
    total_documents: int = 0
    filtered_documents: int = 0
    labeled_documents: int = 0
    group_review_attempts: int = 0
    label_review_attempts: int = 0
    relevant_count: int = 0
    somewhat_relevant_count: int = 0
    acceptable_count: int = 0
    not_sure_count: int = 0
    irrelevant_count: int = 0
    
    def get_label_distribution(self) -> Dict[str, int]:
        """
        Get distribution of labels
        
        Returns:
            Dictionary with label counts
        """
        return {
            "relevant": self.relevant_count,
            "somewhat_relevant": self.somewhat_relevant_count,
            "acceptable": self.acceptable_count,
            "not_sure": self.not_sure_count,
            "irrelevant": self.irrelevant_count
        }
    
    def get_success_rate(self) -> float:
        """
        Calculate labeling success rate
        
        Returns:
            Success rate as percentage
        """
        if self.total_documents == 0:
            return 0.0
        return (self.labeled_documents / self.total_documents) * 100
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"ProcessingStats(total={self.total_documents}, "
            f"labeled={self.labeled_documents}, "
            f"filtered={self.filtered_documents})"
        )


@dataclass
class WorkflowConfig:
    """
    Configuration for the workflow
    """
    max_group_review_attempts: int = 3
    max_label_review_attempts: int = 3
    min_group_size: int = 2
    max_group_size: int = 10
    enable_filtering: bool = True
    enable_grouping: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if self.max_group_review_attempts < 1:
            raise ValueError("max_group_review_attempts must be >= 1")
        if self.max_label_review_attempts < 1:
            raise ValueError("max_label_review_attempts must be >= 1")
        if self.min_group_size < 1:
            raise ValueError("min_group_size must be >= 1")
        if self.max_group_size < self.min_group_size:
            raise ValueError("max_group_size must be >= min_group_size")
