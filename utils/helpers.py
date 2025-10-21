"""
Helper utilities for the document labeling system
"""
import re
from typing import List, Dict, Any
from collections import Counter

def extract_text_from_html(html: str, max_length: int = 5000) -> str:
    """
    Extract plain text from HTML content
    
    Args:
        html: HTML string
        max_length: Maximum length of extracted text (default 5000)
        
    Returns:
        Plain text with HTML tags removed
    """
    if not html:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', html)
    
    # Decode common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    
    # Remove multiple whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Return up to max_length characters
    return text.strip()[:max_length]


def extract_year_from_text(text: str) -> int:
    """
    Extract the most recent year from text (2020-2030 range)
    
    Args:
        text: Text to search for year
        
    Returns:
        Year as integer, or 0 if not found
    """
    # Look for years in format 20XX
    years = re.findall(r'20[2-3][0-9]', text)
    
    if years:
        # Return the maximum (most recent) year found
        return max(int(year) for year in years)
    
    return 0

def extract_date_from_text(text: str) -> str:
    """
    Extract date patterns from text
    
    Args:
        text: Text to search for dates
        
    Returns:
        First date found or empty string
    """
    # Common date patterns
    patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',  # DD/MM/YYYY or DD-MM-YYYY
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD or YYYY-MM-DD
        r'[A-Za-z]+ \d{1,2},? \d{4}',    # Month DD, YYYY
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    
    return ""

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple word overlap similarity between two texts
    Uses Jaccard similarity coefficient
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    # Convert to lowercase and split into words
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    # Remove very common words (basic stopwords)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
                'those', 'it', 'its', 'their', 'there', 'they', 'them'}
    
    words1 = words1 - stopwords
    words2 = words2 - stopwords
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

def find_common_themes(documents: List[Any]) -> List[str]:
    """
    Find common themes/keywords across documents
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of common keywords/themes
    """
    all_words = []
    
    for doc in documents:
        # Extract text from HTML
        text = extract_text_from_html(doc.html)
        
        # Extract meaningful words (longer than 3 chars, alphanumeric)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        all_words.extend(words)
    
    # Remove common stopwords
    stopwords = {'this', 'that', 'with', 'from', 'have', 'will', 'your',
                'they', 'been', 'were', 'their', 'about', 'would', 'there',
                'which', 'when', 'where', 'these', 'those', 'such', 'into',
                'through', 'during', 'before', 'after', 'above', 'below',
                'between', 'under', 'again', 'further', 'then', 'once'}
    
    filtered_words = [w for w in all_words if w not in stopwords]
    
    # Count word frequency
    word_counts = Counter(filtered_words)
    
    # Return top 10 most common words that appear in multiple documents
    common_words = [word for word, count in word_counts.most_common(15) 
                   if count >= 2]  # Must appear at least twice
    
    return common_words[:10]  # Return top 10

def format_label_output(label: str) -> str:
    """
    Format label for consistent display output
    
    Args:
        label: Label string
        
    Returns:
        Formatted label string
    """
    label_mapping = {
        "relevant": "RELEVANT",
        "somewhat_relevant": "SOMEWHAT_RELEVANT",
        "acceptable": "SEMANTICALLY_ACCEPTABLE",
        "not_sure": "NOT_SURE",
        "irrelevant": "IRRELEVANT"
    }
    
    return label_mapping.get(label.lower(), label.upper())

def extract_query_from_data(data: Dict[str, Any]) -> str:
    """
    Extract query text from data structure
    
    Args:
        data: Data dictionary
        
    Returns:
        Query string
    """
    return data.get("text", "")

def check_location_relevance(doc_text: str, location: str) -> bool:
    """
    Check if document is relevant to specified location
    
    Args:
        doc_text: Document text content
        location: Target location
        
    Returns:
        True if document matches location or is global
    """
    if not location:
        return True
    
    doc_lower = doc_text.lower()
    location_lower = location.lower()
    
    # Check if location is mentioned
    if location_lower in doc_lower:
        return True
    
    # Check for global/universal indicators
    global_indicators = ['global', 'all employees', 'everyone', 'worldwide', 
                        'all locations', 'all regions', 'company-wide']
    
    if any(indicator in doc_lower for indicator in global_indicators):
        return True
    
    return False

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length with ellipsis
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def extract_organization_from_data(data: Dict[str, Any]) -> str:
    """
    Extract organization name from data
    
    Args:
        data: Data dictionary
        
    Returns:
        Organization name
    """
    return data.get("organisation", "")

def is_valid_document(doc: Any) -> bool:
    """
    Check if document has valid structure
    
    Args:
        doc: Document object
        
    Returns:
        True if document is valid
    """
    return bool(
        doc.id and 
        doc.title and 
        doc.title.strip() and
        doc.title.lower() not in ['no title', 'untitled', ''] and
        doc.html and
        doc.html.strip()
    )


class Logger:
    """
    Simple logger for agent actions with colored output
    """
    
    # ANSI color codes
    COLORS = {
        "INFO": "\033[94m",      # Blue
        "SUCCESS": "\033[92m",   # Green
        "WARNING": "\033[93m",   # Yellow
        "ERROR": "\033[91m",     # Red
        "RESET": "\033[0m"       # Reset
    }
    
    @staticmethod
    def log(agent_name: str, message: str, level: str = "INFO"):
        """
        Log a message with agent name and level
        
        Args:
            agent_name: Name of the agent
            message: Log message
            level: Log level (INFO, SUCCESS, WARNING, ERROR)
        """
        color = Logger.COLORS.get(level, Logger.COLORS["INFO"])
        reset = Logger.COLORS["RESET"]
        
        # Format the message
        formatted_message = f"{color}[{level}]{reset} {agent_name}: {message}"
        print(formatted_message)
    
    @staticmethod
    def log_decision(agent_name: str, doc_id: str, decision: str, reason: str):
        """
        Log a labeling or review decision with formatting
        
        Args:
            agent_name: Name of the agent
            doc_id: Document ID
            decision: Decision made
            reason: Reason for decision
        """
        print(f"\n{'='*80}")
        print(f"🔍 [{agent_name}] DECISION")
        print(f"{'='*80}")
        print(f"Document ID: {doc_id}")
        print(f"Decision: {decision}")
        print(f"Reason: {reason}")
        print(f"{'='*80}\n")
    
    @staticmethod
    def log_section(title: str):
        """
        Log a section header
        
        Args:
            title: Section title
        """
        print(f"\n{'='*80}")
        print(f"{title}")
        print(f"{'='*80}\n")
    
    @staticmethod
    def log_error(agent_name: str, error: Exception):
        """
        Log an error with traceback
        
        Args:
            agent_name: Name of the agent
            error: Exception object
        """
        Logger.log(agent_name, f"ERROR: {str(error)}", "ERROR")
        
        # Print traceback if available
        import traceback
        traceback.print_exc()


def parse_json_from_response(response_text: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response, handling markdown code blocks
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Parsed JSON dictionary
    """
    import json
    import re
    
    # Try to extract JSON from markdown code blocks using regex
    # Pattern 1: `````` (with json keyword)
    pattern1 = r'``````'
    match = re.search(pattern1, response_text, re.DOTALL)
    if match:
        response_text = match.group(1).strip()
    else:
        # Pattern 2: `````` (plain code blocks)
        pattern2 = r'``````'
        match = re.search(pattern2, response_text, re.DOTALL)
        if match:
            response_text = match.group(1).strip()
    
    # Try to parse JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nResponse: {response_text[:500]}")


def validate_label(label: str) -> str:
    """
    Validate and normalize label name
    
    Args:
        label: Label string
        
    Returns:
        Validated label name
    """
    valid_labels = ["relevant", "somewhat_relevant", "acceptable", "not_sure", "irrelevant"]
    
    label_lower = label.lower().strip()
    
    # Handle variations
    label_mapping = {
        "somewhat relevant": "somewhat_relevant",
        "semantically acceptable": "acceptable",
        "semantically_acceptable": "acceptable",
        "not sure": "not_sure",
        "notsure": "not_sure",
    }
    
    label_lower = label_mapping.get(label_lower, label_lower)
    
    if label_lower in valid_labels:
        return label_lower
    
    # Default to not_sure if invalid
    return "not_sure"
