"""
Configuration for the document labeling system with LLM integration
"""

# ============================================================
# LLM CONFIGURATION
# ============================================================

# Choose your LLM provider: "openai" or "anthropic"
LLM_PROVIDER = "openai"

# OpenAI Configuration
OPENAI_MODEL = "gpt-4"  # Options: "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"

# Anthropic Configuration  
ANTHROPIC_MODEL = "claude-3-sonnet-20240229"  # Options: "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"

# Temperature setting (0.0 = deterministic, 1.0 = creative)
# Lower temperature = more consistent labeling
TEMPERATURE = 0.3

# API Keys (set via environment variables)
# Set these in your .env file or environment:
# OPENAI_API_KEY = "sk-your-key-here"
# ANTHROPIC_API_KEY = "sk-ant-your-key-here"

# ============================================================
# WORKFLOW CONFIGURATION
# ============================================================

# Maximum attempts for review cycles before forcing approval
MAX_GROUP_REVIEW_ATTEMPTS = 3
MAX_LABEL_REVIEW_ATTEMPTS = 3

# ============================================================
# LABEL DEFINITIONS
# ============================================================
LABEL_CRITERIA = {
    "relevant": {
        "description": "Directly answers the query with current, comprehensive information FOR THE USER'S LOCATION",
        "keywords": ["directly answers", "comprehensive", "current", "exact match", "correct location"],
        "examples": [
            "Query='benefits' + Location='India' → 'India Employee Benefits 2025'",
            "Query='holidays' + Location='US' → 'US Holiday Calendar 2025'",
            "Current, comprehensive documentation for the user's specific location"
        ]
    },
    "somewhat_relevant": {
        "description": "Partially addresses query OR good content but older/incomplete",
        "keywords": ["partially addresses", "good but old", "incomplete answer", "related but not perfect"],
        "examples": [
            "Query='benefits' + Location='India' → 'India Benefits 2022' (older)",
            "Partial information about the topic for the correct location",
            "Related but doesn't fully answer the query"
        ]
    },
    "acceptable": {
        "description": "CORRECT TOPIC but WRONG LOCATION, OR related but limited value",
        "keywords": ["correct topic wrong location", "different region", "limited value", "contextual", "background"],
        "examples": [
            "Query='benefits' + Location='India' → 'US Employee Benefits 2025' (wrong location)",
            "Query='holidays' + Location='India' → 'UK Holiday Calendar' (wrong location)",
            "Global/general documents that provide context",
            "Older documents providing historical background"
        ]
    },
    "not_sure": {
        "description": "Missing title, no link, unclear content, or ambiguous relevance",
        "keywords": ["no title", "missing link", "unclear", "ambiguous", "cannot determine"],
        "examples": [
            "Documents with 'No Title' or empty title",
            "Broken or missing links",
            "Insufficient content to determine relevance"
        ]
    },
    "irrelevant": {
        "description": "Completely unrelated to the query (filtered by FilterAgent)",
        "keywords": ["unrelated", "off-topic", "different subject", "no connection", "filtered out"],
        "examples": [
            "Query='benefits' → 'Tax forms for vendors' (completely different topic)",
            "Documents about completely unrelated subjects",
            "System/technical documents with no user value"
        ]
    }
}


# ============================================================
# GROUPING PARAMETERS
# ============================================================

# Group size constraints
MIN_GROUP_SIZE = 1
MAX_GROUP_SIZE = 10
IDEAL_GROUP_SIZE_MIN = 1
IDEAL_GROUP_SIZE_MAX = 7

# Similarity threshold for document grouping (0.0 to 1.0)
SIMILARITY_THRESHOLD = 0.6

# ============================================================
# FILTERING CONFIGURATION
# ============================================================

# Enable/disable filtering stage
ENABLE_FILTERING = True

# Documents with these titles will be filtered out
INVALID_TITLES = [
    "no title",
    "untitled",
    "",
    "none",
    "No Title"
]

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

# Log level options: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL = "INFO"

# Log format
LOG_FORMAT = "[%(levelname)s] %(agent)s: %(message)s"

# Enable colored output (set to False if running in environments that don't support colors)
ENABLE_COLOR_LOGGING = True

# ============================================================
# OUTPUT CONFIGURATION
# ============================================================

# Output file naming
OUTPUT_FILE_PREFIX = "output_id_"
REPORT_FILE_PREFIX = "report_id_"
OUTPUT_FILE_EXTENSION = ".json"

# Pretty print JSON output
JSON_INDENT = 2

# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

# Maximum tokens for LLM responses
MAX_TOKENS = 2000

# Timeout for LLM API calls (seconds)
API_TIMEOUT = 60

# Batch size for processing documents
BATCH_SIZE = 5

# ============================================================
# FEATURE FLAGS
# ============================================================

# Enable/disable various features
ENABLE_GROUPING = True
ENABLE_REVIEW_LOOPS = True
ENABLE_LOCATION_FILTERING = True
ENABLE_RECENCY_DETECTION = True

# ============================================================
# ADVANCED SETTINGS
# ============================================================

# Retry configuration for failed LLM calls
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Cache LLM responses (for development/testing)
ENABLE_CACHING = False
CACHE_DIR = ".cache"

# Debug mode (provides extra logging)
DEBUG_MODE = False

# ============================================================
# CUSTOM PROMPTS (Optional - Override default prompts)
# ============================================================

# You can customize agent prompts here if needed
CUSTOM_PROMPTS = {
    # Example:
    # "filter_agent": "Your custom filter agent prompt...",
    # "labeling_agent": "Your custom labeling agent prompt...",
}

# ============================================================
# LOCATION MAPPING (Optional)
# ============================================================

# Map location variations to standard names
LOCATION_MAPPING = {
    "india": "India",
    "us": "United States",
    "usa": "United States",
    "united states": "United States",
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "canada": "Canada",
}

# ============================================================
# RECENCY CONFIGURATION
# ============================================================

# Define year ranges for recency detection
CURRENT_YEAR_RANGE = [2024, 2025]  # Documents from these years are "current"
RECENT_YEAR_RANGE = [2022, 2023]   # Documents from these years are "recent but older"
OLD_YEAR_RANGE = [2015, 2021]      # Documents from these years are "old"

# ============================================================
# VALIDATION
# ============================================================

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Validate LLM provider
    if LLM_PROVIDER not in ["openai", "anthropic"]:
        errors.append(f"Invalid LLM_PROVIDER: {LLM_PROVIDER}")
    
    # Validate temperature
    if not 0.0 <= TEMPERATURE <= 1.0:
        errors.append(f"TEMPERATURE must be between 0.0 and 1.0, got {TEMPERATURE}")
    
    # Validate group sizes
    if MIN_GROUP_SIZE < 1:
        errors.append(f"MIN_GROUP_SIZE must be >= 1, got {MIN_GROUP_SIZE}")
    if MAX_GROUP_SIZE < MIN_GROUP_SIZE:
        errors.append(f"MAX_GROUP_SIZE must be >= MIN_GROUP_SIZE")
    
    # Validate attempts
    if MAX_GROUP_REVIEW_ATTEMPTS < 1:
        errors.append(f"MAX_GROUP_REVIEW_ATTEMPTS must be >= 1")
    if MAX_LABEL_REVIEW_ATTEMPTS < 1:
        errors.append(f"MAX_LABEL_REVIEW_ATTEMPTS must be >= 1")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))

# Run validation on import
try:
    validate_config()
except ValueError as e:
    print(f"⚠️ Configuration Warning: {e}")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_model_name() -> str:
    """Get the currently configured model name"""
    if LLM_PROVIDER == "openai":
        return OPENAI_MODEL
    elif LLM_PROVIDER == "anthropic":
        return ANTHROPIC_MODEL
    return "unknown"

def is_current_year(year: int) -> bool:
    """Check if year is in current range"""
    return year in CURRENT_YEAR_RANGE

def is_recent_year(year: int) -> bool:
    """Check if year is in recent range"""
    return year in RECENT_YEAR_RANGE

def is_old_year(year: int) -> bool:
    """Check if year is in old range"""
    return year in OLD_YEAR_RANGE

# ============================================================
# DISPLAY CONFIGURATION (for debugging)
# ============================================================

def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("CONFIGURATION SETTINGS")
    print("=" * 80)
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"Model: {get_model_name()}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max Group Review Attempts: {MAX_GROUP_REVIEW_ATTEMPTS}")
    print(f"Max Label Review Attempts: {MAX_LABEL_REVIEW_ATTEMPTS}")
    print(f"Group Size Range: {MIN_GROUP_SIZE}-{MAX_GROUP_SIZE}")
    print(f"Filtering Enabled: {ENABLE_FILTERING}")
    print(f"Grouping Enabled: {ENABLE_GROUPING}")
    print("=" * 80)

# Uncomment to print config on import (for debugging)
# print_config()
