# 🏷️ AI-Powered Multi-Agent Document Labeling System

An intelligent document classification system using multiple AI agents to automatically label and categorize documents based on relevance, location, and recency.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher  
- OpenAI API key  
- pip package manager  
- Git for cloning repository  

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/document-labeling-system.git
cd document-labeling-system
```

### Step 2: Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Includes OpenAI, Anthropic, Streamlit, python-dotenv

### Step 4: Configure Environment

- Create `.env` file in project root  
- Add line:  
  ```
  OPENAI_API_KEY=sk-proj-your-api-key-here
  ```
- Replace with your actual OpenAI API key  
- Ensure `.env` is in `.gitignore`

---

## ⚙️ Configuration

### Edit `config.py` Settings

**LLM Configuration:**
- `LLM_PROVIDER = "openai"` or `"anthropic"`
- `OPENAI_MODEL = "gpt-4"`
- `TEMPERATURE = 0.3` for consistent results

**Review Limits:**
- `MAX_GROUP_REVIEW_ATTEMPTS = 3`
- `MAX_LABEL_REVIEW_ATTEMPTS = 3`

**Grouping Parameters:**
- `MIN_GROUP_SIZE = 1`
- `MAX_GROUP_SIZE = 10`

---

## 📖 Usage

### Command Line Interface

```bash
python3 main.py 1
python3 main.py 2
```

Output saved as:
- `output_id_X.json`
- `report_id_X.json`

### Streamlit Web UI (Recommended)

```bash
streamlit run streamlit_app.py
```

Then open your browser to [http://localhost:8501](http://localhost:8501)

- Enter dataset ID in the input field  
- Click **🚀 Run Labeling**  
- View real-time workflow execution  
- Download JSON reports when complete  

---

## 🧾 Input Data Format

**input_data.json**
```json
[
  {
    "id": 1,
    "data": {
      "items": [
        {
          "id": "doc_1",
          "title": "Employee Benefits 2025",
          "html": "<p>Document content...</p>"
        }
      ],
      "text": "employee benefits",
      "location": "India"
    }
  }
]
```

---

## 📁 Project Structure

```
document-labeling-system/
├── agents/
│   ├── __init__.py
│   ├── filter_agent.py
│   ├── grouping_agent.py
│   ├── group_review_agent.py
│   ├── labeling_agent.py
│   ├── label_review_agent.py
│   ├── regroup_agent.py
│   ├── relabel_agent.py
│   └── superior_agent.py
├── models/
│   └── data_models.py
├── utils/
│   ├── helpers.py
│   └── llm_client.py
├── config.py
├── main.py
├── streamlit_app.py
├── input_data.json
├── requirements.txt
├── .env
└── README.md
```

---

## 🏷️ Label Categories

### ✅ RELEVANT (Maximum 10)
- Directly answers query  
- Correct location match  
- Current year (2025)  
- Most comprehensive information  
- Automatically limited to top 10  

### ⚠️ SOMEWHAT_RELEVANT
- Answers query but older (2024, 2023)  
- Partially addresses query  
- Correct location  
- Still valuable but secondary  

### ℹ️ ACCEPTABLE
- Correct topic but wrong location  
- Provides context or background  
- Example: US docs when user needs India  

### ❓ NOT_SURE
- Missing or invalid title  
- Unclear content  
- Cannot determine relevance confidently  

### 🚫 IRRELEVANT
- Completely unrelated to query  
- Filtered out automatically  
- No connection to topic  

---

## 🤖 AI Agents

### FilterAgent
- Removes irrelevant documents with reasoning  
- Uses LLM to analyze titles and content  
- Conservative approach (keeps if uncertain)

### GroupingAgent
- Groups documents by topic **and year**  
- Example: *Benefits 2025*, *Benefits 2024*

### GroupReviewAgent
- Reviews grouping quality  
- Checks names, themes, coherence  
- Accepts single-document groups if justified  

### LabelingAgent
- Labels entire groups  
- Year-aware: 2025 = RELEVANT, older = SOMEWHAT_RELEVANT  
- Location-aware: wrong location = ACCEPTABLE  

### LabelReviewAgent
- Enforces maximum 10 RELEVANT documents rule  
- Checks label accuracy and consistency  
- Triggers relabeling if needed  

### RegroupAgent
- Reorganizes groups based on feedback  
- Improves group names and themes  

### RelabelAgent
- Selects **TOP 10** most relevant documents  
- Ranks by: Year → Completeness → Quality → Location  
- Downgrades excess to SOMEWHAT_RELEVANT  

---

## 🔄 Workflow

**Nine-Step Process:**
1. Load documents from JSON file  
2. FilterAgent removes irrelevant docs  
3. GroupingAgent organizes by topic/year  
4. GroupReviewAgent checks grouping  
5. RegroupAgent reorganizes if rejected  
6. LabelingAgent assigns labels  
7. LabelReviewAgent validates labels  
8. RelabelAgent ensures top 10 relevance  
9. Generate output JSON reports  

---

## 📦 Requirements

```
openai>=1.0.0
anthropic>=0.20.0
streamlit>=1.28.0
python-dotenv>=1.0.0
```

---

## 📄 License

MIT License – See `LICENSE` file for details.

---

## 📧 Contact

**GitHub:** [https://github.com/yourusername/document-labeling-system](https://github.com/yourusername/document-labeling-system)

---

## ❓ FAQ

### Q: How many documents can I process?
- 100+ efficiently; depends on document size  

### Q: Why limit to 10 RELEVANT?
- Ensures only top-quality documents are marked RELEVANT  

### Q: Can I customize label categories?
- Yes — edit `config.py` → `LABEL_CRITERIA`  

### Q: Can I use Claude instead of GPT-4?
- Yes — set `LLM_PROVIDER = "anthropic"` in `config.py`  

---

⭐ **Star this repo if you find it helpful!**  
🐛 **Report issues** on the Issues page  
