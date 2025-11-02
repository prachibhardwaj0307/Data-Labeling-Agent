# 🏷️ AI-Powered Multi-Agent Document Labeling System

An intelligent document classification system using multiple AI agents to automatically label and categorize documents based on relevance, location, and recency.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Label Studio URL and API key
- pip package manager
- Git for cloning repository

### Step 1: Clone Repository

```bash
git clone https://github.com/prachibhardwaj0307/Data-Labeling-Agent
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

Includes OpenAI, Anthropic, Streamlit, python-dotenv, and requests.

### Step 4: Configure Environment

- Create a `.env` file in the project root.
- Add the following lines to the `.env` file:

```
OPENAI_API_KEY=sk-proj-your-api-key-here
LABEL_STUDIO_URL=https://your-label-studio-instance.com
LABEL_STUDIO_API_KEY=your-label-studio-api-key
```

- Replace with your actual OpenAI API key, Label Studio URL, and API key.
- Ensure `.env` is in your `.gitignore` file to prevent committing secrets.

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

Run the agent on a specific task from Label Studio using the task ID:

```bash
python3 main.py <task_id>
```

For example:

```bash
python3 main.py 35851
```

Output saved as:
- `output_id_<task_id>.json`
- `report_id_<task_id>.json`

### Streamlit Web UI (Recommended)

Launch the Streamlit application:

```bash
streamlit run streamlit_app.py
```

Then open your browser to [http://localhost:8501](http://localhost:8501)

- Enter the Label Studio **Task ID** in the sidebar input field.
- Click **🚀 Run Labeling** to start the process.
- View the real-time workflow execution and interim results.
- Manually override any labels if necessary.
- Save the final results back to Label Studio or download them as JSON files.

---

## 💡 Label Studio Integration

This project is tightly integrated with Label Studio to streamline the data labeling workflow.

- **Data Loading:** The application fetches tasks directly from Label Studio using the provided Task ID. This eliminates the need for local `input_data.json` files.
- **Annotation:** The agent processes the documents and generates labels based on the defined categories.
- **Example-Based Learning:** The system learns from existing annotations in Label Studio to improve the accuracy of its labeling. Documents already labeled as "relevant," "somewhat_relevant," and "acceptable" are used as reference examples.
- **Saving Results:** The updated labels can be saved back to Label Studio, either by updating an existing annotation or creating a new one.

---

## 📁 Project Structure

```
data_labeling_agent/
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
│   ├── __init__.py
│   └── data_models.py
├── utils/
│   ├── __init__.py
│   ├── helpers.py
│   ├── label_studio_client.py
│   └── llm_client.py
├── config.py
├── main.py
├── streamlit_app.py
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

### SuperiorAgent
- Orchestrates the entire workflow, from data loading and learning to final output generation.

### FilterAgent
- Removes irrelevant documents with reasoning.
- Uses LLM to analyze titles and content.

### GroupingAgent
- Groups documents by topic **and year**.

### GroupReviewAgent
- Reviews grouping quality for coherence and correctness.

### RegroupAgent
- Reorganizes groups based on feedback from the review agent.

### LabelingAgent
- Labels entire groups based on the defined criteria and learns from existing examples.

### LabelReviewAgent
- Enforces the "maximum 10 RELEVANT documents" rule and checks for label consistency.

### RelabelAgent
- Selects the **TOP 10** most relevant documents and downgrades others if necessary.

---

## 🔄 Workflow

**Nine-Step Process:**
1.  Load task from Label Studio.
2.  Learn from existing annotations to create reference examples.
3.  FilterAgent removes irrelevant documents.
4.  GroupingAgent organizes documents by topic and year.
5.  GroupReviewAgent checks the grouping.
6.  RegroupAgent reorganizes groups if necessary.
7.  LabelingAgent assigns labels to the groups.
8.  LabelReviewAgent validates the labels.
9.  RelabelAgent ensures the top 10 relevance rule is met.
10. Generate output and save results to Label Studio or download as JSON.

---

## 📦 Requirements

```
openai>=1.0.0
anthropic>=0.20.0
streamlit>=1.28.0
python-dotenv>=1.0.0
requests>=2.28.0
```

---

## 📧 Contact

**GitHub:** [https://github.com/prachibhardwaj0307/Data-Labeling-Agent](https://github.com/prachibhardwaj0307/Data-Labeling-Agent)

---

⭐ **Star this repo if you find it helpful!**
🐛 **Report issues** on the Issues page