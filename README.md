# AI Signs

A simple Streamlit app that analyzes text for AI-generated writing patterns using `ai_analyzer.py` and renders results with a gauge and detailed rubric breakdown.

This app is based on Wikipedia's ["Signs of AI writing"](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing).

## Requirements

- Python 3.10+ (recommended)
- `pip`

## Install

1. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration
Create a `.env` file in the project root with your Google API key:

```
GOOGLE_API_KEY=YOUR-API-KEY
```

This is required for the analyzer to use the Google API.

## Run
```bash
streamlit run app.py
```

## Notes
- Paste your text into the app and click Analyze
- The app shows an AI probability gauge, reasoning, and detailed rubrics
- A few of the categories from the Wikipedia page are included in `categories.yaml`, and this file can be edited to adjust the analyzer
- You can edit the LLM that is being used and the thinking level in `ai_analyzer.py`