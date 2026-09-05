# EduGrade AI

## Run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
streamlit run app.py
```

Add your Google AI Studio and Groq API keys to `.env`, or enter them in the Streamlit sidebar.

Important: Gemini 2.0 Flash and Groq Gemma 2 9B are retired models as of 2026, so this runnable version uses current replacements by default. Model IDs are configurable through `.env`.
