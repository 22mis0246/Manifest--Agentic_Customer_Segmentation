# SegmentIQ

SegmentIQ is a Streamlit-based customer segmentation app for retail banking. It lets you upload a CSV, ask questions in plain English, and get segment analysis, reasoning traces, and downloadable output files.

## What It Does

- Upload a customer transaction dataset in CSV format
- Build customer features such as balance, frequency, recency, and transaction amount
- Segment customers into groups like `priority`, `regular`, and `dormant`
- Show a reasoning trace for each analysis
- Export segmentation results to the `outputs/` folder

## Project Structure

- `app.py` - Streamlit UI
- `agent_core.py` - LLM-driven orchestration logic
- `api/main.py` - FastAPI endpoints
- `tools/` - segmentation, EDA, recommendation, explainability tools
- `pipeline/` - feature engineering and segmentation helpers
- `data/` - default sample dataset
- `outputs/` - generated CSV outputs

## Requirements

- Python 3.10+
- A valid `GEMINI_API_KEY`

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_key_here
```

## How To Run

Open the project in VS Code, open a terminal in the project folder, and run:

```powershell
cd "C:\Users\NAVEENRAJ\Desktop\projects\Manifest Current\Manifest--Agentic_Customer_Segmentation"
streamlit run app.py
```

If `streamlit` is not recognized, use:

```powershell
python -m streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How To Use

1. Upload a CSV from the left sidebar, or keep the sample dataset in `data/customers.csv`
2. Enter a question in the analytics chat
3. Review the answer, reasoning trace, and segment output
4. Download the generated segment CSV if needed

## Example Questions

- `Segment customers into priority, regular, and dormant based on average monthly balance, transaction frequency, and recency.`
- `What is the average transaction size for each segment?`
- `Why were these customers classified as dormant?`
- `Which regular customers are closest to becoming priority customers?`
- `Recommend retention actions for dormant customers.`

## Notes

- Uploaded files are processed in-memory for the current session
- Exported segment files are saved in `outputs/`
- If a prompt is too vague, the app may ask a clarification question

