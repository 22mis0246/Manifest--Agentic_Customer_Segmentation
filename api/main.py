"""FastAPI service for SegmentIQ."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent_core import SegmentIQAgent
from data_store import store
from tools.eda_tool import eda_tool
from tools.segmentation_tool import segmentation_tool

app = FastAPI(title="SegmentIQ API", version="1.0.0")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3)


class SegmentRequest(BaseModel):
    criteria: list[str] = Field(default_factory=lambda: ["avg_monthly_balance", "txn_frequency_monthly"])
    method: str = "rule_based"
    num_segments: int = 3


@app.on_event("startup")
def load_dataset() -> None:
    store.auto_load()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "dataset": store.summary()}


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    try:
        agent = SegmentIQAgent()
        result = agent.run(request.query)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "query": result.query,
        "needs_clarification": result.needs_clarification,
        "clarifying_question": result.clarifying_question,
        "summary": result.summary,
        "tools_used": result.tools_used,
        "trace": result.trace_as_text(),
    }


@app.get("/eda")
def eda() -> dict[str, Any]:
    try:
        return eda_tool()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/segment")
def segment(request: SegmentRequest) -> dict[str, Any]:
    try:
        return segmentation_tool(
            criteria=request.criteria,
            method=request.method,
            num_segments=request.num_segments,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/segments/export")
def export_segments() -> FileResponse:
    path = store.last_export_path
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="No segment export available yet.")
    return FileResponse(path, media_type="text/csv", filename=Path(path).name)