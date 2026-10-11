
import os
import json
from google import genai

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.8-flash").strip()
gemini_client = genai.Client(api_key=API_KEY) if API_KEY else None

app = FastAPI(title="BoardSnap LLM Service", version="1.0.0")


class NotesRequest(BaseModel):
    regions: List[Dict[str, Any]] = Field(default_factory=list)
    diagram: Optional[Dict[str, Any]] = None


class QuizRequest(BaseModel):
    notes: Dict[str, Any]


class ChatRequest(BaseModel):
    query: str
    board_id: str
    regions: List[Dict[str, Any]] = Field(default_factory=list)
    diagram: Optional[Dict[str, Any]] = None
    notes: Optional[Dict[str, Any]] = None


def generate_json(prompt: str) -> Any:
    """Ask Gemini for a JSON response."""
    if gemini_client is None:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is missing from the LLM environment."
        )

    try:
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
            },
        )
        return json.loads(response.text)

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Gemini request failed: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Gemini could not generate a valid response. Check the server logs."
        )


def has_board_evidence(regions, diagram=None, notes=None):
    """Check whether any usable board content was supplied."""
    region_text = any(
        str(region.get("text", "")).strip()
        or str(region.get("label", "")).strip()
        for region in regions
    )

    diagram_content = bool(
        diagram
        and (
            diagram.get("nodes")
            or diagram.get("edges")
            or diagram.get("mermaid")
        )
    )

    notes_content = bool(
        notes
        and (
            notes.get("title")
            or notes.get("keyPoints")
            or notes.get("definitions")
            or notes.get("actionItems")
        )
    )

    return region_text or diagram_content or notes_content


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "BoardSnap LLM Grounded Service",
        "gemini_configured": gemini_client is not None,
        "gemini_model": MODEL_NAME,
    }


@app.post("/llm/notes")
def generate_grounded_notes(req: NotesRequest):
    if not has_board_evidence(req.regions, req.diagram):
        return {
            "title": "No readable whiteboard content",
            "keyPoints": [],
            "definitions": [],
            "actionItems": [],
        }

    prompt = f"""
You generate study notes from a photographed whiteboard.

Use ONLY facts explicitly supported by the supplied OCR regions
and diagram. Do not invent technologies, numbers, requirements,
definitions, or action items. If something is unclear, omit it.

OCR REGIONS:
{json.dumps(req.regions, ensure_ascii=False)}

DIAGRAM:
{json.dumps(req.diagram or {{}}, ensure_ascii=False)}

Return valid JSON with exactly these fields:
{{
  "title": "A short title supported by the evidence",
  "keyPoints": ["Evidence-based point"],
  "definitions": [
    {{"term": "Term written on the board",
     "definition": "Only explain it if supported by the board"}}
  ],
  "actionItems": ["Only an action item explicitly shown on the board"]
}}

Use empty arrays when the evidence does not support any entries.
"""

    result = generate_json(prompt)

    required_fields = ["title", "keyPoints", "definitions", "actionItems"]
    if not isinstance(result, dict) or any(
        field not in result for field in required_fields
    ):
        raise HTTPException(
            status_code=502,
            detail="Gemini returned notes in an unexpected format."
        )

    return result


@app.post("/llm/quiz")
def generate_quiz(req: QuizRequest):
    if not has_board_evidence([], notes=req.notes):
        return []

    prompt = f"""
Create up to 5 multiple-choice study questions from these notes ONLY.
Do not introduce outside facts. If the notes contain too little
information, return fewer questions or an empty array.

NOTES:
{json.dumps(req.notes, ensure_ascii=False)}

Return a JSON array. Each item must have:
{{
  "id": "q1",
  "q": "Question",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correctIndex": 0,
  "a": "Brief explanation supported by the notes"
}}

correctIndex must be an integer from 0 to 3.
Each question must have exactly four options.
Return only the JSON array.
"""

    result = generate_json(prompt)

    if not isinstance(result, list):
        raise HTTPException(
            status_code=502,
            detail="Gemini returned quiz questions in an unexpected format."
        )

    return result


@app.post("/llm/chat")
def chat_with_board(req: ChatRequest):
    if not has_board_evidence(req.regions, req.diagram, req.notes):
        return {
            "answer": "I don't have readable content from this whiteboard yet. Please process the image again.",
            "source": "No board evidence supplied",
        }

    context = {
        "regions": req.regions,
        "diagram": req.diagram,
        "notes": req.notes,
    }

    prompt = f"""
Answer the user's question using ONLY the supplied whiteboard evidence.
Do not invent facts. If the answer is not supported, say that the
whiteboard does not provide enough information.

BOARD EVIDENCE:
{json.dumps(context, ensure_ascii=False)}

USER QUESTION:
{req.query}

Return JSON:
{{
  "answer": "Your grounded answer",
  "source": "The relevant region labels or diagram elements"
}}
"""

    result = generate_json(prompt)

    if not isinstance(result, dict) or "answer" not in result:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an unexpected chat response."
        )

    result.setdefault("source", "Supplied whiteboard evidence")
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
