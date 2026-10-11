# BoardSnap - Live Demo Recording Guide

**Target duration:** 2:30-2:45
**Important:** Record the live-pipeline sections only after a real upload has
successfully passed through Roboflow, TrOCR, and Gemini. Preset sample boards
are preview data; label them as such and do not present them as model output.

## 0:00-0:20 - Problem

Show a real whiteboard photo and explain the problem of preserving handwritten
notes and diagrams from a photograph.

## 0:20-0:40 - Product overview

Show the BoardSnap upload screen. Describe the actual code path: frontend upload
to the Express API, image cleanup and detection in the Vision service, OCR and
diagram construction, then grounded notes and quiz generation.

## 0:40-1:35 - Live pipeline

Only if the live services have been verified:

1. Upload a board photo and show the processing status.
2. Show the cleaned image beside the original.
3. Show detected regions and the generated Mermaid diagram.
4. Compare the recognized text with the visible writing.
5. Show the notes and quiz returned by the language service.

If any stage fails, show the error and explain which integration still needs
configuration. Do not substitute a preset sample and describe it as live
inference.

## 1:35-2:05 - Model transparency

Explain that region detection defaults to fast OpenCV contour/line heuristics;
optional pretrained Grounding DINO is slower on CPU. Neither it nor TrOCR has
been fine-tuned by this project. Share accuracy only after a reproducible,
labeled held-out evaluation has been completed.

## 2:05-2:35 - Wrap-up

Summarize the user workflow and show where source code and setup instructions
are available. Avoid claims of zero hallucinations, model accuracy, or runtime
until measurements support them.

## Pre-recording checklist

- Confirm the OpenCV detector returns useful regions for the selected board
  photo. If using Grounding DINO or Roboflow instead, verify its inference.
- Confirm the Gemini key/model can generate notes, quiz, and chat responses.
- Confirm TrOCR loads and reads the chosen handwritten sample.
- Run the Docker Compose stack and verify frontend, backend, Vision, and LLM.
- Use a board photo that can be shared publicly; do not expose credentials.
- Clearly identify any preset sample content or mocked/test results.
