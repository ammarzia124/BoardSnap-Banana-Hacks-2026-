# BoardSnap - Devpost Draft

> **Submission status:** Draft only. Confirm the event/team details and record a
> successful live end-to-end run before submitting. No unverified model results
> should be added.

**Tagline:** Snap a whiteboard. Get clean notes, a redrawn diagram, and a quiz.

## What BoardSnap is

BoardSnap is a React application and service pipeline intended to turn a
whiteboard photo into a cleaned image, detected regions, a Mermaid diagram,
structured notes, and quiz questions. The repository contains a React/Vite
frontend, an Express orchestration API, a FastAPI vision service, and a
Gemini-backed language service.

The normal upload path sends the image from the frontend to the backend. The
backend orchestrates OpenCV cleanup and fast geometry-based region detection,
handwriting OCR, graph construction, and language-service requests. Grounding
DINO and Roboflow Workflow inference are optional detector providers. Preset
sample cards are previews, not live model predictions.

## How the current implementation works

- **Frontend:** React, Vite, Tailwind CSS, and Mermaid rendering.
- **Backend:** Node.js/Express routes for image upload, asynchronous job status,
  board actions, and calls to Vision and LLM services. MongoDB is supported;
  in-memory storage is only temporary.
- **Image and diagram processing:** OpenCV cleanup and a spatial graph heuristic
  that emits Mermaid.
- **Region detection:** OpenCV contours, Hough lines, and text-line segmentation
  by default. Grounding DINO Tiny or Roboflow Workflow can be enabled as
  optional providers. No detector has been trained by this project.
- **Handwriting OCR:** Microsoft's pretrained TrOCR checkpoint. It has not
  been fine-tuned by this project.
- **Notes, quiz, and chat:** Gemini requests grounded in the extracted board
  evidence.

## Current verification and limitations

The frontend build, frontend API tests, backend contract tests, and mocked
Vision tests pass locally. That does not demonstrate real model quality.

Synthetic OpenCV shape/connector tests pass, but region recall on real
whiteboard images has not been measured. An earlier Roboflow endpoint returned
HTTP 404. Grounding DINO ran locally on CPU but took 57.78 seconds on a
synthetic image, so it is optional rather than the default. Gemini was not
configured and TrOCR had not loaded at the last service check. The dataset
manifest says `not_downloaded`; no project-trained checkpoint or labeled OCR
evaluation set is included. Detector/OCR accuracy and hallucination-rate
figures are **not verified** and are not reported here.

Before submitting a live demo, test the OpenCV detector on representative
boards, load and test the OCR checkpoint, configure the Gemini key in its
ignored local environment file, verify a complete upload through every service,
and evaluate OCR on manually transcribed held-out images. Never publish API
keys or claim a model was trained without reproducible training/evaluation
evidence.

## Links to complete before submission

- **Demo video:** Add the final public or unlisted recording URL.
- **GitHub repository:** https://github.com/ammarzia124/BoardSnap-Banana-Hacks-2026-
