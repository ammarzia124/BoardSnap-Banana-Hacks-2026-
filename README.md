# BoardSnap 📸✨
> **Snap a whiteboard. Get clean notes, a redrawn diagram, and a quiz.**

[![Banana Hacks 2026](https://img.shields.io/badge/Event-Banana%20Hacks%202026-yellow.svg)](https://devpost.com)
[![Theme](https://img.shields.io/badge/Theme-Image%20AI-blue.svg)](https://devpost.com)
[![Frontend](https://img.shields.io/badge/Frontend-React%20%7C%20Tailwind-emerald.svg)](./frontend)
[![Backend](https://img.shields.io/badge/Backend-Node.js%20%7C%20Express-teal.svg)](./backend)
[![Vision](https://img.shields.io/badge/Vision-OpenCV%20%7C%20Roboflow-orange.svg)](./vision)
[![LLM](https://img.shields.io/badge/Language-Gemini%20API-purple.svg)](./llm)

**Team:** Ammar, Muavia, Moyeon, Awais  
**Event:** Banana Hacks 2026 (Devpost, Oct 9 – Oct 12, 2026)  
**Demo Video:** [2:40 YouTube Walkthrough](https://youtube.com)  

---

## 🌟 The Problem
After a university lecture or brainstorm, the whiteboard is the only record. Students photograph it, then the photo sits in a phone gallery forever:
- Handwriting is illegible with skewed angles and harsh glare.
- Hand-drawn diagrams are locked in pixels and cannot be edited.
- Nobody converts the board into structured study materials or quizzes.

## 💡 The Solution
BoardSnap turns raw whiteboard photos into:
1. **Perspective-rectified, glare-attenuated clean imagery**.
2. **AI-detected regions** separating handwritten text blocks from flowchart shapes.
3. **Editable Mermaid vector diagrams** with SVG export.
4. **Evidence-guided study notes** (Key Points, Definitions, Action Items) generated from extracted board content.
5. **Interactive active-recall quizzes, flashcards**, and conversational board chat.

---

## 🏗️ Repository Architecture
```text
boardsnap/
├── frontend/           # React + Tailwind CSS + Mermaid + Lucide UI (Moyeon)
│   ├── src/
│   │   ├── components/ # Comparison slider, diagram viewer, quiz, board chat
│   │   ├── data/       # Preset sample preview boards
│   │   └── utils/      # Backend API client
│   └── package.json
│
├── backend/            # Express.js REST API & Agent Orchestrator (Ammar)
│   ├── server.js       # API startup and graceful shutdown
│   ├── src/            # Routes, pipeline, MongoDB/in-memory repository, service clients
│   ├── test/           # API and end-to-end pipeline contract tests
│   └── package.json
│
├── vision/             # Computer Vision & ML services (Muavia)
│   ├── main.py         # OpenCV cleanup, Roboflow detection, TrOCR & spatial graph builder
│   └── requirements.txt
│
├── llm/                # Grounded language service (Awais)
│   ├── main.py         # Evidence-based note, quiz, and board-chat generation
│   └── requirements.txt
│
├── dataset/            # Roboflow dataset downloader and download manifest
│   └── roboflow/       # Downloaded dataset exports (not committed)
│
└── docs/               # Hackathon submission documentation
    ├── EVALUATION_REPORT.md  # Verified status, test evidence, and evaluation gaps
    ├── DEMO_VIDEO_SCRIPT.md  # Live-demo recording guide
    └── DEVPOST_SUBMISSION.md # Evidence-based submission draft
```

---

## 🚀 Quick Start & How to Run

### Run the complete stack with Docker Compose

Install and start Docker Desktop first. Create the local service environment files
if they do not exist, then edit them on your machine:

```powershell
if (-not (Test-Path vision\.env)) { Copy-Item vision\.env.example vision\.env }
if (-not (Test-Path llm\.env)) { Copy-Item llm\.env.example llm\.env }
```

Vision defaults to fast local OpenCV region detection and TrOCR OCR, so it does
not need a Roboflow key or Workflow URL. TrOCR downloads its pretrained weights
from Hugging Face on first OCR use. Grounding DINO can be selected with
`DETECTOR_PROVIDER=grounding-dino`, but its CPU inference is much slower.
Set `GEMINI_API_KEY` in `llm/.env` for notes, quiz, and chat. Do not commit or
share these files. Roboflow is also optional: set `DETECTOR_PROVIDER=roboflow`,
`ROBOFLOW_API_KEY`, and `ROBOFLOW_WORKFLOW_URL` in `vision/.env` to use it.

From the repository root:

```powershell
docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000). The API is available on
port 5000 and the Vision/LLM health endpoints on ports 8000/8001. MongoDB,
uploaded images, and downloaded Hugging Face model caches use named volumes and
survive `docker compose down`. The initial Vision build and first inference can
take time because the CPU dependencies and pretrained weights are large. Check
container status with `docker compose ps` and service logs with
`docker compose logs -f backend vision llm`. Stop the stack with
`docker compose down`.

GitHub Actions runs the backend/frontend tests, Vision smoke checks, frontend
production build, and Docker Compose configuration validation on pushes and pull
requests. This is continuous integration only; GitHub Actions does not keep the
application hosted after a workflow completes. A public deployment still needs
a hosting provider, provider-managed secrets, and a persistent MongoDB/storage
configuration.

### 1. Run the Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

Sample cards are preset preview data, not live AI results. Uploading a photo uses the
backend API; if the backend is unavailable or a processing service fails, the
frontend shows an error instead of presenting simulated output as a successful
live run. The backend, Vision, and LLM services must be configured for a complete
upload pipeline.

### 2. Run the Express Backend
```bash
cd backend
npm install
# Copy .env.example to .env and configure service URLs / MongoDB if needed
npm run dev
```
Runs the orchestration API on [http://localhost:5000](http://localhost:5000). Without
`MONGODB_URI`, boards are kept in memory for local development and are lost when
the server restarts. Set `MONGODB_URI` to enable MongoDB persistence. Uploaded
images are stored in `backend/uploads` by default; production deployments need
a persistent disk or a configured `UPLOAD_DIR`.

### 3. Run the Vision & ML Service
```powershell
cd vision
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { New-Item -ItemType File -Path .env }
# Edit .env locally; never copy a private key into .env.example
python -m uvicorn main:app --port 8000 --reload
```

The Python virtual environment keeps ML packages isolated from other project
work. The default detector uses fast local OpenCV contour/line geometry and
text-line segmentation; TrOCR reads the detected text regions. This geometry
detector is not a trained ML model. To opt into the slower pretrained
`IDEA-Research/grounding-dino-tiny`, set
`DETECTOR_PROVIDER=grounding-dino` in `vision/.env`. To use Roboflow instead,
set `DETECTOR_PROVIDER=roboflow`, `ROBOFLOW_API_KEY`, and the deployed
`ROBOFLOW_WORKFLOW_URL`. TrOCR weights download from Hugging Face on first use
and are cached locally; no project-specific model has been trained. TrOCR
processes detected text blocks one line at a time. To select the smaller
official OCR checkpoint on memory-limited systems, set
`TROCR_MODEL_ID=microsoft/trocr-small-handwritten` in the local `.env`; its
accuracy has not been benchmarked for this project. Never commit `vision/.env`
or paste private API keys into chat.

The backend calls the vision service at `http://localhost:8000` and the language
service at `http://localhost:8001` by default. Configure `VISION_SERVICE_URL`
and `LLM_SERVICE_URL` in `backend/.env` when these services are hosted elsewhere.
If model loading or inference fails, `/vision/detect` returns a service error
instead of demo regions. Set `ALLOW_DEMO_DETECTOR_FALLBACK=true` only when
intentionally demonstrating fallback data. An image
upload returns a queued board, then the frontend polls `GET /api/boards/:id`
while the backend runs cleanup, detection, OCR, graph construction, grounded
notes, and quiz generation. A quality-gate rejection completes with
`isQualityPass: false` so the UI can ask for a retake. Service or pipeline
failures are reported with `status: "failed"` and an error; they are not
replaced by fabricated sample-board results.

The public API also includes `POST /api/boards/:id/quiz`, `POST /api/chat`,
`GET /api/boards/:id/export?format=md`, and `GET /api/health`. Run backend
contract tests with `cd backend && npm test`.

Check vision setup and run local tests from the active `vision` virtual
environment with `python -m unittest test_vision.py`. The previous hard-coded
OCR benchmark values have been removed. To measure TrOCR on actual held-out,
manually transcribed image crops, create a JSON manifest with `image`,
`ground_truth`, and optional normalized `box: [x, y, width, height]` fields and
run `python benchmark_ocr.py --manifest path\to\test-manifest.json`. These
Roboflow datasets contain region annotations, not OCR transcripts; they are
not evidence of measured handwriting accuracy.

### 4. Run the Language Service
```bash
cd llm
pip install -r requirements.txt
# Set GEMINI_API_KEY in this shell before starting the service
uvicorn main:app --port 8001 --reload
```

The language service requires `GEMINI_API_KEY` in `llm/.env`; copy
`llm/.env.example` first if the file does not exist. `GEMINI_MODEL_NAME` can
select a model available to your API account. Without a valid key, generation
returns a service error and the board job is marked `failed` instead of
returning unrelated sample content. The service uses Google's recommended
`google-genai` SDK.

### 5. Download an annotated whiteboard dataset

The downloader fetches two separate public Roboflow Universe datasets in YOLOv8
format, including their annotations and native train/valid/test splits:

- [Whiteboard](https://universe.roboflow.com/taerua/whiteboard-ctvgi), version
  1: 8,784 images, instance-segmentation classes `pen` and `whiteboard`.
- [Handwriting detection](https://universe.roboflow.com/handwriting-detection-zfowd/handwriting-detection-eoxpj),
  version 8: 361 images, object-detection class `handwriting`.

Both project pages list the dataset license as CC BY 4.0. Keep the source
attribution when using or redistributing the exports. These datasets detect
whiteboard regions and handwriting regions; the handwriting labels do not
provide transcribed text for OCR training.

In PowerShell, install the downloader dependency, set your Roboflow API key,
and run the downloader:

```powershell
pip install -r dataset\requirements.txt
$env:ROBOFLOW_API_KEY = "your-roboflow-api-key"
python dataset\download_samples.py
```

The exports are saved separately under `dataset\roboflow\whiteboard-v1` and
`dataset\roboflow\handwriting-v8`. `dataset\manifest.json` records the source,
license, and downloaded image counts. The API key is read from the environment
and is not written to the manifest. Use `--format` to select another export
format, `--output-dir` to choose a different destination root, or
`--overwrite` to replace existing exports.

---

## 📊 Evaluation & Benchmark Results
No quantitative results are claimed for the Roboflow exports yet. Evaluate each
task against its matching held-out split and report those measurements before
using them as benchmark results.

---

## 👥 Team Split & Credits
- **Moyeon**: Frontend Lead, UI/UX, Camera integration, Test Data curation & Demo Video.
- **Ammar**: Backend Lead, Pipeline Orchestrator, REST API, Deck & Submission lead.
- **Muavia**: Vision & ML Lead, OpenCV cleanup/region detection, TrOCR integration, optional Grounding DINO and Roboflow providers, Graph builder. No project-trained detector weights are included in this repository.
- **Awais**: Language service lead, evidence-based prompt engineering, quiz generation, quality review.

---
*Created for Banana Hacks 2026. All code and datasets created during the hackathon period (Oct 9–12, 2026).*
