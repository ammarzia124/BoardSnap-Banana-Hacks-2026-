# BoardSnap Implementation and Evaluation Status

**Last checked:** October 11, 2026
**Status:** Local code checks pass; live end-to-end processing and model accuracy are not verified.

## Evidence-based implementation status

| Component | What is in this checkout | Verification and limitation |
| --- | --- | --- |
| Frontend | React/Vite upload UI and API client | Production build passes. Uploaded images use the backend; sample cards are preset previews, not live inference. |
| Backend | Express API orchestrates cleanup, detection, OCR, graph, notes, and quiz stages | Four backend pipeline/API tests pass. The local run used in-memory storage; MongoDB persistence is now described in Docker Compose but has not been run here. |
| Image cleanup | OpenCV-based cleanup and quality scoring | Synthetic smoke checks pass. This is not a real-photo quality or OCR benchmark. |
| Region detector | Local OpenCV geometry/line heuristics by default; optional Grounding DINO or hosted Roboflow | Synthetic shape/connector tests pass. Real-photo performance and detection recall have not been measured; OpenCV heuristics are not a trained detector. |
| Handwriting OCR | Loads Microsoft's pretrained `microsoft/trocr-base-handwritten` checkpoint on demand | The project has not fine-tuned TrOCR. The last Vision health check reported the model unloaded; unit tests mock recognition. Real OCR output and accuracy are unverified. |
| Diagram graph | Spatial heuristic connects detected shapes and arrows and emits Mermaid | Synthetic graph tests pass; no labeled real-image graph benchmark is present. |
| Notes, quiz, and chat | Gemini-backed generation uses supplied board evidence | The last LLM health check reported no Gemini key configured. Live generation and grounding quality have not been verified. |

## Model training and dataset

- No project-trained YOLO/model weight files are tracked in this repository.
- No project-trained detector is included. Grounding DINO is an optional,
  upstream pretrained zero-shot model; Roboflow is also optional.
- TrOCR is an upstream pretrained checkpoint; there is no project fine-tuning
  code, training run, or trained checkpoint in this checkout.
- `dataset/manifest.json` records `not_downloaded`. The downloader's target
  datasets contain region annotations, not transcribed text for OCR training.
- No held-out image/transcription manifest or reproducible real-image
  evaluation is checked in.

**Verified model accuracy figures: none.** Previous accuracy, latency, dataset
size, and hallucination-rate figures in project materials lacked supporting
evaluation artifacts and have been removed. Do not present them as measured
results.

## Checks run

- Backend contract tests: 4 passed.
- Vision unit tests: 12 passed; external inference and OCR were mocked.
- Vision smoke checks: 4 passed; OCR was mocked.
- Frontend API failure tests: 3 passed.
- Frontend production build: passed, with a large JavaScript chunk warning.
- Docker Compose YAML structure parses. Docker itself was unavailable locally,
  so container builds and runtime health have not been verified.

## Required before claiming a complete live pipeline

1. Test the default OpenCV detector on representative real board images and
   measure region recall/precision. If using Grounding DINO or Roboflow instead,
   configure and benchmark the selected provider as well.
2. Configure a valid Gemini key locally and verify the selected model/SDK with
   notes, quiz, and chat requests.
3. Load TrOCR and test representative handwritten photos against manually
   transcribed ground truth; report CER/WER, sample count, and split protocol.
4. If claiming a trained detector, download and document the dataset version,
   train/fine-tune a model, preserve its reproducible configuration/checkpoint,
   and evaluate on an untouched labeled test split.
5. Run Docker Compose on a machine with Docker Desktop and exercise a real
   frontend upload through every backend and model stage.
