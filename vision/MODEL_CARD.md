# BoardSnap Vision Pipeline Model Card

Updated: October 11, 2026

## What the repository actually runs

1. **Image cleanup:** OpenCV contour-based board corner detection, perspective
   warp, glare inpainting, CLAHE contrast enhancement, and a Laplacian-variance
   quality score.
2. **Region detection:** OpenCV contour, line, and row-projection heuristics
   by default. It finds common closed diagram shapes, connectors between
   detected shapes, and standalone text lines without a model download. This
   is a geometry heuristic, not a trained detector. Optional providers are
   zero-shot `IDEA-Research/grounding-dino-tiny` (`DETECTOR_PROVIDER=grounding-dino`)
   and hosted Roboflow (`DETECTOR_PROVIDER=roboflow`).
3. **Handwriting recognition:** Hugging Face
   `microsoft/trocr-base-handwritten`, loaded on first OCR request. It is a
   pretrained model, not a model fine-tuned by this project. Detected text
   blocks are split into line crops because TrOCR is line-oriented; recognized
   lines are joined with line breaks. `TROCR_MODEL_ID` can select another
   compatible checkpoint, including Microsoft's smaller handwritten model.
4. **Diagram graph:** a spatial heuristic associates each arrow with the two
   nearest detected shapes and compiles a Mermaid flowchart.

## Dataset status and model claims

The dataset downloader targets Roboflow Universe exports in YOLOv8 format:

- `taerua/whiteboard-ctvgi`, version 1: classes `pen` and `whiteboard`.
- `handwriting-detection-zfowd/handwriting-detection-eoxpj`, version 8:
  class `handwriting`.

These are annotation datasets, not OCR transcriptions. The repository's
`dataset/manifest.json` records `not_downloaded`, so the exports and labels are
not present in this checkout. No project-trained detector weights or
ground-truth OCR evaluation corpus are present here.

**Verified accuracy figures: none.** Prior benchmark numbers and the earlier
OCR benchmark script were not backed by model inference over a labeled test
set. The current `benchmark_ocr.py` runs the configured TrOCR model over a
user-provided held-out manifest and reports measured predictions, WER, CER,
and latency. Do not report accuracy until that benchmark has been run against
real handwritten images with manually checked transcripts.

## Setup requirements

- The default OpenCV detector requires no Roboflow credentials, Workflow URL,
  or model download.
- Grounding DINO and TrOCR weights download from Hugging Face on first use and
  are cached for later runs.
- To opt into the hosted provider, set `DETECTOR_PROVIDER=roboflow`,
  `ROBOFLOW_API_KEY`, and the deployed Workflow inference URL in `vision/.env`.
- Install `vision/requirements.txt`.
- The detector provider and model load state are reported by `/health`.
- Model download or inference errors produce an explicit HTTP error rather than
  sample regions.
- OCR dependency/model errors produce an explicit HTTP error; they are not
  reported as successful empty OCR results.

## Known limitations

- OpenCV heuristics can miss shapes, arrows, and text on cluttered, faint, or
  perspective-distorted boards; evaluate them on representative board images.
- OpenCV `confidence` values are heuristic placeholders, not calibrated model
  probabilities.
- Grounding DINO is general-purpose open-vocabulary detection, not a
  whiteboard/flowchart-specific model. CPU inference can be slow; GPU is used
  when available.
- Roboflow results still depend on the externally deployed Workflow and its
  class mapping, which cannot be confirmed from this source tree.
- The two downloaded datasets have region labels, not transcribed handwriting.
- `microsoft/trocr-base-handwritten` is primarily intended for English
  handwriting. The line-splitting heuristic can miss tightly packed or
  low-contrast lines, and the recognizer may perform poorly on diagrams or
  non-Latin scripts.
- A detector's `handwriting` region is a crop proposal, not recognized text.
- The graph association is a nearest-centre heuristic; crossing and diagonal
  arrows can be connected incorrectly.
- The Laplacian quality score is a sharpness heuristic, not OCR confidence.

## API

All endpoints are served by the FastAPI application on port 8000 by default:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Reports detector provider/configuration/load state and OCR status |
| `POST /vision/clean` | Rectifies and scores an uploaded image |
| `POST /vision/detect` | Uses local OpenCV heuristics by default; optional Grounding DINO or Roboflow Workflow |
| `POST /vision/ocr` | Runs TrOCR on detected text/shape crops |
| `POST /vision/graph` | Builds nodes, edges, and Mermaid from regions |
