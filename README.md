# Experimental Project

AI-generated sample that performs OCR on classical Japanese documents using the models from [NDLkotenOCR-Lite](https://github.com/ndl-lab/ndlocr-lite).

## Pipeline

1. **Layout detection** — DEIM (DEIMv2) detects text-line regions, filtered by confidence threshold and class-agnostic NMS (IoU ≥ 0.2 among `line_*` classes only).
2. **Text recognition** — PARSeq reads each detected line. Vertically oriented lines are rotated 90° before inference.
