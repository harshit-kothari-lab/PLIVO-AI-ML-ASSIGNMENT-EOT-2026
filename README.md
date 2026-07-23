# End-of-Turn Detection

A causal, feature-engineered end-of-turn detection system designed to distinguish **thinking pauses** from **true conversational turn endings** using only acoustic information available before the pause.

The objective is to minimize response latency while maintaining an interrupted-turn rate below the assignment threshold.

---

## Overview

Unlike a fixed silence threshold, this approach models the conversational state preceding every pause. The system combines multiple prosodic and temporal cues—including speech dynamics, energy, pitch, pause context, and speaking rhythm—to estimate whether the current pause represents:

- **Continue Speaking** (thinking pause)
- **End of Turn** (safe for the assistant to respond)

The implementation is fully **causal** and uses **no future audio**, satisfying the assignment constraints.

---

## Design Philosophy

The model is built around three principles:

- **Causality First** – Every prediction uses only information available up to `pause_start`.
- **Generalization Over Memorization** – Features are designed to capture speaker-independent conversational patterns rather than dataset-specific heuristics.
- **Latency-Oriented Decision Making** – The goal is not merely classification accuracy, but minimizing response delay while respecting the interruption constraint.

---

## Repository Contents

| File | Description |
|------|-------------|
| `predict.py`  Main inference pipeline 
| `RUNLOG.md`  Development timeline and experiments 
| `NOTES.md`  Design decisions, assumptions, and implementation notes 
| `SUMMARY.html`  Executive summary of the solution 

---

## Running

```bash
python predict.py --data_dir <dataset_directory> --out predictions.csv
```

The script generates a prediction for every pause provided in the evaluation dataset.

---

## Technical Highlights

- Fully causal inference
- Acoustic feature engineering
- Prosodic and temporal modeling
- Speaker-independent decision logic
- No external language models
- Lightweight inference pipeline

---

## Acknowledgements

This solution was developed as part of the Plivo End-of-Turn Detection assignment. The implementation focuses on interpretable feature engineering and efficient inference while adhering to all assignment constraints.
