# STAR: Semantic-Aware Organisation and Adaptive-Context Reasoning for Training-Free Hateful Video Detection

Training-free hateful video detection with a frozen VLM (default: Qwen3-VL-8B-Instruct). STAR groups clip descriptions and transcript utterances into semantic segments, then labels each segment with adaptively selected context (none / temporal / same-group). A video is predicted as hateful if any segment is hateful.

> **Disclaimer:** this project deals with hateful content for research purposes.

## Files

| File | Purpose |
|---|---|
| `code_multi_choice_context_reasoning_v5.py` | Main pipeline (Qwen3-VL) |
| `code_multi_choice_context_reasoning_v5_internvl35.py` / `_llava.py` | Same pipeline with InternVL3.5 / LLaVA-1.5 |
| `evidence_thread_segmentation_v2.py` | Semantic-aware temporal organisation |
| `framev2.py` | Clip construction and clip descriptions |
| `ETS_ablation.py` | Modality ablations (`multimodal`, `visual_only`, `transcript_only`) |
| `LlaVA_Next_Video.py` | LLaVA helper module |
| `result_hatemm.py` | Evaluation |
| `HateMM_datasplit_5folds.json` | 5-fold split used for HateMM |

## Requirements

Python 3.10+, `torch`, `transformers`, `pandas`, `openpyxl`, `Pillow`, and a GPU.

## Data

Place the inputs next to the scripts:

```
Video_frames_CLIP/<video_id>/frame_000000.jpg ...   # frames at 1 fps, numbered from 0
whisper_transcript/<video_id>.xlsx                   # columns: start, end, text (seconds)
```

Videos and transcripts are not redistributed; obtain HateMM, MultiHateClip and DeHate from their original sources.

## Run

```bash
python code_multi_choice_context_reasoning_v5.py
```

Settings (`K=10`, `S=5`, model name, GPU, `LIMIT`, `OVERWRITE`, output directory) are constants at the top of each script. Each video produces a folder containing segments, segment-level reasoning, `video_result.json` and a readable report.

## License

MIT
