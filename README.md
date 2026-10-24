# Semantic-Aware Organisation and Adaptive-Context Reasoning for Training-Free Hateful Video Detection

## Datasets

* [HateMM](https://github.com/hate-alert/HateMM)
* [MultiHateClip (MHC)](https://github.com/Social-AI-Studio/MultiHateClip)
* [DeHate](https://github.com/Multimodal-Intelligence-Lab-MIL/DeHate)

All datasets can be obtained from their respective GitHub repositories.

## Data Preprocessing

1. Extract video frames using **ffmpeg** or **OpenCV**:
   * Sample at **1 frame per second** and save each video as a folder of frames named from `frame_000000.jpg`.

2. Transcribe video audio into text using **Whisper**:
   * Save one `.xlsx` file per video with `start`, `end` (seconds) and `text` columns.

```
code/
├── Video_frames_CLIP/<video_id>/frame_000000.jpg ...
└── whisper_transcript/<video_id>.xlsx
```

## Repository Structure

```
.
├── code/          # Inference code
│   ├── code_multi_choice_context_reasoning_v5.py   # Main pipeline
│   ├── evidence_thread_segmentation_v2.py          # Semantic-aware temporal organisation
│   └── framev2.py                                  # Clip construction and description
├── data_split/    # Five-fold splits for HateMM, MHC-CN, MHC-EN and DeHate
└── README.md      # Documentation
```

## Usage

```bash
cd code
python code_multi_choice_context_reasoning_v5.py
```
