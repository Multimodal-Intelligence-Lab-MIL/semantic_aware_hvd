# all_data_code 使用说明

这个目录包含 HateMM 实验所需的主要代码、数据和辅助模块。代码中的默认路径已经改为基于当前文件夹自动解析，因此只要保持本目录内部结构不变，`all_data_code` 放到其他位置后也可以运行。

## 目录内容

- `Video_frames_CLIP/`: 视频抽帧数据。
- `whisper_transcript/`: Whisper 转写结果，按 `video_id.xlsx` 命名。
- `HateMM_datasplit_5folds.json`: 数据划分文件。
- `code_multi_choice_context_reasoning_v5_llava.py`: LLaVA 主实验脚本。
- `code_multi_choice_context_reasoning_v5_internvl35.py`: InternVL3.5 主实验脚本。
- `ETS_ablation.py`: ETS 与单模态消融实验脚本。
- `evidence_thread_segmentation_v2.py`: evidence-thread segmentation 辅助模块。
- `framev2.py`: frame-level unit segmentation 辅助模块。
- `LlaVA_Next_Video.py`: LLaVA 推理辅助模块。
- `result_hatemm.py`: 结果汇总/评估相关脚本。

## 推荐运行顺序

建议优先跑两个主实验：

```bash
cd /path/to/all_data_code
python code_multi_choice_context_reasoning_v5_llava.py
python code_multi_choice_context_reasoning_v5_internvl35.py
```

这两个脚本分别对应 LLaVA 和 InternVL3.5 的完整多模态实验。默认输入路径为当前目录下的 `Video_frames_CLIP/` 和 `whisper_transcript/`，默认输出分别写入：

- `HateMM_LLaVA_1_5_7B_K=10_S=5/`
- `HateMM_InternVL3_5_8B_HF_K=10_S=5/`

如需只测试少量样本，可在脚本顶部修改：

```python
LIMIT = 5
DRY_RUN = True
```

`DRY_RUN = True` 只用于检查流程和路径，不会真正加载大模型推理。

## ETS 与单模态消融实验

论文中的单一模态消融实验在 `ETS_ablation.py` 中完成。该脚本顶部有：

```python
ABLATION_MODE = "multimodal"
```

可选值包括：

- `"multimodal"`: 视觉 + 转写文本，完整多模态流程。
- `"visual_only"`: 只使用视觉模态，不读取或传入 transcript/title 文本。
- `"transcript_only"`: 只使用转写文本，不读取或传入视觉帧信息。

运行视觉单模态消融时，将其改为：

```python
ABLATION_MODE = "visual_only"
```

运行文本单模态消融时，将其改为：

```python
ABLATION_MODE = "transcript_only"
```

然后执行：

```bash
cd /path/to/all_data_code
python ETS_ablation.py
```

默认输出目录为：

```text
HateMM_Qwen3VL_2B_K=10_S=5/
```

建议每种消融模式使用不同的 `OUTPUT_ROOT`，避免不同实验结果互相覆盖。例如：

```python
OUTPUT_ROOT = local_path("HateMM_Qwen3VL_visual_only_K=10_S=5")
OUTPUT_ROOT = local_path("HateMM_Qwen3VL_transcript_only_K=10_S=5")
```

## 路径说明

这些脚本默认使用如下相对资源：

```python
INPUT_ROOT = local_path("Video_frames_CLIP")
TRANSCRIPT_ROOT = local_path("whisper_transcript")
SEGMENTER_PATH = local_path("evidence_thread_segmentation_v2.py")
FRAMEV2_PATH = local_path("framev2.py")
```

因此移动整个 `all_data_code` 文件夹时，不需要再修改 `/home/...` 或 `/imc-gpu01/...` 这类绝对路径。只要不要拆散目录内的数据和辅助模块即可。

## 输出文件

每个视频的结果通常会写入对应输出目录下的子文件夹，常见文件包括：

- `video_result.json`: 单视频最终预测结果。
- `evidence_segments.json`: ETS 分段结果。
- `evidence_segments_for_reasoning.json`: 供后续推理使用的结构化分段。
- `segment_reasoning.json`: 分段级推理结果。
- `readable_report.txt`: 便于人工检查的文本报告。
- `*_inference_time_report.md`: 推理耗时统计。

## 注意事项

- 首次正式运行会加载大模型，请确认当前环境已经安装对应模型依赖，并且 GPU 可用。
- 如需指定 GPU，可修改脚本顶部的 `CUDA_VISIBLE_DEVICES`。
- 如需只跑某一个视频，可修改 `VIDEO_ID`。
- 如需重新生成已有结果，可将 `OVERWRITE = True`。
- LLaVA 脚本依赖同目录下的 `LlaVA_Next_Video.py`，移动文件夹时需要一起保留。
