import os
import re
import json
import math
import time
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"


# =========================
# 数据结构
# =========================

@dataclass
class FrameInfo:
    path: str
    filename: str
    frame_order: int          # 在排序后的序列中位置（从1开始）
    frame_number: Optional[int]  # 从文件名中解析出的数字；解析不到则为None


@dataclass
class UnitInfo:
    unit_id: int
    video_id: str
    frame_paths: List[str]
    frame_filenames: List[str]
    start_frame_order: int
    end_frame_order: int
    start_frame_number: Optional[int]
    end_frame_number: Optional[int]
    start_time_sec: float
    end_time_sec: float
    description: str = ""


@dataclass
class MergeDecision:
    left_unit_id: int
    right_unit_id: int
    merge: bool
    reason: str


@dataclass
class MergedSegment:
    merged_segment_id: int
    video_id: str
    unit_ids: List[int]
    start_unit_id: int
    end_unit_id: int
    start_frame_order: int
    end_frame_order: int
    start_frame_number: Optional[int]
    end_frame_number: Optional[int]
    start_time_sec: float
    end_time_sec: float
    descriptions: List[str]
    merged_description: str


# =========================
# 工具函数
# =========================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def natural_sort_key(s: str):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]


def extract_first_int_from_filename(filename: str) -> Optional[int]:
    nums = re.findall(r"\d+", filename)
    if not nums:
        return None
    return int(nums[-1])


def list_video_dirs(root_dir: str) -> List[str]:
    subdirs = []
    for name in os.listdir(root_dir):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full):
            subdirs.append(full)
    subdirs.sort(key=natural_sort_key)
    return subdirs


def list_frames_in_video_dir(video_dir: str) -> List[FrameInfo]:
    files = [f for f in os.listdir(video_dir) if is_image_file(f)]
    files.sort(key=natural_sort_key)

    frames: List[FrameInfo] = []
    for idx, fname in enumerate(files, start=1):
        frames.append(
            FrameInfo(
                path=os.path.join(video_dir, fname),
                filename=fname,
                frame_order=idx,
                frame_number=extract_first_int_from_filename(fname)
            )
        )
    return frames


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def format_time_range(start_sec: float, end_sec: float) -> str:
    return f"({int(round(start_sec))}, {int(round(end_sec))})"


def safe_json_loads(text: str) -> Optional[Any]:
    text = text.strip()

    # 先直接试
    try:
        return json.loads(text)
    except Exception:
        pass

    # 再提取 ```json ... ```
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 再提取最外层 {} 或 []
    obj_match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if obj_match:
        candidate = obj_match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


# =========================
# Qwen3-VL 推理封装
# =========================

class Qwen3VLPipeline:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: Optional[str] = None,
        max_new_tokens_desc: int = 256,
        max_new_tokens_merge: int = 256,
        load_in_8bit: bool = False,
        load_in_8bit_cpu_offload: bool = False,
        offload_folder: Optional[str] = None,
        clear_cuda_cache_after_batch: bool = False,
    ):
        self.model_name = model_name
        self.max_new_tokens_desc = max_new_tokens_desc
        self.max_new_tokens_merge = max_new_tokens_merge
        self.load_in_8bit = load_in_8bit
        self.load_in_8bit_cpu_offload = load_in_8bit_cpu_offload
        self.clear_cuda_cache_after_batch = clear_cuda_cache_after_batch

        model_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "device_map": device_map,
        }

        if torch_dtype == "auto":
            model_kwargs["dtype"] = "auto"
        else:
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)

        if attn_implementation is not None and attn_implementation.strip():
            model_kwargs["attn_implementation"] = attn_implementation

        if offload_folder is not None and str(offload_folder).strip():
            ensure_dir(offload_folder)
            model_kwargs["offload_folder"] = offload_folder
            model_kwargs["offload_state_dict"] = True

        if load_in_8bit:
            if BitsAndBytesConfig is not None:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=load_in_8bit_cpu_offload,
                )
            else:
                model_kwargs["load_in_8bit"] = True
                if load_in_8bit_cpu_offload:
                    model_kwargs["llm_int8_enable_fp32_cpu_offload"] = True

        self.model = AutoModelForImageTextToText.from_pretrained(**model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _generate_batch_once(
        self,
        messages_batch: List[List[Dict[str, Any]]],
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> List[str]:
        # Decoder-only batch generation requires left padding. Each outer-list
        # item remains an independent conversation; no prompt content is shared.
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        inputs = self.processor.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )

        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature

        generated_ids = None
        generated_ids_trimmed = None
        try:
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return [text.strip() for text in output_texts]
        finally:
            del generated_ids_trimmed, generated_ids, inputs
            if self.clear_cuda_cache_after_batch and torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate_batch_from_messages(
        self,
        messages_batch: List[List[Dict[str, Any]]],
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> List[str]:
        """Generate a tensor batch, splitting it automatically after CUDA OOM."""
        if not messages_batch:
            return []
        try:
            return self._generate_batch_once(
                messages_batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
        except torch.OutOfMemoryError:
            if len(messages_batch) == 1:
                raise
            midpoint = len(messages_batch) // 2
        # Retry only after leaving the except block so the OOM traceback cannot
        # keep failed-generation tensors alive during the smaller retry.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[CudaOOM] Qwen batch_size={len(messages_batch)}; "
            f"retrying as {midpoint}+{len(messages_batch) - midpoint}"
        )
        return self.generate_batch_from_messages(
            messages_batch[:midpoint],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        ) + self.generate_batch_from_messages(
            messages_batch[midpoint:],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

    @torch.inference_mode()
    def generate_from_messages(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.0,
    ) -> str:
        """兼容单条调用，并复用 batch 生成实现。"""
        return self.generate_batch_from_messages(
            messages_batch=[messages],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )[0]

    def build_unit_description_messages(self, unit: UnitInfo) -> List[Dict[str, Any]]:
        """Build one independent visual-description conversation."""
        content = []

        for img_path in unit.frame_paths:
            content.append({
                "type": "image",
                "image": os.path.abspath(img_path)
            })

        prompt = f"""
You are given {len(unit.frame_paths)} chronologically ordered frames from the same short video unit.

Task:
1. Understand ONLY the visible content in these frames.
2. Write a concise but detailed description of what is consistently happening in this unit.
3. Focus on stable content shared across the frames. Do NOT speculate about unseen audio or future/past events.
4. Keep the description useful for later adjacent-unit merging.

Return STRICT JSON only with this schema:
{{
  "summary": "one concise paragraph",
  "scene": ["..."],
  "subjects": ["..."],
  "actions": ["..."],
  "tone": "...",
  "salient_objects": ["..."],
  "aggression_or_conflict_signal": "none / low / medium / high",
  "notes_for_merging": ["short phrases about what makes this unit continuous"]
}}

Requirements:
- summary must be short but informative.
- Prefer factual visual content.
- If uncertain, say "unclear" rather than inventing.
- Output JSON only.
""".strip()

        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def describe_unit(
        self,
        video_id: str,
        unit: UnitInfo,
    ) -> str:
        """对一个 unit 的多张图做简明但细致描述。"""
        return self.generate_from_messages(
            messages=self.build_unit_description_messages(unit),
            max_new_tokens=self.max_new_tokens_desc,
            do_sample=False,
            temperature=0.0,
        )

    def describe_units_batch(self, units: List[UnitInfo]) -> List[str]:
        """Describe units independently while computing them in one batch."""
        return self.generate_batch_from_messages(
            messages_batch=[self.build_unit_description_messages(unit) for unit in units],
            max_new_tokens=self.max_new_tokens_desc,
            do_sample=False,
            temperature=0.0,
        )

    def judge_adjacent_merge(
        self,
        left_unit: UnitInfo,
        right_unit: UnitInfo,
        left_desc: str,
        right_desc: str,
    ) -> MergeDecision:
        """
        只判断相邻 unit 是否可以合并。
        不让模型重写描述，只输出 merge / no_merge + reason。
        """
        prompt = f"""
You are deciding whether two ADJACENT video units should be merged into one larger segment.

Important principles:
1. Consider only whether they are likely to form part of the same continuous content.
2. Small camera motion or slight pose change is okay.
3. Units may be merged if the visual composition changes but the content remains the same.
4. The units are adjacent in time.

Left unit metadata:
- unit_id: {left_unit.unit_id}
- time_range_sec: [{left_unit.start_time_sec}, {left_unit.end_time_sec}]
- frame_order_range: [{left_unit.start_frame_order}, {left_unit.end_frame_order}]
- description_json:
{left_desc}

Right unit metadata:
- unit_id: {right_unit.unit_id}
- time_range_sec: [{right_unit.start_time_sec}, {right_unit.end_time_sec}]
- frame_order_range: [{right_unit.start_frame_order}, {right_unit.end_frame_order}]
- description_json:
{right_desc}

Return STRICT JSON only:
{{
  "merge": true,
  "reason": "short reason"
}}

or

{{
  "merge": false,
  "reason": "short reason"
}}

Output JSON only.
""".strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        raw = self.generate_from_messages(
            messages=messages,
            max_new_tokens=self.max_new_tokens_merge,
            do_sample=False,
            temperature=0.0,
        )

        parsed = safe_json_loads(raw)
        if isinstance(parsed, dict) and "merge" in parsed:
            merge_flag = bool(parsed["merge"])
            reason = str(parsed.get("reason", "")).strip()
            return MergeDecision(
                left_unit_id=left_unit.unit_id,
                right_unit_id=right_unit.unit_id,
                merge=merge_flag,
                reason=reason,
            )

        # 兜底解析
        lower = raw.lower()
        merge_flag = False
        if '"merge": true' in lower or re.search(r"\btrue\b", lower):
            merge_flag = True

        return MergeDecision(
            left_unit_id=left_unit.unit_id,
            right_unit_id=right_unit.unit_id,
            merge=merge_flag,
            reason=raw[:300].strip(),
        )


# =========================
# 主流程
# =========================

def build_units_from_frames(
    video_id: str,
    frames: List[FrameInfo],
    k: int,
    fps: float,
) -> List[UnitInfo]:
    if fps <= 0:
        raise ValueError("fps must be > 0")

    chunks = chunk_list(frames, k)
    units: List[UnitInfo] = []

    for unit_idx, chunk in enumerate(chunks, start=1):
        start_order = chunk[0].frame_order
        end_order = chunk[-1].frame_order

        # 帧文件从 frame_000000 开始编号，与 Whisper 转写一样从 0 秒起算。
        # unit 覆盖半开区间 [首帧时间, 末帧时间 + 1/fps)，相邻 unit 首尾相接、没有空隙。
        start_index = chunk[0].frame_number if chunk[0].frame_number is not None else start_order - 1
        end_index = chunk[-1].frame_number if chunk[-1].frame_number is not None else end_order - 1
        start_time_sec = float(start_index) / fps
        end_time_sec = float(end_index + 1) / fps

        units.append(
            UnitInfo(
                unit_id=unit_idx,
                video_id=video_id,
                frame_paths=[x.path for x in chunk],
                frame_filenames=[x.filename for x in chunk],
                start_frame_order=start_order,
                end_frame_order=end_order,
                start_frame_number=chunk[0].frame_number,
                end_frame_number=chunk[-1].frame_number,
                start_time_sec=start_time_sec,
                end_time_sec=end_time_sec,
            )
        )
    return units


def describe_video_units(
    qwen: Qwen3VLPipeline,
    units: List[UnitInfo],
    save_every: int = 10,
    unit_output_json_path: Optional[str] = None,
) -> List[UnitInfo]:
    for idx, unit in enumerate(units, start=1):
        desc = qwen.describe_unit(unit.video_id, unit)
        unit.description = desc

        print(
            f"[Describe] video={unit.video_id} unit={unit.unit_id} "
            f"time={format_time_range(unit.start_time_sec, unit.end_time_sec)}"
        )

        if unit_output_json_path and (idx % save_every == 0 or idx == len(units)):
            dump_units_json(units, unit_output_json_path)

    return units


def merge_adjacent_units(
    qwen: Qwen3VLPipeline,
    units: List[UnitInfo],
    save_merge_json_path: Optional[str] = None,
) -> Tuple[List[MergeDecision], List[MergedSegment]]:
    decisions: List[MergeDecision] = []

    if len(units) == 0:
        return decisions, []

    for i in range(len(units) - 1):
        left = units[i]
        right = units[i + 1]
        decision = qwen.judge_adjacent_merge(
            left_unit=left,
            right_unit=right,
            left_desc=left.description,
            right_desc=right.description,
        )
        decisions.append(decision)
        print(
            f"[MergeJudge] video={left.video_id} "
            f"{left.unit_id}->{right.unit_id}: {decision.merge} | {decision.reason}"
        )

    merged_segments: List[MergedSegment] = []
    current_group = [units[0]]

    for i, decision in enumerate(decisions):
        if decision.merge:
            current_group.append(units[i + 1])
        else:
            merged_segments.append(build_merged_segment(current_group, len(merged_segments) + 1))
            current_group = [units[i + 1]]

    if current_group:
        merged_segments.append(build_merged_segment(current_group, len(merged_segments) + 1))

    if save_merge_json_path:
        dump_merge_json(decisions, merged_segments, save_merge_json_path)

    return decisions, merged_segments


def build_merged_segment(group: List[UnitInfo], merged_segment_id: int) -> MergedSegment:
    merged_desc = " ".join(
        [f"[unit_{u.unit_id}]{extract_summary_from_desc(u.description)}" for u in group]
    ).strip()

    return MergedSegment(
        merged_segment_id=merged_segment_id,
        video_id=group[0].video_id,
        unit_ids=[u.unit_id for u in group],
        start_unit_id=group[0].unit_id,
        end_unit_id=group[-1].unit_id,
        start_frame_order=group[0].start_frame_order,
        end_frame_order=group[-1].end_frame_order,
        start_frame_number=group[0].start_frame_number,
        end_frame_number=group[-1].end_frame_number,
        start_time_sec=group[0].start_time_sec,
        end_time_sec=group[-1].end_time_sec,
        descriptions=[u.description for u in group],
        merged_description=merged_desc,
    )


def extract_summary_from_desc(desc_text: str) -> str:
    parsed = safe_json_loads(desc_text)
    if isinstance(parsed, dict):
        summary = parsed.get("summary", "")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
    return desc_text.strip().replace("\n", " ")[:300]


def dump_units_json(units: List[UnitInfo], path: str) -> None:
    serializable = []
    for u in units:
        item = asdict(u)
        serializable.append(item)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def dump_merge_json(
    decisions: List[MergeDecision],
    merged_segments: List[MergedSegment],
    path: str,
) -> None:
    data = {
        "merge_decisions": [asdict(x) for x in decisions],
        "merged_segments": [asdict(x) for x in merged_segments],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_single_video(
    qwen: Qwen3VLPipeline,
    video_dir: str,
    output_root: str,
    k: int,
    fps: float,
    overwrite: bool = False,
) -> None:
    video_id = os.path.basename(video_dir.rstrip("/"))
    save_dir = os.path.join(output_root, video_id)
    ensure_dir(save_dir)

    units_json_path = os.path.join(save_dir, "units.json")
    merge_json_path = os.path.join(save_dir, "merged_segments.json")

    if (not overwrite) and os.path.exists(units_json_path) and os.path.exists(merge_json_path):
        print(f"[Skip] {video_id} already done.")
        return

    frames = list_frames_in_video_dir(video_dir)
    if len(frames) == 0:
        print(f"[Warning] No image frames found in {video_dir}")
        return

    units = build_units_from_frames(video_id=video_id, frames=frames, k=k, fps=fps)
    units = describe_video_units(
        qwen=qwen,
        units=units,
        save_every=5,
        unit_output_json_path=units_json_path,
    )

    decisions, merged_segments = merge_adjacent_units(
        qwen=qwen,
        units=units,
        save_merge_json_path=merge_json_path,
    )

    # 再额外输出一个纯文本版，方便你直接看
    txt_path = os.path.join(save_dir, "merged_segments_readable.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"video_id: {video_id}\n")
        f.write("=" * 100 + "\n\n")

        f.write("UNITS\n")
        f.write("-" * 100 + "\n")
        for u in units:
            f.write(
                f"unit_id={u.unit_id} "
                f"time={format_time_range(u.start_time_sec, u.end_time_sec)} "
                f"frame_order=({u.start_frame_order},{u.end_frame_order})\n"
            )
            f.write(f"{u.description}\n\n")

        f.write("\nMERGE_DECISIONS\n")
        f.write("-" * 100 + "\n")
        for d in decisions:
            f.write(
                f"{d.left_unit_id}->{d.right_unit_id} "
                f"merge={d.merge} reason={d.reason}\n"
            )

        f.write("\nMERGED_SEGMENTS\n")
        f.write("-" * 100 + "\n")
        for seg in merged_segments:
            f.write(
                f"segment_id={seg.merged_segment_id} "
                f"units={seg.unit_ids} "
                f"time={format_time_range(seg.start_time_sec, seg.end_time_sec)}\n"
            )
            f.write(f"{seg.merged_description}\n\n")

    print(f"[Done] video={video_id}, output_dir={save_dir}")


def process_all_videos(
    input_root: str,
    output_root: str,
    k: int,
    fps: float,
    overwrite: bool,
    model_name: str,
    device_map: str,
    torch_dtype: str,
    attn_implementation: Optional[str],
    max_new_tokens_desc: int,
    max_new_tokens_merge: int,
) -> None:
    ensure_dir(output_root)

    qwen = Qwen3VLPipeline(
        model_name=model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        max_new_tokens_desc=max_new_tokens_desc,
        max_new_tokens_merge=max_new_tokens_merge,
    )

    video_dirs = list_video_dirs(input_root)
    print(f"[Info] Found {len(video_dirs)} video folders under {input_root}")

    for i, video_dir in enumerate(video_dirs, start=1):
        print("\n" + "=" * 120)
        print(f"[Video {i}/{len(video_dirs)}] {video_dir}")
        print("=" * 120)
        try:
            process_single_video(
                qwen=qwen,
                video_dir=video_dir,
                output_root=output_root,
                k=k,
                fps=fps,
                overwrite=overwrite,
            )
        except Exception as e:
            print(f"[Error] Failed on {video_dir}: {repr(e)}")


if __name__ == "__main__":
    # ✅ 强制使用 GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # ✅ 手动指定参数（替代 args）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_root = os.path.join(script_dir, "Video_frames_CLIP")
    output_root = os.path.join(script_dir, "segment_outputs_qwen3vl")
    k = 5
    fps = 1.0
    overwrite = False

    model_name = MODEL_NAME
    device_map = "auto"
    torch_dtype = "auto"
    attn_implementation = None
    max_new_tokens_desc = 256
    max_new_tokens_merge = 128

    process_all_videos(
        input_root=input_root,
        output_root=output_root,
        k=k,
        fps=fps,
        overwrite=overwrite,
        model_name=model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        max_new_tokens_desc=max_new_tokens_desc,
        max_new_tokens_merge=max_new_tokens_merge,
    )
