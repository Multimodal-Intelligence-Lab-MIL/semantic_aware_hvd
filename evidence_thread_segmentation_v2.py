import importlib.util
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def local_path(*parts: str) -> str:
    return os.path.join(SCRIPT_DIR, *parts)


FRAMEV2_PATH = local_path("framev2.py")


# =========================
# Manual config area
# =========================

CUDA_VISIBLE_DEVICES = "1"

FRAME_ROOT = local_path("Video_frames_CLIP")
TRANSCRIPT_ROOT = local_path("whisper_transcript")
OUTPUT_ROOT = local_path("outputs_evidence_thread_v2")

VIDEO_ID = ""
LIMIT = 10
OVERWRITE = True
DRY_RUN = False

# Visual unit stage. This keeps the framev2 unit-description pipeline, but it
# intentionally does not run framev2 adjacent-unit merge.
K = 10
S = 5
FPS = 1.0
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DEVICE_MAP = "auto"
TORCH_DTYPE = "auto"
ATTN_IMPLEMENTATION = None
MAX_NEW_TOKENS_FRAME_DESC = 1024
UNIT_DESCRIPTION_BATCH_SIZE = 1

# Thread stage.
MAX_NEW_TOKENS_THREAD_ASSIGNMENT = 4096
MAX_VISUAL_DESC_CHARS = 500
MAX_SENTENCE_TEXT_CHARS = 350


# =========================
# Data structures
# =========================


@dataclass
class TranscriptItem:
    sentence_id: int
    start_sec: float
    end_sec: float
    text: str
    gap_before: Optional[float] = None
    gap_before_bucket: str = "START"
    original_row_index: int = -1


@dataclass
class VisualUnit:
    unit_id: int
    start_sec: float
    end_sec: float
    description: str
    frame_filenames: List[str] = field(default_factory=list)


@dataclass
class EvidenceNode:
    node_id: str
    node_type: str
    start_sec: float
    end_sec: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceThread:
    thread_id: str
    label: str
    thread_description: str
    evidence_question: str
    node_ids: List[str]
    primary_node_ids: List[str] = field(default_factory=list)
    context_node_ids: List[str] = field(default_factory=list)
    thread_type: str = ""
    reasoning_focus: str = ""
    reason: str = ""


@dataclass
class NodeAssignment:
    node_id: str
    bucket: str
    thread_id: Optional[str] = None
    role: str = "none"
    source: str = "llm"


@dataclass
class TimelineCell:
    cell_id: str
    start_sec: float
    end_sec: float
    unit_id: Optional[int]
    node_ids: List[str]
    sentence_ids: List[int]
    primary_thread_id: Optional[str]
    secondary_thread_ids: List[str]
    role: str
    assignment_votes: Dict[str, int] = field(default_factory=dict)


# =========================
# Generic helpers
# =========================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def natural_sort_key(s: Any) -> List[Any]:
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def safe_json_loads(text: str) -> Optional[Any]:
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except Exception:
            pass

    obj = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if obj:
        try:
            return json.loads(obj.group(1).strip())
        except Exception:
            pass
    return None


def truncate_text(text: Any, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def format_time(start_sec: float, end_sec: float) -> str:
    return f"[{start_sec:.2f}-{end_sec:.2f}]"


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return float(a_end) > float(b_start) and float(b_end) > float(a_start)


def unique_preserve_order(values: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def uniform_sample_indices(n: int, sample_size: int) -> List[int]:
    if n <= 0 or sample_size <= 0:
        return []
    if sample_size >= n:
        return list(range(n))
    if sample_size == 1:
        return [n // 2]
    return sorted(
        {
            int(round(i * (n - 1) / (sample_size - 1)))
            for i in range(sample_size)
        }
    )


def classify_gap(gap: Optional[float]) -> str:
    if gap is None:
        return "START"
    if gap <= 0.2:
        return "none_or_overlap"
    if gap <= 0.8:
        return "short_pause"
    if gap <= 2.0:
        return "medium_pause"
    if gap <= 5.0:
        return "long_pause"
    if gap <= 8.0:
        return "very_long_pause"
    return "hard_pause"


def extract_summary_from_desc(desc_text: str) -> str:
    parsed = safe_json_loads(desc_text)
    if isinstance(parsed, dict):
        summary = parsed.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
    return truncate_text(desc_text, 500)


def load_framev2_module() -> Any:
    if not os.path.exists(FRAMEV2_PATH):
        raise FileNotFoundError(f"framev2.py not found: {FRAMEV2_PATH}")
    spec = importlib.util.spec_from_file_location("framev2_evidence_thread", FRAMEV2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import framev2.py from {FRAMEV2_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["framev2_evidence_thread"] = module
    spec.loader.exec_module(module)
    return module


# =========================
# Transcript loading
# =========================


def normalize_col(name: Any) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    mapping = {normalize_col(c): c for c in df.columns}
    for cand in candidates:
        if normalize_col(cand) in mapping:
            return mapping[normalize_col(cand)]
    return None


def transcript_path_for_video(video_id: str) -> str:
    return os.path.join(TRANSCRIPT_ROOT, f"{video_id}.xlsx")


def read_transcript(video_id: str) -> Tuple[List[TranscriptItem], List[str]]:
    path = transcript_path_for_video(video_id)
    warnings: List[str] = []
    if not os.path.exists(path):
        return [], [f"missing_transcript:{path}"]

    df = pd.read_excel(path)
    if df.empty:
        return [], ["empty_transcript_xlsx"]

    start_col = find_col(df, ["start", "segment_start", "begin"])
    end_col = find_col(df, ["end", "segment_end", "finish"])
    text_col = find_col(df, ["text", "transcript", "utterance", "content"])
    if start_col is None or end_col is None or text_col is None:
        return [], ["missing_required_transcript_columns"]

    rows: List[Tuple[float, float, str, int]] = []
    for row_idx, row in df.iterrows():
        try:
            start = float(row[start_col])
            end = float(row[end_col])
        except Exception:
            warnings.append(f"skip_row_{row_idx}:bad_time")
            continue
        text = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
        if not text:
            warnings.append(f"skip_row_{row_idx}:empty_text")
            continue
        if end < start:
            warnings.append(f"skip_row_{row_idx}:end_before_start")
            continue
        rows.append((start, end, text, int(row_idx)))

    rows.sort(key=lambda x: (x[0], x[1], x[3]))
    items: List[TranscriptItem] = []
    prev_end: Optional[float] = None
    for idx, (start, end, text, row_idx) in enumerate(rows, start=1):
        raw_gap = None if prev_end is None else start - prev_end
        gap = None if raw_gap is None else max(0.0, raw_gap)
        items.append(
            TranscriptItem(
                sentence_id=idx,
                start_sec=start,
                end_sec=end,
                text=text,
                gap_before=gap,
                gap_before_bucket=classify_gap(gap),
                original_row_index=row_idx,
            )
        )
        prev_end = end
    return items, warnings


def transcript_lines(items: List[TranscriptItem], max_chars: int = 5000) -> str:
    if not items:
        return "No transcript."
    lines = [
        f"S{item.sentence_id:03d} {format_time(item.start_sec, item.end_sec)} "
        f"gap={item.gap_before_bucket} text={truncate_text(item.text, MAX_SENTENCE_TEXT_CHARS)}"
        for item in items
    ]
    return truncate_text("\n".join(lines), max_chars)


def transcript_items_for_output(video_id: str, items: List[TranscriptItem]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for item in items:
        output.append(
            {
                "sentence_id": item.sentence_id,
                "video_id": video_id,
                "start": item.start_sec,
                "end": item.end_sec,
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "duration": max(0.0, item.end_sec - item.start_sec),
                "text": item.text,
                "time_pair": format_time(item.start_sec, item.end_sec),
                "original_row_index": item.original_row_index,
                "gap_before_raw": item.gap_before,
                "gap_before": item.gap_before,
                "gap_before_bucket": item.gap_before_bucket,
            }
        )
    return output


def sentences_starting_in_range(
    items: List[TranscriptItem],
    start_sec: float,
    end_sec: float,
    include_end: bool = False,
) -> List[TranscriptItem]:
    out: List[TranscriptItem] = []
    for item in items:
        if include_end:
            if start_sec <= item.start_sec <= end_sec:
                out.append(item)
        elif start_sec <= item.start_sec < end_sec:
            out.append(item)
    return out


# =========================
# Visual unit stage
# =========================


def video_dirs(framev2: Any) -> List[str]:
    if VIDEO_ID:
        return [os.path.join(FRAME_ROOT, VIDEO_ID)]
    dirs = framev2.list_video_dirs(FRAME_ROOT)
    if LIMIT and LIMIT > 0:
        dirs = dirs[:LIMIT]
    return dirs


def describe_video_units_with_sampling(
    framev2: Any,
    qwen: Any,
    units: List[Any],
    sample_size: int,
    save_every: int,
    unit_output_json_path: Optional[str],
    sample_output_json_path: Optional[str],
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    sample_records: List[Dict[str, Any]] = []
    batch_size = max(1, int(UNIT_DESCRIPTION_BATCH_SIZE))
    for batch_start in range(0, len(units), batch_size):
        batch_units = units[batch_start: batch_start + batch_size]
        original_values: List[Tuple[Any, List[str], List[str]]] = []
        batch_sample_counts: Dict[int, Tuple[int, int]] = {}

        for unit in batch_units:
            original_paths = list(unit.frame_paths)
            original_filenames = list(unit.frame_filenames)
            sample_indices = uniform_sample_indices(len(original_paths), min(sample_size, len(original_paths)))
            sampled_paths = [original_paths[i] for i in sample_indices]
            sampled_filenames = [original_filenames[i] for i in sample_indices]
            original_values.append((unit, original_paths, original_filenames))
            batch_sample_counts[unit.unit_id] = (len(sampled_paths), len(original_paths))
            sample_records.append(
                {
                    "unit_id": unit.unit_id,
                    "unit_time": [unit.start_time_sec, unit.end_time_sec],
                    "full_frame_count": len(original_paths),
                    "sample_size": len(sampled_paths),
                    "sample_indices_0based": sample_indices,
                    "sampled_frame_filenames": sampled_filenames,
                }
            )
            unit.frame_paths = sampled_paths
            unit.frame_filenames = sampled_filenames

        try:
            if DRY_RUN:
                descriptions = [
                    json.dumps(
                        {
                            "summary": (
                                f"Dry-run visual unit {unit.unit_id} covering "
                                f"{format_time(unit.start_time_sec, unit.end_time_sec)}."
                            ),
                            "notes_for_merging": ["model_not_called"],
                        },
                        ensure_ascii=False,
                    )
                    for unit in batch_units
                ]
            elif hasattr(qwen, "describe_units_batch"):
                descriptions = qwen.describe_units_batch(batch_units)
            else:
                descriptions = [qwen.describe_unit(unit.video_id, unit) for unit in batch_units]

            if len(descriptions) != len(batch_units):
                raise RuntimeError(
                    f"Unit-description batch size mismatch: inputs={len(batch_units)}, outputs={len(descriptions)}"
                )
            for unit, description in zip(batch_units, descriptions):
                unit.description = description
        finally:
            for unit, original_paths, original_filenames in original_values:
                unit.frame_paths = original_paths
                unit.frame_filenames = original_filenames

        for unit in batch_units:
            sampled_count, full_count = batch_sample_counts[unit.unit_id]
            print(
                f"[DescribeBatch] video={unit.video_id} unit={unit.unit_id} "
                f"time={framev2.format_time_range(unit.start_time_sec, unit.end_time_sec)} "
                f"sampled={sampled_count}/{full_count} batch_size={len(batch_units)}"
            )

        completed = batch_start + len(batch_units)
        crossed_save_boundary = (
            save_every > 0
            and completed // save_every > batch_start // save_every
        )
        if unit_output_json_path and (crossed_save_boundary or completed == len(units)):
            framev2.dump_units_json(units, unit_output_json_path)
        if sample_output_json_path and (crossed_save_boundary or completed == len(units)):
            write_json(sample_output_json_path, sample_records)

    return units, sample_records


def run_visual_units_in_memory(framev2: Any, qwen: Any, video_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    video_id = os.path.basename(video_dir.rstrip("/"))
    save_dir = os.path.join(OUTPUT_ROOT, video_id, "framev2_units")
    ensure_dir(save_dir)
    units_json = os.path.join(save_dir, "units.json")
    sample_json = os.path.join(save_dir, "frame_sampling.json")

    frames = framev2.list_frames_in_video_dir(video_dir)
    if len(frames) == 0:
        raise RuntimeError(f"No image frames found in {video_dir}")

    units = framev2.build_units_from_frames(video_id=video_id, frames=frames, k=K, fps=FPS)
    units, sample_records = describe_video_units_with_sampling(
        framev2=framev2,
        qwen=qwen,
        units=units,
        sample_size=S,
        save_every=5,
        unit_output_json_path=units_json,
        sample_output_json_path=sample_json,
    )

    readable_path = os.path.join(save_dir, "units_readable.txt")
    with open(readable_path, "w", encoding="utf-8") as f:
        f.write(f"video_id: {video_id}\n")
        f.write("=" * 100 + "\n\n")
        for unit in units:
            f.write(
                f"unit_id={unit.unit_id} "
                f"time={framev2.format_time_range(unit.start_time_sec, unit.end_time_sec)} "
                f"frame_order=({unit.start_frame_order},{unit.end_frame_order})\n"
            )
            f.write(f"{unit.description}\n\n")

    return [asdict(x) for x in units], sample_records


def build_visual_units(units_raw: List[Dict[str, Any]]) -> List[VisualUnit]:
    units: List[VisualUnit] = []
    for item in units_raw:
        units.append(
            VisualUnit(
                unit_id=int(item["unit_id"]),
                start_sec=float(item.get("start_time_sec", 0.0)),
                end_sec=float(item.get("end_time_sec", 0.0)),
                description=str(item.get("description", "")),
                frame_filenames=[str(x) for x in item.get("frame_filenames", [])],
            )
        )
    units.sort(key=lambda x: (x.start_sec, x.end_sec, x.unit_id))
    return units


# =========================
# Evidence node and thread stage
# =========================


def unit_node_id(unit_id: int) -> str:
    return f"U{unit_id:03d}"


def sentence_node_id(sentence_id: int) -> str:
    return f"S{sentence_id:03d}"


def build_evidence_nodes(units: List[VisualUnit], transcript: List[TranscriptItem]) -> List[EvidenceNode]:
    nodes: List[EvidenceNode] = []
    for unit in units:
        nodes.append(
            EvidenceNode(
                node_id=unit_node_id(unit.unit_id),
                node_type="visual_unit",
                start_sec=unit.start_sec,
                end_sec=unit.end_sec,
                content=truncate_text(extract_summary_from_desc(unit.description), MAX_VISUAL_DESC_CHARS),
                metadata={
                    "unit_id": unit.unit_id,
                    "frame_filenames": unit.frame_filenames,
                    "raw_description": unit.description,
                },
            )
        )
    for item in transcript:
        nodes.append(
            EvidenceNode(
                node_id=sentence_node_id(item.sentence_id),
                node_type="transcript_sentence",
                start_sec=item.start_sec,
                end_sec=item.end_sec,
                content=truncate_text(item.text, MAX_SENTENCE_TEXT_CHARS),
                metadata={
                    "sentence_id": item.sentence_id,
                    "gap_before": item.gap_before,
                    "gap_before_bucket": item.gap_before_bucket,
                    "original_row_index": item.original_row_index,
                },
            )
        )
    nodes.sort(key=lambda x: (x.start_sec, x.end_sec, x.node_type, natural_sort_key(x.node_id)))
    return nodes


def node_block(nodes: List[EvidenceNode], node_type: str) -> str:
    selected = [x for x in nodes if x.node_type == node_type]
    if not selected:
        return "None"
    lines = []
    for node in selected:
        extra = ""
        if node.node_type == "transcript_sentence":
            extra = f" gap={node.metadata.get('gap_before_bucket', 'unknown')}"
        lines.append(
            f"{node.node_id} {format_time(node.start_sec, node.end_sec)}{extra}\n"
            f"  {node.content}"
        )
    return "\n".join(lines)


def build_thread_assignment_prompt(video_id: str, nodes: List[EvidenceNode]) -> str:
    return f"""You are doing EVIDENCE THREADING for downstream video hate/non-hate reasoning.

Important: You are NOT segmenting the video and you must NOT output time segments.

Task:
Assign the smallest nodes below to evidence/context threads.

Definition:
- A thread is a set of visual/text nodes that should be remembered together as one coherent event, claim, narrative, referent, setup/callback, or reasoning context.
- A thread may include non-adjacent nodes.
- Same-thread adjacent nodes will later be merged by code.
- Same-thread non-adjacent nodes will later remain separate timeline segments with remote context links.
- A node does not need to prove hate/non-hate by itself to be threaded. It only needs to provide meaningful context that a downstream reasoner should remember.
- Nodes that are filler, empty visual context, silence, generic thanks/outro, unrelated movement, or unusable content should be background, transition, or unclear.

A good evidence thread usually shares at least one of:
- same target or referenced group/person/object
- same claim, insult, stereotype, or harmful/degrading expression
- same stance question, especially quote/rebuttal/satire/reporting vs endorsement
- same visual meme/text/symbol referent
- same setup and later callback that must be remembered together
- same reported incident, accusation, news-like narrative, or on-screen text story

Rules:
1. Every node id should appear exactly once, either in one thread's node_ids or in background_node_ids / transition_node_ids / unclear_node_ids.
2. Do not classify any thread as hate or non-hate.
3. Use primary_node_ids for nodes that directly contain the main target/claim/stance/visual evidence.
4. Use context_node_ids for setup, callback, title, transition-with-meaning, or clarifying nodes that belong to the thread but are not primary evidence.
5. It is acceptable to return zero threads only when there is genuinely no meaningful evidence or reasoning context in any node.

Field meanings:
- thread_description: describe what happens or what is communicated in this thread, using the visual/text evidence.
- evidence_question: the downstream reasoning question this thread can help answer. This is not a node; it is a reasoning lens.
- reasoning_focus: the short reasoning issue, such as reporting-vs-endorsement, target/reference, stereotype/claim, context-only, or unclear-stance.
- thread_type: a short category, such as reported_incident, direct_claim, visual_context, quote_or_report, rebuttal_or_satire, meme_or_symbol, background_context, or other.

Return STRICT JSON only:
{{
  "threads": [
    {{
      "thread_id": "T1",
      "label": "short label",
      "thread_description": "what happens or is communicated in this thread",
      "evidence_question": "what this thread helps the downstream reasoner judge",
      "reasoning_focus": "short reasoning issue",
      "thread_type": "short category",
      "node_ids": ["U001", "S003"],
      "primary_node_ids": ["S003"],
      "context_node_ids": ["U001"],
      "reason": "why these nodes belong together"
    }}
  ],
  "background_node_ids": ["U004"],
  "transition_node_ids": [],
  "unclear_node_ids": []
}}

Video id: {video_id}

VISUAL UNIT NODES:
{node_block(nodes, "visual_unit")}

TRANSCRIPT SENTENCE NODES:
{node_block(nodes, "transcript_sentence")}
"""


def normalize_id_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return unique_preserve_order(out)


def normalize_thread_id(raw: Any, idx: int) -> str:
    text = str(raw or "").strip().upper()
    if re.fullmatch(r"T\d+", text):
        return text
    return f"T{idx}"


def fallback_threads_by_unit(
    nodes: List[EvidenceNode],
    reason: str = "Model output could not be parsed; fallback preserves one visual unit and its starting transcript sentences.",
    warning: str = "fallback_threads_by_unit_used_parse_error",
) -> Tuple[List[EvidenceThread], Dict[str, NodeAssignment], List[str]]:
    warnings = [warning]
    threads: List[EvidenceThread] = []
    assignments: Dict[str, NodeAssignment] = {}
    visual_nodes = [x for x in nodes if x.node_type == "visual_unit"]
    sentence_nodes = [x for x in nodes if x.node_type == "transcript_sentence"]
    for idx, visual in enumerate(visual_nodes, start=1):
        thread_id = f"T{idx}"
        attached = [
            s.node_id for s in sentence_nodes
            if visual.start_sec <= s.start_sec < visual.end_sec
        ]
        node_ids = [visual.node_id] + attached
        threads.append(
            EvidenceThread(
                thread_id=thread_id,
                label=f"Fallback unit thread {idx}",
                thread_description=(
                    "Fallback thread created because the model thread assignment could not be used. "
                    "It preserves one visual unit and any transcript sentences that start inside it."
                ),
                evidence_question="What, if anything, does this fallback visual unit contribute to downstream reasoning?",
                node_ids=node_ids,
                primary_node_ids=node_ids,
                context_node_ids=[],
                thread_type="fallback_unit",
                reasoning_focus="fallback_preservation",
                reason=reason,
            )
        )
        for node_id in node_ids:
            assignments[node_id] = NodeAssignment(
                node_id=node_id,
                bucket="thread",
                thread_id=thread_id,
                role="primary",
                source="fallback",
            )

    for node in nodes:
        if node.node_id not in assignments:
            assignments[node.node_id] = NodeAssignment(
                node_id=node.node_id,
                bucket="background",
                thread_id=None,
                role="none",
                source="fallback_unassigned",
            )
    return threads, assignments, warnings


def parse_thread_assignment(
    parsed: Any,
    nodes: List[EvidenceNode],
) -> Tuple[List[EvidenceThread], Dict[str, NodeAssignment], List[str]]:
    if not isinstance(parsed, dict):
        return fallback_threads_by_unit(
            nodes,
            reason="Model thread assignment output was unavailable or could not be parsed as JSON.",
            warning="fallback_threads_by_unit_used_unparseable_output",
        )

    warnings: List[str] = []
    valid_node_ids = {x.node_id for x in nodes}
    threads: List[EvidenceThread] = []
    assignments: Dict[str, NodeAssignment] = {}

    raw_threads = parsed.get("threads", [])
    if not isinstance(raw_threads, list):
        warnings.append("invalid_threads_field_default_empty")
        raw_threads = []

    for idx, raw_thread in enumerate(raw_threads, start=1):
        if not isinstance(raw_thread, dict):
            continue
        thread_id = normalize_thread_id(raw_thread.get("thread_id"), idx)
        node_ids = [x for x in normalize_id_list(raw_thread.get("node_ids")) if x in valid_node_ids]
        if not node_ids:
            warnings.append(f"drop_empty_thread:{thread_id}")
            continue
        primary_node_ids = [x for x in normalize_id_list(raw_thread.get("primary_node_ids")) if x in node_ids]
        context_node_ids = [x for x in normalize_id_list(raw_thread.get("context_node_ids")) if x in node_ids]
        threads.append(
            EvidenceThread(
                thread_id=thread_id,
                label=str(raw_thread.get("label", "")).strip() or thread_id,
                thread_description=str(
                    raw_thread.get("thread_description")
                    or raw_thread.get("description")
                    or ""
                ).strip(),
                evidence_question=str(raw_thread.get("evidence_question", "")).strip(),
                node_ids=node_ids,
                primary_node_ids=primary_node_ids,
                context_node_ids=context_node_ids,
                thread_type=str(raw_thread.get("thread_type", "")).strip(),
                reasoning_focus=str(raw_thread.get("reasoning_focus", "")).strip(),
                reason=str(raw_thread.get("reason", "")).strip(),
            )
        )
        for node_id in node_ids:
            if node_id in assignments:
                warnings.append(f"duplicate_node_assignment_ignored:{node_id}:{thread_id}")
                continue
            role = "support"
            if node_id in primary_node_ids:
                role = "primary"
            elif node_id in context_node_ids:
                role = "context"
            assignments[node_id] = NodeAssignment(
                node_id=node_id,
                bucket="thread",
                thread_id=thread_id,
                role=role,
                source="llm_thread",
            )

    special_buckets = [
        ("background_node_ids", "background"),
        ("transition_node_ids", "transition"),
        ("unclear_node_ids", "unclear"),
    ]
    for key, bucket in special_buckets:
        for node_id in normalize_id_list(parsed.get(key)):
            if node_id not in valid_node_ids:
                continue
            if node_id in assignments:
                warnings.append(f"special_bucket_duplicate_ignored:{node_id}:{bucket}")
                continue
            assignments[node_id] = NodeAssignment(
                node_id=node_id,
                bucket=bucket,
                thread_id=None,
                role="none",
                source="llm_special_bucket",
            )

    for node_id in sorted(valid_node_ids, key=natural_sort_key):
        if node_id not in assignments:
            warnings.append(f"unassigned_node_default_background:{node_id}")
            assignments[node_id] = NodeAssignment(
                node_id=node_id,
                bucket="background",
                thread_id=None,
                role="none",
                source="repair_unassigned",
            )

    if not threads:
        warnings.append("model_returned_no_valid_threads_preserved_node_buckets")

    return threads, assignments, warnings


def assign_evidence_threads(
    qwen: Any,
    video_id: str,
    nodes: List[EvidenceNode],
) -> Tuple[List[EvidenceThread], Dict[str, NodeAssignment], Dict[str, Any]]:
    prompt = build_thread_assignment_prompt(video_id, nodes)
    if DRY_RUN:
        raw = json.dumps({"threads": [], "background_node_ids": [], "transition_node_ids": [], "unclear_node_ids": []})
        parsed = safe_json_loads(raw)
    else:
        raw = qwen.generate_from_messages(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            max_new_tokens=MAX_NEW_TOKENS_THREAD_ASSIGNMENT,
            do_sample=False,
            temperature=0.0,
        )
        parsed = safe_json_loads(raw)

    threads, assignments, warnings = parse_thread_assignment(parsed, nodes)
    artifact = {
        "prompt": prompt,
        "raw_output": raw,
        "parsed_output": parsed,
        "warnings": warnings,
    }
    return threads, assignments, artifact


# =========================
# Timeline materialization
# =========================


def build_timeline_cells(
    units: List[VisualUnit],
    transcript: List[TranscriptItem],
    assignments: Dict[str, NodeAssignment],
) -> List[TimelineCell]:
    cells: List[TimelineCell] = []
    assigned_sentence_ids = set()

    for idx, unit in enumerate(units, start=1):
        is_last = idx == len(units)
        sents = sentences_starting_in_range(transcript, unit.start_sec, unit.end_sec, include_end=is_last)
        assigned_sentence_ids.update(item.sentence_id for item in sents)
        node_ids = [unit_node_id(unit.unit_id)] + [sentence_node_id(item.sentence_id) for item in sents]
        cell = make_timeline_cell(
            cell_id=f"C{len(cells) + 1:03d}",
            start_sec=unit.start_sec,
            end_sec=unit.end_sec,
            unit_id=unit.unit_id,
            node_ids=node_ids,
            sentence_ids=[item.sentence_id for item in sents],
            assignments=assignments,
        )
        cells.append(cell)

    for item in transcript:
        if item.sentence_id in assigned_sentence_ids:
            continue
        cell = make_timeline_cell(
            cell_id=f"C{len(cells) + 1:03d}",
            start_sec=item.start_sec,
            end_sec=item.end_sec,
            unit_id=None,
            node_ids=[sentence_node_id(item.sentence_id)],
            sentence_ids=[item.sentence_id],
            assignments=assignments,
        )
        cells.append(cell)

    cells.sort(key=lambda x: (x.start_sec, x.end_sec, x.cell_id))
    for idx, cell in enumerate(cells, start=1):
        cell.cell_id = f"C{idx:03d}"
    return cells


def node_assignment_weight(node_id: str, assignment: NodeAssignment) -> int:
    if assignment.thread_id is None:
        return 0
    role_weight = {
        "primary": 4,
        "context": 2,
        "support": 1,
    }.get(assignment.role, 1)
    if node_id.startswith("S") and assignment.role in {"primary", "context"}:
        role_weight += 1
    return role_weight


def make_timeline_cell(
    cell_id: str,
    start_sec: float,
    end_sec: float,
    unit_id: Optional[int],
    node_ids: List[str],
    sentence_ids: List[int],
    assignments: Dict[str, NodeAssignment],
) -> TimelineCell:
    votes: Dict[str, int] = {}
    special_buckets: List[str] = []
    for node_id in node_ids:
        assignment = assignments.get(node_id)
        if assignment is None:
            continue
        if assignment.thread_id:
            votes[assignment.thread_id] = votes.get(assignment.thread_id, 0) + node_assignment_weight(node_id, assignment)
        else:
            special_buckets.append(assignment.bucket)

    primary_thread_id: Optional[str] = None
    secondary_thread_ids: List[str] = []
    role = "background"
    if votes:
        unit_assignment = assignments.get(unit_node_id(unit_id)) if unit_id is not None else None
        max_vote = max(votes.values())
        winners = [tid for tid, count in votes.items() if count == max_vote]
        winners.sort(key=natural_sort_key)
        if unit_assignment and unit_assignment.thread_id in winners:
            primary_thread_id = unit_assignment.thread_id
        else:
            primary_thread_id = winners[0]
        secondary_thread_ids = [tid for tid in sorted(votes, key=natural_sort_key) if tid != primary_thread_id]
        role = "thread"
    elif "transition" in special_buckets:
        role = "transition"
    elif "unclear" in special_buckets:
        role = "unclear"

    return TimelineCell(
        cell_id=cell_id,
        start_sec=start_sec,
        end_sec=end_sec,
        unit_id=unit_id,
        node_ids=unique_preserve_order(node_ids),
        sentence_ids=sentence_ids,
        primary_thread_id=primary_thread_id,
        secondary_thread_ids=secondary_thread_ids,
        role=role,
        assignment_votes=votes,
    )


def cell_key(cell: TimelineCell) -> str:
    if cell.primary_thread_id:
        return f"thread:{cell.primary_thread_id}"
    return f"role:{cell.role}"


def infer_anchor_modality(transcript_text: str, frame_descriptions: List[str]) -> str:
    has_text = bool(transcript_text.strip())
    has_visual = bool(frame_descriptions)
    if has_text and has_visual:
        return "both"
    if has_text:
        return "transcript"
    if has_visual:
        return "frame"
    return "unclear"


def materialize_segments(
    cells: List[TimelineCell],
    threads: List[EvidenceThread],
    assignments: Dict[str, NodeAssignment],
    units: List[VisualUnit],
    transcript: List[TranscriptItem],
) -> List[Dict[str, Any]]:
    if not cells:
        return []

    thread_by_id = {thread.thread_id: thread for thread in threads}
    unit_by_id = {unit.unit_id: unit for unit in units}
    sent_by_id = {item.sentence_id: item for item in transcript}

    groups: List[List[TimelineCell]] = []
    current: List[TimelineCell] = [cells[0]]
    current_key = cell_key(cells[0])
    for cell in cells[1:]:
        key = cell_key(cell)
        if key == current_key:
            current.append(cell)
            continue
        groups.append(current)
        current = [cell]
        current_key = key
    groups.append(current)

    segments: List[Dict[str, Any]] = []
    for idx, group in enumerate(groups, start=1):
        primary_thread_id = group[0].primary_thread_id
        role = group[0].role if primary_thread_id is None else "thread"
        node_ids = unique_preserve_order([node_id for cell in group for node_id in cell.node_ids])
        source_unit_ids = [
            int(cell.unit_id) for cell in group
            if cell.unit_id is not None
        ]
        sentence_ids = sorted(set(sid for cell in group for sid in cell.sentence_ids))
        sent_items = [sent_by_id[sid] for sid in sentence_ids if sid in sent_by_id]
        transcript_text = "\n".join(
            f"S{item.sentence_id:03d} {format_time(item.start_sec, item.end_sec)} {item.text}"
            for item in sent_items
        )
        frame_descriptions = [
            f"[U{unit_id:03d}] {extract_summary_from_desc(unit_by_id[unit_id].description)}"
            for unit_id in source_unit_ids
            if unit_id in unit_by_id
        ]
        thread = thread_by_id.get(primary_thread_id) if primary_thread_id else None
        node_roles = [
            {
                "node_id": node_id,
                "bucket": assignments[node_id].bucket,
                "thread_id": assignments[node_id].thread_id,
                "role": assignments[node_id].role,
            }
            for node_id in node_ids
            if node_id in assignments
        ]
        segment_thread_role = infer_segment_thread_role(primary_thread_id, node_ids, assignments)
        start_sec = float(group[0].start_sec)
        end_sec = float(group[-1].end_sec)
        cell_numeric_ids = [int(re.sub(r"\D", "", cell.cell_id) or "0") for cell in group]
        text_proposal_ids = [sentence_node_id(sid) for sid in sentence_ids]
        frame_proposal_ids = [unit_node_id(unit_id) for unit_id in source_unit_ids]
        segments.append(
            {
                "segment_id": idx,
                "segment_type": "evidence_thread_unit" if primary_thread_id else role,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": max(0.0, end_sec - start_sec),
                "candidate_fragment_ids": cell_numeric_ids,
                "text_proposal_ids": text_proposal_ids,
                "frame_proposal_ids": frame_proposal_ids,
                "cell_ids": [cell.cell_id for cell in group],
                "source_node_ids": node_ids,
                "source_unit_ids": source_unit_ids,
                "sentence_ids": sentence_ids,
                "primary_thread_id": primary_thread_id,
                "secondary_thread_ids": unique_preserve_order(
                    [tid for cell in group for tid in cell.secondary_thread_ids]
                ),
                "thread_label": thread.label if thread else "",
                "thread_description": thread.thread_description if thread else "",
                "thread_evidence_question": thread.evidence_question if thread else "",
                "thread_type": thread.thread_type if thread else "",
                "thread_reasoning_focus": thread.reasoning_focus if thread else "",
                "thread_reason": thread.reason if thread else "",
                "segment_thread_role": segment_thread_role,
                "remote_thread_context": [],
                "transcript": [asdict(item) for item in sent_items],
                "transcript_text": transcript_text,
                "frame_descriptions": frame_descriptions,
                "visual_description": " ".join(frame_descriptions),
                "anchor_modality": infer_anchor_modality(transcript_text, frame_descriptions),
                "cross_modal_signal": {
                    "has_signal": bool(primary_thread_id),
                    "signal_types": ["evidence_thread"] if primary_thread_id else [],
                    "has_contradiction": False,
                    "signals": [],
                },
                "node_thread_roles": node_roles,
            }
        )

    attach_remote_thread_context(segments)
    return segments


def infer_segment_thread_role(
    primary_thread_id: Optional[str],
    node_ids: List[str],
    assignments: Dict[str, NodeAssignment],
) -> str:
    if primary_thread_id is None:
        return "none"
    roles = [
        assignments[node_id].role
        for node_id in node_ids
        if node_id in assignments and assignments[node_id].thread_id == primary_thread_id
    ]
    if "primary" in roles:
        return "primary_evidence"
    if "context" in roles:
        return "thread_context"
    if roles:
        return "thread_support"
    return "thread"


def attach_remote_thread_context(segments: List[Dict[str, Any]]) -> None:
    by_thread: Dict[str, List[Dict[str, Any]]] = {}
    for segment in segments:
        thread_id = segment.get("primary_thread_id")
        if not thread_id:
            continue
        by_thread.setdefault(str(thread_id), []).append(segment)

    for thread_id, same_thread_segments in by_thread.items():
        if len(same_thread_segments) <= 1:
            continue
        for segment in same_thread_segments:
            links = []
            for other in same_thread_segments:
                if other["segment_id"] == segment["segment_id"]:
                    continue
                links.append(
                    {
                        "segment_id": other["segment_id"],
                        "thread_id": thread_id,
                        "relation": "same_evidence_thread_non_adjacent",
                        "time": [other["start_sec"], other["end_sec"]],
                        "thread_label": other.get("thread_label", ""),
                        "thread_description": other.get("thread_description", ""),
                        "remote_segment_thread_role": other.get("segment_thread_role", ""),
                    }
                )
            segment["remote_thread_context"] = links


# =========================
# Output
# =========================


def readable_output(
    video_id: str,
    threads: List[EvidenceThread],
    assignments: Dict[str, NodeAssignment],
    cells: List[TimelineCell],
    segments: List[Dict[str, Any]],
    warnings: List[str],
) -> str:
    lines = [
        "[Method] evidence threading v2: descriptive threads first, timeline materialization second",
        f"video_id: {video_id}",
        "",
        "[Threads]",
    ]
    for thread in threads:
        lines.append(
            f"{thread.thread_id} label={thread.label} nodes={thread.node_ids} "
            f"primary={thread.primary_node_ids} context={thread.context_node_ids}"
        )
        if thread.thread_type or thread.reasoning_focus:
            lines.append(
                f"  type={thread.thread_type or 'unknown'} "
                f"focus={thread.reasoning_focus or 'unknown'}"
            )
        if thread.thread_description:
            lines.append(f"  description={truncate_text(thread.thread_description, 320)}")
        if thread.evidence_question:
            lines.append(f"  question={truncate_text(thread.evidence_question, 260)}")
        if thread.reason:
            lines.append(f"  reason={truncate_text(thread.reason, 260)}")

    lines.extend(["", "[Special assignments]"])
    for node_id in sorted(assignments, key=natural_sort_key):
        assignment = assignments[node_id]
        if assignment.thread_id:
            continue
        lines.append(f"{node_id}: {assignment.bucket}")

    lines.extend(["", "[Timeline cells]"])
    for cell in cells:
        lines.append(
            f"{cell.cell_id} {format_time(cell.start_sec, cell.end_sec)} "
            f"unit={cell.unit_id} primary_thread={cell.primary_thread_id} "
            f"role={cell.role} nodes={cell.node_ids} votes={cell.assignment_votes}"
        )

    lines.extend(["", "[Final segments]"])
    for segment in segments:
        remote = segment.get("remote_thread_context", [])
        lines.append(
            f"Sg{segment['segment_id']:03d} {format_time(segment['start_sec'], segment['end_sec'])} "
            f"type={segment['segment_type']} thread={segment.get('primary_thread_id')} "
            f"thread_role={segment.get('segment_thread_role')} remote={[x['segment_id'] for x in remote]}"
        )
        if segment.get("thread_label"):
            lines.append(f"  thread_label={segment['thread_label']}")
        if segment.get("thread_description"):
            lines.append(f"  thread_description={truncate_text(segment['thread_description'], 320)}")
        if segment.get("sentence_ids"):
            lines.append(f"  sentences={segment['sentence_ids']}")
        if segment.get("transcript_text"):
            lines.append(f"  transcript={truncate_text(segment['transcript_text'], 500)}")
        if segment.get("frame_descriptions"):
            lines.append(f"  visual={truncate_text(' '.join(segment['frame_descriptions']), 500)}")

    if warnings:
        lines.extend(["", "[Warnings]"])
        lines.extend(warnings)
    return "\n".join(lines)


# =========================
# Main process
# =========================


def process_single_video(framev2: Any, qwen: Any, video_dir: str) -> None:
    video_id = os.path.basename(video_dir.rstrip("/"))
    save_dir = os.path.join(OUTPUT_ROOT, video_id)
    ensure_dir(save_dir)

    output_json = os.path.join(save_dir, "evidence_segments.json")
    if (not OVERWRITE) and os.path.exists(output_json):
        print(f"[Skip] {video_id} already threaded.")
        return

    warnings: List[str] = []
    transcript, transcript_warnings = read_transcript(video_id)
    warnings.extend(transcript_warnings)

    units_raw, sample_records = run_visual_units_in_memory(
        framev2=framev2,
        qwen=qwen,
        video_dir=video_dir,
    )
    units = build_visual_units(units_raw)
    if not units:
        raise RuntimeError(f"No visual units for {video_id}")

    nodes = build_evidence_nodes(units, transcript)
    threads, assignments, thread_artifact = assign_evidence_threads(
        qwen=qwen,
        video_id=video_id,
        nodes=nodes,
    )
    warnings.extend(thread_artifact.get("warnings", []))

    cells = build_timeline_cells(units, transcript, assignments)
    segments = materialize_segments(cells, threads, assignments, units, transcript)

    transcript_output = transcript_items_for_output(video_id, transcript)
    write_json(os.path.join(save_dir, "transcript_items.json"), transcript_output)
    write_json(os.path.join(save_dir, "frame_units.json"), units_raw)
    write_json(os.path.join(save_dir, "frame_sampling.json"), sample_records)
    write_json(os.path.join(save_dir, "evidence_nodes.json"), [asdict(x) for x in nodes])
    write_json(os.path.join(save_dir, "evidence_threads.json"), [asdict(x) for x in threads])
    write_json(os.path.join(save_dir, "node_assignments.json"), {k: asdict(v) for k, v in assignments.items()})
    write_json(os.path.join(save_dir, "timeline_cells.json"), [asdict(x) for x in cells])
    write_text(os.path.join(save_dir, "thread_assignment_prompt.txt"), thread_artifact["prompt"])
    write_text(os.path.join(save_dir, "raw_thread_assignment_output.txt"), thread_artifact["raw_output"])

    write_json(
        output_json,
        {
            "video_id": video_id,
            "method": "evidence_thread_v2_node_assignment_then_timeline_materialization",
            "framev2_path": FRAMEV2_PATH,
            "config": {
                "visual_unit_stage": {
                    "k": K,
                    "s": S,
                    "fps": FPS,
                    "model_name": MODEL_NAME,
                    "max_new_tokens_desc": MAX_NEW_TOKENS_FRAME_DESC,
                    "framev2_adjacent_merge_used": False,
                },
                "thread_stage": {
                    "max_new_tokens_thread_assignment": MAX_NEW_TOKENS_THREAD_ASSIGNMENT,
                    "thread_schema": "label_thread_description_evidence_question_reasoning_focus_thread_type_node_roles",
                    "materialization_rule": "weighted_node_role_votes_then_same_thread_adjacent_cells_merge_same_thread_non_adjacent_segments_link",
                    "empty_thread_policy": "preserve_model_special_assignments_no_unit_fallback",
                    "parse_failure_policy": "fallback_one_thread_per_visual_unit",
                    "unassigned_nodes": "background_segments",
                },
            },
            "warnings": warnings,
            "frame_units": units_raw,
            "transcript_items": transcript_output,
            "evidence_nodes": [asdict(x) for x in nodes],
            "evidence_threads": [asdict(x) for x in threads],
            "node_assignments": {k: asdict(v) for k, v in assignments.items()},
            "timeline_cells": [asdict(x) for x in cells],
            "thread_assignment": {
                "raw_output": thread_artifact["raw_output"],
                "parsed_output": thread_artifact["parsed_output"],
            },
            "segments": segments,
        },
    )
    write_text(
        os.path.join(save_dir, "evidence_segments_readable.txt"),
        readable_output(video_id, threads, assignments, cells, segments, warnings),
    )
    print(f"[Done] video={video_id}, threads={len(threads)}, segments={len(segments)}, output={save_dir}")


def main() -> None:
    if CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    ensure_dir(OUTPUT_ROOT)
    framev2 = load_framev2_module()
    qwen = None
    if not DRY_RUN:
        qwen = framev2.Qwen3VLPipeline(
            model_name=MODEL_NAME,
            device_map=DEVICE_MAP,
            torch_dtype=TORCH_DTYPE,
            attn_implementation=ATTN_IMPLEMENTATION,
            max_new_tokens_desc=MAX_NEW_TOKENS_FRAME_DESC,
            max_new_tokens_merge=MAX_NEW_TOKENS_THREAD_ASSIGNMENT,
        )

    dirs = video_dirs(framev2)
    print(f"[Info] Found {len(dirs)} video dirs")
    for idx, video_dir in enumerate(dirs, start=1):
        print("\n" + "=" * 120)
        print(f"[Video {idx}/{len(dirs)}] {video_dir}")
        print("=" * 120)
        try:
            process_single_video(framev2, qwen, video_dir)
        except Exception as exc:
            print(f"[Error] {video_dir}: {repr(exc)}")
            err_dir = os.path.join(OUTPUT_ROOT, os.path.basename(video_dir.rstrip("/")))
            ensure_dir(err_dir)
            write_json(os.path.join(err_dir, "error.json"), {"error": repr(exc)})


if __name__ == "__main__":
    main()
