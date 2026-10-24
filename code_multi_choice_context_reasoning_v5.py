import importlib.util
import gc
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# Must be set before torch initializes CUDA. This reduces allocator
# fragmentation for batches whose padded sequence lengths vary substantially.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def local_path(*parts: str) -> str:
    return os.path.join(SCRIPT_DIR, *parts)


SEGMENTER_PATH = local_path("evidence_thread_segmentation_v2.py")


# =========================
# Manual config area
# =========================

CUDA_VISIBLE_DEVICES = "0"

INPUT_ROOT = local_path("Video_frames_CLIP")
TRANSCRIPT_ROOT = local_path("whisper_transcript")
OUTPUT_ROOT = local_path("HateMM_Qwen3VL_8B_K=10_S=5")
INFERENCE_TIME_REPORT_FILE = "hatemm_qwen3vl8b_inference_time_report.md"

VIDEO_ID = ""
LIMIT = 0
OVERWRITE = False
DRY_RUN = False

MODEL_NAME_TO_USE = "Qwen/Qwen3-VL-8B-Instruct" #Qwen/Qwen3-VL-30B-A3B-Instruct #Qwen/Qwen3-VL-235B-A22B-Instruct
DEVICE_MAP = "auto"
TORCH_DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "sdpa"

# Forwarded into evidence_thread_segmentation_v2.py.
K = 10
S = 5
FPS = 1.0
MAX_NEW_TOKENS_FRAME_DESC = 1024
MAX_NEW_TOKENS_THREAD_ASSIGNMENT = 4096
MAX_VISUAL_DESC_CHARS = 1024
MAX_SENTENCE_TEXT_CHARS = 1024

# Local forced reasoning.
# 1 approximates the old single representative-frame behavior.
# Values like 3 or 5 uniformly sample frames inside the target segment.
LOCAL_REPRESENTATIVE_FRAME_COUNT = 1
LOCAL_REPRESENTATIVE_FRAME_SAMPLING = "middle"  # "uniform" or "middle"
MAX_REPRESENTATIVE_IMAGES_PER_PROMPT = 6

# Context controls.
NEIGHBOR_CONTEXT_WINDOW = 1
MAX_THREAD_CONTEXT_SEGMENTS = 8
ENABLE_TITLE_REASONING = True

# Reasoning budgets.
MAX_NEW_TOKENS_LOCAL_REASON = 1024
MAX_NEW_TOKENS_CONTEXT_CHOICE = 384
MAX_NEW_TOKENS_CONTEXT_REASON = 1024
MAX_NEW_TOKENS_VIDEO_CONTEXT = 1024
MAX_NEW_TOKENS_FALLBACK = 256
MAX_NEW_TOKENS_TITLE_REASON = 256

# True tensor batching. Every request remains an independent conversation.
# Tune these values to fit the selected model and available GPU memory.
UNIT_DESCRIPTION_BATCH_SIZE = 32
SEGMENT_REASONING_BATCH_SIZE = 32
FALLBACK_REASONING_BATCH_SIZE = 32
CLEAR_CUDA_CACHE_AFTER_BATCH = False
EARLY_EXIT_ON_HATE = True


# =========================
# Data structures
# =========================


@dataclass
class EvidenceSegment:
    segment_id: int
    video_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    candidate_fragment_ids: List[int] = field(default_factory=list)
    text_proposal_ids: List[str] = field(default_factory=list)
    frame_proposal_ids: List[str] = field(default_factory=list)
    anchor_modality: str = "unclear"
    cross_modal_signal: Dict[str, Any] = field(default_factory=dict)
    primary_sentence_ids: List[int] = field(default_factory=list)
    boundary_sentence_ids: List[int] = field(default_factory=list)
    primary_transcript: str = ""
    boundary_context_not_primary: str = ""
    transcript_ownership_note: str = ""
    representative_frame_paths: List[str] = field(default_factory=list)
    representative_frame_filenames: List[str] = field(default_factory=list)
    representative_frame_path: str = ""
    representative_frame_filename: str = ""
    frame_descriptions: List[str] = field(default_factory=list)
    primary_thread_id: Optional[str] = None
    secondary_thread_ids: List[str] = field(default_factory=list)
    thread_label: str = ""
    thread_description: str = ""
    thread_evidence_question: str = ""
    thread_type: str = ""
    thread_reasoning_focus: str = ""
    thread_reason: str = ""
    segment_thread_role: str = "none"
    remote_thread_context: List[Dict[str, Any]] = field(default_factory=list)
    proposal_trace: Dict[str, Any] = field(default_factory=dict)
    node_thread_roles: List[Dict[str, Any]] = field(default_factory=list)
    raw_segment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalHypothesisResult:
    hypothesis_label: str
    evidence: List[str] = field(default_factory=list)
    target_clues: List[str] = field(default_factory=list)
    transcript_evidence: List[str] = field(default_factory=list)
    visual_evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5
    rationale: str = ""
    raw_model_output: str = ""


@dataclass
class LocalEvidenceResult:
    evidence_for_hate: List[str] = field(default_factory=list)
    evidence_against_hate: List[str] = field(default_factory=list)
    target_clues: List[str] = field(default_factory=list)
    transcript_evidence_for_hate: List[str] = field(default_factory=list)
    transcript_evidence_against_hate: List[str] = field(default_factory=list)
    visual_evidence_for_hate: List[str] = field(default_factory=list)
    visual_evidence_against_hate: List[str] = field(default_factory=list)
    multimodal_consistency_note: str = ""
    hate_hypothesis_confidence: float = 0.5
    nonhate_hypothesis_confidence: float = 0.5
    hate_confidence: float = 0.5
    rationale: str = ""
    raw_model_output: str = ""


@dataclass
class InitialJudgmentResult:
    verdict_label: str
    hate_confidence: float
    context_need: str = "none"
    rationale: str = ""
    raw_model_output: str = ""


@dataclass
class ContextChoice:
    context_type: str
    rationale: str
    context_types: List[str] = field(default_factory=list)
    missing_information_needed: str = ""
    available_context: Dict[str, bool] = field(default_factory=dict)
    resolved_context_type: str = ""
    resolved_context_types: List[str] = field(default_factory=list)
    resolution_note: str = ""
    raw_model_output: str = ""


@dataclass
class VideoContextResult:
    video_id: str
    objective_video_description: str = ""
    discourse_structure: str = ""
    stance_cues: str = ""
    limitations: str = ""
    raw_model_output: str = ""


@dataclass
class ContextReasoningStep:
    round_id: int
    context_type: str
    used_left_segment_ids: List[int] = field(default_factory=list)
    used_right_segment_ids: List[int] = field(default_factory=list)
    used_thread_segment_ids: List[int] = field(default_factory=list)
    used_video_context: bool = False
    verdict_label: str = "uncertain"
    hate_confidence: float = 0.5
    rationale: str = ""
    raw_model_output: str = ""


@dataclass
class FallbackVote:
    prompt_variant: str
    verdict_label: str
    raw_model_output: str
    rationale: str


@dataclass
class TitleReasoningResult:
    title_present: bool
    title_text: str
    verdict_label: str = "nonhate"
    rationale: str = ""
    raw_model_output: str = ""


@dataclass
class SegmentInferenceResult:
    merged_segment_id: int
    evidence_segment_id: int
    video_id: str
    representative_frame_path: str
    representative_frame_filename: str
    representative_frame_paths: List[str]
    representative_frame_filenames: List[str]
    time_range_sec: Tuple[float, float]
    unit_ids: List[int]
    merged_description: str
    transcript_text: str
    has_transcript: bool
    local_evidence: LocalEvidenceResult
    initial_judgment: InitialJudgmentResult
    context_choice: Optional[ContextChoice] = None
    context_steps: List[ContextReasoningStep] = field(default_factory=list)
    final_label_before_fallback: str = "uncertain"
    final_hate_confidence_before_fallback: float = 0.5
    used_context: bool = False
    used_context_type: str = ""
    used_neighbor_segment_ids: List[int] = field(default_factory=list)
    used_thread_segment_ids: List[int] = field(default_factory=list)
    context_available: bool = True
    context_skip_reason: str = ""
    final_rationale_before_fallback: str = ""
    fallback_votes: List[FallbackVote] = field(default_factory=list)
    fallback_label: Optional[str] = None
    final_label_after_fallback: Optional[str] = None
    final_rationale_after_fallback: str = ""
    anchor_modality: str = "unclear"
    cross_modal_signal: Dict[str, Any] = field(default_factory=dict)
    primary_sentence_ids: List[int] = field(default_factory=list)
    boundary_sentence_ids: List[int] = field(default_factory=list)
    boundary_context_not_primary: str = ""
    transcript_ownership_note: str = ""
    primary_thread_id: Optional[str] = None
    thread_label: str = ""
    thread_description: str = ""
    segment_thread_role: str = "none"


@dataclass
class VideoInferenceResult:
    video_id: str
    video_title: str
    final_video_label: str
    hate_source: str
    decision_rule: str
    hate_segment_ids_before_fallback: List[int]
    nonhate_segment_ids_before_fallback: List[int]
    uncertain_segment_ids_before_fallback: List[int]
    title_segment_ids_before_fallback: List[int]
    fallback_used: bool
    fallback_segment_ids: List[int]
    hate_segment_ids_after_fallback: List[int]
    nonhate_segment_ids_after_fallback: List[int]
    title_segment_ids_after_fallback: List[int]
    coarse_location_segment_ids: List[int]
    coarse_location_time_ranges: List[Tuple[float, float]] = field(default_factory=list)
    coarse_location_spans: List[Dict[str, float]] = field(default_factory=list)
    early_exit_triggered: bool = False
    early_exit_segment_id: Optional[int] = None
    unprocessed_segment_ids: List[int] = field(default_factory=list)
    localization_complete: bool = True
    title_reasoning: Optional[TitleReasoningResult] = None
    video_context: Optional[VideoContextResult] = None
    segment_results: List[SegmentInferenceResult] = field(default_factory=list)
    segmentation_method: str = "Evidence-thread segmentation v2"
    reasoning_method: str = "forced local + multi-context selection before judgment + single-pass selected-context final reasoning + fallback"


# =========================
# Utilities
# =========================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def release_cuda_memory(label: str = "") -> None:
    """Release unreachable tensors and cached allocator blocks at safe boundaries."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    if label:
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[CudaCleanup] {label} allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB")


def dump_json(data: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def truncate_text(text: Any, max_chars: int) -> str:
    s = "" if text is None else str(text)
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 3)] + "..."


def compact_text(text: Any, max_chars: int) -> str:
    return truncate_text(re.sub(r"\s+", " ", str(text or "")).strip(), max_chars)


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def clamp01(value: Any, default: float = 0.5) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x


def safe_json_loads(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", s, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    start_candidates = [i for i in [s.find("{"), s.find("[")] if i >= 0]
    end_candidates = [i for i in [s.rfind("}"), s.rfind("]")] if i >= 0]
    if start_candidates and end_candidates:
        start = min(start_candidates)
        end = max(end_candidates)
        if end > start:
            try:
                return json.loads(s[start: end + 1])
            except Exception:
                return None
    return None


def normalize_label(value: Any, allowed: Optional[set] = None, default: str = "uncertain") -> str:
    if allowed is None:
        allowed = {"hate", "nonhate", "uncertain"}
    s = re.sub(r"[\s_-]+", "", str(value or "").strip().lower())
    mapping = {
        "hate": "hate",
        "hateful": "hate",
        "nonhate": "nonhate",
        "nothate": "nonhate",
        "nothateful": "nonhate",
        "benign": "nonhate",
        "uncertain": "uncertain",
        "unsure": "uncertain",
        "unknown": "uncertain",
        "title": "title",
    }
    out = mapping.get(s, str(value or "").strip().lower())
    return out if out in allowed else default


ALLOWED_V5_CONTEXT_TYPES = ["no_context", "temporal_context", "semantic_thread_context"]


def normalize_context_type(value: Any, default: str = "no_context") -> str:
    s = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "none": "no_context",
        "no": "no_context",
        "local": "no_context",
        "target_only": "no_context",
        "no_context": "no_context",
        "temporal": "temporal_context",
        "neighbor": "temporal_context",
        "neighbor_context": "temporal_context",
        "temporal_context": "temporal_context",
        "semantic": "semantic_thread_context",
        "thread": "semantic_thread_context",
        "thread_context": "semantic_thread_context",
        "semantic_context": "semantic_thread_context",
        "semantic_thread": "semantic_thread_context",
        "semantic_thread_context": "semantic_thread_context",
    }
    return mapping.get(s, default)


def normalize_context_type_list(value: Any, default: Optional[List[str]] = None) -> List[str]:
    if default is None:
        default = ["no_context"]
    raw_items: List[Any] = []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, tuple):
        raw_items = list(value)
    elif value is not None:
        text = str(value).strip()
        if text:
            raw_items = [
                part.strip()
                for part in re.split(r"[,;/+|&]|\band\b|\bor\b", text, flags=re.IGNORECASE)
                if part.strip()
            ]

    normalized: List[str] = []
    for item in raw_items:
        context_type = normalize_context_type(item, "")
        if context_type in ALLOWED_V5_CONTEXT_TYPES and context_type not in normalized:
            normalized.append(context_type)
    return normalized or default[:]


def natural_sort_key(value: Any) -> List[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def format_time_range(start_sec: float, end_sec: float) -> str:
    return f"[{round(float(start_sec), 2)}-{round(float(end_sec), 2)}]"


def normalize_column_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lookup = {normalize_column_name(col): col for col in df.columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in lookup:
            return lookup[key]
    return None


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def list_frames_in_dir(frame_dir: str) -> List[str]:
    if not os.path.isdir(frame_dir):
        return []
    paths = [
        os.path.join(frame_dir, name)
        for name in os.listdir(frame_dir)
        if is_image_file(name)
    ]
    return sorted(paths, key=lambda p: natural_sort_key(os.path.basename(p)))


def frame_time_from_filename(path: str, fps: float = FPS) -> Optional[float]:
    nums = re.findall(r"\d+", os.path.basename(path))
    if not nums:
        return None
    frame_number = int(nums[-1])
    if fps <= 0:
        return float(frame_number)
    return float(frame_number) / fps


def frames_in_time_range(
    frame_paths: List[str],
    start_sec: float,
    end_sec: float,
    fps: float = FPS,
) -> List[str]:
    selected: List[str] = []
    for path in frame_paths:
        t = frame_time_from_filename(path, fps)
        if t is None:
            continue
        if float(start_sec) <= t <= float(end_sec):
            selected.append(path)
    return selected


def closest_frame_to_time(frame_paths: List[str], target_sec: float, fps: float = FPS) -> str:
    best_path = ""
    best_dist: Optional[float] = None
    for path in frame_paths:
        t = frame_time_from_filename(path, fps)
        if t is None:
            continue
        dist = abs(t - float(target_sec))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_path = path
    return best_path


def sample_representative_frames_for_time_range(
    frame_paths: List[str],
    start_sec: float,
    end_sec: float,
    sample_size: int,
    strategy: str,
    fps: float = FPS,
) -> List[str]:
    if not frame_paths or sample_size <= 0:
        return []
    candidates = frames_in_time_range(frame_paths, start_sec, end_sec, fps)
    if not candidates:
        midpoint = (float(start_sec) + float(end_sec)) / 2.0
        closest = closest_frame_to_time(frame_paths, midpoint, fps)
        return [closest] if closest else []
    if strategy == "middle" or sample_size == 1:
        midpoint = (float(start_sec) + float(end_sec)) / 2.0
        closest = closest_frame_to_time(candidates, midpoint, fps)
        return [closest] if closest else candidates[:1]
    capped = min(sample_size, MAX_REPRESENTATIVE_IMAGES_PER_PROMPT)
    return [candidates[i] for i in uniform_sample_indices(len(candidates), capped)]


def list_video_ids_from_frame_root(frame_root: str) -> List[str]:
    if not os.path.isdir(frame_root):
        return []
    return sorted(
        [
            name for name in os.listdir(frame_root)
            if os.path.isdir(os.path.join(frame_root, name))
        ],
        key=natural_sort_key,
    )


def list_video_ids_from_transcript_root(transcript_root: str) -> List[str]:
    if not os.path.isdir(transcript_root):
        return []
    ids = []
    for name in os.listdir(transcript_root):
        if name.lower().endswith(".xlsx"):
            ids.append(os.path.splitext(name)[0])
    return sorted(ids, key=natural_sort_key)


def list_video_ids(frame_root: str, transcript_root: str, video_id: str, limit: int) -> List[str]:
    if video_id.strip():
        return [video_id.strip()]
    ids = list_video_ids_from_frame_root(frame_root)
    if not ids:
        ids = list_video_ids_from_transcript_root(transcript_root)
    if limit and limit > 0:
        return ids[:limit]
    return ids


def read_video_title(transcript_root: str, video_id: str) -> str:
    path = os.path.join(transcript_root, f"{video_id}.xlsx")
    if not os.path.exists(path):
        return ""
    try:
        df = pd.read_excel(path)
    except Exception:
        return ""
    if df.empty:
        return ""
    title_col = find_column(df, ["video title", "videotitle", "title"])
    if not title_col:
        return ""
    seen = set()
    for value in df[title_col].tolist():
        title = str(value).strip()
        if title and title.lower() != "nan" and title not in seen:
            return title
        seen.add(title)
    return ""


# =========================
# Evidence-thread segmentation v2
# =========================


def load_segmenter_module() -> Any:
    if not os.path.exists(SEGMENTER_PATH):
        raise FileNotFoundError(f"Missing segmenter: {SEGMENTER_PATH}")
    spec = importlib.util.spec_from_file_location("evidence_thread_segmentation_v2_imported", SEGMENTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import segmenter from {SEGMENTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["evidence_thread_segmentation_v2_imported"] = module
    spec.loader.exec_module(module)
    return module


def configure_segmenter(
    segmenter: Any,
    frame_root: str,
    transcript_root: str,
    output_root: str,
    overwrite: bool,
    dry_run: bool,
    model_name: str,
) -> None:
    segmenter.FRAME_ROOT = frame_root
    segmenter.TRANSCRIPT_ROOT = transcript_root
    segmenter.OUTPUT_ROOT = output_root
    segmenter.OVERWRITE = overwrite
    segmenter.DRY_RUN = dry_run
    segmenter.K = K
    segmenter.S = S
    segmenter.FPS = FPS
    segmenter.MODEL_NAME = model_name
    segmenter.MAX_NEW_TOKENS_FRAME_DESC = MAX_NEW_TOKENS_FRAME_DESC
    segmenter.UNIT_DESCRIPTION_BATCH_SIZE = UNIT_DESCRIPTION_BATCH_SIZE
    segmenter.MAX_NEW_TOKENS_THREAD_ASSIGNMENT = MAX_NEW_TOKENS_THREAD_ASSIGNMENT
    segmenter.MAX_VISUAL_DESC_CHARS = MAX_VISUAL_DESC_CHARS
    segmenter.MAX_SENTENCE_TEXT_CHARS = MAX_SENTENCE_TEXT_CHARS


def run_evidence_thread_segmentation(
    segmenter: Any,
    framev2: Any,
    qwen: Any,
    video_id: str,
    frame_root: str,
    output_root: str,
    overwrite: bool,
) -> Dict[str, Any]:
    save_dir = os.path.join(output_root, video_id)
    ensure_dir(save_dir)
    output_json = os.path.join(save_dir, "evidence_segments.json")
    if (not overwrite) and os.path.exists(output_json):
        print(f"[EvidenceThreadSkip] video={video_id} already has evidence_segments.json.")
        return read_json(output_json)

    video_dir = os.path.join(frame_root, video_id)
    segmenter.process_single_video(framev2=framev2, qwen=qwen, video_dir=video_dir)
    return read_json(output_json)


# =========================
# Segment normalization and formatting
# =========================


def transcript_lines_from_raw(raw_seg: Dict[str, Any]) -> str:
    raw_text = str(raw_seg.get("transcript_text", "")).strip()
    if raw_text:
        return raw_text
    lines: List[str] = []
    for raw in raw_seg.get("transcript", []):
        if not isinstance(raw, dict):
            continue
        sid = raw.get("sentence_id", "")
        start = safe_float(raw.get("start_sec", raw.get("start")), 0.0) or 0.0
        end = safe_float(raw.get("end_sec", raw.get("end")), start) or start
        text = str(raw.get("text", "")).strip()
        if text:
            try:
                sid_text = f"S{int(sid):03d}"
            except Exception:
                sid_text = f"S{sid}"
            lines.append(f"{sid_text} {format_time_range(start, end)} {text}")
    return "\n".join(lines)


def normalize_evidence_thread_segments(
    data: Dict[str, Any],
    frame_root: str,
) -> List[EvidenceSegment]:
    video_id = str(data.get("video_id", ""))
    frame_dir = os.path.join(frame_root, video_id)
    frames = list_frames_in_dir(frame_dir)
    out: List[EvidenceSegment] = []

    for idx, raw_seg in enumerate(data.get("segments", []), start=1):
        if not isinstance(raw_seg, dict):
            continue
        start = float(raw_seg.get("start_sec", 0.0))
        end = float(raw_seg.get("end_sec", start))
        rep_frames = sample_representative_frames_for_time_range(
            frame_paths=frames,
            start_sec=start,
            end_sec=end,
            sample_size=LOCAL_REPRESENTATIVE_FRAME_COUNT,
            strategy=LOCAL_REPRESENTATIVE_FRAME_SAMPLING,
            fps=FPS,
        )
        representative = rep_frames[len(rep_frames) // 2] if rep_frames else ""

        frame_descriptions = [
            str(x).strip()
            for x in raw_seg.get("frame_descriptions", [])
            if str(x).strip()
        ]
        if not frame_descriptions and str(raw_seg.get("visual_description", "")).strip():
            frame_descriptions = [str(raw_seg.get("visual_description", "")).strip()]

        thread_id_raw = raw_seg.get("primary_thread_id")
        primary_thread_id = str(thread_id_raw) if thread_id_raw is not None else None

        out.append(
            EvidenceSegment(
                segment_id=int(raw_seg.get("segment_id", idx)),
                video_id=video_id,
                start_sec=start,
                end_sec=end,
                duration_sec=float(raw_seg.get("duration_sec", max(0.0, end - start))),
                candidate_fragment_ids=[int(x) for x in raw_seg.get("candidate_fragment_ids", [])],
                text_proposal_ids=[str(x) for x in raw_seg.get("text_proposal_ids", [])],
                frame_proposal_ids=[str(x) for x in raw_seg.get("frame_proposal_ids", [])],
                anchor_modality=str(raw_seg.get("anchor_modality", "unclear")),
                cross_modal_signal=raw_seg.get("cross_modal_signal", {}) or {},
                primary_sentence_ids=[int(x) for x in raw_seg.get("sentence_ids", [])],
                primary_transcript=transcript_lines_from_raw(raw_seg),
                representative_frame_paths=rep_frames,
                representative_frame_filenames=[os.path.basename(x) for x in rep_frames],
                representative_frame_path=representative,
                representative_frame_filename=os.path.basename(representative) if representative else "",
                frame_descriptions=frame_descriptions,
                primary_thread_id=primary_thread_id,
                secondary_thread_ids=[str(x) for x in raw_seg.get("secondary_thread_ids", [])],
                thread_label=str(raw_seg.get("thread_label", "")),
                thread_description=str(raw_seg.get("thread_description", "")),
                thread_evidence_question=str(raw_seg.get("thread_evidence_question", "")),
                thread_type=str(raw_seg.get("thread_type", "")),
                thread_reasoning_focus=str(raw_seg.get("thread_reasoning_focus", "")),
                thread_reason=str(raw_seg.get("thread_reason", "")),
                segment_thread_role=str(raw_seg.get("segment_thread_role", "none")),
                remote_thread_context=raw_seg.get("remote_thread_context", []) or [],
                proposal_trace={
                    "segmenter": "evidence_thread_segmentation_v2",
                    "segment_type": raw_seg.get("segment_type", ""),
                    "cell_ids": raw_seg.get("cell_ids", []),
                    "source_node_ids": raw_seg.get("source_node_ids", []),
                    "source_unit_ids": raw_seg.get("source_unit_ids", []),
                    "primary_thread_id": raw_seg.get("primary_thread_id"),
                    "secondary_thread_ids": raw_seg.get("secondary_thread_ids", []),
                    "thread_label": raw_seg.get("thread_label", ""),
                    "thread_description": raw_seg.get("thread_description", ""),
                    "thread_evidence_question": raw_seg.get("thread_evidence_question", ""),
                    "thread_type": raw_seg.get("thread_type", ""),
                    "thread_reasoning_focus": raw_seg.get("thread_reasoning_focus", ""),
                    "thread_reason": raw_seg.get("thread_reason", ""),
                    "segment_thread_role": raw_seg.get("segment_thread_role", ""),
                    "remote_thread_context": raw_seg.get("remote_thread_context", []),
                },
                node_thread_roles=raw_seg.get("node_thread_roles", []) or [],
                raw_segment=raw_seg,
            )
        )
    return out


def visual_descriptions_block(segment: EvidenceSegment, max_chars: int = 2600) -> str:
    if not segment.frame_descriptions:
        return "<none>"
    return truncate_text("\n".join(f"- {desc}" for desc in segment.frame_descriptions), max_chars)


def cross_modal_signal_brief(signal: Dict[str, Any]) -> str:
    if not signal:
        return "<none>"
    return truncate_text(json.dumps(signal, ensure_ascii=False), 900)


def format_target_segment_local(segment: EvidenceSegment, max_chars: int = 6200) -> str:
    block = f"""
segment_id: {segment.segment_id}
time_range_sec: {format_time_range(segment.start_sec, segment.end_sec)}
anchor_modality: {segment.anchor_modality}
sampled_representative_frames: {segment.representative_frame_filenames if segment.representative_frame_filenames else []}

primary_transcript:
{segment.primary_transcript if segment.primary_transcript else "<none>"}

visual_evidence_descriptions:
{visual_descriptions_block(segment)}

cross_modal_signal:
{cross_modal_signal_brief(segment.cross_modal_signal)}
""".strip()
    return truncate_text(block, max_chars)


def summarize_segment_for_context(segment: EvidenceSegment, max_chars: int = 1000) -> str:
    transcript = compact_text(segment.primary_transcript, 360)
    visual = compact_text(" ".join(segment.frame_descriptions), 500)
    return truncate_text(
        (
            f"segment_id={segment.segment_id}; time={format_time_range(segment.start_sec, segment.end_sec)}; "
            f"transcript={transcript or '<none>'}; visual={visual or '<none>'}"
        ),
        max_chars,
    )


def evidence_segment_compat_description(segment: EvidenceSegment) -> str:
    visual = compact_text(" ".join(segment.frame_descriptions), 800)
    transcript = compact_text(segment.primary_transcript, 500)
    return (
        f"Evidence-thread segment {segment.segment_id} {format_time_range(segment.start_sec, segment.end_sec)}. "
        f"Transcript: {transcript or '<none>'}. Visual: {visual or '<none>'}."
    )


def same_thread_segments(segment: EvidenceSegment, all_segments: List[EvidenceSegment]) -> List[EvidenceSegment]:
    if not segment.primary_thread_id:
        return []
    return [
        s for s in all_segments
        if s.primary_thread_id == segment.primary_thread_id
    ]


def get_temporal_context(
    segments: List[EvidenceSegment],
    idx: int,
    window: int = NEIGHBOR_CONTEXT_WINDOW,
) -> Tuple[List[EvidenceSegment], List[EvidenceSegment]]:
    if window <= 0:
        return [], []
    left = segments[max(0, idx - window):idx]
    right = segments[idx + 1: idx + 1 + window]
    return left, right


def context_availability(
    segment: EvidenceSegment,
    segments: List[EvidenceSegment],
    idx: int,
) -> Dict[str, bool]:
    left, right = get_temporal_context(segments, idx)
    has_thread = bool(segment.primary_thread_id)
    return {
        "no_context": True,
        "temporal_context": bool(left or right),
        "semantic_thread_context": has_thread,
    }


def resolve_context_choice(choice: ContextChoice, available: Dict[str, bool]) -> ContextChoice:
    requested = choice.context_types or normalize_context_type_list(choice.context_type)
    resolved: List[str] = []
    unavailable: List[str] = []
    for context_type in requested:
        if available.get(context_type, False):
            resolved.append(context_type)
        else:
            unavailable.append(context_type)

    if not resolved and available.get("no_context", False):
        resolved = ["no_context"]

    choice.context_types = requested
    choice.resolved_context_types = resolved
    choice.resolved_context_type = "+".join(resolved)
    if resolved and not unavailable:
        choice.resolution_note = "used_model_choices"
    elif resolved:
        choice.resolution_note = f"dropped_unavailable:{unavailable}; used:{resolved}"
    else:
        choice.resolution_note = f"no_context_available_for_model_choices:{requested}"
    return choice


# =========================
# Reasoner
# =========================


class MultiChoiceContextReasoner:
    def __init__(
        self,
        llm: Any,
        dry_run: bool,
        max_new_tokens_local_reason: int,
        max_new_tokens_context_choice: int,
        max_new_tokens_context_reason: int,
        max_new_tokens_video_context: int,
        max_new_tokens_fallback: int,
        max_new_tokens_title_reason: int,
    ):
        self.llm = llm
        self.dry_run = dry_run
        self.max_new_tokens_local_reason = max_new_tokens_local_reason
        self.max_new_tokens_context_choice = max_new_tokens_context_choice
        self.max_new_tokens_context_reason = max_new_tokens_context_reason
        self.max_new_tokens_video_context = max_new_tokens_video_context
        self.max_new_tokens_fallback = max_new_tokens_fallback
        self.max_new_tokens_title_reason = max_new_tokens_title_reason
        self.video_context_cache: Dict[str, VideoContextResult] = {}

    def generate_text(self, prompt: str, max_new_tokens: int, dry_payload: Optional[Dict[str, Any]] = None) -> str:
        if self.dry_run or self.llm is None:
            return json.dumps(dry_payload or {"verdict_label": "uncertain", "rationale": "dry_run"}, ensure_ascii=False)
        return self.llm.generate_from_messages(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    def generate_text_batch(
        self,
        prompts: List[str],
        max_new_tokens: int,
        dry_payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        if not prompts:
            return []
        if self.dry_run or self.llm is None:
            payloads = dry_payloads or [
                {"verdict_label": "uncertain", "rationale": "dry_run"}
                for _ in prompts
            ]
            return [json.dumps(payload, ensure_ascii=False) for payload in payloads]

        messages_batch = [
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            for prompt in prompts
        ]
        if hasattr(self.llm, "generate_batch_from_messages"):
            outputs = self.llm.generate_batch_from_messages(
                messages_batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        else:
            outputs = [
                self.llm.generate_from_messages(
                    messages,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
                for messages in messages_batch
            ]
        if len(outputs) != len(prompts):
            raise RuntimeError(f"Text batch size mismatch: inputs={len(prompts)}, outputs={len(outputs)}")
        return outputs

    def generate_with_representative_frames(
        self,
        prompt: str,
        frame_paths: List[str],
        max_new_tokens: int,
        dry_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self.dry_run or self.llm is None:
            return json.dumps(dry_payload or {"verdict_label": "uncertain", "rationale": "dry_run"}, ensure_ascii=False)
        content: List[Dict[str, Any]] = []
        for path in frame_paths[:MAX_REPRESENTATIVE_IMAGES_PER_PROMPT]:
            if path and os.path.exists(path):
                content.append({"type": "image", "image": os.path.abspath(path)})
        content.append({"type": "text", "text": prompt})
        return self.llm.generate_from_messages(
            [{"role": "user", "content": content}],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    def generate_batch_with_representative_frames(
        self,
        prompts: List[str],
        frame_paths_batch: List[List[str]],
        max_new_tokens: int,
        dry_payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        if len(prompts) != len(frame_paths_batch):
            raise ValueError("prompts and frame_paths_batch must have the same length")
        if not prompts:
            return []
        if self.dry_run or self.llm is None:
            payloads = dry_payloads or [
                {"verdict_label": "uncertain", "rationale": "dry_run"}
                for _ in prompts
            ]
            return [json.dumps(payload, ensure_ascii=False) for payload in payloads]

        messages_batch: List[List[Dict[str, Any]]] = []
        for prompt, frame_paths in zip(prompts, frame_paths_batch):
            content: List[Dict[str, Any]] = []
            for path in frame_paths[:MAX_REPRESENTATIVE_IMAGES_PER_PROMPT]:
                if path and os.path.exists(path):
                    content.append({"type": "image", "image": os.path.abspath(path)})
            content.append({"type": "text", "text": prompt})
            messages_batch.append([{"role": "user", "content": content}])

        if hasattr(self.llm, "generate_batch_from_messages"):
            outputs = self.llm.generate_batch_from_messages(
                messages_batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        else:
            outputs = [
                self.llm.generate_from_messages(
                    messages,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                )
                for messages in messages_batch
            ]
        if len(outputs) != len(prompts):
            raise RuntimeError(f"Multimodal batch size mismatch: inputs={len(prompts)}, outputs={len(outputs)}")
        return outputs

    def build_local_hate_prompt(self, segment: EvidenceSegment) -> str:
        prompt = f"""
You are performing LOCAL ADVERSARIAL REASONING under a FORCED HATE hypothesis.

Assumption:
- Assume the TARGET segment contains hateful content.

Definition:
- "hate" means the TARGET segment itself contains visible or transcript evidence that likely expresses,
  promotes, glorifies, threatens, dehumanizes, excludes, or directly targets a protected group or identity
  with hateful hostility, slurs, stereotypes, or violence.

Important rules:
1. In this step, you MUST reason under the assumption that the TARGET segment is hateful.
2. Your task is NOT to be balanced here.
3. Extract the strongest evidence that could support a hateful interpretation of THIS TARGET SEGMENT.
4. Use only the TARGET segment itself: sampled representative frames, primary transcript, visual descriptions, and cross-modal signal.
5. Do NOT use neighbor segments, remote same-thread context, thread description, full video context, or video title.
6. confidence means how strongly the hateful interpretation is supported under this forced hypothesis.

TARGET segment:
{format_target_segment_local(segment)}

Return STRICT JSON only:
{{
  "evidence": ["..."],
  "target_clues": ["..."],
  "transcript_evidence": ["..."],
  "visual_evidence": ["..."],
  "confidence": 0.50,
  "rationale": "brief explanation of the strongest hate-supporting interpretation"
}}
""".strip()
        return prompt

    def reason_segment_local_hate(self, segment: EvidenceSegment) -> LocalHypothesisResult:
        raw = self.generate_with_representative_frames(
            self.build_local_hate_prompt(segment),
            segment.representative_frame_paths,
            self.max_new_tokens_local_reason,
            {
                "evidence": [],
                "target_clues": [],
                "transcript_evidence": [],
                "visual_evidence": [],
                "confidence": 0.5,
                "rationale": "dry_run forced hate",
            },
        )
        return parse_local_hypothesis_output(raw, "hate")

    def build_local_nonhate_prompt(self, segment: EvidenceSegment) -> str:
        prompt = f"""
You are performing LOCAL ADVERSARIAL REASONING under a FORCED NON-HATE hypothesis.

Assumption:
- Assume the TARGET segment does NOT contain hateful content.

Important rules:
1. In this step, you MUST reason under the assumption that the TARGET segment is not hateful.
2. Your task is NOT to be balanced here.
3. Extract the strongest evidence supporting a non-hateful interpretation of THIS TARGET SEGMENT.
4. Consider benign explanations such as neutral reporting, quotation, satire, artistic usage,
   interpersonal conflict without protected-group targeting, or insufficient evidence of hateful intent.
5. Use only the TARGET segment itself: sampled representative frames, primary transcript, visual descriptions, and cross-modal signal.
6. Do NOT use neighbor segments, remote same-thread context, thread description, full video context, or video title.
7. confidence means how strongly the non-hateful interpretation is supported under this forced hypothesis.

TARGET segment:
{format_target_segment_local(segment)}

Return STRICT JSON only:
{{
  "evidence": ["..."],
  "target_clues": ["..."],
  "transcript_evidence": ["..."],
  "visual_evidence": ["..."],
  "confidence": 0.50,
  "rationale": "brief explanation of the strongest non-hate-supporting interpretation"
}}
""".strip()
        return prompt

    def reason_segment_local_nonhate(self, segment: EvidenceSegment) -> LocalHypothesisResult:
        raw = self.generate_with_representative_frames(
            self.build_local_nonhate_prompt(segment),
            segment.representative_frame_paths,
            self.max_new_tokens_local_reason,
            {
                "evidence": [],
                "target_clues": [],
                "transcript_evidence": [],
                "visual_evidence": [],
                "confidence": 0.5,
                "rationale": "dry_run forced nonhate",
            },
        )
        return parse_local_hypothesis_output(raw, "nonhate")

    def combine_local_hypotheses(
        self,
        segment: EvidenceSegment,
        hate_result: LocalHypothesisResult,
        nonhate_result: LocalHypothesisResult,
    ) -> LocalEvidenceResult:
        multimodal_note = (
            f"hate_hypothesis_confidence={hate_result.confidence:.2f}; "
            f"nonhate_hypothesis_confidence={nonhate_result.confidence:.2f}; "
            f"sampled_frames={segment.representative_frame_filenames}; "
            f"anchor_modality={segment.anchor_modality}; "
            f"cross_modal_signal={cross_modal_signal_brief(segment.cross_modal_signal)}"
        )
        return LocalEvidenceResult(
            evidence_for_hate=hate_result.evidence,
            evidence_against_hate=nonhate_result.evidence,
            target_clues=list(dict.fromkeys(hate_result.target_clues + nonhate_result.target_clues)),
            transcript_evidence_for_hate=hate_result.transcript_evidence,
            transcript_evidence_against_hate=nonhate_result.transcript_evidence,
            visual_evidence_for_hate=hate_result.visual_evidence,
            visual_evidence_against_hate=nonhate_result.visual_evidence,
            multimodal_consistency_note=multimodal_note,
            hate_hypothesis_confidence=hate_result.confidence,
            nonhate_hypothesis_confidence=nonhate_result.confidence,
            hate_confidence=hate_result.confidence,
            rationale=f"[HATE] {hate_result.rationale} [NONHATE] {nonhate_result.rationale}".strip(),
            raw_model_output=json.dumps(
                {
                    "hate_hypothesis": asdict(hate_result),
                    "nonhate_hypothesis": asdict(nonhate_result),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    def reason_segments_local_batch(self, segments: List[EvidenceSegment]) -> List[LocalEvidenceResult]:
        if not segments:
            return []
        hate_payload = {
            "evidence": [],
            "target_clues": [],
            "transcript_evidence": [],
            "visual_evidence": [],
            "confidence": 0.5,
            "rationale": "dry_run forced hate",
        }
        nonhate_payload = {
            "evidence": [],
            "target_clues": [],
            "transcript_evidence": [],
            "visual_evidence": [],
            "confidence": 0.5,
            "rationale": "dry_run forced nonhate",
        }
        frame_paths_batch = [segment.representative_frame_paths for segment in segments]
        hate_outputs = self.generate_batch_with_representative_frames(
            [self.build_local_hate_prompt(segment) for segment in segments],
            frame_paths_batch,
            self.max_new_tokens_local_reason,
            [hate_payload.copy() for _ in segments],
        )
        nonhate_outputs = self.generate_batch_with_representative_frames(
            [self.build_local_nonhate_prompt(segment) for segment in segments],
            frame_paths_batch,
            self.max_new_tokens_local_reason,
            [nonhate_payload.copy() for _ in segments],
        )
        return [
            self.combine_local_hypotheses(
                segment,
                parse_local_hypothesis_output(hate_raw, "hate"),
                parse_local_hypothesis_output(nonhate_raw, "nonhate"),
            )
            for segment, hate_raw, nonhate_raw in zip(segments, hate_outputs, nonhate_outputs)
        ]

    def reason_segment_local(self, segment: EvidenceSegment) -> LocalEvidenceResult:
        hate_result = self.reason_segment_local_hate(segment)
        nonhate_result = self.reason_segment_local_nonhate(segment)
        return self.combine_local_hypotheses(segment, hate_result, nonhate_result)

    def build_context_choice_prompt(
        self,
        segment: EvidenceSegment,
        local_evidence: LocalEvidenceResult,
        available: Dict[str, bool],
    ) -> str:
        prompt = f"""
You are choosing the context policy for one TARGET segment before final hate attribution.

TARGET segment, local only:
{format_target_segment_local(segment, max_chars=3600)}

Local forced evidence:
{json.dumps(asdict(local_evidence), ensure_ascii=False, indent=2)}

Available context classes:
{json.dumps(available, ensure_ascii=False, indent=2)}


Context choices:
1. temporal_context
   - Provides only nearby segments before/after the target in time.
   - Useful for sentence continuation, local visual transition, immediate setup or consequence.
   - It will provide neighbor segment visual descriptions and transcripts.

2. semantic_thread_context
   - Provides the evidence-thread semantic context from segment division.
   - Useful when the target depends on a repeated referent, setup/callback, quotation/reporting-vs-endorsement,
     same-thread transcript, or thread-level evidence question.
   - It will provide thread description/metadata plus transcripts and visual descriptions inside the same thread.

3. no_context
   - Provides no additional context.
   - The final judgment will use only the TARGET segment and the local forced evidence.
   - Use it when the TARGET segment itself already contains enough information for final attribution.

Task:
- Choose one or more context policies for the next final judgment step.
- Select both temporal_context and semantic_thread_context if both immediate continuity and same-thread semantic evidence are needed.
- Do not output full_video_context; it is disabled in this V5 pipeline.
- Do not classify the target in this step.
- Do not choose a context class marked unavailable unless all options are unavailable.

Return STRICT JSON only:
{{
  "valid_context_types": "choose one or more from: no_context, temporal_context, semantic_thread_context",
  "context_types": ["temporal_context", "semantic_thread_context"],
  "rationale": "brief reason for choosing these context classes"
}}
""".strip()
        return prompt

    def choose_context_type(
        self,
        segment: EvidenceSegment,
        local_evidence: LocalEvidenceResult,
        available: Dict[str, bool],
    ) -> ContextChoice:
        raw = self.generate_text(
            self.build_context_choice_prompt(segment, local_evidence, available),
            self.max_new_tokens_context_choice,
            {
                "context_types": ["no_context"],
                "rationale": "dry_run context choice",
            },
        )
        choice = parse_context_choice_output(raw, available)
        return resolve_context_choice(choice, available)

    def choose_context_types_batch(
        self,
        segments: List[EvidenceSegment],
        local_evidences: List[LocalEvidenceResult],
        availabilities: List[Dict[str, bool]],
    ) -> List[ContextChoice]:
        if not (len(segments) == len(local_evidences) == len(availabilities)):
            raise ValueError("Context-choice batch inputs must have the same length")
        outputs = self.generate_text_batch(
            [
                self.build_context_choice_prompt(segment, local_evidence, available)
                for segment, local_evidence, available in zip(segments, local_evidences, availabilities)
            ],
            self.max_new_tokens_context_choice,
            [
                {"context_types": ["no_context"], "rationale": "dry_run context choice"}
                for _ in segments
            ],
        )
        return [
            resolve_context_choice(parse_context_choice_output(raw, available), available)
            for raw, available in zip(outputs, availabilities)
        ]

    def build_reason_without_context_prompt(
        self,
        segment: EvidenceSegment,
        local_evidence: LocalEvidenceResult,
        context_choice: ContextChoice,
    ) -> str:
        prompt = f"""
You are making the final attribution judgment for one TARGET segment with NO extra context.

TARGET segment:
{format_target_segment_local(segment)}

Local forced evidence:
{json.dumps(asdict(local_evidence), ensure_ascii=False, indent=2)}

Selected context policy:
{json.dumps(asdict(context_choice), ensure_ascii=False, indent=2)}

Task:
- Judge whether the TARGET segment itself is hate, nonhate, or uncertain.
- Keep attribution on the TARGET segment itself.

Return STRICT JSON only:
{{
  "verdict_label": "hate or nonhate or uncertain",
  "hate_confidence": 0.50,
  "rationale": "brief explanation"
}}
""".strip()
        return prompt

    def reason_without_context(
        self,
        segment: EvidenceSegment,
        local_evidence: LocalEvidenceResult,
        context_choice: ContextChoice,
    ) -> ContextReasoningStep:
        raw = self.generate_text(
            self.build_reason_without_context_prompt(segment, local_evidence, context_choice),
            self.max_new_tokens_context_reason,
            {"verdict_label": "uncertain", "hate_confidence": 0.5, "rationale": "dry_run no context"},
        )
        return parse_context_reasoning_output(
            raw=raw,
            round_id=1,
            context_type="no_context",
        )

    def prepare_selected_context_reasoning(
        self,
        segment: EvidenceSegment,
        local_evidence: LocalEvidenceResult,
        context_choice: ContextChoice,
        selected_context_types: List[str],
        left_context: List[EvidenceSegment],
        right_context: List[EvidenceSegment],
        thread_segments: List[EvidenceSegment],
    ) -> Dict[str, Any]:
        use_temporal = "temporal_context" in selected_context_types
        use_thread = "semantic_thread_context" in selected_context_types
        left_text = "\n".join(summarize_segment_for_context(s) for s in left_context) if left_context else "<none>"
        right_text = "\n".join(summarize_segment_for_context(s) for s in right_context) if right_context else "<none>"

        limited_thread_segments = thread_segments[:MAX_THREAD_CONTEXT_SEGMENTS]
        thread_segment_lines = []
        for seg in limited_thread_segments:
            marker = "TARGET" if seg.segment_id == segment.segment_id else "SAME_THREAD"
            thread_segment_lines.append(
                f"[{marker}] role={seg.segment_thread_role}; {summarize_segment_for_context(seg, max_chars=1100)}"
            )

        temporal_block = "Not selected."
        if use_temporal:
            temporal_block = f"""
Left temporal neighbor segments:
{left_text}

Right temporal neighbor segments:
{right_text}
""".strip()

        thread_block = "Not selected."
        if use_thread:
            thread_block = f"""
Evidence-thread metadata:
- primary_thread_id: {segment.primary_thread_id or "<none>"}
- thread_label: {segment.thread_label or "<none>"}
- thread_description: {segment.thread_description or "<none>"}
- thread_evidence_question: {segment.thread_evidence_question or "<none>"}
- thread_type: {segment.thread_type or "<none>"}
- thread_reasoning_focus: {segment.thread_reasoning_focus or "<none>"}
- target_segment_thread_role: {segment.segment_thread_role}

Same-thread segment content:
{chr(10).join(thread_segment_lines) if thread_segment_lines else "<none>"}
""".strip()

        context_type = "+".join(selected_context_types)
        prompt = f"""
You are doing final attribution reasoning for one TARGET segment.

TARGET segment:
{format_target_segment_local(segment)}

Local forced evidence:
{json.dumps(asdict(local_evidence), ensure_ascii=False, indent=2)}

Selected context policy:
{json.dumps(asdict(context_choice), ensure_ascii=False, indent=2)}

Executable selected context types for this single pass:
{json.dumps(selected_context_types, ensure_ascii=False)}

TEMPORAL CONTEXT:
{temporal_block}

SEMANTIC THREAD CONTEXT:
{thread_block}

Rules:
1. This is a single final reasoning pass. Consider all selected context content together.
2. Use temporal neighbors only to understand immediate continuity, setup, continuation, or local visual transition.
3. Use semantic thread context only to clarify referent, setup/callback, quote/report/rebuttal/satire vs endorsement, and same-thread meaning.
4. Context may clarify TARGET-owned evidence, but final attribution must remain on the TARGET segment itself.
5. Do not label the TARGET as hate if hateful evidence appears only in neighbor or same-thread context.
6. Judge only whether the TARGET segment itself is hate / nonhate / uncertain.
7. hate_confidence means confidence that the TARGET segment itself is hate.

Return STRICT JSON only:
{{
  "verdict_label": "hate or nonhate or uncertain",
  "hate_confidence": 0.50,
  "rationale": "brief explanation"
}}
""".strip()
        return {
            "prompt": prompt,
            "context_type": context_type,
            "used_left_segment_ids": [s.segment_id for s in left_context] if use_temporal else [],
            "used_right_segment_ids": [s.segment_id for s in right_context] if use_temporal else [],
            "used_thread_segment_ids": [s.segment_id for s in limited_thread_segments] if use_thread else [],
        }

    def reason_with_selected_contexts(
        self,
        segment: EvidenceSegment,
        local_evidence: LocalEvidenceResult,
        context_choice: ContextChoice,
        selected_context_types: List[str],
        left_context: List[EvidenceSegment],
        right_context: List[EvidenceSegment],
        thread_segments: List[EvidenceSegment],
    ) -> ContextReasoningStep:
        prepared = self.prepare_selected_context_reasoning(
            segment,
            local_evidence,
            context_choice,
            selected_context_types,
            left_context,
            right_context,
            thread_segments,
        )
        raw = self.generate_text(
            prepared["prompt"],
            self.max_new_tokens_context_reason,
            {"verdict_label": "uncertain", "hate_confidence": 0.5, "rationale": "dry_run selected contexts"},
        )
        return parse_context_reasoning_output(
            raw=raw,
            round_id=1,
            context_type=prepared["context_type"],
            used_left_segment_ids=prepared["used_left_segment_ids"],
            used_right_segment_ids=prepared["used_right_segment_ids"],
            used_thread_segment_ids=prepared["used_thread_segment_ids"],
        )

    def reason_final_batch(self, prepared_requests: List[Dict[str, Any]]) -> List[ContextReasoningStep]:
        """Run independent final-reason prompts in one text batch."""
        outputs = self.generate_text_batch(
            [request["prompt"] for request in prepared_requests],
            self.max_new_tokens_context_reason,
            [
                {
                    "verdict_label": "uncertain",
                    "hate_confidence": 0.5,
                    "rationale": "dry_run final reasoning",
                }
                for _ in prepared_requests
            ],
        )
        return [
            parse_context_reasoning_output(
                raw=raw,
                round_id=1,
                context_type=request["context_type"],
                used_left_segment_ids=request.get("used_left_segment_ids", []),
                used_right_segment_ids=request.get("used_right_segment_ids", []),
                used_thread_segment_ids=request.get("used_thread_segment_ids", []),
            )
            for raw, request in zip(outputs, prepared_requests)
        ]

    def build_fallback_prompt(
        self,
        segment: EvidenceSegment,
        local_evidence: LocalEvidenceResult,
        initial_judgment: InitialJudgmentResult,
        context_choice: Optional[ContextChoice],
        context_steps: List[ContextReasoningStep],
        variant_instruction: str,
    ) -> str:
        prompt = f"""
You are performing FORCED ATTRIBUTION fallback for one TARGET segment that remained uncertain.

TARGET segment:
{format_target_segment_local(segment)}

Local forced evidence:
{json.dumps(asdict(local_evidence), ensure_ascii=False, indent=2)}

Pre-final judgment placeholder:
{json.dumps(asdict(initial_judgment), ensure_ascii=False, indent=2)}

Selected context choice:
{json.dumps(asdict(context_choice), ensure_ascii=False, indent=2) if context_choice else "<none>"}

Context reasoning history:
{json.dumps([asdict(x) for x in context_steps], ensure_ascii=False, indent=2)}

Decision policy for this run:
{variant_instruction}

Output constraint:
- You MUST choose exactly one label: hate or nonhate.
- Do not output uncertain.
- Do not use title-level attribution here.
- Keep attribution on the TARGET segment itself.

Return STRICT JSON only:
{{
  "verdict_label": "hate or nonhate",
  "rationale": "brief explanation"
}}
""".strip()
        return prompt

    def parse_fallback_vote(self, raw: str, variant_name: str) -> FallbackVote:
        parsed = safe_json_loads(raw)
        verdict = "nonhate"
        rationale = raw[:300].strip()
        if isinstance(parsed, dict):
            verdict = normalize_label(parsed.get("verdict_label"), {"hate", "nonhate"}, "nonhate")
            rationale = str(parsed.get("rationale", rationale)).strip()
        return FallbackVote(
            prompt_variant=variant_name,
            verdict_label=verdict,
            raw_model_output=raw,
            rationale=rationale,
        )

    def forced_multiclass_fallback(
        self,
        segment: EvidenceSegment,
        local_evidence: LocalEvidenceResult,
        initial_judgment: InitialJudgmentResult,
        context_choice: Optional[ContextChoice],
        context_steps: List[ContextReasoningStep],
        variant_name: str,
        variant_instruction: str,
    ) -> FallbackVote:
        raw = self.generate_text(
            self.build_fallback_prompt(
                segment,
                local_evidence,
                initial_judgment,
                context_choice,
                context_steps,
                variant_instruction,
            ),
            self.max_new_tokens_fallback,
            {"verdict_label": "nonhate", "rationale": "dry_run fallback"},
        )
        return self.parse_fallback_vote(raw, variant_name)

    def forced_multiclass_fallback_batch(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[FallbackVote]:
        votes: List[FallbackVote] = []
        batch_size = max(1, int(FALLBACK_REASONING_BATCH_SIZE))
        for batch_start in range(0, len(requests), batch_size):
            batch_requests = requests[batch_start: batch_start + batch_size]
            outputs = self.generate_text_batch(
                [
                    self.build_fallback_prompt(
                        request["segment"],
                        request["local_evidence"],
                        request["initial_judgment"],
                        request["context_choice"],
                        request["context_steps"],
                        request["variant_instruction"],
                    )
                    for request in batch_requests
                ],
                self.max_new_tokens_fallback,
                [
                    {"verdict_label": "nonhate", "rationale": "dry_run fallback"}
                    for _ in batch_requests
                ],
            )
            votes.extend(
                self.parse_fallback_vote(raw, request["variant_name"])
                for raw, request in zip(outputs, batch_requests)
            )
        return votes

    def reason_video_title(self, video_title: str) -> TitleReasoningResult:
        title_text = (video_title or "").strip()
        if not title_text or not ENABLE_TITLE_REASONING:
            return TitleReasoningResult(
                title_present=bool(title_text),
                title_text=title_text,
                verdict_label="nonhate",
                rationale="Title reasoning disabled or title absent.",
                raw_model_output="",
            )
        prompt = f"""
You are judging whether a VIDEO TITLE itself contains hateful content.

Definition:
- "hate" means the title likely expresses, promotes, glorifies, threatens, or directly targets a protected group or identity
  with hateful hostility, slurs, dehumanization, exclusion, or violence.

Important rules:
1. Judge ONLY the title text itself.
2. Do NOT use video frames, transcript, segments, or other context.
3. Output exactly one label: hate or nonhate.

Title:
{title_text}

Return STRICT JSON only:
{{
  "verdict_label": "hate or nonhate",
  "rationale": "brief explanation"
}}
""".strip()
        raw = self.generate_text(
            prompt,
            self.max_new_tokens_title_reason,
            {"verdict_label": "nonhate", "rationale": "dry_run title"},
        )
        parsed = safe_json_loads(raw)
        if isinstance(parsed, dict):
            verdict = normalize_label(parsed.get("verdict_label"), {"hate", "nonhate"}, "nonhate")
            rationale = str(parsed.get("rationale", "")).strip()
            return TitleReasoningResult(True, title_text, verdict, rationale, raw)
        low = raw.lower()
        verdict = "hate" if ("hate" in low and "nonhate" not in low and "non-hate" not in low) else "nonhate"
        return TitleReasoningResult(True, title_text, verdict, raw[:300].strip(), raw)


# =========================
# Parse model outputs
# =========================


def parse_local_hypothesis_output(raw: str, hypothesis_label: str) -> LocalHypothesisResult:
    parsed = safe_json_loads(raw)
    if isinstance(parsed, dict):
        return LocalHypothesisResult(
            hypothesis_label=hypothesis_label,
            evidence=[str(x).strip() for x in parsed.get("evidence", []) if str(x).strip()],
            target_clues=[str(x).strip() for x in parsed.get("target_clues", []) if str(x).strip()],
            transcript_evidence=[str(x).strip() for x in parsed.get("transcript_evidence", []) if str(x).strip()],
            visual_evidence=[str(x).strip() for x in parsed.get("visual_evidence", []) if str(x).strip()],
            confidence=clamp01(parsed.get("confidence"), 0.5),
            rationale=str(parsed.get("rationale", "")).strip(),
            raw_model_output=raw,
        )
    return LocalHypothesisResult(hypothesis_label=hypothesis_label, rationale=raw[:300].strip(), raw_model_output=raw)


def parse_initial_judgment_output(raw: str) -> InitialJudgmentResult:
    parsed = safe_json_loads(raw)
    if isinstance(parsed, dict):
        return InitialJudgmentResult(
            verdict_label=normalize_label(parsed.get("verdict_label"), {"hate", "nonhate", "uncertain"}, "uncertain"),
            hate_confidence=clamp01(parsed.get("hate_confidence"), 0.5),
            context_need=str(parsed.get("context_need", "none")).strip(),
            rationale=str(parsed.get("rationale", "")).strip(),
            raw_model_output=raw,
        )
    return InitialJudgmentResult("uncertain", 0.5, "unclear", raw[:300].strip(), raw)


def parse_context_choice_output(raw: str, available: Dict[str, bool]) -> ContextChoice:
    parsed = safe_json_loads(raw)
    if isinstance(parsed, dict):
        context_types = normalize_context_type_list(
            parsed.get("context_types", parsed.get("context_type")),
            ["no_context"],
        )
        return ContextChoice(
            context_type="+".join(context_types),
            rationale=str(parsed.get("rationale", "")).strip(),
            context_types=context_types,
            missing_information_needed=str(parsed.get("missing_information_needed", "")).strip(),
            available_context=available,
            raw_model_output=raw,
        )
    return ContextChoice(
        context_type="no_context",
        rationale=raw[:300].strip(),
        context_types=["no_context"],
        available_context=available,
        raw_model_output=raw,
    )


def parse_context_reasoning_output(
    raw: str,
    round_id: int,
    context_type: str,
    used_left_segment_ids: Optional[List[int]] = None,
    used_right_segment_ids: Optional[List[int]] = None,
    used_thread_segment_ids: Optional[List[int]] = None,
    used_video_context: bool = False,
) -> ContextReasoningStep:
    parsed = safe_json_loads(raw)
    if isinstance(parsed, dict):
        label = normalize_label(parsed.get("verdict_label"), {"hate", "nonhate", "uncertain"}, "uncertain")
        conf = clamp01(parsed.get("hate_confidence"), 0.5)
        rationale = str(parsed.get("rationale", "")).strip()
    else:
        label = "uncertain"
        conf = 0.5
        rationale = raw[:300].strip()
    return ContextReasoningStep(
        round_id=round_id,
        context_type=context_type,
        used_left_segment_ids=used_left_segment_ids or [],
        used_right_segment_ids=used_right_segment_ids or [],
        used_thread_segment_ids=used_thread_segment_ids or [],
        used_video_context=used_video_context,
        verdict_label=label,
        hate_confidence=conf,
        rationale=rationale,
        raw_model_output=raw,
    )


# =========================
# Reasoning flow
# =========================


def build_segment_inference_result(
    video_id: str,
    seg: EvidenceSegment,
    local_evidence: LocalEvidenceResult,
) -> SegmentInferenceResult:
    initial = InitialJudgmentResult(
        verdict_label="uncertain",
        hate_confidence=local_evidence.hate_confidence,
        context_need="selected_before_final_judgment",
        rationale="V5 skips initial hard labeling; one or more context policies are selected before final judgment.",
        raw_model_output="",
    )
    return SegmentInferenceResult(
        merged_segment_id=seg.segment_id,
        evidence_segment_id=seg.segment_id,
        video_id=video_id,
        representative_frame_path=seg.representative_frame_path,
        representative_frame_filename=seg.representative_frame_filename,
        representative_frame_paths=seg.representative_frame_paths,
        representative_frame_filenames=seg.representative_frame_filenames,
        time_range_sec=(seg.start_sec, seg.end_sec),
        unit_ids=seg.candidate_fragment_ids,
        merged_description=evidence_segment_compat_description(seg),
        transcript_text=seg.primary_transcript,
        has_transcript=bool(seg.primary_transcript.strip()),
        local_evidence=local_evidence,
        initial_judgment=initial,
        final_label_before_fallback=initial.verdict_label,
        final_hate_confidence_before_fallback=initial.hate_confidence,
        final_rationale_before_fallback=initial.rationale,
        anchor_modality=seg.anchor_modality,
        cross_modal_signal=seg.cross_modal_signal,
        primary_sentence_ids=seg.primary_sentence_ids,
        boundary_sentence_ids=seg.boundary_sentence_ids,
        boundary_context_not_primary=seg.boundary_context_not_primary,
        transcript_ownership_note=seg.transcript_ownership_note,
        primary_thread_id=seg.primary_thread_id,
        thread_label=seg.thread_label,
        thread_description=seg.thread_description,
        segment_thread_role=seg.segment_thread_role,
    )


def prepare_final_reason_request(
    reasoner: MultiChoiceContextReasoner,
    seg: EvidenceSegment,
    local_evidence: LocalEvidenceResult,
    choice: ContextChoice,
    evidence_segments: List[EvidenceSegment],
    idx: int,
) -> Optional[Dict[str, Any]]:
    selected_context_types = [
        context_type
        for context_type in choice.resolved_context_types
        if context_type != "no_context"
    ]
    if not selected_context_types:
        selected_context_types = ["no_context"]

    if selected_context_types == ["no_context"]:
        return {
            "prompt": reasoner.build_reason_without_context_prompt(seg, local_evidence, choice),
            "context_type": "no_context",
            "used_left_segment_ids": [],
            "used_right_segment_ids": [],
            "used_thread_segment_ids": [],
        }

    left: List[EvidenceSegment] = []
    right: List[EvidenceSegment] = []
    thread_segments: List[EvidenceSegment] = []
    if "temporal_context" in selected_context_types:
        left, right = get_temporal_context(evidence_segments, idx)
    if "semantic_thread_context" in selected_context_types:
        thread_segments = same_thread_segments(seg, evidence_segments)
        if not thread_segments:
            selected_context_types = [
                context_type
                for context_type in selected_context_types
                if context_type != "semantic_thread_context"
            ]
            choice.resolution_note = (
                f"{choice.resolution_note}; semantic_thread_context_unavailable_at_execution"
            )

    if not selected_context_types:
        return {
            "prompt": reasoner.build_reason_without_context_prompt(seg, local_evidence, choice),
            "context_type": "no_context",
            "used_left_segment_ids": [],
            "used_right_segment_ids": [],
            "used_thread_segment_ids": [],
        }
    return reasoner.prepare_selected_context_reasoning(
        seg,
        local_evidence,
        choice,
        selected_context_types,
        left,
        right,
        thread_segments,
    )


def infer_segment_batch(
    reasoner: MultiChoiceContextReasoner,
    video_id: str,
    evidence_segments: List[EvidenceSegment],
    indexed_segments: List[Tuple[int, EvidenceSegment]],
) -> List[SegmentInferenceResult]:
    batch_segments = [seg for _, seg in indexed_segments]
    local_evidences = reasoner.reason_segments_local_batch(batch_segments)
    segment_results = [
        build_segment_inference_result(video_id, seg, local_evidence)
        for seg, local_evidence in zip(batch_segments, local_evidences)
    ]

    for seg, local_evidence in zip(batch_segments, local_evidences):
        print(
            f"[LocalForcedReasoningBatch] video={video_id} segment={seg.segment_id} "
            f"hate_hyp={local_evidence.hate_hypothesis_confidence:.2f} "
            f"nonhate_hyp={local_evidence.nonhate_hypothesis_confidence:.2f} "
            f"batch_size={len(batch_segments)}"
        )

    availabilities = [
        context_availability(seg, evidence_segments, idx)
        for idx, seg in indexed_segments
    ]
    choices = reasoner.choose_context_types_batch(batch_segments, local_evidences, availabilities)

    prepared_requests: List[Dict[str, Any]] = []
    prepared_result_indices: List[int] = []
    for result_idx, ((idx, seg), local_evidence, choice) in enumerate(
        zip(indexed_segments, local_evidences, choices)
    ):
        seg_result = segment_results[result_idx]
        seg_result.context_choice = choice
        if not choice.resolved_context_types:
            seg_result.context_available = False
            seg_result.context_skip_reason = choice.resolution_note
            continue
        prepared = prepare_final_reason_request(
            reasoner,
            seg,
            local_evidence,
            choice,
            evidence_segments,
            idx,
        )
        if prepared is None:
            seg_result.context_available = False
            seg_result.context_skip_reason = f"no_executable_context_from:{choice.resolved_context_types}"
            continue
        prepared_requests.append(prepared)
        prepared_result_indices.append(result_idx)

    steps = reasoner.reason_final_batch(prepared_requests)
    for result_idx, step in zip(prepared_result_indices, steps):
        seg_result = segment_results[result_idx]
        seg_result.context_steps.append(step)
        seg_result.used_context = step.context_type != "no_context"
        seg_result.used_context_type = step.context_type
        seg_result.used_neighbor_segment_ids = sorted(set(step.used_left_segment_ids + step.used_right_segment_ids))
        seg_result.used_thread_segment_ids = step.used_thread_segment_ids
        seg_result.final_label_before_fallback = step.verdict_label
        seg_result.final_hate_confidence_before_fallback = step.hate_confidence
        seg_result.final_rationale_before_fallback = step.rationale
        print(
            f"[FinalReasonBatch] video={video_id} segment={seg_result.merged_segment_id} "
            f"context={step.context_type} -> {step.verdict_label} "
            f"conf={step.hate_confidence:.2f} batch_size={len(prepared_requests)}"
        )
    return segment_results


def infer_all_segments(
    reasoner: MultiChoiceContextReasoner,
    video_id: str,
    evidence_segments: List[EvidenceSegment],
) -> List[SegmentInferenceResult]:
    results: List[SegmentInferenceResult] = []
    batch_size = max(1, int(SEGMENT_REASONING_BATCH_SIZE))
    indexed = list(enumerate(evidence_segments))
    for batch_start in range(0, len(indexed), batch_size):
        current_indexed = indexed[batch_start: batch_start + batch_size]
        current_results = infer_segment_batch(
            reasoner,
            video_id,
            evidence_segments,
            current_indexed,
        )
        direct_hate_results = [
            result for result in current_results
            if result.final_label_before_fallback == "hate"
        ]
        if EARLY_EXIT_ON_HATE and direct_hate_results:
            for result in current_results:
                if result.final_label_before_fallback != "uncertain":
                    result.final_label_after_fallback = result.final_label_before_fallback
                    result.final_rationale_after_fallback = result.final_rationale_before_fallback
            results.extend(current_results)
            trigger = direct_hate_results[0]
            remaining = (
                len(indexed) - (batch_start + len(current_indexed))
                + sum(result.final_label_after_fallback is None for result in current_results)
            )
            print(
                f"[EarlyExit] video={video_id} trigger_segment={trigger.merged_segment_id} "
                f"unprocessed_or_unfinalized_segments={remaining}"
            )
            break

        run_fallback_for_uncertain_segments(reasoner, evidence_segments, current_results)
        results.extend(current_results)

        hate_results = [
            result for result in current_results
            if result.final_label_after_fallback == "hate"
        ]
        if EARLY_EXIT_ON_HATE and hate_results:
            trigger = hate_results[0]
            remaining = len(indexed) - (batch_start + len(current_indexed))
            print(
                f"[EarlyExit] video={video_id} trigger_segment={trigger.merged_segment_id} "
                f"unprocessed_segments={remaining}"
            )
            break

    return results


def run_fallback_for_uncertain_segments(
    reasoner: MultiChoiceContextReasoner,
    evidence_segments: List[EvidenceSegment],
    segment_results: List[SegmentInferenceResult],
) -> None:
    vote_specs = [
        (
            "conservative_evidence",
            "Only classify as hate if there is clear, direct, and target-specific evidence within the TARGET segment itself. "
            "Avoid relying on weak, indirect, or purely contextual signals. Do not over-interpret ambiguous content.",
        ),
        (
            "balanced_contextual",
            "Evaluate the TARGET segment using the target evidence and the selected context reasoning history. "
            "Consider direct and indirect signals, but keep attribution on the target segment.",
        ),
        (
            "implicit_sensitive",
            "Pay special attention to implicit or context-dependent hate signals. "
            "Classify as hate only if a coherent hateful interpretation is grounded in the target segment.",
        ),
    ]
    seg_map = {seg.segment_id: seg for seg in evidence_segments}
    batch_requests: List[Dict[str, Any]] = []
    for sr in segment_results:
        if sr.final_label_after_fallback in {"hate", "nonhate"}:
            continue
        if sr.final_label_before_fallback != "uncertain":
            sr.final_label_after_fallback = sr.final_label_before_fallback
            sr.final_rationale_after_fallback = sr.final_rationale_before_fallback
            continue

        seg = seg_map[sr.merged_segment_id]
        for variant_name, variant_instruction in vote_specs:
            batch_requests.append(
                {
                    "segment_result": sr,
                    "segment": seg,
                    "local_evidence": sr.local_evidence,
                    "initial_judgment": sr.initial_judgment,
                    "context_choice": sr.context_choice,
                    "context_steps": sr.context_steps,
                    "variant_name": variant_name,
                    "variant_instruction": variant_instruction,
                }
            )

    votes_by_segment: Dict[int, List[FallbackVote]] = {}
    if batch_requests:
        batch_votes = reasoner.forced_multiclass_fallback_batch(batch_requests)
        for request, vote in zip(batch_requests, batch_votes):
            sr = request["segment_result"]
            votes_by_segment.setdefault(sr.merged_segment_id, []).append(vote)
            print(
                f"[FallbackVoteBatch] segment={sr.merged_segment_id} "
                f"variant={vote.prompt_variant} -> {vote.verdict_label} "
                f"micro_batch_size={min(FALLBACK_REASONING_BATCH_SIZE, len(batch_requests))} "
                f"total_requests={len(batch_requests)}"
            )

    for sr in segment_results:
        votes = votes_by_segment.get(sr.merged_segment_id, [])
        if not votes:
            continue
        sr.fallback_votes = votes
        counts = {"hate": 0, "nonhate": 0}
        for vote in votes:
            counts[vote.verdict_label] += 1
        max_count = max(counts.values())
        winners = [label for label, count in counts.items() if count == max_count]
        sr.fallback_label = winners[0] if len(winners) == 1 else "hate"
        sr.final_label_after_fallback = sr.fallback_label
        sr.final_rationale_after_fallback = (
            f"fallback_majority={sr.fallback_label}; votes={[v.verdict_label for v in votes]}"
        )


def build_video_coarse_locations(
    evidence_segments: List[EvidenceSegment],
    segment_results: List[SegmentInferenceResult],
    final_video_label: str,
    hate_source: str,
) -> Tuple[List[int], List[Tuple[float, float]], List[Dict[str, float]]]:
    if final_video_label == "nonhate":
        return [], [], []
    seg_map = {seg.segment_id: seg for seg in evidence_segments}
    hate_ids = [
        sr.merged_segment_id
        for sr in segment_results
        if sr.final_label_after_fallback == "hate"
    ]
    time_ranges: List[Tuple[float, float]] = []
    span_dicts: List[Dict[str, float]] = []
    for seg_id in hate_ids:
        seg = seg_map.get(seg_id)
        if not seg:
            continue
        time_ranges.append((seg.start_sec, seg.end_sec))
        span_dicts.append({"start_time_sec": seg.start_sec, "end_time_sec": seg.end_sec})
    if isinstance(hate_source, str) and "title" in hate_source:
        time_ranges.append((0.0, 0.0))
        span_dicts.append({"start_time_sec": 0.0, "end_time_sec": 0.0})
    return hate_ids, time_ranges, span_dicts


def aggregate_video_result(
    reasoner: MultiChoiceContextReasoner,
    video_id: str,
    video_title: str,
    evidence_segments: List[EvidenceSegment],
    segment_results: List[SegmentInferenceResult],
) -> VideoInferenceResult:
    hate_ids_before = [x.merged_segment_id for x in segment_results if x.final_label_before_fallback == "hate"]
    nonhate_ids_before = [x.merged_segment_id for x in segment_results if x.final_label_before_fallback == "nonhate"]
    uncertain_ids_before = [x.merged_segment_id for x in segment_results if x.final_label_before_fallback == "uncertain"]
    title_ids_before: List[int] = []

    pending_fallback_ids = [
        x.merged_segment_id
        for x in segment_results
        if x.final_label_before_fallback == "uncertain"
        and x.final_label_after_fallback not in {"hate", "nonhate"}
    ]
    early_hate_already_final = EARLY_EXIT_ON_HATE and any(
        x.final_label_after_fallback == "hate" for x in segment_results
    )
    if pending_fallback_ids and not early_hate_already_final:
        run_fallback_for_uncertain_segments(reasoner, evidence_segments, segment_results)
    else:
        for sr in segment_results:
            if sr.final_label_after_fallback is not None:
                continue
            if sr.final_label_before_fallback == "uncertain":
                continue
            sr.final_label_after_fallback = sr.final_label_before_fallback
            sr.final_rationale_after_fallback = sr.final_rationale_before_fallback

    fallback_segment_ids = [
        x.merged_segment_id for x in segment_results if x.fallback_votes
    ]

    hate_ids_after = [x.merged_segment_id for x in segment_results if x.final_label_after_fallback == "hate"]
    nonhate_ids_after = [x.merged_segment_id for x in segment_results if x.final_label_after_fallback == "nonhate"]
    finalized_segment_ids = {
        x.merged_segment_id
        for x in segment_results
        if x.final_label_after_fallback in {"hate", "nonhate"}
    }
    unprocessed_segment_ids = [
        segment.segment_id
        for segment in evidence_segments
        if segment.segment_id not in finalized_segment_ids
    ]
    early_exit_triggered = bool(unprocessed_segment_ids and hate_ids_after)
    early_exit_segment_id = hate_ids_after[0] if early_exit_triggered else None

    title_reasoning: Optional[TitleReasoningResult] = None
    title_ids_after: List[int] = []
    if video_title and video_title.strip() and ENABLE_TITLE_REASONING and not early_exit_triggered:
        title_reasoning = reasoner.reason_video_title(video_title)
        print(f"[TitleReason] video={video_id} -> {title_reasoning.verdict_label} | {title_reasoning.rationale}")
        if title_reasoning.verdict_label == "hate":
            title_ids_after = [0]

    if hate_ids_after and title_ids_after:
        final_video_label = "hate"
        hate_source = "segment+title"
        decision_rule = "Rule C2: at least one evidence-thread segment is hate after fallback, and the title is also hateful."
    elif hate_ids_after:
        final_video_label = "hate"
        hate_source = "segment"
        if early_exit_triggered:
            decision_rule = (
                "Early-exit rule: at least one processed evidence-thread segment is hate after fallback; "
                "remaining segments were not processed."
            )
        elif hate_ids_before:
            decision_rule = "Rule A2: at least one evidence-thread segment is hate before fallback."
        else:
            decision_rule = "Rule B2: no segment is hate before fallback, but at least one segment becomes hate after fallback."
    elif title_ids_after:
        final_video_label = "hate"
        hate_source = "title"
        decision_rule = "Rule C: all evidence-thread segments are nonhate after fallback, but the video title is hateful."
    else:
        final_video_label = "nonhate"
        hate_source = "none"
        decision_rule = "Rule D: no hate segment after fallback and title is nonhate or absent."

    coarse_ids, coarse_ranges, coarse_spans = build_video_coarse_locations(
        evidence_segments=evidence_segments,
        segment_results=segment_results,
        final_video_label=final_video_label,
        hate_source=hate_source,
    )

    video_context = reasoner.video_context_cache.get(video_id)
    return VideoInferenceResult(
        video_id=video_id,
        video_title=video_title,
        final_video_label=final_video_label,
        hate_source=hate_source,
        decision_rule=decision_rule,
        hate_segment_ids_before_fallback=hate_ids_before,
        nonhate_segment_ids_before_fallback=nonhate_ids_before,
        uncertain_segment_ids_before_fallback=uncertain_ids_before,
        title_segment_ids_before_fallback=title_ids_before,
        fallback_used=bool(fallback_segment_ids),
        fallback_segment_ids=fallback_segment_ids,
        hate_segment_ids_after_fallback=hate_ids_after,
        nonhate_segment_ids_after_fallback=nonhate_ids_after,
        title_segment_ids_after_fallback=title_ids_after,
        coarse_location_segment_ids=coarse_ids,
        coarse_location_time_ranges=coarse_ranges,
        coarse_location_spans=coarse_spans,
        early_exit_triggered=early_exit_triggered,
        early_exit_segment_id=early_exit_segment_id,
        unprocessed_segment_ids=unprocessed_segment_ids,
        localization_complete=not early_exit_triggered,
        title_reasoning=title_reasoning,
        video_context=video_context,
        segment_results=segment_results,
    )


# =========================
# Output
# =========================


def write_readable_txt(
    video_id: str,
    video_title: str,
    segmentation_data: Dict[str, Any],
    evidence_segments: List[EvidenceSegment],
    segment_results: List[SegmentInferenceResult],
    video_result: VideoInferenceResult,
    path: str,
) -> None:
    lines: List[str] = []
    lines.append(f"video_id: {video_id}")
    lines.append(f"video_title: {video_title}")
    lines.append("=" * 120)
    lines.append("")
    lines.append("[0] SEGMENTATION TRACE")
    lines.append("-" * 120)
    lines.append(f"method: {segmentation_data.get('method', 'evidence_thread_segmentation_v2')}")
    lines.append(f"thread_count: {len(segmentation_data.get('evidence_threads', []))}")
    lines.append(f"segment_count: {len(evidence_segments)}")
    lines.append(f"local_representative_frame_count: {LOCAL_REPRESENTATIVE_FRAME_COUNT}")
    lines.append(f"local_representative_frame_sampling: {LOCAL_REPRESENTATIVE_FRAME_SAMPLING}")
    lines.append(f"unit_description_batch_size: {UNIT_DESCRIPTION_BATCH_SIZE}")
    lines.append(f"segment_reasoning_batch_size: {SEGMENT_REASONING_BATCH_SIZE}")
    lines.append(f"fallback_reasoning_batch_size: {FALLBACK_REASONING_BATCH_SIZE}")
    lines.append(f"clear_cuda_cache_after_batch: {CLEAR_CUDA_CACHE_AFTER_BATCH}")
    lines.append(f"early_exit_on_hate: {EARLY_EXIT_ON_HATE}")
    lines.append("")

    lines.append("[1] EVIDENCE-THREAD SEGMENTS")
    lines.append("-" * 120)
    for seg in evidence_segments:
        lines.append(
            f"segment_id={seg.segment_id} time={format_time_range(seg.start_sec, seg.end_sec)} "
            f"anchor={seg.anchor_modality} thread={seg.primary_thread_id} role={seg.segment_thread_role}"
        )
        lines.append(f"thread_label: {seg.thread_label or '<none>'}")
        if seg.thread_description:
            lines.append(f"thread_description: {truncate_text(seg.thread_description, 500)}")
        lines.append(f"representative_frames: {seg.representative_frame_filenames or []}")
        lines.append("primary_transcript:")
        lines.append(seg.primary_transcript if seg.primary_transcript else "<none>")
        lines.append("visual_descriptions:")
        lines.append(visual_descriptions_block(seg, max_chars=1800))
        lines.append(f"remote_thread_context_ids: {[x.get('segment_id') for x in seg.remote_thread_context if isinstance(x, dict)]}")
        lines.append("")

    lines.append("[2] SEGMENT-LEVEL INFERENCE")
    lines.append("-" * 120)
    for sr in segment_results:
        lines.append(
            f"segment_id={sr.merged_segment_id} time={format_time_range(sr.time_range_sec[0], sr.time_range_sec[1])} "
            f"thread={sr.primary_thread_id} role={sr.segment_thread_role}"
        )
        lines.append(f"representative_frames: {sr.representative_frame_filenames or []}")
        lines.append("LOCAL EVIDENCE:")
        lines.append(json.dumps(asdict(sr.local_evidence), ensure_ascii=False, indent=2))
        lines.append("V5 PRE-FINAL PLACEHOLDER:")
        lines.append(json.dumps(asdict(sr.initial_judgment), ensure_ascii=False, indent=2))
        if sr.context_choice:
            lines.append("CONTEXT CHOICE:")
            lines.append(json.dumps(asdict(sr.context_choice), ensure_ascii=False, indent=2))
        if sr.context_steps:
            lines.append("CONTEXT REASONING:")
            for step in sr.context_steps:
                lines.append(json.dumps(asdict(step), ensure_ascii=False, indent=2))
        lines.append(
            f"BEFORE_FALLBACK: label={sr.final_label_before_fallback}, "
            f"hate_confidence={sr.final_hate_confidence_before_fallback:.3f}, "
            f"used_context={sr.used_context}, context_type={sr.used_context_type}"
        )
        if not sr.context_available:
            lines.append(f"context_skip_reason: {sr.context_skip_reason}")
        if sr.fallback_votes:
            lines.append("FALLBACK VOTES:")
            for vote in sr.fallback_votes:
                lines.append(json.dumps(asdict(vote), ensure_ascii=False, indent=2))
        lines.append(f"FINAL_AFTER_FALLBACK: {sr.final_label_after_fallback}")
        lines.append(f"after_fallback_rationale: {sr.final_rationale_after_fallback}")
        lines.append("")

    lines.append("[3] TITLE-LEVEL REASONING")
    lines.append("-" * 120)
    if video_result.title_reasoning is None:
        lines.append("title_reasoning: not executed")
    else:
        lines.append(json.dumps(asdict(video_result.title_reasoning), ensure_ascii=False, indent=2))
    lines.append("")

    lines.append("[4] FULL VIDEO CONTEXT")
    lines.append("-" * 120)
    if video_result.video_context is None:
        lines.append("full_video_context: disabled in V5 context selection")
    else:
        lines.append(json.dumps(asdict(video_result.video_context), ensure_ascii=False, indent=2))
    lines.append("")

    lines.append("[5] VIDEO-LEVEL RESULT")
    lines.append("-" * 120)
    lines.append(f"final_video_label: {video_result.final_video_label}")
    lines.append(f"hate_source: {video_result.hate_source}")
    lines.append(f"decision_rule: {video_result.decision_rule}")
    lines.append(f"hate_segment_ids_before_fallback: {video_result.hate_segment_ids_before_fallback}")
    lines.append(f"nonhate_segment_ids_before_fallback: {video_result.nonhate_segment_ids_before_fallback}")
    lines.append(f"uncertain_segment_ids_before_fallback: {video_result.uncertain_segment_ids_before_fallback}")
    lines.append(f"fallback_used: {video_result.fallback_used}")
    lines.append(f"fallback_segment_ids: {video_result.fallback_segment_ids}")
    lines.append(f"hate_segment_ids_after_fallback: {video_result.hate_segment_ids_after_fallback}")
    lines.append(f"title_segment_ids_after_fallback: {video_result.title_segment_ids_after_fallback}")
    lines.append(f"nonhate_segment_ids_after_fallback: {video_result.nonhate_segment_ids_after_fallback}")
    lines.append(f"coarse_location_segment_ids: {video_result.coarse_location_segment_ids}")
    lines.append(f"coarse_location_time_ranges: {video_result.coarse_location_time_ranges}")
    lines.append(f"coarse_location_spans: {json.dumps(video_result.coarse_location_spans, ensure_ascii=False)}")
    lines.append(f"early_exit_triggered: {video_result.early_exit_triggered}")
    lines.append(f"early_exit_segment_id: {video_result.early_exit_segment_id}")
    lines.append(f"unprocessed_segment_ids: {video_result.unprocessed_segment_ids}")
    lines.append(f"localization_complete: {video_result.localization_complete}")
    write_text(path, "\n".join(lines))


def format_seconds(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "-"


def mean_seconds(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [
        float(record[key])
        for record in records
        if record.get("status") == "success" and record.get(key) is not None
    ]
    return sum(values) / len(values) if values else None


def escape_md_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def resolve_inference_time_report_path(output_root: str, report_file: str) -> str:
    if os.path.isabs(report_file):
        return report_file
    return os.path.join(output_root, report_file)


def write_inference_time_report(
    timing_records: List[Dict[str, Any]],
    output_root: str,
    report_file: str,
    total_video_count: int,
    model_name: str,
) -> None:
    successful_records = [record for record in timing_records if record.get("status") == "success"]
    skipped_records = [record for record in timing_records if record.get("status") == "skipped"]
    error_records = [record for record in timing_records if record.get("status") == "error"]

    total_success_time = sum(float(record.get("total_sec") or 0.0) for record in successful_records)
    report_path = resolve_inference_time_report_path(output_root, report_file)

    lines = [
        "# Multi-Choice Context Reasoning Inference Time Report",
        "",
        "## Summary",
        "",
        f"- Model: `{model_name}`",
        f"- Output root: `{output_root}`",
        f"- Total video ids: {total_video_count}",
        f"- Records written: {len(timing_records)}",
        f"- Timed successful videos: {len(successful_records)}",
        f"- Skipped existing results: {len(skipped_records)}",
        f"- Errors: {len(error_records)}",
        f"- Mean segment division time per video: {format_seconds(mean_seconds(successful_records, 'segment_division_sec'))} sec",
        f"- Mean reasoning time per video: {format_seconds(mean_seconds(successful_records, 'reasoning_sec'))} sec",
        f"- Mean total inference time per video: {format_seconds(mean_seconds(successful_records, 'total_sec'))} sec",
        f"- Total measured successful video time: {format_seconds(total_success_time)} sec ({format_seconds(total_success_time / 60.0)} min)",
        "",
        "## Per-Video Inference Time",
        "",
        "| video_id | status | segment_division_sec | reasoning_sec | total_sec | total_min | note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for record in timing_records:
        total_sec = record.get("total_sec")
        total_min = float(total_sec) / 60.0 if total_sec is not None else None
        note = record.get("note") or record.get("error") or ""
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md_cell(record.get("video_id")),
                    escape_md_cell(record.get("status")),
                    format_seconds(record.get("segment_division_sec")),
                    format_seconds(record.get("reasoning_sec")),
                    format_seconds(total_sec),
                    format_seconds(total_min),
                    escape_md_cell(note),
                ]
            )
            + " |"
        )

    write_text(report_path, "\n".join(lines))


# =========================
# Main process
# =========================


def process_single_video(
    segmenter: Any,
    framev2: Any,
    video_id: str,
    transcript_root: str,
    frame_root: str,
    output_root: str,
    llm: Any,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    video_start = time.perf_counter()
    save_dir = os.path.join(output_root, video_id)
    ensure_dir(save_dir)
    video_result_json_path = os.path.join(save_dir, "video_result.json")
    if (not overwrite) and os.path.exists(video_result_json_path):
        print(f"[Skip] {video_id} already has video_result.json.")
        return {
            "video_id": video_id,
            "status": "skipped",
            "segment_division_sec": None,
            "reasoning_sec": None,
            "total_sec": time.perf_counter() - video_start,
            "note": "existing video_result.json",
        }

    segment_division_start = time.perf_counter()
    segmentation_data = run_evidence_thread_segmentation(
        segmenter=segmenter,
        framev2=framev2,
        qwen=llm,
        video_id=video_id,
        frame_root=frame_root,
        output_root=output_root,
        overwrite=overwrite,
    )
    segment_division_sec = time.perf_counter() - segment_division_start

    evidence_segments = normalize_evidence_thread_segments(segmentation_data, frame_root)
    dump_json([asdict(x) for x in evidence_segments], os.path.join(save_dir, "evidence_segments_for_reasoning.json"))

    video_title = read_video_title(transcript_root, video_id)
    if video_title:
        print(f"[TranscriptTitle] video={video_id} title={video_title}")
    else:
        print(f"[TranscriptTitle] video={video_id} title=<empty>")

    reasoning_start = time.perf_counter()
    reasoner = MultiChoiceContextReasoner(
        llm=llm,
        dry_run=dry_run,
        max_new_tokens_local_reason=MAX_NEW_TOKENS_LOCAL_REASON,
        max_new_tokens_context_choice=MAX_NEW_TOKENS_CONTEXT_CHOICE,
        max_new_tokens_context_reason=MAX_NEW_TOKENS_CONTEXT_REASON,
        max_new_tokens_video_context=MAX_NEW_TOKENS_VIDEO_CONTEXT,
        max_new_tokens_fallback=MAX_NEW_TOKENS_FALLBACK,
        max_new_tokens_title_reason=MAX_NEW_TOKENS_TITLE_REASON,
    )

    segment_results = infer_all_segments(reasoner, video_id, evidence_segments)
    video_result = aggregate_video_result(
        reasoner=reasoner,
        video_id=video_id,
        video_title=video_title,
        evidence_segments=evidence_segments,
        segment_results=segment_results,
    )
    reasoning_sec = time.perf_counter() - reasoning_start

    dump_json([asdict(x) for x in segment_results], os.path.join(save_dir, "segment_reasoning.json"))
    dump_json(asdict(video_result), video_result_json_path)
    write_readable_txt(
        video_id=video_id,
        video_title=video_title,
        segmentation_data=segmentation_data,
        evidence_segments=evidence_segments,
        segment_results=segment_results,
        video_result=video_result,
        path=os.path.join(save_dir, "readable_report.txt"),
    )
    print(f"[Done] video={video_id}, output_dir={save_dir}")
    return {
        "video_id": video_id,
        "status": "success",
        "segment_division_sec": segment_division_sec,
        "reasoning_sec": reasoning_sec,
        "total_sec": time.perf_counter() - video_start,
        "note": f"segments={len(evidence_segments)}",
    }


def process_all_videos(
    input_root: str,
    output_root: str,
    inference_time_report_file: str,
    transcript_root: str,
    video_id: str,
    limit: int,
    overwrite: bool,
    dry_run: bool,
    model_name: str,
    device_map: str,
    torch_dtype: str,
    attn_implementation: Optional[str],
) -> None:
    ensure_dir(output_root)
    video_ids = list_video_ids(input_root, transcript_root, video_id, limit)
    dump_json(
        {
            "input_root": input_root,
            "output_root": output_root,
            "inference_time_report_file": inference_time_report_file,
            "transcript_root": transcript_root,
            "video_ids": video_ids,
            "model_name": model_name,
            "torch_dtype": torch_dtype,
            "attn_implementation": attn_implementation,
            "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "pipeline": "evidence_thread_segmentation_v2 + multi_choice_context_reasoning_v5",
            "reasoning_policy": "forced_hate_and_nonhate_then_multi_context_selection_then_single_pass_selected_context_final_judgment",
            "local_representative_frame_count": LOCAL_REPRESENTATIVE_FRAME_COUNT,
            "local_representative_frame_sampling": LOCAL_REPRESENTATIVE_FRAME_SAMPLING,
            "unit_description_batch_size": UNIT_DESCRIPTION_BATCH_SIZE,
            "segment_reasoning_batch_size": SEGMENT_REASONING_BATCH_SIZE,
            "fallback_reasoning_batch_size": FALLBACK_REASONING_BATCH_SIZE,
            "clear_cuda_cache_after_batch": CLEAR_CUDA_CACHE_AFTER_BATCH,
            "early_exit_on_hate": EARLY_EXIT_ON_HATE,
        },
        os.path.join(output_root, "run_inputs.json"),
    )
    print(f"[Info] Found {len(video_ids)} video ids under {input_root}")
    if not video_ids:
        return

    segmenter = load_segmenter_module()
    configure_segmenter(
        segmenter=segmenter,
        frame_root=input_root,
        transcript_root=transcript_root,
        output_root=output_root,
        overwrite=overwrite,
        dry_run=dry_run,
        model_name=model_name,
    )
    framev2 = segmenter.load_framev2_module()
    llm = None
    if not dry_run:
        print(f"[Info] Loading Qwen3-VL model: {model_name}")
        llm = framev2.Qwen3VLPipeline(
            model_name=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            max_new_tokens_desc=MAX_NEW_TOKENS_FRAME_DESC,
            max_new_tokens_merge=max(MAX_NEW_TOKENS_THREAD_ASSIGNMENT, MAX_NEW_TOKENS_CONTEXT_REASON),
            clear_cuda_cache_after_batch=CLEAR_CUDA_CACHE_AFTER_BATCH,
        )

    timing_records: List[Dict[str, Any]] = []
    for idx, current_video_id in enumerate(video_ids, start=1):
        print("\n" + "=" * 120)
        print(f"[Video {idx}/{len(video_ids)}] {current_video_id}")
        print("=" * 120)
        loop_start = time.perf_counter()
        try:
            timing_record = process_single_video(
                segmenter=segmenter,
                framev2=framev2,
                video_id=current_video_id,
                transcript_root=transcript_root,
                frame_root=input_root,
                output_root=output_root,
                llm=llm,
                overwrite=overwrite,
                dry_run=dry_run,
            )
            timing_records.append(timing_record)
        except Exception as exc:
            print(f"[Error] Failed on {current_video_id}: {repr(exc)}")
            err_dir = os.path.join(output_root, current_video_id)
            ensure_dir(err_dir)
            dump_json({"video_id": current_video_id, "error": repr(exc)}, os.path.join(err_dir, "error.json"))
            timing_records.append(
                {
                    "video_id": current_video_id,
                    "status": "error",
                    "segment_division_sec": None,
                    "reasoning_sec": None,
                    "total_sec": time.perf_counter() - loop_start,
                    "error": repr(exc),
                }
            )
        finally:
            release_cuda_memory(f"video={current_video_id}")
            write_inference_time_report(
                timing_records=timing_records,
                output_root=output_root,
                report_file=inference_time_report_file,
                total_video_count=len(video_ids),
                model_name=model_name,
            )


def print_runtime_info(dry_run: bool) -> None:
    print(f"DRY_RUN={dry_run}")
    print(f"MODEL_NAME_TO_USE={MODEL_NAME_TO_USE}")
    print(f"TORCH_DTYPE={TORCH_DTYPE}")
    print(f"ATTN_IMPLEMENTATION={ATTN_IMPLEMENTATION}")
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}")
    print(f"INPUT_ROOT={INPUT_ROOT}")
    print(f"TRANSCRIPT_ROOT={TRANSCRIPT_ROOT}")
    print(f"OUTPUT_ROOT={OUTPUT_ROOT}")
    print(f"INFERENCE_TIME_REPORT_FILE={INFERENCE_TIME_REPORT_FILE}")
    print(f"SEGMENTER_PATH={SEGMENTER_PATH}")
    print(f"LOCAL_REPRESENTATIVE_FRAME_COUNT={LOCAL_REPRESENTATIVE_FRAME_COUNT}")
    print(f"LOCAL_REPRESENTATIVE_FRAME_SAMPLING={LOCAL_REPRESENTATIVE_FRAME_SAMPLING}")
    print(f"UNIT_DESCRIPTION_BATCH_SIZE={UNIT_DESCRIPTION_BATCH_SIZE}")
    print(f"SEGMENT_REASONING_BATCH_SIZE={SEGMENT_REASONING_BATCH_SIZE}")
    print(f"FALLBACK_REASONING_BATCH_SIZE={FALLBACK_REASONING_BATCH_SIZE}")
    print(f"CLEAR_CUDA_CACHE_AFTER_BATCH={CLEAR_CUDA_CACHE_AFTER_BATCH}")
    print(f"EARLY_EXIT_ON_HATE={EARLY_EXIT_ON_HATE}")
    print("V5 policy: forced local hypotheses -> multi-context selection -> single-pass selected-context final judgment")


if __name__ == "__main__":
    if CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    print_runtime_info(DRY_RUN)
    process_all_videos(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        inference_time_report_file=INFERENCE_TIME_REPORT_FILE,
        transcript_root=TRANSCRIPT_ROOT,
        video_id=VIDEO_ID,
        limit=LIMIT,
        overwrite=OVERWRITE,
        dry_run=DRY_RUN,
        model_name=MODEL_NAME_TO_USE,
        device_map=DEVICE_MAP,
        torch_dtype=TORCH_DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
