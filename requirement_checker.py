import base64
import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Tuple

import cv2
import requests
import torch
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Ollama / LLaVA configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
LLAVA_MODEL = os.getenv("LLAVA_MODEL_ID", "llava")  # Ollama model name (e.g., llava, llava:13b, etc.)


def sample_frames(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """Uniformly sample frames from a video and return as PIL Images (RGB)."""
    cap = cv2.VideoCapture(video_path)
    frames: List[Image.Image] = []
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length == 0:
        cap.release()
        raise ValueError(f"No frames found in video: {video_path}")

    indices = torch.linspace(0, length - 1, steps=num_frames).long().tolist()
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()

    if not frames:
        raise ValueError(f"Failed to extract frames from video: {video_path}")

    return frames


def _normalize_requirements(video_requirements: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Convert incoming requirements into a list of {id, text}.
    Supports:
    - {"requirements": [...]} where each item is str or {"id", "text"}
    - a plain list of strings or dicts
    - legacy flags (require_top_down_view, require_clean_workspace, allowed_objects, forbidden_objects)
    """
    req = video_requirements or {}

    def to_list(item: Any) -> List[Any]:
        if isinstance(item, list):
            return item
        return []

    raw_reqs: List[Any] = []
    if isinstance(req, dict) and "requirements" in req:
        raw_reqs = to_list(req.get("requirements", []))
    elif isinstance(req, list):
        raw_reqs = req

    normalized: List[Dict[str, Any]] = []
    if raw_reqs:
        for idx, item in enumerate(raw_reqs, 1):
            if isinstance(item, dict):
                text = item.get("text") or item.get("requirement") or ""
                rid = item.get("id", idx)
            else:
                text = str(item)
                rid = idx
            if text:
                normalized.append({"id": rid, "text": text})

    # Fallback to legacy flags if no explicit requirements list is provided
    if not normalized and isinstance(req, dict):
        if req.get("require_top_down_view"):
            normalized.append({"id": 1, "text": "Video is shot from a top-down perspective."})
        if req.get("require_clean_workspace"):
            normalized.append({"id": 2, "text": "Workspace is clean and uncluttered."})
        allowed = req.get("allowed_objects") or []
        if allowed:
            normalized.append({"id": 3, "text": f"Only allowed objects visible: {allowed}."})
        forbidden = req.get("forbidden_objects") or []
        if forbidden:
            normalized.append({"id": 4, "text": f"No forbidden objects present: {forbidden}."})

    # Absolute fallback to avoid empty prompt
    if not normalized:
        normalized.append({"id": 1, "text": "Follow the task description closely."})

    task_description = req.get("task_description", "No description provided.")
    return task_description, normalized


def build_prompt(video_requirements: Dict[str, Any]) -> str:
    """Create a frame-specific prompt that reflects dynamic requirements."""
    task_description, requirements = _normalize_requirements(video_requirements)
    req_lines = "\n".join(f"{r['id']}. {r['text']}" for r in requirements)
    prompt = f"""You are verifying whether this video frame satisfies the company's video requirements.

Task Description:
{task_description}

Requirements to Check (label each as fulfilled / violated / unknown):
{req_lines}

Be strict: if evidence is missing, mark the requirement as "unknown".

Return ONLY valid JSON with the following structure:
{{
  "requirements": [
    {{
      "id": <number>,
      "text": "the exact text of the requirement",
      "status": "fulfilled/violated/unknown",
      "reason": "short explanation"
    }}
  ]
}}
"""
    return prompt


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """Robustly extract JSON from model output."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    fence_match = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        inner = fence_match.group(1).strip()
        return json.loads(inner)

    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        inner = brace_match.group(0)
        return json.loads(inner)

    # Fallback
    return json.loads(text)


def encode_image_to_base64(img: Image.Image) -> str:
    """Encode a PIL image to base64 for Ollama."""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def call_llava_via_ollama(prompt_text: str, frame: Image.Image) -> str:
    """Send a single frame + prompt to Ollama and return raw text."""
    image_b64 = encode_image_to_base64(frame)
    payload = {
        "model": LLAVA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "images": [image_b64],
    }

    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    resp_data = resp.json()
    return resp_data["message"]["content"]


def run_llava_on_frame(frame: Image.Image, prompt_text: str) -> str:
    """Run LLaVA via Ollama on a single frame with the given prompt and return decoded text."""
    return call_llava_via_ollama(prompt_text, frame)


def parse_llava_json(response: str) -> Dict[str, Any]:
    """
    Extract JSON from the model response. Tries direct parse; falls back to brace matching.
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    try:
        return _extract_json_from_text(response)
    except Exception:
        return {}


def evaluate_frame_against_requirements(
    frame: Image.Image, frame_index: int, video_requirements: Dict[str, Any]
) -> Dict[str, Any]:
    """Run a single-frame check with LLaVA and normalize the result."""
    prompt = build_prompt(video_requirements)
    raw_output = run_llava_on_frame(frame, prompt)
    parsed = parse_llava_json(raw_output)

    reqs = parsed.get("requirements") if isinstance(parsed, dict) else None
    normalized_reqs: List[Dict[str, Any]] = []
    if isinstance(reqs, list):
        for idx, item in enumerate(reqs, 1):
            if not isinstance(item, dict):
                continue
            normalized_reqs.append(
                {
                    "id": item.get("id", idx),
                    "text": item.get("text", ""),
                    "status": str(item.get("status", "unknown")).lower(),
                    "reason": item.get("reason", ""),
                }
            )

    frame_result = {
        "frame_index": frame_index,
        "requirements": normalized_reqs,
        "raw_output": raw_output,
    }
    return frame_result


def aggregate_results(frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall pass/fail and aggregate violations across frames."""
    per_req_statuses: Dict[Any, Dict[str, Any]] = {}

    for fr in frame_results:
        for item in fr.get("requirements", []):
            rid = item.get("id")
            if rid is None:
                continue
            entry = per_req_statuses.setdefault(
                rid,
                {"text": item.get("text", ""), "statuses": [], "reasons": []},
            )
            status = str(item.get("status", "unknown")).lower()
            entry["statuses"].append(status)
            if item.get("reason"):
                entry["reasons"].append(item["reason"])

    summary = []
    aggregated_violations: List[str] = []
    for rid, data in per_req_statuses.items():
        statuses = data["statuses"]
        if "violated" in statuses:
            final_status = "violated"
        elif "fulfilled" in statuses:
            final_status = "fulfilled"
        else:
            final_status = "unknown"

        summary.append(
            {
                "id": rid,
                "text": data.get("text", ""),
                "final_status": final_status,
                "frame_statuses": statuses,
                "reasons": data.get("reasons", []),
            }
        )

        if final_status != "fulfilled":
            aggregated_violations.append(f"Requirement {rid}: {data.get('text', '')} -> {final_status}")

    summary.sort(key=lambda x: x["id"])
    overall_pass = all(item["final_status"] == "fulfilled" for item in summary) if summary else False

    return {
        "overall_pass": bool(overall_pass),
        "requirements_summary": summary,
        "frame_results": frame_results,
        "aggregated_violations": aggregated_violations,
    }


def check_video_requirements(video_path: str, video_requirements: Dict[str, Any], num_frames: int = 8) -> Dict[str, Any]:
    """
    High-level entry point: sample frames, run LLaVA checks per frame via Ollama, aggregate.
    """
    frames = sample_frames(video_path, num_frames=num_frames)
    frame_results: List[Dict[str, Any]] = []

    for idx, frame in enumerate(frames):
        try:
            fr = evaluate_frame_against_requirements(frame, idx, video_requirements)
            frame_results.append(fr)
        except Exception as e:
            logger.warning(f"Frame {idx} evaluation failed: {e}")
            frame_results.append(
                {
                    "frame_index": idx,
                    "top_down_view": False,
                    "only_task_visible": False,
                    "allowed_objects_ok": False,
                    "forbidden_objects_present": [],
                    "workspace_clean": False,
                    "violations": [f"model_error: {e}"],
                    "raw_output": "",
                }
            )

    return aggregate_results(frame_results)


if __name__ == "__main__":
    # Example usage placeholder. Replace video_path and requirements as needed.
    sample_requirements = {
        "require_top_down_view": True,
        "require_clean_workspace": True,
        "allowed_objects": ["screwdriver", "screws", "metal plate"],
        "forbidden_objects": ["phone", "cup", "laptop"],
        "task_description": "Assembly demonstration recorded from top-down on a clean workspace with only the required tools.",
    }
    # result = check_video_requirements("path/to/video.mp4", sample_requirements, num_frames=4)
    # print(json.dumps(result, indent=2))
    pass
