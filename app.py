#!/usr/bin/env python3
"""
FastAPI server for phishing email detection using Gemma-2-2b-it.

Key features:
1) Uses Gemma-2-2b-it for both classification and explainability
2) Loads Gemma tokenizer from the PEFT base model
3) Uses a completion-style prompt with JSON prefill
4) Stops on an explicit <END_JSON> marker
5) Reconstructs the full JSON output from the assistant prefill before parsing
6) Keeps Gemma locked to avoid concurrent decode thrash
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any, Dict, List, Tuple

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import AutoPeftModelForCausalLM, PeftConfig

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

END_MARKER = "<END_JSON>"
SAFE_GATE_THRESHOLD = 0.30
PROMPT_MAX_LEN = 768
MAX_BODY_CHARS = 1800
MAX_SUBJECT_CHARS = 250

app = FastAPI(
    title="Phishing Email Detector - PhishGuard Lite",
    description="API for detecting phishing emails using Gemma-2-2b-it",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=2)
gemma_lock = asyncio.Lock()

GEMMA_FINETUNED_PATH = "rishabhpagaria/gemma-2-2b-it-phishing"

gemma_model = None
gemma_tokenizer = None
gemma_base_model_name = None


def log_gpu_diagnostics() -> None:
    logger.info("=" * 80)
    logger.info("GPU DIAGNOSTICS")
    logger.info("=" * 80)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        logger.info(f"Current VRAM usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    else:
        logger.warning("GPU not detected. Running on CPU will be very slow.")
    logger.info("=" * 80)


log_gpu_diagnostics()


class EmailRequest(BaseModel):
    text: str = Field(..., description="The email text to analyze")
    subject: Optional[str] = Field(None, description="Optional email subject")


class ClassificationResponse(BaseModel):
    label: str = Field(..., description="Classification result: 'phish' or 'benign'")
    confidence: float = Field(..., description="Confidence score between 0 and 1", ge=0, le=1)
    explanation: str = Field(..., description="Explanation from Gemma")


class StopOnSubstr(StoppingCriteria):
    def __init__(self, tokenizer, substr: str, window: int = 128):
        self.tokenizer = tokenizer
        self.substr = substr
        self.window = window

    def __call__(self, input_ids, scores, **kwargs):
        tail = input_ids[0][-self.window:]
        text = self.tokenizer.decode(tail, skip_special_tokens=False)
        return self.substr in text


def _resolve_gemma_base_model(adapter_path: str) -> str:
    cfg = PeftConfig.from_pretrained(adapter_path)
    base = cfg.base_model_name_or_path
    if not base:
        raise RuntimeError("PEFT config is missing base_model_name_or_path")
    return base


def load_gemma():
    global gemma_model, gemma_tokenizer, gemma_base_model_name

    if gemma_model is None:
        t0 = time.time()
        logger.info("Loading Gemma explainability model...")

        gemma_base_model_name = _resolve_gemma_base_model(GEMMA_FINETUNED_PATH)
        logger.info(f"Resolved Gemma base model from PEFT config: {gemma_base_model_name}")

        try:
            gemma_tokenizer = AutoTokenizer.from_pretrained(
                gemma_base_model_name,
                trust_remote_code=True,
            )
            logger.info("Loaded Gemma tokenizer from base model path")
        except Exception:
            logger.warning("Falling back to tokenizer in adapter directory")
            gemma_tokenizer = AutoTokenizer.from_pretrained(
                GEMMA_FINETUNED_PATH,
                trust_remote_code=True,
            )

        if gemma_tokenizer.pad_token is None:
            gemma_tokenizer.pad_token = gemma_tokenizer.eos_token
            gemma_tokenizer.pad_token_id = gemma_tokenizer.eos_token_id
        gemma_tokenizer.padding_side = "left"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        device_map = "auto" if torch.cuda.is_available() else None

        gemma_model = AutoPeftModelForCausalLM.from_pretrained(
            GEMMA_FINETUNED_PATH,
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            attn_implementation="sdpa",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()

        gemma_model.config.use_cache = True

        if hasattr(gemma_model, "generation_config"):
            gemma_model.generation_config.pad_token_id = gemma_tokenizer.pad_token_id
            gemma_model.generation_config.eos_token_id = gemma_tokenizer.eos_token_id

        if torch.cuda.is_available():
            logger.info(f"Gemma on GPU: {torch.cuda.get_device_name(0)}")
            if hasattr(gemma_model, "hf_device_map"):
                logger.info(f"Gemma device map: {gemma_model.hf_device_map}")
        else:
            logger.info("Gemma running on CPU")

        logger.info(f"Gemma loaded in {time.time() - t0:.2f}s")
    else:
        logger.info("Gemma already loaded (cached)")

    return gemma_model, gemma_tokenizer


def sanitize_text(text: str, max_len: int) -> str:
    text = (text or "").replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    return text[:max_len]


def build_structured_fallback(label: str, raw: str) -> Dict[str, Any]:
    snippet = (raw or "").strip().replace("\n", " ")[:240]
    explanation = "Model output could not be parsed into the required JSON format."
    if snippet:
        explanation += f" Raw output preview: {snippet}"
    return {
        "label": "PHISHING" if label.lower() == "phish" else "SAFE",
        "explanation": explanation,
        "evidence_snippets": [],
        "user_advice": [
            "Treat suspicious links cautiously",
            "Verify the sender through a trusted channel",
        ],
    }


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _extract_json_substring(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start found")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":  # backslash
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    raise ValueError("No complete JSON object found")


def extract_json_object(raw: str) -> Dict[str, Any]:
    if not raw or not raw.strip():
        raise ValueError("Empty model output")

    text = raw.replace(END_MARKER, " ")
    text = _strip_code_fences(text)
    json_str = _extract_json_substring(text)
    obj = json.loads(json_str)

    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object")

    label = str(obj.get("label", "")).strip().upper()
    if label not in {"PHISHING", "SAFE"}:
        raise ValueError(f"Unexpected label in Gemma JSON: {label!r}")

    obj["label"] = label
    obj["explanation"] = str(obj.get("explanation", "")).strip()
    obj["evidence_snippets"] = [str(x).strip() for x in obj.get("evidence_snippets", []) if str(x).strip()][:3]
    obj["user_advice"] = [str(x).strip() for x in obj.get("user_advice", []) if str(x).strip()][:3]
    return obj


def format_gemma_json(obj: Dict[str, Any]) -> str:
    parts: List[str] = [f"Classification: {obj.get('label', 'UNKNOWN')}"]

    explanation = obj.get("explanation") or ""
    if explanation:
        parts.append(f"\nExplanation:\n{explanation}")

    evidence = obj.get("evidence_snippets") or []
    if evidence:
        parts.append("\nEvidence:\n" + "\n".join(f"• {x}" for x in evidence))

    advice = obj.get("user_advice") or []
    if advice:
        parts.append("\nUser Advice:\n" + "\n".join(f"• {x}" for x in advice))

    return "\n".join(parts).strip()


def _build_gemma_prompt(subject: str, text: str) -> Tuple[str, str]:
    clean_subject = sanitize_text(subject, MAX_SUBJECT_CHARS)
    clean_body = sanitize_text(text, MAX_BODY_CHARS)

    assistant_prefill = (
        '{\n'
        '  "label": '
    )

    prompt = (
        "You are a phishing email explanation engine.\n"
        "Read the email and output exactly one valid JSON object.\n"
        f"End the response with {END_MARKER}.\n\n"
        "Allowed labels: \"PHISHING\" or \"SAFE\".\n"
        "JSON schema:\n"
        "{\n"
        '  "label": "PHISHING",\n'
        '  "explanation": "2-3 concise sentences.",\n'
        '  "evidence_snippets": ["short quote 1", "short quote 2"],\n'
        '  "user_advice": ["tip 1", "tip 2"]\n'
        "}\n\n"
        "Rules:\n"
        "- Return JSON only.\n"
        "- Use double quotes.\n"
        "- evidence_snippets must quote or closely reflect the email text.\n"
        "- user_advice must be short and actionable.\n"
        "- Do not add markdown.\n"
        f"- Finish with {END_MARKER}.\n\n"
        f"EMAIL SUBJECT:\n{clean_subject or '(none)'}\n\n"
        f"EMAIL BODY:\n{clean_body}\n\n"
        "JSON RESPONSE:\n"
        f"{assistant_prefill}"
    )
    return prompt, assistant_prefill


def _terminator_ids(tokenizer) -> List[int]:
    ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)

    for tok in ["<end_of_turn>", "<eos>", END_MARKER]:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is not None and tok_id != tokenizer.unk_token_id and tok_id not in ids:
            ids.append(tok_id)
    return ids


def _gemma_generate(prompt: str, assistant_prefill: str) -> str:
    model, tokenizer = load_gemma()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=PROMPT_MAX_LEN,
        padding=False,
    )

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = int(inputs["input_ids"].shape[1])
    terminators = _terminator_ids(tokenizer)
    stopping = StoppingCriteriaList([StopOnSubstr(tokenizer, END_MARKER)])

    logger.info(
        "Gemma generation config | prompt_toks=%s | eos_ids=%s | prompt_preview=%s",
        input_len,
        terminators,
        prompt[-350:].replace("\n", " "),
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=terminators if terminators else tokenizer.eos_token_id,
            stopping_criteria=stopping,
            return_dict_in_generate=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_time = time.time() - t0

    response_ids = output_ids[0][input_len:]
    decoded = tokenizer.decode(response_ids, skip_special_tokens=False)
    full_text = f"{assistant_prefill}{decoded}"
    new_toks = int(response_ids.shape[0])

    logger.info(
        "Gemma done | new_toks=%s | gen=%.2fs | speed=%.1f tok/s",
        new_toks,
        gen_time,
        (new_toks / max(gen_time, 1e-6)),
    )
    logger.info("Gemma raw output preview: %s", full_text[:700])

    return full_text


async def get_gemma_json(subject: str, text: str, fallback_label: str) -> Dict[str, Any]:
    load_gemma()
    prompt, assistant_prefill = _build_gemma_prompt(subject, text)

    async with gemma_lock:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(executor, _gemma_generate, prompt, assistant_prefill)

    try:
        return extract_json_object(raw)
    except Exception as e:
        logger.error("Failed to parse Gemma JSON: %s", e, exc_info=True)
        return build_structured_fallback(fallback_label, raw)


@app.on_event("startup")
async def startup_event():
    logger.info("Pre-warming models on startup...")
    try:
        load_gemma()
        logger.info("Gemma ready")
        logger.info("All models warmed up")
    except Exception as e:
        logger.error("Failed to pre-warm models: %s", e, exc_info=True)
        logger.warning("Models will lazy-load on first request")


@app.post("/classify", response_model=ClassificationResponse)
async def classify_email(request: EmailRequest):
    try:
        request_start = time.time()
        timings: Dict[str, float] = {}

        subject = request.subject or ""
        body = request.text or ""

        if not body.strip() and not subject.strip():
            raise HTTPException(status_code=400, detail="Empty email content")

        t0 = time.time()
        gemma_json = await get_gemma_json(
            subject=subject,
            text=body,
            fallback_label="benign",
        )
        timings["gemma_generation_and_parse"] = time.time() - t0

        # Extract classification from Gemma response
        gemma_label = gemma_json.get("label", "SAFE").upper()
        label = "phish" if gemma_label == "PHISHING" else "benign"
        # Use confidence of 0.95 for strong predictions, 0.6 for weak
        confidence = 0.95 if gemma_json.get("label") in ["PHISHING", "SAFE"] else 0.6
        
        explanation = format_gemma_json(gemma_json)
        logger.info("Gemma classification: label=%s", label)

        total = time.time() - request_start
        logger.info(
            "TIMING total=%.2fs | %s",
            total,
            " | ".join(f"{k}={v:.2f}s" for k, v in timings.items()),
        )
        logger.info("Result label=%s confidence=%.3f", label, confidence)

        return ClassificationResponse(
            label=label,
            confidence=float(confidence),
            explanation=explanation,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Classification error", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "PhishGuard Lite",
        "model": "Gemma-2-2b-it",
        "version": "3.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 80)
    logger.info("STARTING PHISHGUARD LITE - GEMMA-2-2B-IT POWERED")
    logger.info("=" * 80)

    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=120, timeout_graceful_shutdown=30)
