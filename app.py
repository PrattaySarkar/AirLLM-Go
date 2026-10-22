from __future__ import annotations

import os
import json
import logging
import re
import shutil
import threading
import uuid
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request, send_from_directory
from werkzeug.utils import secure_filename


app = Flask(__name__)

CONFIG_FILE = Path(__file__).with_name("airllm-go-config.json")
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def read_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        value = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def cache_path_from_config(config: dict[str, Any] | None = None) -> Path:
    config = config or read_config()
    configured = str(config.get("cache_path", "")).strip()
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "huggingface"


def apply_cache_config(config: dict[str, Any] | None = None) -> Path:
    cache_path = cache_path_from_config(config)
    cache_path.mkdir(parents=True, exist_ok=True)
    # These are read by huggingface_hub when AirLLM is imported lazily.
    os.environ["HF_HOME"] = str(cache_path)
    os.environ["HF_HUB_CACHE"] = str(cache_path / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "transformers")
    return cache_path


apply_cache_config()


class AirLLMEngine:
    """Small lazy wrapper around AirLLM's AutoModel API."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_name: str | None = None
        self.lock = threading.Lock()

    def load(self, model_name: str, compression: str | None = None) -> None:
        if self.model is not None and self.model_name == model_name:
            return
        with self.lock:
            if self.model is not None and self.model_name == model_name:
                return
            try:
                update_status("loading", 12, f"Loading {model_name}")
                from airllm import AutoModel
                import torch
            except ImportError as exc:
                raise RuntimeError(
                    "AirLLM is not installed. Run: pip install -r requirements.txt"
                ) from exc

            kwargs = {}
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            kwargs["device"] = device
            terminal_log(f"AirLLM inference device: {device}", "INFO")
            if compression in {"4bit", "8bit"}:
                kwargs["compression"] = compression
            cache_path = cache_path_from_config()
            free_gb = shutil.disk_usage(cache_path).free / (1024 ** 3)
            estimated_gb = 1.2 if "0.6B" in model_name else 2.5 if any(x in model_name for x in ("4B", "3B")) else 5.5 if any(x in model_name for x in ("7B", "8B")) else 8.0
            if free_gb < estimated_gb:
                raise RuntimeError(
                    f"Not enough space in {cache_path}. Free: {free_gb:.2f} GB; "
                    f"estimated model cache requirement: {estimated_gb:.2f} GB. "
                    "Choose another drive in AirLLM-Go setup."
                )
            update_status("loading", 28, "Preparing AirLLM layer shards")
            terminal_log("Entering AirLLM model preparation; live downloader output follows", "AIRLLM")
            with capture_runtime_output():
                self.model = AutoModel.from_pretrained(model_name, **kwargs)
            self.model_name = model_name
            update_status("ready", 60, "Model loaded and ready")

    def reply(
        self,
        messages: list[dict[str, str]],
        model_name: str,
        compression: str | None,
        max_new_tokens: int,
    ) -> str:
        self.load(model_name, compression)
        assert self.model is not None
        update_status("tokenizing", 68, "Tokenizing conversation")
        tokenizer = self.model.tokenizer
        prompt = "\n".join(
            f"{item['role'].title()}: {item['content']}" for item in messages
        ) + "\nAssistant:"
        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=4096,
            padding=False,
        )
        update_status("generating", 78, "Generating response")
        with capture_runtime_output():
            generation = self.model.generate(
                inputs["input_ids"].to(self.model.device),
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
            )
        full = tokenizer.decode(generation.sequences[0], skip_special_tokens=True)
        if "Assistant:" in full:
            full = full.split("Assistant:", 1)[-1]
        update_status("complete", 100, "Response complete")
        return full.strip() or "I generated an empty response. Try a little more context."


engine = AirLLMEngine()
HISTORY_FILE = Path(__file__).with_name("conversations.json")
UPLOAD_DIR = Path(__file__).with_name("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
TERMINAL_LOG_FILE = Path(__file__).with_name("airllm-go-terminal.log")
TERMINAL_LOG_LOCK = threading.Lock()


class RuntimeStream:
    def __init__(self, original):
        self.original = original
        self.buffer = ""

    def write(self, value: str) -> int:
        if not value:
            return 0
        self.original.write(value)
        self.original.flush()
        self.buffer += value.replace("\r", "\n")
        parts = self.buffer.split("\n")
        self.buffer = parts.pop()
        for part in parts:
            cleaned = " ".join(part.split())
            if cleaned:
                terminal_log(cleaned, "AIRLLM")
        return len(value)

    def flush(self) -> None:
        self.original.flush()

    def isatty(self) -> bool:
        return False


@contextmanager
def capture_runtime_output():
    import sys
    out = RuntimeStream(sys.stdout)
    err = RuntimeStream(sys.stderr)
    with redirect_stdout(out), redirect_stderr(err):
        yield
    for stream in (out, err):
        if stream.buffer.strip():
            terminal_log(" ".join(stream.buffer.split()), "AIRLLM")


def terminal_log(message: str, level: str = "INFO") -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} [{level}] {message}"
    with TERMINAL_LOG_LOCK:
        with TERMINAL_LOG_FILE.open("a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")
    shard = re.search(r"Loading shard\s+(\d+)\s*/\s*(\d+)", message, re.IGNORECASE)
    match = re.search(r"(\d{1,3})\s*%", message)
    if match and "RUNTIME_STATUS" in globals():
        with STATUS_LOCK:
            percent = int(match.group(1))
            # AirLLM prints an inner tqdm counter as 0% while it is still
            # reading/converting the current shard. Do not make the whole UI
            # look idle during that CPU-heavy phase.
            if percent == 0 and "0%|" in message and RUNTIME_STATUS.get("phase") == "loading":
                RUNTIME_STATUS["progress"] = max(32, int(RUNTIME_STATUS.get("progress", 0)))
                RUNTIME_STATUS["message"] = "Preparing model weights on CPU — the first shard can take a few minutes"
            else:
                RUNTIME_STATUS["progress"] = max(0, min(99, percent))
                RUNTIME_STATUS["message"] = message[:180]
            RUNTIME_STATUS["busy"] = True
    if shard and "RUNTIME_STATUS" in globals():
        current, total = int(shard.group(1)), int(shard.group(2))
        with STATUS_LOCK:
            RUNTIME_STATUS["progress"] = max(32, min(58, 32 + int((current - 1) / max(total, 1) * 26)))
            RUNTIME_STATUS["message"] = f"Loading model shard {current} of {total} — preparing CPU weights"
            RUNTIME_STATUS["busy"] = True
HISTORY_LOCK = threading.Lock()
STATUS_LOCK = threading.Lock()
RUNTIME_STATUS: dict[str, Any] = {"phase": "idle", "progress": 0, "message": "AirLLM ready", "busy": False, "logs": []}


def update_status(phase: str, progress: int, message: str) -> None:
    terminal_log(message, "ERROR" if phase == "error" else "INFO")
    with STATUS_LOCK:
        RUNTIME_STATUS.update(phase=phase, progress=max(0, min(100, progress)), message=message, busy=phase not in {"idle", "complete", "error"})
        RUNTIME_STATUS["logs"].append({"time": datetime.now().strftime("%H:%M:%S"), "message": message})
        RUNTIME_STATUS["logs"] = RUNTIME_STATUS["logs"][-80:]


def current_status() -> dict[str, Any]:
    with STATUS_LOCK:
        return dict(RUNTIME_STATUS, logs=list(RUNTIME_STATUS["logs"]))


def terminal_tail(limit: int = 160) -> list[str]:
    if not TERMINAL_LOG_FILE.exists():
        return []
    with TERMINAL_LOG_LOCK:
        return TERMINAL_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]


terminal_log("AirLLM-Go backend initialized")


@app.after_request
def log_request(response):
    if request.path.startswith("/api/") and request.path != "/api/status" and request.path != "/api/terminal":
        terminal_log(f"{request.method} {request.path} → {response.status_code}", "ERROR" if response.status_code >= 400 else "HTTP")
    return response


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_conversations() -> list[dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def write_conversations(conversations: list[dict[str, Any]]) -> None:
    temp = HISTORY_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(conversations, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(HISTORY_FILE)


def save_conversation(conversation: dict[str, Any]) -> None:
    with HISTORY_LOCK:
        conversations = read_conversations()
        for index, item in enumerate(conversations):
            if item["id"] == conversation["id"]:
                conversations[index] = conversation
                break
        else:
            conversations.append(conversation)
        write_conversations(conversations)


def new_conversation(title: str = "New conversation") -> dict[str, Any]:
    stamp = now_iso()
    return {"id": uuid.uuid4().hex, "title": title, "created_at": stamp, "updated_at": stamp, "model": read_config().get("default_model", DEFAULT_MODEL), "messages": []}


PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AirLLM-Go — local AI workspace</title>
  <style>
    :root { --bg:#f7f7f5; --panel:#fff; --ink:#262624; --muted:#77756e; --line:#e6e4df; --accent:#d97757; --accent-soft:#f2e7df; --shadow:0 18px 50px rgba(46,38,30,.08); }
    body.dark { --bg:#20211f; --panel:#2b2c29; --ink:#f2f0eb; --muted:#aaa79f; --line:#42433e; --accent:#e59a78; --accent-soft:#49362e; --shadow:0 18px 50px rgba(0,0,0,.25); } body.dark aside { background:#292a27; } body.dark header { background:rgba(38,39,36,.78); } body.dark .composer, body.dark .card { background:#2b2c29; } body.dark .new, body.dark input, body.dark select, body.dark .model-menu { background:#30312d; color:var(--ink); border-color:#4a4b45; } body.dark .model-option:hover, body.dark .model-option.selected { background:#49362e; } body.dark .icon:hover, body.dark .tool:hover { background:#3a3b36; }
    body.sepia { --bg:#f4ecdf; --panel:#fff9ef; --ink:#3c3028; --muted:#88766a; --line:#dfd1c1; --accent:#ad624a; --accent-soft:#ead9c8; --shadow:0 18px 50px rgba(85,55,35,.1); } body.sepia aside { background:#eee3d4; } body.sepia header { background:rgba(255,249,239,.72); }
    body.ocean { --bg:#eaf4f7; --panel:rgba(255,255,255,.62); --ink:#15313b; --muted:#66818a; --line:rgba(82,145,164,.24); --accent:#2b9ab2; --accent-soft:#d7edf2; --shadow:0 20px 60px rgba(39,111,131,.13); } body.ocean aside { background:rgba(218,239,244,.68); }
    body.lavender { --bg:#f1effa; --panel:rgba(255,255,255,.68); --ink:#2f2948; --muted:#7f7899; --line:rgba(126,113,179,.23); --accent:#8170c7; --accent-soft:#e5e0f6; --shadow:0 20px 60px rgba(93,76,157,.13); } body.lavender aside { background:rgba(228,223,247,.68); }
    body.forest { --bg:#edf4ed; --panel:rgba(255,255,255,.65); --ink:#213629; --muted:#718879; --line:rgba(78,125,88,.22); --accent:#4c9360; --accent-soft:#dceede; --shadow:0 20px 60px rgba(56,112,68,.13); } body.forest aside { background:rgba(220,238,221,.7); }
    body.sunset { --bg:#fbefea; --panel:rgba(255,255,255,.66); --ink:#432d2c; --muted:#96736d; --line:rgba(185,107,94,.23); --accent:#d46e59; --accent-soft:#f6dcd4; --shadow:0 20px 60px rgba(180,92,75,.14); } body.sunset aside { background:rgba(248,222,215,.7); }
    body.obsidian { --bg:#151619; --panel:rgba(38,40,45,.68); --ink:#f3f2ed; --muted:#aaaab0; --line:rgba(255,255,255,.12); --accent:#c6a6ff; --accent-soft:rgba(198,166,255,.16); --shadow:0 25px 80px rgba(0,0,0,.35); } body.obsidian aside, body.obsidian header { background:rgba(25,27,31,.72); } body.obsidian .new, body.obsidian input, body.obsidian select, body.obsidian .model-menu { background:rgba(44,46,52,.78); color:var(--ink); border-color:var(--line); }
    body.compact .conversation { width:min(980px,calc(100% - 38px)); } body.compact .message { margin-bottom:18px; } body.large-text { font-size:17px; }
    * { box-sizing:border-box; } body { margin:0; color:var(--ink); background:var(--bg); font:15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { overflow-x:hidden; transition:background .6s ease,color .35s ease; } body::before { content:""; position:fixed; z-index:-2; inset:0; pointer-events:none; background:radial-gradient(circle at 12% 8%, color-mix(in srgb, var(--accent) 12%, transparent), transparent 28%),radial-gradient(circle at 88% 18%, color-mix(in srgb, var(--accent) 9%, transparent), transparent 25%); animation:ambientDrift 18s ease-in-out infinite alternate; } body.pattern-grid::after { content:""; position:fixed; z-index:-1; inset:0; pointer-events:none; opacity:.27; background-image:linear-gradient(var(--line) 1px,transparent 1px),linear-gradient(90deg,var(--line) 1px,transparent 1px); background-size:42px 42px; mask-image:linear-gradient(to bottom,black,transparent 88%); } body.pattern-dots::after { content:""; position:fixed; z-index:-1; inset:0; pointer-events:none; opacity:.3; background-image:radial-gradient(var(--accent) 1px,transparent 1.5px); background-size:24px 24px; mask-image:linear-gradient(to bottom,black,transparent 80%); } body.pattern-diamonds::after { content:""; position:fixed; z-index:-1; inset:0; pointer-events:none; opacity:.22; background-image:linear-gradient(45deg,var(--line) 25%,transparent 25%,transparent 75%,var(--line) 75%),linear-gradient(45deg,var(--line) 25%,transparent 25%,transparent 75%,var(--line) 75%); background-position:0 0,12px 12px; background-size:24px 24px; mask-image:linear-gradient(to bottom,black,transparent 85%); } body.pattern-waves::after { content:""; position:fixed; z-index:-1; inset:0; pointer-events:none; opacity:.2; background-image:radial-gradient(ellipse at 100% 0%,transparent 0 55%,var(--accent-soft) 56% 57%,transparent 58%),radial-gradient(ellipse at 0% 100%,transparent 0 55%,var(--accent-soft) 56% 57%,transparent 58%); background-size:360px 240px; animation:patternFloat 25s linear infinite; }
    body.reduced *, body.reduced *::before, body.reduced *::after { animation-duration:.001ms !important; animation-iteration-count:1 !important; transition-duration:.001ms !important; }
    button, textarea, input, select { font:inherit; } button { cursor:pointer; border:0; }
    .app { min-height:100vh; display:grid; grid-template-columns:270px 1fr; }
    aside { background:color-mix(in srgb, var(--panel) 76%, transparent); border-right:1px solid var(--line); display:flex; flex-direction:column; padding:22px 16px 16px; backdrop-filter:blur(24px) saturate(135%); -webkit-backdrop-filter:blur(24px) saturate(135%); }
    .brand { display:flex; align-items:center; gap:10px; padding:2px 10px 25px; font-family:Georgia,serif; font-size:24px; letter-spacing:-.03em; }
    .mark { width:30px; height:30px; border-radius:9px; background:linear-gradient(135deg,var(--accent),color-mix(in srgb,var(--accent) 55%,#fff)); display:grid; place-items:center; color:#fff; box-shadow:0 5px 16px color-mix(in srgb,var(--accent) 35%,transparent); animation:floatMark 4s ease-in-out infinite; overflow:hidden; } .mark svg { width:24px; height:24px; filter:drop-shadow(0 1px 1px rgba(0,0,0,.15)); } .mark .cut { fill:var(--accent-soft); }
    .new { width:100%; text-align:left; padding:11px 13px; border:1px solid #d8d5ce; background:rgba(255,255,255,.5); border-radius:9px; color:var(--ink); transition:.2s; }
    .new:hover { background:#fff; box-shadow:var(--shadow); } .section { margin:27px 10px 10px; color:#97938b; font-size:11px; text-transform:uppercase; letter-spacing:.11em; }
    .history { color:#5e5b55; font-size:14px; } .history-row { display:flex; align-items:center; gap:3px; margin:2px 0; } .history-row button:first-child { flex:1; text-align:left; padding:9px 10px; border-radius:7px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; background:transparent; color:inherit; } .history-row button:first-child:hover, .history-row.active button:first-child { background:#e6e4de; color:var(--ink); } .history-row .delete { opacity:0; padding:5px 7px; border-radius:6px; background:transparent; color:#999; } .history-row:hover .delete { opacity:1; } .history-row .delete:hover { background:#e6e4de; color:#b15b45; } .empty-history { color:#9b978f; padding:9px 10px; font-size:13px; }
    .bottom { margin-top:auto; border-top:1px solid var(--line); padding-top:15px; color:var(--muted); font-size:13px; } .status { display:flex; align-items:center; gap:8px; padding:8px 10px; }
    .dot { width:8px; height:8px; border-radius:50%; background:#71a67a; box-shadow:0 0 0 4px #dcebdc; }
    main { display:flex; flex-direction:column; min-width:0; } header { height:70px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; padding:0 38px; background:color-mix(in srgb, var(--panel) 72%, transparent); backdrop-filter:blur(24px) saturate(140%); -webkit-backdrop-filter:blur(24px) saturate(140%); position:relative; z-index:5; }
    .model { display:flex; align-items:center; gap:9px; color:#66635d; font-size:14px; } .model strong { color:var(--ink); font-weight:600; } .chev { color:#aaa59c; }
    .header-actions { display:flex; gap:8px; } .icon { background:transparent; padding:7px 9px; color:#77736b; border-radius:7px; } .icon:hover { background:#eceae5; }
    .conversation { width:min(820px, calc(100% - 52px)); margin:0 auto; padding:58px 0 30px; flex:1; }
    .welcome { text-align:center; padding:35px 0 45px; } .welcome h1 { margin:0 0 10px; font:400 38px/1.15 Georgia,serif; letter-spacing:-.04em; } .welcome p { margin:0; color:var(--muted); }
    .message { display:flex; gap:15px; margin:0 0 31px; } .avatar { flex:0 0 31px; height:31px; border-radius:9px; display:grid; place-items:center; font-size:13px; } .user .avatar { background:#363633; color:#fff; } .assistant .avatar { background:var(--accent); color:#fff; font-family:Georgia,serif; }
    .message-body { min-width:0; padding-top:3px; white-space:pre-wrap; } .role { font-weight:600; margin-bottom:5px; font-size:13px; } .assistant .role { color:var(--accent); } .user .role { color:#4b4a46; }
    .composer-wrap { padding:10px 0 32px; } .composer { border:1px solid color-mix(in srgb, var(--line) 85%, var(--accent)); background:color-mix(in srgb, var(--panel) 82%, transparent); border-radius:18px; box-shadow:0 12px 35px color-mix(in srgb, var(--accent) 12%, transparent), inset 0 1px 0 rgba(255,255,255,.55); padding:13px 14px 10px; backdrop-filter:blur(24px) saturate(145%); -webkit-backdrop-filter:blur(24px) saturate(145%); transition:box-shadow .3s ease,transform .3s ease,border-color .3s ease; } .composer:focus-within { transform:translateY(-2px); border-color:var(--accent); box-shadow:0 18px 45px color-mix(in srgb, var(--accent) 18%, transparent),inset 0 1px 0 rgba(255,255,255,.65); }
    textarea { width:100%; min-height:54px; resize:none; outline:0; border:0; color:var(--ink); background:transparent; line-height:1.5; } textarea::placeholder { color:#aaa69f; }
    .composer-footer { display:flex; align-items:center; justify-content:space-between; color:#a09c94; font-size:12px; } .tools { display:flex; gap:3px; align-items:center; } .tool { padding:5px 7px; border-radius:6px; background:transparent; color:#858078; } .tool:hover { background:#f3f1ed; }
    .model-picker { position:relative; } .model-trigger { display:flex; align-items:center; gap:5px; padding:6px 8px; border-radius:7px; background:transparent; color:#77736b; max-width:270px; } .model-trigger:hover, .model-trigger.open { background:#f3f1ed; color:var(--ink); } .model-trigger span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .model-menu { display:none; position:absolute; z-index:10; left:0; bottom:39px; width:330px; padding:8px; background:#fff; border:1px solid #dedbd4; border-radius:11px; box-shadow:0 15px 40px rgba(48,40,30,.15); } .model-menu.open { display:block; }
    .model-search { width:100%; padding:9px 10px; border:1px solid #e1ded8; border-radius:7px; background:#faf9f7; outline:0; font-size:13px; } .model-search:focus { border-color:#c9a590; box-shadow:0 0 0 3px var(--accent-soft); }
    .model-list { max-height:245px; overflow:auto; margin-top:6px; } .model-option { width:100%; display:block; text-align:left; padding:9px 10px; border-radius:7px; background:transparent; color:var(--ink); } .model-option:hover, .model-option.selected { background:var(--accent-soft); } .model-option small { display:block; color:#969189; font-size:11px; margin-top:1px; }
    .send { width:31px; height:31px; border-radius:8px; background:var(--ink); color:#fff; } .send:disabled { background:#d9d7d1; cursor:default; }
    .view-tabs { display:flex; gap:4px; align-items:center; } .view-tab { background:transparent; color:var(--muted); padding:8px 11px; border-radius:8px; transition:.2s; position:relative; } .view-tab:hover, .view-tab.active { color:var(--ink); background:var(--accent-soft); } .view-tab.active::after { content:""; position:absolute; left:14px; right:14px; bottom:2px; height:2px; border-radius:2px; background:var(--accent); animation:tabIn .25s ease both; } .view-panel { display:none; flex:1; min-height:0; } .view-panel.active { display:flex; flex-direction:column; }
    .progress-wrap { height:3px; background:transparent; overflow:hidden; } .progress-bar { width:0; height:100%; background:var(--accent); transition:width .35s ease; } .progress-bar.busy { animation:progressGlow 1.3s ease-in-out infinite; } .terminal-shell { width:min(980px,calc(100% - 52px)); margin:0 auto; padding:30px 0; flex:1; display:flex; flex-direction:column; } .terminal-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; color:var(--muted); font-size:13px; } .terminal-head::before { content:'●  ●  ●'; letter-spacing:3px; color:#d97757; margin-right:auto; font-size:9px; } .terminal { flex:1; min-height:340px; overflow:auto; padding:18px; border-radius:12px; background:linear-gradient(145deg,#1a1d1b,#202a24); color:#b4d6b5; font:12px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace; box-shadow:inset 0 0 0 1px rgba(255,255,255,.08),0 15px 45px rgba(0,0,0,.16); position:relative; } .terminal::after { content:""; position:absolute; inset:0; pointer-events:none; background:repeating-linear-gradient(0deg,transparent 0 3px,rgba(255,255,255,.018) 4px); opacity:.5; } .terminal-line { animation:terminalIn .25s ease both; position:relative; z-index:1; } .terminal-line::before { content:'› '; color:#e59a78; } .terminal-empty { color:#7f9580; position:relative; z-index:1; } .terminal.active-output { animation:terminalGlow 1.8s ease-in-out infinite; } .uncensored-badge { display:inline-block; margin-left:7px; padding:2px 5px; border-radius:4px; color:#9c4c39; background:#f3d9d1; font:700 9px/1.2 ui-sans-serif,sans-serif; letter-spacing:.07em; vertical-align:middle; } body.dark .uncensored-badge { color:#ffc0aa; background:#5a332b; }
    .message { animation:messageIn .35s cubic-bezier(.2,.8,.2,1) both; } .status .dot { animation:statusPulse 2s infinite; } .loading-dots::after { content:' '; animation:dots 1.2s steps(4,end) infinite; } @keyframes messageIn { from { opacity:0; transform:translateY(8px) scale(.99); } to { opacity:1; transform:none; } } @keyframes terminalIn { from { opacity:0; transform:translateX(-5px); } to { opacity:1; transform:none; } } @keyframes statusPulse { 0%,100% { opacity:1; } 50% { opacity:.45; } } @keyframes progressGlow { 0%,100% { filter:brightness(1); } 50% { filter:brightness(1.35); } } @keyframes terminalGlow { 0%,100% { box-shadow:inset 0 0 0 1px rgba(255,255,255,.08),0 15px 45px rgba(0,0,0,.16); } 50% { box-shadow:inset 0 0 0 1px rgba(180,214,181,.28),0 15px 55px rgba(73,148,91,.18); } } @keyframes dots { 0% { content:''; } 25% { content:'.'; } 50% { content:'..'; } 75%,100% { content:'...'; } } @keyframes ambientDrift { from { transform:scale(1) translate3d(0,0,0); } to { transform:scale(1.08) translate3d(1.5%, -1%, 0); } } @keyframes patternFloat { from { background-position:0 0,0 0; } to { background-position:360px 240px,-360px -240px; } } @keyframes floatMark { 0%,100% { transform:translateY(0) rotate(0); } 50% { transform:translateY(-3px) rotate(3deg); } } @keyframes tabIn { from { transform:scaleX(.2); opacity:0; } to { transform:scaleX(1); opacity:1; } }
    .thinking-box { display:none; width:min(820px,calc(100% - 52px)); margin:18px auto 0; padding:10px 13px; align-items:center; gap:10px; color:var(--muted); font-size:13px; border-left:2px solid var(--accent); background:var(--accent-soft); border-radius:0 8px 8px 0; animation:thinkingIn .3s ease both; } .thinking-box.visible { display:flex; } .thinking-orb { width:9px; height:9px; border-radius:50%; background:var(--accent); animation:thinkingPulse 1s ease-in-out infinite; } .thinking-box strong { color:var(--ink); margin-right:4px; } @keyframes thinkingIn { from { opacity:0; transform:translateY(-5px); } to { opacity:1; transform:none; } } @keyframes thinkingPulse { 0%,100% { transform:scale(.75); opacity:.5; } 50% { transform:scale(1.2); opacity:1; } }
    .image-modal { display:none; position:fixed; inset:0; z-index:25; align-items:center; justify-content:center; background:rgba(30,28,24,.3); } .image-modal.open { display:flex; } .image-card { width:min(520px,calc(100% - 35px)); background:var(--panel); border-radius:15px; padding:24px; box-shadow:0 25px 80px rgba(0,0,0,.2); animation:thinkingIn .25s ease both; } .image-card h2 { margin:0 0 6px; font:400 24px Georgia,serif; } .image-card p { margin:0 0 16px; color:var(--muted); font-size:13px; } .image-card textarea { min-height:100px; border:1px solid var(--line); border-radius:8px; padding:10px; background:var(--bg); } .image-card .save { margin-top:12px; } .generated-image { max-width:min(520px,100%); border-radius:12px; margin-top:10px; box-shadow:var(--shadow); animation:messageIn .45s ease both; }
    .settings { display:none; position:fixed; inset:0; background:rgba(30,28,24,.22); align-items:center; justify-content:center; z-index:20; } .settings.open { display:flex; } .card { width:min(490px, calc(100% - 35px)); background:var(--panel); border-radius:15px; padding:24px; box-shadow:0 25px 80px rgba(0,0,0,.18); } .card h2 { margin:0 0 20px; font:400 25px Georgia,serif; } .field { display:grid; gap:7px; margin:16px 0; } label { color:var(--muted); font-size:13px; } input, select { width:100%; border:1px solid #d8d5ce; border-radius:8px; padding:10px; background:#faf9f7; color:var(--ink); outline:0; } .close { float:right; background:transparent; color:var(--muted); font-size:21px; } .save { width:100%; background:var(--accent); color:#fff; padding:11px; border-radius:8px; margin-top:8px; } .settings-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; } .check { display:flex; align-items:center; gap:9px; color:var(--ink); font-size:13px; margin:12px 0; } .check input { width:auto; accent-color:var(--accent); } .secondary { width:100%; background:transparent; border:1px solid var(--line); color:var(--ink); padding:10px; border-radius:8px; margin-top:10px; }
    @media (max-width:700px) { .app { grid-template-columns:1fr; } aside { display:none; } header { padding:0 18px; } .conversation, .terminal-shell { width:calc(100% - 34px); padding-top:28px; } .welcome h1 { font-size:32px; } }
    .onboarding { position:fixed; inset:0; z-index:30; display:none; place-items:center; background:rgba(20,18,15,.42); backdrop-filter:blur(18px); }
    .onboarding.open { display:grid; animation:fadeIn .25s ease; }
    .onboarding-card { width:min(570px,calc(100vw - 32px)); padding:30px; border:1px solid rgba(255,255,255,.62); border-radius:24px; background:rgba(255,253,249,.88); box-shadow:0 30px 100px rgba(45,35,25,.28); }
    .onboarding-card h1 { margin:0 0 8px; font-size:27px; letter-spacing:-.04em; } .onboarding-card p { color:#77736b; line-height:1.55; }
    .setup-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:22px 0; } .setup-grid .field { margin:0; }
    .drive-option { padding:12px; border:1px solid var(--line); border-radius:12px; background:rgba(255,255,255,.54); cursor:pointer; } .drive-option.selected { border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-soft); }
    .setup-note { font-size:12px; color:#88837b; margin-top:12px; }
    @media(max-width:600px){ .setup-grid { grid-template-columns:1fr; } .onboarding-card { padding:22px; } }
    .sidebar-search { margin:8px 10px 4px; width:calc(100% - 20px); padding:9px 10px; font-size:12px; }
    .terminal-actions { display:flex; gap:7px; margin-left:12px; } .terminal-actions button { color:var(--muted); background:transparent; border:1px solid var(--line); border-radius:6px; padding:4px 8px; font-size:11px; } .terminal-actions button:hover { color:var(--ink); background:var(--accent-soft); }
    .toast { position:fixed; right:22px; bottom:22px; z-index:50; max-width:360px; padding:12px 15px; border:1px solid var(--line); border-radius:12px; color:var(--ink); background:var(--panel); box-shadow:var(--shadow); opacity:0; transform:translateY(12px); pointer-events:none; transition:.25s ease; font-size:13px; } .toast.show { opacity:1; transform:none; }
    .palette { display:none; position:fixed; inset:0; z-index:40; place-items:start center; padding-top:16vh; background:rgba(20,18,15,.28); backdrop-filter:blur(8px); } .palette.open { display:grid; } .palette-card { width:min(510px,calc(100vw - 30px)); background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:10px; box-shadow:var(--shadow); animation:thinkingIn .2s ease both; } .palette-card input { border:0; background:transparent; } .command { display:block; width:100%; text-align:left; padding:11px; border-radius:8px; color:var(--ink); background:transparent; } .command:hover { background:var(--accent-soft); }
    body.midnight { --bg:#0d1220; --panel:rgba(22,29,45,.82); --ink:#eef4ff; --muted:#9ca9c0; --line:rgba(194,211,241,.16); --accent:#7c9cff; --accent-soft:rgba(124,156,255,.18); --shadow:0 25px 80px rgba(0,0,0,.5); } body.midnight aside, body.midnight header { background:rgba(13,18,32,.78); } body.midnight input, body.midnight select, body.midnight .new, body.midnight .model-menu { background:#172035; color:var(--ink); border-color:var(--line); }
    body.rose { --bg:#fff5f7; --panel:rgba(255,250,251,.78); --ink:#39232a; --muted:#94717b; --line:rgba(174,103,125,.2); --accent:#d77896; --accent-soft:rgba(215,120,150,.16); --shadow:0 25px 80px rgba(153,78,104,.16); } body.rose aside, body.rose header { background:rgba(255,246,248,.78); }
    body.graphite { --bg:#17191b; --panel:rgba(34,37,40,.82); --ink:#f2f3f3; --muted:#a4a9ad; --line:rgba(255,255,255,.13); --accent:#d0d3d5; --accent-soft:rgba(208,211,213,.15); --shadow:0 25px 80px rgba(0,0,0,.42); } body.graphite aside, body.graphite header { background:rgba(24,27,29,.8); } body.graphite input, body.graphite select, body.graphite .new, body.graphite .model-menu { background:#282c2f; color:var(--ink); border-color:var(--line); }
    body.aurora { --bg:#f1f7f7; --panel:rgba(248,255,253,.74); --ink:#183133; --muted:#668083; --line:rgba(91,142,143,.2); --accent:#3b9b9b; --accent-soft:rgba(59,155,155,.14); --shadow:0 25px 80px rgba(54,113,115,.15); } body.aurora::before { background:conic-gradient(from 140deg at 50% 50%,rgba(145,224,203,.25),transparent 28%,rgba(173,192,248,.25),transparent 58%,rgba(255,226,177,.2)); } body.aurora aside, body.aurora header { background:rgba(245,255,252,.72); }
    @keyframes fadeIn { from{opacity:0} to{opacity:1} }
    /* AirLLM-Go premium surface system */
    :root { --radius-xl:22px; --radius-lg:16px; --ease:cubic-bezier(.22,.75,.24,1); }
    * { scrollbar-width:thin; scrollbar-color:rgba(130,122,112,.32) transparent; }
    ::selection { color:#fff; background:var(--accent); }
    body { font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text",Inter,ui-sans-serif,sans-serif; letter-spacing:-.012em; background:radial-gradient(circle at 8% 0%,rgba(255,255,255,.95),transparent 32%),radial-gradient(circle at 92% 12%,rgba(238,218,201,.42),transparent 34%),var(--bg); }
    body::before { content:""; position:fixed; inset:-20%; z-index:-2; pointer-events:none; background:conic-gradient(from 200deg at 50% 50%,rgba(255,255,255,.15),transparent 23%,rgba(221,193,177,.12),transparent 54%,rgba(255,255,255,.18)); filter:blur(40px); animation:premiumDrift 22s ease-in-out infinite alternate; }
    .app { gap:0; background:rgba(255,255,255,.12); }
    aside { width:278px; padding:25px 17px 18px; border-right:1px solid rgba(120,110,100,.14); background:linear-gradient(180deg,rgba(255,255,255,.72),rgba(247,244,239,.42)); backdrop-filter:blur(26px) saturate(135%); box-shadow:14px 0 50px rgba(74,59,45,.04); }
    .brand { font-size:16px; font-weight:650; letter-spacing:-.035em; color:var(--ink); }
    .brand .mark { width:29px; height:29px; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,.15); transition:transform .5s var(--ease),box-shadow .5s var(--ease); }
    .brand:hover .mark { transform:rotate(8deg) scale(1.08); box-shadow:0 10px 26px rgba(0,0,0,.2); }
    .new { margin-top:28px; padding:11px 13px; border:1px solid rgba(150,140,130,.18); border-radius:12px; background:rgba(255,255,255,.46); box-shadow:0 6px 18px rgba(65,50,38,.04); transition:transform .25s var(--ease),box-shadow .25s var(--ease),background .25s; }
    .new:hover { transform:translateY(-2px); background:rgba(255,255,255,.86); box-shadow:0 12px 28px rgba(65,50,38,.1); }
    .section { margin:30px 10px 8px; font-weight:650; font-size:10px; letter-spacing:.14em; }
    .sidebar-search { border-radius:11px; background:rgba(255,255,255,.38); border-color:rgba(150,140,130,.14); transition:box-shadow .2s,background .2s; }
    .sidebar-search:focus { background:rgba(255,255,255,.76); box-shadow:0 0 0 4px var(--accent-soft); }
    .history-row button:first-child { border-radius:10px; transition:background .2s,transform .2s; }
    .history-row button:first-child:hover { transform:translateX(2px); }
    main { min-width:0; background:rgba(255,255,255,.18); }
    header { height:68px; padding:0 30px; border-bottom:1px solid rgba(130,120,110,.13); background:rgba(255,255,255,.38); backdrop-filter:blur(24px) saturate(145%); }
    .view-tab { font-weight:600; font-size:13px; transition:transform .25s var(--ease),background .25s; }
    .view-tab:hover { transform:translateY(-1px); }
    .model { margin-left:10px; padding-left:14px; border-left:1px solid var(--line); font-size:12px; }
    .model strong { font-weight:600; }
    .icon { width:34px; height:34px; border-radius:10px; transition:transform .25s var(--ease),background .2s; }
    .icon:hover { transform:translateY(-2px) rotate(4deg); }
    .conversation { padding-top:60px; }
    .welcome { animation:premiumRise .7s var(--ease) both; }
    .welcome h1 { font-size:clamp(36px,5vw,58px); font-weight:520; letter-spacing:-.065em; line-height:1.02; text-wrap:balance; }
    .welcome p { max-width:420px; margin:18px auto; font-size:15px; line-height:1.6; }
    .message { padding:5px 0; }
    .message-body { padding:13px 16px; border-radius:15px; transition:box-shadow .25s,transform .25s; }
    .message.user .message-body { background:rgba(255,255,255,.48); border:1px solid rgba(150,140,130,.12); box-shadow:0 8px 24px rgba(75,59,44,.04); }
    .message.user:hover .message-body, .message.assistant:hover .message-body { transform:translateY(-1px); box-shadow:0 12px 30px rgba(75,59,44,.07); }
    .avatar { transition:transform .3s var(--ease); }
    .message:hover .avatar { transform:scale(1.08) rotate(-4deg); }
    .composer-wrap { padding-bottom:28px; }
    .composer { border:1px solid rgba(140,130,120,.2); border-radius:18px; background:rgba(255,255,255,.68); box-shadow:0 18px 45px rgba(69,51,35,.1),inset 0 1px rgba(255,255,255,.8); backdrop-filter:blur(22px) saturate(140%); transition:box-shadow .3s var(--ease),border-color .3s; }
    .composer:focus-within { border-color:rgba(182,137,112,.5); box-shadow:0 20px 54px rgba(69,51,35,.14),0 0 0 4px var(--accent-soft),inset 0 1px rgba(255,255,255,.9); }
    .composer textarea { font-size:15px; line-height:1.55; }
    .tool, .send, .model-trigger { transition:transform .22s var(--ease),background .22s,box-shadow .22s; }
    .tool:hover, .model-trigger:hover { transform:translateY(-2px); }
    .send { width:36px; height:36px; border-radius:11px; box-shadow:0 5px 14px rgba(0,0,0,.16); }
    .send:hover { transform:translateY(-2px) scale(1.04); box-shadow:0 9px 20px rgba(0,0,0,.2); }
    .thinking-box { border:1px solid var(--line); border-left:2px solid var(--accent); border-radius:12px; box-shadow:0 10px 28px rgba(70,50,35,.06); backdrop-filter:blur(16px); }
    .terminal-shell { padding-top:44px; }
    .terminal { border:1px solid rgba(169,218,173,.16); border-radius:16px; box-shadow:0 22px 60px rgba(24,37,27,.2),inset 0 1px rgba(255,255,255,.08); }
    .settings .card, .image-card, .onboarding-card { border:1px solid rgba(255,255,255,.7); backdrop-filter:blur(30px) saturate(145%); }
    .save, .secondary, input, select { transition:transform .2s var(--ease),box-shadow .2s,background .2s; }
    .save:hover, .secondary:hover { transform:translateY(-2px); box-shadow:0 8px 18px rgba(60,45,34,.1); }
    input:focus, select:focus { box-shadow:0 0 0 4px var(--accent-soft); }
    @keyframes premiumRise { from{opacity:0;transform:translateY(16px) scale(.985)} to{opacity:1;transform:none} }
    @keyframes premiumDrift { from{transform:translate3d(-2%,0,0) rotate(0deg)} to{transform:translate3d(2%,1%,0) rotate(4deg)} }
    @media (prefers-reduced-motion:reduce) { *,*::before,*::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; } }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand"><span class="mark" aria-label="Sierpinski triangle logo"><svg viewBox="0 0 36 36" role="img"><path fill="white" fill-rule="evenodd" d="M18 2 34 32H2L18 2ZM18 13 24 24H12L18 13ZM10 17 15 26H5l5-9ZM26 17l5 9H21l5-9Z"/><path class="cut" d="M18 13 21 18.5h-6L18 13Z"/></svg></span> AirLLM-Go</div>
      <button class="new" onclick="newChat()">＋&nbsp; New conversation</button>
      <div class="section">Conversations</div>
      <input class="sidebar-search" id="historySearch" placeholder="Search conversations…" oninput="renderHistory()" />
      <div class="history" id="history"><div class="empty-history">No saved conversations yet</div></div>
      <div class="bottom"><div class="status"><span class="dot"></span><span id="status">AirLLM ready</span></div><div class="status">⌘ &nbsp; Local inference</div></div>
    </aside>
    <main>
      <header><div class="view-tabs"><button class="view-tab active" id="chatTab" onclick="switchView('chat')">Chat</button><button class="view-tab" id="terminalTab" onclick="switchView('terminal')">Terminal</button><span class="model">Model&nbsp; <strong id="modelLabel">Qwen/Qwen3-32B</strong> <span class="chev">⌄</span></span></div><div class="header-actions"><button class="icon" title="Settings" onclick="openSettings()">⚙</button><button class="icon" title="About">?</button></div></header>
      <div class="progress-wrap"><div class="progress-bar" id="progressBar"></div></div>
      <div class="view-panel active" id="chatPanel"><section class="conversation" id="conversation"><div class="welcome" id="welcome"><h1>What’s on your mind?</h1><p>A quiet space for thoughtful answers, powered locally.</p></div></section>
      <div class="thinking-box" id="thinkingBox"><span class="thinking-orb"></span><strong>Thinking</strong><span id="thinkingText">AirLLM is standing by.</span></div>
      <div class="composer-wrap"><div class="conversation"><div class="composer"><textarea id="prompt" rows="1" placeholder="Message AirLLM-Go…" onkeydown="keySend(event)" oninput="resize(this)"></textarea><div class="composer-footer"><div class="tools"><button class="tool" title="Attach files" onclick="document.getElementById('fileInput').click()">＋</button><button class="tool" title="Generate an image" onclick="openImageModal()">✦</button><input id="fileInput" type="file" multiple hidden onchange="handleFiles(event)" /><span id="attachmentLabel">AirLLM · local</span><div class="model-picker"><button class="model-trigger" id="modelTrigger" onclick="toggleModelMenu(event)">◈ <span id="composerModel">Qwen/Qwen3-32B</span>⌄</button><div class="model-menu" id="modelMenu"><input class="model-search" id="modelSearch" placeholder="Search models…" oninput="filterModels()" onkeydown="modelKeydown(event)" /><div class="model-list" id="modelList"></div></div></div></div><button class="send" id="send" onclick="sendMessage()">↑</button></div></div></div></div>
      <div class="view-panel" id="terminalPanel"><div class="terminal-shell"><div class="terminal-head"><span id="terminalStatus">AirLLM runtime</span><span id="progressLabel">Idle</span><span class="terminal-actions"><button onclick="refreshTerminal()">Refresh</button><button onclick="copyTerminal()">Copy</button><button onclick="downloadTerminal()">Save log</button></span></div><div class="terminal" id="terminal"><div class="terminal-empty">Terminal output will appear here when AirLLM loads a model.</div></div></div></div>
    </main>
  </div>
  <div class="onboarding" id="onboarding"><div class="onboarding-card"><div class="eyebrow">WELCOME TO AIRLLM-GO</div><h1>Let’s set up your local workspace</h1><p>Choose where downloaded model files should live. Your C: drive is low on space, so AirLLM-Go will keep the Hugging Face cache on the drive you choose.</p><div class="setup-grid"><div class="field"><label>Model cache drive</label><select id="setupDrive"></select></div><div class="field"><label>Default model</label><select id="setupModel"><option value="Qwen/Qwen3-0.6B">Qwen 0.6B · easiest start</option><option value="Qwen/Qwen3-4B">Qwen 4B · balanced</option><option value="Qwen/Qwen3-8B">Qwen 8B · higher quality</option><option value="cognitivecomputations/Dolphin3.0-Llama3.1-8B">Dolphin 8B · UNFILTERED</option></select></div></div><div class="field"><label>Cache folder</label><input id="setupPath" placeholder="D:\\AirLLM-Go\\huggingface" /></div><div class="setup-note" id="setupSpace">Checking available drives…</div><button class="save" id="setupSave" onclick="finishOnboarding()">Save setup and continue</button></div></div>
  <div class="image-modal" id="imageModal"><div class="image-card"><button class="close" onclick="closeImageModal()">×</button><h2>Create an image</h2><p>Uses optional local Diffusers support. The first image model download can be very large.</p><textarea id="imagePrompt" placeholder="A cinematic watercolor of a quiet library on Mars…"></textarea><button class="save" id="imageButton" onclick="generateImage()">Generate image</button></div></div>
  <div class="settings" id="settings"><div class="card"><button class="close" onclick="closeSettings()">×</button><h2>Settings</h2><div class="settings-grid"><div class="field"><label>Theme</label><select id="themeInput"><option value="light">Warm light</option><option value="dark">Dark</option><option value="sepia">Sepia</option><option value="ocean">Ocean glass</option><option value="lavender">Lavender glass</option><option value="forest">Forest glass</option><option value="sunset">Sunset glass</option><option value="obsidian">Obsidian neon</option></select></div><div class="field"><label>Background pattern</label><select id="patternInput"><option value="none">None</option><option value="grid">Fine grid</option><option value="dots">Soft dots</option><option value="diamonds">Diamonds</option><option value="waves">Flowing waves</option></select></div></div><div class="settings-grid"><div class="field"><label>Text size</label><select id="fontInput"><option value="normal">Comfortable</option><option value="large">Large</option></select></div><div class="field"><label>Motion</label><select id="motionInput"><option value="full">Full motion</option><option value="reduced">Reduced motion</option></select></div></div><label class="check"><input id="compactInput" type="checkbox" /> Compact conversation layout</label><label class="check"><input id="enterInput" type="checkbox" checked /> Enter sends messages</label><div class="field"><label>Hugging Face model ID</label><input id="modelInput" value="Qwen/Qwen3-32B" /></div><div class="settings-grid"><div class="field"><label>Compression</label><select id="compression"><option value="">None</option><option value="4bit">4-bit</option><option value="8bit">8-bit</option></select></div><div class="field"><label>Max new tokens</label><input id="tokens" type="number" value="256" min="16" max="2048" /></div></div><button class="save" onclick="saveSettings()">Save settings</button><button class="secondary" onclick="exportHistory()">Export conversations</button><button class="secondary" onclick="clearHistory()">Clear all saved conversations</button></div></div>
  <div class="toast" id="toast" role="status"></div>
  <script>
    let messages = []; let model = 'Qwen/Qwen3-0.6B'; let modelIndex = 0; let conversationId = null; let conversations = []; let attachments = [];
    async function loadSetup(){ try { const [configResponse, drivesResponse]=await Promise.all([fetch('/api/config'),fetch('/api/drives')]); const config=await configResponse.json(); const driveData=await drivesResponse.json(); const driveSelect=document.getElementById('setupDrive'); driveSelect.innerHTML=driveData.drives.map(d=>'<option value="'+d.drive+'">'+d.drive+' · '+d.free_gb+' GB free</option>').join(''); const preferred=config.drive||'D:\\'; if([...driveSelect.options].some(o=>o.value===preferred)) driveSelect.value=preferred; updateSetupPath(); driveSelect.onchange=updateSetupPath; model=config.default_model||model; document.getElementById('setupModel').value=model; document.getElementById('modelInput').value=model; document.getElementById('modelLabel').textContent=model; document.getElementById('composerModel').textContent=model; if(!config.configured) document.getElementById('onboarding').classList.add('open'); } catch(e){ document.getElementById('status').textContent='Setup information unavailable'; } }
    function updateSetupPath(){ const drive=document.getElementById('setupDrive').value||'D:\\'; document.getElementById('setupPath').value=drive+'AirLLM-Go\\huggingface'; const selected=document.getElementById('setupDrive').selectedOptions[0]; document.getElementById('setupSpace').textContent=selected?selected.textContent+' · model files will stay on this drive.':'Choose a drive with enough free space.'; }
    async function finishOnboarding(){ const button=document.getElementById('setupSave'); button.disabled=true; button.textContent='Saving workspace…'; try { const drive=document.getElementById('setupDrive').value; const cachePath=document.getElementById('setupPath').value.trim(); const defaultModel=document.getElementById('setupModel').value; const r=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({drive,cache_path:cachePath,default_model:defaultModel})}); const data=await r.json(); if(!r.ok) throw new Error(data.error||'Could not save setup'); model=defaultModel; document.getElementById('modelLabel').textContent=model; document.getElementById('composerModel').textContent=model; document.getElementById('modelInput').value=model; document.getElementById('onboarding').classList.remove('open'); } catch(e){ alert(e.message); } finally { button.disabled=false; button.textContent='Save setup and continue'; } }
    const models = [
      ['Qwen/Qwen3-32B', 'Balanced · high memory', false],
      ['Qwen/Qwen3-8B', 'Fast · lower memory', false],
      ['Qwen/Qwen3-4B', 'Fastest · local testing', false],
      ['Qwen/Qwen3-0.6B', 'Quick start · lowest memory', false],
      ['cognitivecomputations/Dolphin3.0-Llama3.1-8B', 'Community unfiltered checkpoint', true],
      ['huihui-ai/Qwen2.5-7B-Instruct-abliterated', 'Community unfiltered checkpoint', true],
      ['NousResearch/Hermes-3-Llama-3.1-8B', 'Community unfiltered checkpoint', true],
      ['mistralai/Mistral-7B-Instruct-v0.3', 'Compact instruction model', false],
      ['meta-llama/Llama-3.1-8B-Instruct', 'Requires Hugging Face access', false],
      ['deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'Reasoning model', false],
      ['google/gemma-3-4b-it', 'Compact multimodal family', false],
      ['custom…', 'Enter any Hugging Face model ID in Settings', false]
    ];
    function resize(el){ el.style.height='auto'; el.style.height=Math.min(el.scrollHeight,170)+'px'; }
    function keySend(e){ if(e.key==='Enter' && !e.shiftKey && document.getElementById('enterInput').checked){ e.preventDefault(); sendMessage(); } }
    function toggleModelMenu(e){ e.stopPropagation(); const menu=document.getElementById('modelMenu'); const open=menu.classList.toggle('open'); document.getElementById('modelTrigger').classList.toggle('open',open); if(open){ renderModels(); setTimeout(()=>document.getElementById('modelSearch').focus(),0); } }
    function renderModels(){ const query=document.getElementById('modelSearch').value.toLowerCase(); const list=document.getElementById('modelList'); const filtered=models.filter(item=>item[0].toLowerCase().includes(query)||item[1].toLowerCase().includes(query)); list.innerHTML=filtered.map((item,i)=>{ const tag=item[2]?'<span class="uncensored-badge">UNFILTERED</span>':''; return '<button class="model-option '+(item[0]===model?'selected':'')+'" onclick="selectModel(\''+item[0].replace(/'/g,"\\'")+'\')">'+escapeHtml(item[0])+tag+'<small>'+escapeHtml(item[1])+'</small></button>'; }).join('') || '<div style="padding:12px;color:#999;font-size:13px">No matching models</div>'; modelIndex=0; }
    function filterModels(){ renderModels(); }
    function modelKeydown(e){ const options=[...document.querySelectorAll('.model-option')]; if(e.key==='ArrowDown'||e.key==='ArrowUp'){ e.preventDefault(); modelIndex=(modelIndex+(e.key==='ArrowDown'?1:-1)+options.length)%options.length; options.forEach((o,i)=>o.classList.toggle('selected',i===modelIndex)); } if(e.key==='Enter'&&options[modelIndex]){ e.preventDefault(); options[modelIndex].click(); } if(e.key==='Escape'){ document.getElementById('modelMenu').classList.remove('open'); document.getElementById('modelTrigger').classList.remove('open'); } }
    function selectModel(value){ if(value==='custom…'){ openSettings(); return; } model=value; document.getElementById('modelLabel').textContent=model; document.getElementById('composerModel').textContent=model; document.getElementById('modelMenu').classList.remove('open'); document.getElementById('modelTrigger').classList.remove('open'); }
    function addMessage(role, content){ const welcome=document.getElementById('welcome'); if(welcome) welcome.style.display='none'; const el=document.createElement('div'); el.className='message '+role; el.innerHTML='<div class="avatar">'+(role==='assistant'?'a':'you')+'</div><div class="message-body"><div class="role">'+(role==='assistant'?'air':'You')+'</div><div>'+escapeHtml(content)+'</div></div>'; document.getElementById('conversation').appendChild(el); window.scrollTo(0,document.body.scrollHeight); }
    function escapeHtml(s){ return s.replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c])); }
    function switchView(view){ const chat=view==='chat'; document.getElementById('chatPanel').classList.toggle('active',chat); document.getElementById('terminalPanel').classList.toggle('active',!chat); document.getElementById('chatTab').classList.toggle('active',chat); document.getElementById('terminalTab').classList.toggle('active',!chat); }
    async function pollStatus(){ try { const s=await (await fetch('/api/status?ts='+Date.now(),{cache:'no-store'})).json(); const terminalResponse=await fetch('/api/terminal?ts='+Date.now(),{cache:'no-store'}); const terminalData=await terminalResponse.json(); const bar=document.getElementById('progressBar'); bar.style.width=s.progress+'%'; bar.classList.toggle('busy',s.busy); document.getElementById('status').textContent=s.message; document.getElementById('progressLabel').textContent=s.message; document.getElementById('terminalStatus').textContent=s.phase.toUpperCase()+' · '+(s.progress||0)+'% · LIVE'; document.getElementById('thinkingText').textContent=s.message; document.getElementById('thinkingBox').classList.toggle('visible',s.busy); const terminal=document.getElementById('terminal'); terminal.classList.toggle('active-output',s.busy); terminal.innerHTML=terminalData.lines.length?terminalData.lines.map(line=>'<div class="terminal-line">'+escapeHtml(line)+'</div>').join(''):'<div class="terminal-empty">Waiting for backend output…</div>'; terminal.scrollTop=terminal.scrollHeight; } catch(e){ document.getElementById('terminalStatus').textContent='DISCONNECTED · retrying'; document.getElementById('terminal').innerHTML='<div class="terminal-empty">Terminal stream reconnecting…</div>'; } }
    async function handleFiles(event){ for(const file of event.target.files){ const form=new FormData(); form.append('file',file); try { const data=await (await fetch('/api/upload',{method:'POST',body:form})).json(); if(data.error) throw new Error(data.error); attachments.push(data); } catch(e){ alert(e.message); } } document.getElementById('attachmentLabel').textContent=attachments.length?attachments.length+' file'+(attachments.length>1?'s':'')+' attached':'AirLLM · local'; event.target.value=''; }
    function openImageModal(){ document.getElementById('imageModal').classList.add('open'); document.getElementById('imagePrompt').focus(); } function closeImageModal(){ document.getElementById('imageModal').classList.remove('open'); }
    async function generateImage(){ const prompt=document.getElementById('imagePrompt').value.trim(); if(!prompt) return; const button=document.getElementById('imageButton'); button.disabled=true; button.textContent='Loading image pipeline…'; document.getElementById('thinkingBox').classList.add('visible'); try { const r=await fetch('/api/image',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,model:'stabilityai/sdxl-turbo'})}); const data=await r.json(); if(!r.ok) throw new Error(data.error||'Image generation failed'); const wrap=document.createElement('div'); wrap.className='message assistant'; wrap.innerHTML='<div class="avatar">a</div><div class="message-body"><div class="role">AirLLM-Go image</div><div>'+escapeHtml(prompt)+'</div><img class="generated-image" src="'+data.url+'" alt="Generated image" /></div>'; document.getElementById('conversation').appendChild(wrap); closeImageModal(); switchView('chat'); } catch(e){ alert(e.message); } finally { button.disabled=false; button.textContent='Generate image'; document.getElementById('thinkingBox').classList.remove('visible'); } }
    async function loadConversations(){ try { const r=await fetch('/api/conversations'); conversations=await r.json(); renderHistory(); } catch(e){ document.getElementById('status').textContent='History unavailable'; } }
    function renderHistory(){ const el=document.getElementById('history'); if(!conversations.length){ el.innerHTML='<div class="empty-history">No saved conversations yet</div>'; return; } el.innerHTML=conversations.map(c=>'<div class="history-row '+(c.id===conversationId?'active':'')+'"><button onclick="openConversation(\''+c.id+'\')">'+escapeHtml(c.title||'New conversation')+'</button><button class="delete" title="Delete conversation" onclick="deleteConversation(event,\''+c.id+'\')">×</button></div>').join(''); }
    function renderMessages(){ const area=document.getElementById('conversation'); if(!messages.length){ area.innerHTML='<div class="welcome" id="welcome"><h1>What’s on your mind?</h1><p>A quiet space for thoughtful answers, powered locally.</p></div>'; return; } area.innerHTML=''; messages.forEach(m=>addMessage(m.role,m.content)); }
    async function openConversation(id){ const r=await fetch('/api/conversations/'+id); if(!r.ok) return; const data=await r.json(); conversationId=data.id; messages=data.messages||[]; if(data.model){ model=data.model; document.getElementById('modelLabel').textContent=model; document.getElementById('composerModel').textContent=model; document.getElementById('modelInput').value=model; } renderMessages(); renderHistory(); document.getElementById('prompt').focus(); }
    async function newChat(){ conversationId=null; messages=[]; renderMessages(); renderHistory(); document.getElementById('prompt').focus(); }
    async function deleteConversation(event,id){ event.stopPropagation(); if(!confirm('Delete this conversation?')) return; await fetch('/api/conversations/'+id,{method:'DELETE'}); if(conversationId===id){ conversationId=null; messages=[]; renderMessages(); } await loadConversations(); }
    async function sendMessage(){ const box=document.getElementById('prompt'), text=box.value.trim(); if(!text && !attachments.length) return; const attachmentBlock=attachments.length?' [Attached files: '+attachments.map(a=>a.filename+'\\n'+a.text).join('\\n')+']':''; box.value=''; resize(box); addMessage('user',text+(attachments.length?'\\n\\n📎 '+attachments.map(a=>a.filename).join(', '):'')); messages.push({role:'user',content:text+attachmentBlock}); attachments=[]; document.getElementById('attachmentLabel').textContent='AirLLM · local'; const send=document.getElementById('send'); send.disabled=true; document.getElementById('thinkingBox').classList.add('visible'); document.getElementById('thinkingText').textContent='Starting AirLLM…'; try { const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({conversation_id:conversationId,messages,model,compression:document.getElementById('compression').value,max_new_tokens:Number(document.getElementById('tokens').value)})}); const data=await r.json(); if(!r.ok) throw new Error(data.error||'Backend error'); conversationId=data.conversation_id; const reply=data.reply || 'No response.'; addMessage('assistant',reply); messages.push({role:'assistant',content:reply}); await loadConversations(); } catch(e){ const detail=e.name==='TypeError'?'The AirLLM-Go backend connection was interrupted. Refresh the page once and retry; the Terminal tab will show the live server state.':e.message; addMessage('assistant','The local model returned an error: '+detail); } finally { send.disabled=false; document.getElementById('status').textContent='AirLLM ready'; setTimeout(()=>document.getElementById('thinkingBox').classList.remove('visible'),1200); } }
    function openSettings(){ document.getElementById('settings').classList.add('open'); } function closeSettings(){ document.getElementById('settings').classList.remove('open'); }
    function applyPreferences(){ const theme=localStorage.getItem('air-theme')||'light'; const pattern=localStorage.getItem('air-pattern')||'none'; const font=localStorage.getItem('air-font')||'normal'; const motion=localStorage.getItem('air-motion')||'full'; const compact=localStorage.getItem('air-compact')==='true'; document.body.classList.remove('dark','sepia','ocean','lavender','forest','sunset','obsidian','pattern-grid','pattern-dots','pattern-diamonds','pattern-waves','large-text','compact','reduced'); if(theme!=='light') document.body.classList.add(theme); if(pattern!=='none') document.body.classList.add('pattern-'+pattern); if(font==='large') document.body.classList.add('large-text'); if(compact) document.body.classList.add('compact'); if(motion==='reduced') document.body.classList.add('reduced'); document.getElementById('themeInput').value=theme; document.getElementById('patternInput').value=pattern; document.getElementById('fontInput').value=font; document.getElementById('motionInput').value=motion; document.getElementById('compactInput').checked=compact; document.getElementById('enterInput').checked=localStorage.getItem('air-enter')!=='false'; }
    async function saveSettings(){ model=document.getElementById('modelInput').value.trim()||model; document.getElementById('modelLabel').textContent=model; document.getElementById('composerModel').textContent=model; localStorage.setItem('air-theme',document.getElementById('themeInput').value); localStorage.setItem('air-pattern',document.getElementById('patternInput').value); localStorage.setItem('air-font',document.getElementById('fontInput').value); localStorage.setItem('air-motion',document.getElementById('motionInput').value); localStorage.setItem('air-compact',document.getElementById('compactInput').checked); localStorage.setItem('air-enter',document.getElementById('enterInput').checked); try { const current=await (await fetch('/api/config')).json(); if(current.configured) await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({drive:current.drive,cache_path:current.cache_path,default_model:model})}); } catch(e){} applyPreferences(); closeSettings(); }
    function exportHistory(){ const blob=new Blob([JSON.stringify(conversations,null,2)],{type:'application/json'}); const link=document.createElement('a'); link.href=URL.createObjectURL(blob); link.download='air-conversations.json'; link.click(); URL.revokeObjectURL(link.href); }
    async function clearHistory(){ if(!confirm('Clear every saved conversation?')) return; await fetch('/api/conversations',{method:'DELETE'}); conversationId=null; messages=[]; renderMessages(); await loadConversations(); closeSettings(); }
    function showToast(message){ const toast=document.getElementById('toast'); if(!toast) return; toast.textContent=message; toast.classList.add('show'); clearTimeout(window.toastTimer); window.toastTimer=setTimeout(()=>toast.classList.remove('show'),3200); }
    function refreshTerminal(){ pollStatus(); showToast('Terminal refreshed'); }
    function copyTerminal(){ navigator.clipboard?.writeText([...document.querySelectorAll('.terminal-line')].map(x=>x.textContent).join('\n')); showToast('Terminal log copied'); }
    function downloadTerminal(){ const text=[...document.querySelectorAll('.terminal-line')].map(x=>x.textContent).join('\n'); const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([text],{type:'text/plain'})); a.download='airllm-go-terminal.log'; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),500); showToast('Terminal log saved'); }
    const originalRenderHistory=renderHistory; renderHistory=function(){ const query=(document.getElementById('historySearch')?.value||'').toLowerCase(); const old=conversations; conversations=query?old.filter(c=>(c.title||'').toLowerCase().includes(query)||JSON.stringify(c.messages||[]).toLowerCase().includes(query)):old; originalRenderHistory(); conversations=old; };
    document.addEventListener('keydown', e=>{ if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){ e.preventDefault(); document.getElementById('modelSearch').focus(); toggleModelMenu({stopPropagation(){}}); } if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='j'){ e.preventDefault(); switchView('terminal'); } if(e.key==='Escape'){ document.getElementById('palette')?.classList.remove('open'); } });
    document.addEventListener('click', e=>{ if(!e.target.closest('.model-picker')){ document.getElementById('modelMenu').classList.remove('open'); document.getElementById('modelTrigger').classList.remove('open'); } });
    [['midnight','Midnight blue'],['rose','Rose glass'],['graphite','Graphite'],['aurora','Aurora glass']].forEach(([value,label])=>{ const select=document.getElementById('themeInput'); if(select&&!select.querySelector('option[value="'+value+'"]')) select.insertAdjacentHTML('beforeend','<option value="'+value+'">'+label+'</option>'); });
    const stableApplyPreferences=()=>{ const theme=localStorage.getItem('air-theme')||'light'; const pattern=localStorage.getItem('air-pattern')||'none'; const font=localStorage.getItem('air-font')||'normal'; const motion=localStorage.getItem('air-motion')||'full'; const compact=localStorage.getItem('air-compact')==='true'; document.body.classList.remove('dark','sepia','ocean','lavender','forest','sunset','obsidian','midnight','rose','graphite','aurora','pattern-grid','pattern-dots','pattern-diamonds','pattern-waves','large-text','compact','reduced'); if(theme!=='light') document.body.classList.add(theme); if(pattern!=='none') document.body.classList.add('pattern-'+pattern); if(font==='large') document.body.classList.add('large-text'); if(compact) document.body.classList.add('compact'); if(motion==='reduced') document.body.classList.add('reduced'); document.getElementById('themeInput').value=theme; document.getElementById('patternInput').value=pattern; document.getElementById('fontInput').value=font; document.getElementById('motionInput').value=motion; document.getElementById('compactInput').checked=compact; document.getElementById('enterInput').checked=localStorage.getItem('air-enter')!=='false'; };
    applyPreferences=stableApplyPreferences;
    let pollInFlight=false;
    pollStatus=async function(){ if(pollInFlight) return; pollInFlight=true; const controller=new AbortController(); const timeout=setTimeout(()=>controller.abort(),9000); try { const s=await (await fetch('/api/status?ts='+Date.now(),{cache:'no-store',signal:controller.signal})).json(); const terminalData=await (await fetch('/api/terminal?ts='+Date.now(),{cache:'no-store',signal:controller.signal})).json(); const lines=Array.isArray(terminalData.lines)?terminalData.lines:[]; document.getElementById('progressBar').style.width=(Number(s.progress)||0)+'%'; document.getElementById('progressBar').classList.toggle('busy',!!s.busy); document.getElementById('status').textContent=s.message||'AirLLM ready'; document.getElementById('progressLabel').textContent=s.message||'Idle'; document.getElementById('terminalStatus').textContent=(s.phase||'idle').toUpperCase()+' · '+(s.progress||0)+'% · LIVE'; document.getElementById('thinkingText').textContent=s.message||'AirLLM is standing by.'; document.getElementById('thinkingBox').classList.toggle('visible',!!s.busy); const terminal=document.getElementById('terminal'); terminal.classList.toggle('active-output',!!s.busy); terminal.innerHTML=lines.length?lines.map(line=>'<div class="terminal-line">'+escapeHtml(String(line))+'</div>').join(''):'<div class="terminal-empty">Waiting for backend output…</div>'; terminal.scrollTop=terminal.scrollHeight; } catch(e){ document.getElementById('terminalStatus').textContent='RECONNECTING · retrying'; if(!document.querySelector('.terminal-line')) document.getElementById('terminal').innerHTML='<div class="terminal-empty">Waiting for AirLLM-Go backend output…</div>'; } finally { clearTimeout(timeout); pollInFlight=false; setTimeout(pollStatus,800); } };
    applyPreferences();
    loadSetup();
    loadConversations();
    pollStatus();
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/api/conversations")
def conversations():
    items = read_conversations()
    return jsonify(sorted(items, key=lambda item: item.get("updated_at", ""), reverse=True))


@app.post("/api/conversations")
def create_conversation():
    item = new_conversation()
    save_conversation(item)
    return jsonify(item), 201


@app.delete("/api/conversations")
def clear_conversations():
    with HISTORY_LOCK:
        write_conversations([])
    return jsonify(ok=True)


@app.get("/api/conversations/<conversation_id>")
def get_conversation(conversation_id: str):
    item = next((x for x in read_conversations() if x["id"] == conversation_id), None)
    if item is None:
        return jsonify(error="Conversation not found."), 404
    return jsonify(item)


@app.put("/api/conversations/<conversation_id>")
def update_conversation(conversation_id: str):
    data = request.get_json(force=True) or {}
    items = read_conversations()
    item = next((x for x in items if x["id"] == conversation_id), None)
    if item is None:
        return jsonify(error="Conversation not found."), 404
    if isinstance(data.get("title"), str):
        item["title"] = data["title"].strip()[:100] or "New conversation"
    if isinstance(data.get("messages"), list):
        item["messages"] = data["messages"]
    item["updated_at"] = now_iso()
    save_conversation(item)
    return jsonify(item)


@app.delete("/api/conversations/<conversation_id>")
def delete_conversation(conversation_id: str):
    with HISTORY_LOCK:
        items = [x for x in read_conversations() if x["id"] != conversation_id]
        write_conversations(items)
    return jsonify(ok=True)


@app.post("/api/upload")
def upload_file():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify(error="No file selected."), 400
    filename = secure_filename(uploaded.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".md", ".csv", ".json", ".py", ".js", ".html", ".css", ".log"}:
        return jsonify(error="Supported files: txt, md, csv, json, py, js, html, css, log."), 415
    content = uploaded.read()
    if len(content) > 2_000_000:
        return jsonify(error="Files must be smaller than 2 MB."), 413
    try:
        text_content = content.decode("utf-8", errors="replace")
    except Exception as exc:
        return jsonify(error=f"Could not read file: {exc}"), 400
    stored_name = f"{uuid.uuid4().hex[:12]}-{filename}"
    (UPLOAD_DIR / stored_name).write_bytes(content)
    update_status("ready", 0, f"Attached {filename}")
    return jsonify(filename=filename, stored_name=stored_name, size=len(content), text=text_content[:120_000])


@app.post("/api/image")
def generate_image():
    data = request.get_json(force=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify(error="Describe the image you want to create."), 400
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        return jsonify(error="Image generation is optional. Install it with: pip install diffusers"), 503
    try:
        update_status("image-loading", 15, "Loading image generation pipeline")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(data.get("model", "stabilityai/sdxl-turbo"), torch_dtype=dtype)
        pipe = pipe.to(device)
        update_status("image-generating", 65, "Rendering image")
        image = pipe(prompt=prompt, num_inference_steps=4 if "turbo" in data.get("model", "").lower() else 20, guidance_scale=0.0 if "turbo" in data.get("model", "").lower() else 7.0).images[0]
        filename = f"{uuid.uuid4().hex}.png"
        image.save(UPLOAD_DIR / filename)
        update_status("complete", 100, "Image ready")
        return jsonify(url=f"/generated/{filename}", prompt=prompt)
    except Exception as exc:
        update_status("error", 0, f"Image generation failed: {exc}")
        return jsonify(error=str(exc)), 500


@app.get("/generated/<path:filename>")
def generated_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.post("/api/chat")
def chat():
    data = request.get_json(force=True) or {}
    messages = data.get("messages", [])
    if not messages or not messages[-1].get("content"):
        return jsonify(error="Please enter a message."), 400
    try:
        conversation_id = data.get("conversation_id")
        item = next((x for x in read_conversations() if x["id"] == conversation_id), None)
        if item is None:
            item = new_conversation()
        selected_model = str(data.get("model") or item.get("model") or read_config().get("default_model", DEFAULT_MODEL))
        reply = engine.reply(
            messages,
            selected_model,
            data.get("compression") or None,
            int(data.get("max_new_tokens", 256)),
        )
        item["messages"] = messages + [{"role": "assistant", "content": reply}]
        item["model"] = selected_model
        if item["messages"]:
            first = item["messages"][0].get("content", "").strip()
            item["title"] = first[:48] + ("…" if len(first) > 48 else "")
        item["updated_at"] = now_iso()
        save_conversation(item)
        return jsonify(reply=reply, conversation_id=item["id"], conversation=item)
    except Exception as exc:
        update_status("error", 0, str(exc))
        return jsonify(error=str(exc)), 500


@app.get("/api/health")
def health():
    return jsonify(
        status="ok",
        python=os.sys.version.split()[0],
        airllm_loaded=engine.model is not None,
        model=engine.model_name,
    )


@app.get("/api/config")
def get_config():
    config = read_config()
    cache_path = cache_path_from_config(config)
    usage = shutil.disk_usage(cache_path if cache_path.exists() else cache_path.anchor or cache_path)
    return jsonify(
        configured=bool(config.get("onboarded")),
        default_model=config.get("default_model", DEFAULT_MODEL),
        cache_path=str(cache_path),
        drive=cache_path.drive or str(cache_path),
        free_gb=round(usage.free / (1024 ** 3), 2),
    )


@app.get("/api/drives")
def drives():
    values = []
    for code in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        root = Path(f"{code}:\\")
        if root.exists():
            try:
                usage = shutil.disk_usage(root)
                values.append({"drive": f"{code}:\\", "free_gb": round(usage.free / (1024 ** 3), 2), "total_gb": round(usage.total / (1024 ** 3), 2)})
            except OSError:
                pass
    return jsonify(drives=values)


@app.post("/api/config")
def save_config():
    data = request.get_json(force=True) or {}
    drive = str(data.get("drive", "")).strip().rstrip("\\/")
    if not re.fullmatch(r"[A-Za-z]:", drive):
        return jsonify(error="Choose a valid drive, such as D:"), 400
    root = Path(drive + "\\")
    if not root.exists():
        return jsonify(error=f"Drive {drive} is not available."), 400
    cache_path = Path(str(data.get("cache_path") or (drive + "\\AirLLM-Go\\huggingface")))
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(cache_path)
        config = {
            "onboarded": True,
            "drive": drive + "\\",
            "cache_path": str(cache_path),
            "default_model": str(data.get("default_model") or DEFAULT_MODEL).strip(),
        }
        CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
        apply_cache_config(config)
        terminal_log(f"Cache configured at {cache_path} ({usage.free / (1024 ** 3):.2f} GB free)")
        return jsonify(config=config, free_gb=round(usage.free / (1024 ** 3), 2))
    except OSError as exc:
        return jsonify(error=f"Could not create cache folder: {exc}"), 400


@app.get("/api/status")
def status():
    return jsonify(current_status())


@app.get("/api/terminal")
def terminal_output():
    return jsonify(lines=terminal_tail())


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "7860")), debug=False, threaded=True)
