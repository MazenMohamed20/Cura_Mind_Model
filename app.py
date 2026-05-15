# -*- coding: utf-8 -*-
import os
import re
import sys
import time
import asyncio
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from aiohttp import web
from aiohttp.web_middlewares import middleware

# ============================================================
# MODEL PATH (HUGGINGFACE FIXED)
# ============================================================
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="MazenMohamed10/Cura_Mind_v2",
    filename="model.gguf",
    cache_dir="/tmp"
)

# optional
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
PORT = int(os.environ.get("PORT", 10000))

N_THREADS = 2
N_CTX = 256
N_BATCH = 128
MAX_TOKENS = 80
MAX_Q_LEN = 350
RATE_LIMIT = 10
TIMEOUT_SEC = 25

# ============================================================
# EXECUTOR
# ============================================================
_executor = ThreadPoolExecutor(max_workers=2)
_start_time = time.monotonic()

_rate_store = {}

# ============================================================
# RATE LIMIT
# ============================================================
def is_rate_limited(ip: str) -> bool:
    now = time.monotonic()
    dq = _rate_store.setdefault(ip, deque())

    while dq and now - dq[0] > 60:
        dq.popleft()

    if len(dq) >= RATE_LIMIT:
        return True

    dq.append(now)
    return False

# ============================================================
# LOAD MODEL (LAZY)
# ============================================================
from llama_cpp import Llama

llama = None

def get_model():
    global llama
    if llama is None:
        logger.info("📦 Loading model...")
        llama = Llama(
            model_path=model_path,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_batch=N_BATCH,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        logger.info("✅ Model loaded successfully")
    return llama

# ============================================================
# SYSTEM PROMPT
# ============================================================
BASE_SYSTEM = (
    "You are a helpful AI assistant. "
    "Be concise and accurate. "
    "Reply in the same language as the user."
)

# ============================================================
# LANGUAGE DETECTION
# ============================================================
def detect_language(text: str) -> str:
    ar = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total = sum(1 for c in text if c.isalpha())
    return "arabic" if total and ar / total > 0.3 else "english"

# ============================================================
# CLASSIFIER
# ============================================================
def classify(q: str) -> str:
    q = q.lower()

    if any(x in q for x in ["calorie", "protein", "سعرات"]):
        return "nutrition"
    if any(x in q for x in ["how", "steps", "طريقة", "كيف"]):
        return "steps"
    if any(x in q for x in ["compare", "vs", "بين"]):
        return "compare"
    if any(x in q for x in ["recipe", "cook", "وصفة"]):
        return "recipe"
    return "general"

# ============================================================
# PROMPT
# ============================================================
def build_prompt(question: str, lang: str, qtype: str) -> str:
    lang_rule = (
        "Reply ONLY in Arabic." if lang == "arabic"
        else "Reply ONLY in English."
    )

    return (
        "<|system|>\n"
        f"{BASE_SYSTEM}\n{lang_rule}\n"
        "<|end|>\n"
        "<|user|>\n"
        f"{question}\n"
        "<|end|>\n"
        "<|assistant|>\n"
    )

# ============================================================
# STOP TOKENS
# ============================================================
_STOP = ["<|end|>"]

# ============================================================
# INFERENCE
# ============================================================
def _run(llm, prompt: str):
    return llm(
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.2,
        top_p=0.9,
        top_k=20,
        repeat_penalty=1.1,
        stop=_STOP,
        echo=False,
    )

async def generate(prompt: str):
    loop = asyncio.get_running_loop()
    llm = get_model()

    return await asyncio.wait_for(
        loop.run_in_executor(_executor, _run, llm, prompt),
        timeout=TIMEOUT_SEC
    )

# ============================================================
# CLEAN OUTPUT
# ============================================================
def clean(out) -> str:
    if isinstance(out, dict):
        out = out.get("choices", [{}])[0].get("text", "")

    text = out.replace("\\n", "\n").strip()

    lines = []
    prev_blank = False

    for l in text.splitlines():
        blank = not l.strip()
        if blank and prev_blank:
            continue
        lines.append(l)
        prev_blank = blank

    return "\n".join(lines).strip()

# ============================================================
# CORS
# ============================================================
@middleware
async def cors(req, handler):
    if req.method == "OPTIONS":
        return web.Response(headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET",
            "Access-Control-Allow-Headers": "Content-Type",
        })

    resp = await handler(req)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

# ============================================================
# MAIN API
# ============================================================
async def ask(req):
    ip = req.remote or "x"

    if is_rate_limited(ip):
        return web.json_response({"error": "rate limit"}, status=429)

    try:
        body = await req.json()
        q = (body.get("question") or "").strip()[:MAX_Q_LEN]

        if not q:
            return web.json_response({"error": "empty"}, status=400)

        lang = detect_language(q)
        qtype = classify(q)

        prompt = build_prompt(q, lang, qtype)

        t0 = time.monotonic()
        out = await generate(prompt)

        text = clean(out)

        return web.json_response({
            "response": text,
            "type": qtype,
            "lang": lang,
            "time_ms": int((time.monotonic() - t0) * 1000)
        })

    except asyncio.TimeoutError:
        return web.json_response({"error": "timeout"}, status=504)

    except Exception as e:
        logger.exception(e)
        return web.json_response({"error": "server error"}, status=500)

# ============================================================
# HEALTH
# ============================================================
async def health(req):
    mem = None
    if HAS_PSUTIL:
        mem = psutil.Process().memory_info().rss / 1024 / 1024

    return web.json_response({
        "status": "ok",
        "ctx": N_CTX,
        "threads": N_THREADS,
        "memory_mb": mem,
        "uptime": round(time.monotonic() - _start_time)
    })

# ============================================================
# APP
# ============================================================
app = web.Application(middlewares=[cors])
app.router.add_post("/ask", ask)
app.router.add_get("/health", health)

if __name__ == "__main__":
    logger.info("🚀 Starting server...")
    web.run_app(app, host="0.0.0.0", port=PORT, access_log=False)
