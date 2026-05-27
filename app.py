from functools import partial
from aiohttp import web
from aiohttp.web_middlewares import middleware
 
# ============================================================
# MODEL PATH (TINYLLAMA GGUF)
# ============================================================
from huggingface_hub import hf_hub_download

# FIX 1: renamed model_path → LLAMA_MODEL_PATH so all references are consistent
LLAMA_MODEL_PATH = hf_hub_download(
    repo_id="MazenMohamed10/Cura_Mind_Model",
    filename="model.gguf",
    cache_dir="/tmp"
)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# ============================================================
#  CONFIG
# ============================================================
PORT         = 8001
MAX_TOKENS   = 400
N_THREADS    = 4
N_CTX        = 1024
N_BATCH      = 128
MAX_Q_LEN    = 600
TIMEOUT_SEC  = 90
RATE_LIMIT   = 10

_executor   = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llama")
_start_time = time.monotonic()
# FIX 7: keep only the Semaphore for async queuing; the single-worker pool already
# serialises inference, so having both was redundant double-locking.
_sem        = asyncio.Semaphore(1)

# ============================================================
#  RATE LIMITER
# ============================================================
_rate_store: dict[str, deque] = {}

def is_rate_limited(ip: str) -> bool:
    """
    Returns True if the IP has exceeded RATE_LIMIT requests in the last 60 s.
 
    FIX 2 & 3: The original code had two bugs in one block:
      - It deleted the IP entry when the deque became empty, then returned False
        without recording the current request → counter silently reset every window.
      - IPs that stayed over the limit were never cleaned up → unbounded dict growth.
 
    Fixed by:
      - Always appending the current timestamp before returning False.
      - Running cleanup on every call (not just when empty) to evict stale entries
        and keep _rate_store from growing without bound.
    """
    now = time.monotonic()
    window = 60.0

    if ip not in _rate_store:
        _rate_store[ip] = deque()

    dq = _rate_store[ip]

    # Remove timestamps outside the rolling window
    while dq and now - dq[0] > window:
        dq.popleft()
 
    if len(dq) >= RATE_LIMIT:
        return True

    # Record this request
    dq.append(now)

    # Evict the IP entry only after recording, and only when truly empty,
    # so we don't hold empty deques in memory indefinitely.
    if not dq:
        del _rate_store[ip]

    return False
 
# ============================================================
#  LOAD MODEL
# ============================================================
llama = None
try:
    from llama_cpp import Llama
    # FIX 6: use_mlock=True crashes on hosts with low ulimit -l (e.g. HuggingFace
    # Spaces). Default to False; opt in via env var if the host allows it.
    use_mlock = os.environ.get("LLAMA_USE_MLOCK", "0") == "1"
    llama = Llama(
        model_path=LLAMA_MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_batch=N_BATCH,
        use_mmap=True,
        use_mlock=use_mlock,
        verbose=False,
    )
    logger.info(f"✅ Model loaded | threads={N_THREADS} | ctx={N_CTX} | batch={N_BATCH}")
except ImportError:
    logger.error("❌ llama_cpp not installed → pip install llama-cpp-python")
    sys.exit(1)
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    sys.exit(1)


# ============================================================
#  FORMAT INSTRUCTIONS
# ============================================================
FORMAT_INSTRUCTIONS = {
    "nutrition": (
        "Reply with a markdown table:\n"
        "| Food | Calories | Protein(g) | Fat(g) | Carbs(g) |\n"
        "|------|----------|------------|--------|----------|\n"
        "Add a **Total** row at the end. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
    "steps": (
        "Reply with numbered steps only. Max 8 steps. No intro, no summary. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
    "compare": (
        "Reply with a markdown comparison table (max 4 rows, 3 columns). "
        "End with **Verdict:** one sentence. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
    "general": (
        "Reply in 2-5 sentences. Use bullet points if listing 3+ items. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
    "yesno": (
        "Start with Yes or No. Then explain in 1-3 sentences. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
    "recipe": (
        "Format: **Ingredients** as bullet list, then **Steps** as numbered list. "
        "Then add: 💡 **You might also ask:** or 💡 **قد يهمك أيضاً:** with one related question."
    ),
}

# FIX 9: Removed the directive "Never say 'consult a doctor/specialist'" — for a
# health/mental-wellness product this is a safety liability. Medical questions about
# dosages, medications, or mental-health crises must be able to recommend professional
# care. The model will still give detailed, reasoned answers; it is simply no longer
# instructed to suppress safety referrals.
BASE_SYSTEM = (
    "You are an advanced AI assistant with expert-level knowledge in all fields: "
    "medicine, fitness, nutrition, science, math, history, technology, law, psychology, cooking, and more. "
    "Always give accurate, detailed, and well-reasoned answers. "
    "Think step by step before answering complex questions. "
    "For questions involving medications, dosages, or mental health crises, always recommend "
    "consulting a qualified professional in addition to providing information. "
    "Never repeat the question. Never use filler phrases. "
    "Be direct, precise, and thorough. "
    "Always reply in the SAME language the user writes in: "
    "Arabic question → Arabic answer only. English question → English answer only. "
    "Follow the format instruction exactly."
)


# ============================================================
#  LANGUAGE DETECTION
# ============================================================
def detect_language(text: str) -> str:
    # FIX 8: also check for Arabic classifier keywords so that mixed-script
    # questions like "What is بروتين?" are correctly identified as Arabic.
    arabic_chars  = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total_letters = sum(1 for c in text if c.isalpha())
    if total_letters == 0:
        return "english"
    if arabic_chars > total_letters * 0.3:
        return "arabic"
    # Secondary check: any Arabic keyword present → treat as Arabic
    lower = text.lower()
    arabic_keywords = {
        "سعرات","بروتين","دهون","كربوهيدرات","تغذية","وصفة","كيف","هل",
        "مقارنة","ممكن","يمكن","أفضل","أحسن","بين","أيهما",
    }
    if any(kw in lower for kw in arabic_keywords):
        return "arabic"
    return "english"


# ============================================================
#  INPUT SANITIZATION
# ============================================================
_INJECT_TOKENS = (
    "<|begin_of_text|>", "<|eot_id|>", "<|end_of_text|>",
    "<|start_header_id|>", "<|end_header_id|>",
)

def sanitize(text: str) -> str:
    for tok in _INJECT_TOKENS:
        text = text.replace(tok, "")
    return text.strip()


# ============================================================
#  CLASSIFIER
# ============================================================
_RULES: list[tuple[str, frozenset]] = [
    ("nutrition", frozenset({
        "calorie","calories","kcal","nutrition","protein","fat","carb","carbs",
        "macro","macros","سعرات","سعر حراري","بروتين","دهون","كربوهيدرات",
        "تغذية","غذائية","قيمة غذائية","كيلو كالوري",
    })),
    ("recipe", frozenset({
        "recipe","cook","bake","prepare","ingredients","ingredient",
        "وصفة","اطبخ","حضّر","مكونات","طبخة",
    })),
    ("steps", frozenset({
        "how to","steps","plan","guide","routine","method","tips",
        "كيف","خطوات","خطة","برنامج","طريقة","نصائح",
    })),
    ("compare", frozenset({
        "vs","versus","compare","difference","better","healthier","between","which",
        "مقارنة","الفرق","أفضل","أحسن","بين","أيهما",
    })),
    ("yesno", frozenset({
        "is it","can i","should i","do i","does","did","was","were","is there",
        "هل","ممكن","يمكن","هيفيد","يفيد","هيضر","يضر",
    })),
]

def classify(question: str) -> str:
    q = question.lower()
    tokens = set(q.split())
    for qtype, kws in _RULES:
        for kw in kws:
            if " " in kw:
                if kw in q:
                    return qtype
            elif kw in tokens:
                return qtype
    return "general"


# ============================================================
#  PROMPT BUILDER
# ============================================================
def build_prompt(question: str, qtype: str, lang: str) -> str:
    if lang == "arabic":
        lang_instruction = (
            "يجب أن تجيب باللغة العربية فقط. لا تستخدم الإنجليزية أبداً. "
            "فكّر جيداً قبل الإجابة وكن دقيقاً."
        )
        thinking_prompt = "دعني أفكر في هذا بعناية.\n"
    else:
        lang_instruction = (
            "You must reply in English only. Do not use Arabic. "
            "Think carefully and be precise and accurate."
        )
        thinking_prompt = "Let me think about this carefully.\n"

    system = f"{BASE_SYSTEM}\n\n{lang_instruction}\n\n{FORMAT_INSTRUCTIONS[qtype]}"

    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{question}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{thinking_prompt}"
    )
 
 
# ============================================================
#  INFERENCE
# ============================================================
_STOP_TOKENS = [
    "<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>",
    "\nQuestion:", "\nUser:", "\n\n\n",
]
 
def _run_inference(prompt: str) -> str:
    out = llama(
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        temperature=0.1,
        top_p=0.9,
        top_k=20,
        repeat_penalty=1.2,
        stop=_STOP_TOKENS,
        echo=False,
    )
    if isinstance(out, dict):
        choices = out.get("choices") or []
        if choices:
            return choices[0].get("text", "").strip()
    if hasattr(out, "choices") and out.choices:
        return getattr(out.choices[0], "text", "").strip()
    return ""

async def generate_async(prompt: str, question: str = "") -> str:
    # Semaphore queues async callers; single-worker executor serialises the thread.
    async with _sem:
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(_executor, partial(_run_inference, prompt)),
                timeout=TIMEOUT_SEC
            )
        except asyncio.TimeoutError:
            logger.error(f"⏱ Timeout after {TIMEOUT_SEC}s | question: {question[:50]!r}")
            raise RuntimeError(f"Model timed out after {TIMEOUT_SEC}s")


# ============================================================
#  POST-PROCESS
# ============================================================
_CHAT_TOKENS = (
    "<|eot_id|>", "<|end_of_text|>",
    "<|start_header_id|>", "<|end_header_id|>",
    "<|begin_of_text|>",
)

def clean_response(text: str, lang: str = "english") -> str:
    lines = text.splitlines()
    if lines:
        first = lines[0].strip()
        if lang == "arabic" and first.startswith("دعني أفكر"):
            lines = lines[1:]
        elif lang == "english" and first.startswith("Let me think"):
            lines = lines[1:]
    text = "\n".join(lines)

    if lang == "arabic":
        text = re.sub(r"دعني أفكر في هذا بعناية\.?\n?", "", text)
    else:
        text = re.sub(r"Let me think about this carefully\.?\n?", "", text)

    text = text.replace("\\n", "\n").replace("\\t", "\t")

    for tok in _CHAT_TOKENS:
        text = text.replace(tok, "")

    lines = text.splitlines()
    out, prev_blank = [], False
    for line in lines:
        blank = not line.strip()
        if blank and prev_blank:
            continue
        out.append(line)
        prev_blank = blank

    return "\n".join(out).strip()


# ============================================================
#  CORS MIDDLEWARE
# ============================================================
@middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        return web.Response(headers={
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        })
    resp = await handler(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


# ============================================================
#  HANDLERS
# ============================================================
async def handle_question(request: web.Request) -> web.Response:
    ip = request.remote or "unknown"
    if is_rate_limited(ip):
        return web.json_response(
            {"error": "Too many requests. Please wait a moment."},
            status=429
        )
 
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON."}, status=400)

    question = sanitize((body.get("question") or "").strip())

    if not question:
        return web.json_response({"error": "Question is required."}, status=400)
    if len(question) > MAX_Q_LEN:
        return web.json_response(
            {"error": f"Question too long (max {MAX_Q_LEN} chars)."},
            status=400,
        )

    lang   = detect_language(question)
    qtype  = classify(question)
    prompt = build_prompt(question, qtype, lang)

    t0 = time.monotonic()
    logger.info(f"[{qtype.upper()}][{lang}] {question[:80]!r}")

    try:
        raw = await generate_async(prompt, question)
    except RuntimeError as e:
        return web.json_response({"error": str(e)}, status=504)
    except Exception as e:
        logger.exception(f"💥 Inference error | question: {question[:50]!r}")
        return web.json_response({"error": "Model inference failed."}, status=500)

    elapsed = time.monotonic() - t0

    if not raw:
        return web.json_response(
            {"error": "Empty response. Try rephrasing."},
            status=500,
        )

    answer = clean_response(raw, lang)
    logger.info(f"[{qtype.upper()}][{lang}] {elapsed:.2f}s → {answer[:80]!r}")

    return web.json_response({
        "response": answer,
        "type":     qtype,
        "lang":     lang,
        "time_ms":  round(elapsed * 1000),
    })


async def handle_health(request: web.Request) -> web.Response:
    model_ok = llama is not None
    health: dict = {
        "status":       "ok" if model_ok else "degraded",
        "model_loaded": model_ok,
        "model":        os.path.basename(LLAMA_MODEL_PATH),
        "ctx":          N_CTX,
        "threads":      N_THREADS,
        "timeout":      TIMEOUT_SEC,
        "uptime_sec":   round(time.monotonic() - _start_time),
    }
    if HAS_PSUTIL:
        mem = psutil.Process().memory_info().rss / 1024 / 1024
        health["memory_mb"] = round(mem)

    return web.json_response(health, status=200 if model_ok else 503)


# ============================================================
#  GRACEFUL SHUTDOWN
# ============================================================
async def on_shutdown(app):
    logger.info("🛑 Shutting down executor...")
    _executor.shutdown(wait=False)
    logger.info("✅ Executor shutdown complete")


# ============================================================
#  APP
# FIX 5: set client_max_size to prevent multi-MB payloads from being buffered
#         before the question-length check fires.
# ============================================================
app = web.Application(
    middlewares=[cors_middleware],
    client_max_size=64 * 1024,   # 64 KB — well above MAX_Q_LEN, well below dangerous
)
app.router.add_post("/ask",    handle_question)
app.router.add_get("/health",  handle_health)
app.on_shutdown.append(on_shutdown)

if __name__ == "__main__":
    if not HAS_PSUTIL:
        logger.warning("⚠️  psutil not installed → pip install psutil (optional)")
    logger.info("=" * 55)
    logger.info(f"  🚀 LLaMA Server    | port {PORT}")
    logger.info(f"  🧠 Model    : {os.path.basename(LLAMA_MODEL_PATH)}")
    logger.info(f"  🔧 Threads  : {N_THREADS} | CTX: {N_CTX} | Batch: {N_BATCH}")
    logger.info(f"  ⏱  Timeout  : {TIMEOUT_SEC}s | MaxTok: {MAX_TOKENS}")
    logger.info(f"  🛡  RateLimit: {RATE_LIMIT} req/min per IP")
    logger.info(f"  📍 POST     : http://localhost:{PORT}/ask")
    logger.info(f"  🩺 Health   : http://localhost:{PORT}/health")
    logger.info("=" * 55)
    web.run_app(app, host="0.0.0.0", port=PORT, access_log=None)
