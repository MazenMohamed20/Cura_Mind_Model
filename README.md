# 🧠 Cura Mind AI Server

Advanced AI assistant API powered by GGUF models using `llama-cpp-python` + `aiohttp`.

Supports:
- Arabic & English
- Nutrition analysis
- Recipes
- Comparisons
- Step-by-step guides
- Health & wellness responses
- Lightweight local inference

---

# ✨ Features

- ⚡ Fast GGUF inference
- 🌍 Automatic Arabic / English detection
- 🧠 Smart question classification
- 🛡 Rate limiting
- 🔒 Prompt sanitization
- 📊 Health monitoring endpoint
- 🚀 Ready for Render & Hugging Face

---

# 📦 Model

Model used:

```txt
MazenMohamed10/Cura_Mind_Model
```

Automatically downloaded from Hugging Face Hub.

---

# 🚀 Installation

## Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

---

## Install requirements

```bash
pip install -r requirements.txt
```

---

# ▶️ Run

```bash
python app.py
```

Server will run on:

```txt
http://localhost:8001
```

---

# 📡 API Endpoints

## POST `/ask`

### Request

```json
{
  "question": "How many calories are in 2 eggs?"
}
```

### Response

```json
{
  "response": "Answer here...",
  "type": "nutrition",
  "lang": "english",
  "time_ms": 1200
}
```

---

## GET `/health`

### Response

```json
{
  "status": "ok",
  "model_loaded": true,
  "ctx": 1024,
  "threads": 4
}
```

---

# 🌍 Language Support

- Arabic question → Arabic answer
- English question → English answer

---

# ⚙️ Configuration

You can edit settings inside `app.py`:

```python
PORT = 8001
MAX_TOKENS = 400
N_THREADS = 4
N_CTX = 1024
N_BATCH = 128
TIMEOUT_SEC = 90
RATE_LIMIT = 10
```

---

# 🛡 Security

Includes:
- Prompt injection sanitization
- Request size limiting
- Timeout handling
- Async queue protection
- IP rate limiting

---

# ☁️ Deployment

Compatible with:
- Render
- Hugging Face Spaces
- Railway
- VPS

---

# 👨‍💻 Author

Mazen Mohamed Fayez
