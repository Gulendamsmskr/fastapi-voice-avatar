# fastapi-voice-avatar

A lightweight Voice Avatar backend built with **FastAPI** and powered by a local **Ollama** model.

---

## Features

- **FastAPI:** High-performance, asynchronous web framework.
- **Local LLM (Ollama):** Runs open-source models (like Llama 3) locally for private, fast response generation.
- Asynchronous Processing: FastAPI handles requests concurrently, ensuring low latency.
- Local Inference: No API keys (like OpenAI) are required. The entire pipeline runs 100% offline and securely on localhost.

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama:** Download and install from [ollama.com](https://ollama.com/). Run your preferred model locally:
   ```bash
   ollama run llama3
