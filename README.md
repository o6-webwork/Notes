# Running Local LLMs with VLLM and Accessing via an OpenAI‑Compatible API

_Local deployment of large language models (LLMs) offers privacy, cost savings, and offline access.  
[VLLM](https://github.com/vllm-project/vllm) is a high‑throughput inference engine that can expose an **OpenAI‑compatible** REST endpoint when run with the `vllm/vllm-openai` Docker image.  
This guide shows you how to_

1. **Spin up** a VLLM server in Docker  
2. **Query** it with or without LangChain  
3. Use **LM Studio** as a GUI alternative

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Docker** | Engine running on the host |
| **NVIDIA GPU** | VLLM is CUDA‑only today |
| **NVIDIA drivers** | Proper version for your card |
| **nvidia‑container‑toolkit** | Gives containers GPU access |
| **Python (optional)** | Needed for the example clients (`pip`, etc.) |
| **Hugging Face cache (optional)** | Mount `~/.cache/huggingface` to avoid redownloading |

---

## 1 – Start the VLLM Server in Docker

### Generic command

```bash
docker run   --gpus all   --name <container_name>   -v ~/.cache/huggingface:/root/.cache/huggingface   -p <host_port>:8000   --ipc=host   vllm/vllm-openai:latest   --model <model_card>   [--max-model-len <tokens>]   [--gpu-memory-utilization <0‑1>]   [other_vllm_flags]
```

**Flag breakdown**

| Flag | Purpose |
|------|---------|
| `--gpus all` | Expose all GPUs (or `--gpus '"device=0,1"'` for specific IDs) |
| `-v ~/.cache/huggingface:/root/.cache/huggingface` | Re‑use host model cache |
| `-p <host_port>:8000` | Map container’s port 8000 to host |
| `--ipc=host` | Better performance with multi‑GPU |
| `--model` | Hugging Face model ID, e.g. `mistralai/Mistral-7B-Instruct-v0.1` |
| `--max-model-len` | Override context window if needed |
| `--gpu-memory-utilization` | Fraction of VRAM VLLM may claim |

### Concrete example

```bash
export CONTAINER_NAME="my-mistral-server"
export MODEL_ID="mistralai/Mistral-7B-Instruct-v0.1"
export HOST_PORT=8000
export MAX_LEN=4096
export GPU_MEM_UTIL=0.90

docker run -d   --gpus all   --name $CONTAINER_NAME   -v ~/.cache/huggingface:/root/.cache/huggingface   -p $HOST_PORT:8000   --ipc=host   vllm/vllm-openai:latest   --model $MODEL_ID   --max-model-len $MAX_LEN   --gpu-memory-utilization $GPU_MEM_UTIL
```

### Using Docker Compose

```
version: "3.8"

services:
    vllm:
        container_name: <container-name>
        image: vllm/vllm-openai:latest
        runtime: nvidia
        ipc: host
        volumes:
         - <model-directory-path-on-dgx>:/mnt/model/
         - /home/dsta/.cache/hugging-face:/root/.cache/huggingface
        ports:
         - "<host-port>:8000"
        environment:
         - NVIDIA_VISIBLE_DEVICES=<devices-used>
        command: [
         "--model", "/mnt/model/",
         "--max-model-len", "<max-model-len>"
         "--gpu-memory-utilization", "<gpu-mem-utilization>",
        ]
```

*View logs with `docker logs -f my-mistral-server`.*

---

## 2 – Interact with the API

The server listens at:

```
http://<host>:<host_port>/v1
```

Most tools merely need:

* **`openai_api_base` / `base_url`** ➜ your VLLM URL  
* **`model`** ➜ the Hugging Face ID you launched with  
* **`api_key`** ➜ any placeholder (VLLM does not check it)

### A. LangChain

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    openai_api_key="not-needed",
    openai_api_base="http://127.0.0.1:8000/v1",
    temperature=0.7,
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain quantization in LLMs.")
]
print(llm.invoke(messages).content)
```

### B. Official `OpenAI` Library

```python
import openai
client = openai.OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"
)

resp = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    messages=[
        {"role": "system", "content": "You are a helpful coder."},
        {"role": "user", "content": "Write a Python add-two‑numbers function."}
    ],
    temperature=0.5,
    max_tokens=100
)
print(resp.choices[0].message.content)
```

### C. Raw `requests`

```python
import requests, json

payload = {
    "model": "mistralai/Mistral-7B-Instruct-v0.1",
    "messages": [
        {"role": "system", "content": "You are a travel advisor."},
        {"role": "user", "content": "Suggest 3 activities in Kyoto."}
    ],
    "temperature": 0.8,
    "max_tokens": 200
}

r = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json=payload, timeout=60
)
print(r.json()["choices"][0]["message"]["content"])
```

### D. `curl`

```bash
curl http://127.0.0.1:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.1",
    "messages": [
      {"role": "system", "content": "You are a science explainer."},
      {"role": "user", "content": "What is a black hole?"}
    ],
    "temperature": 0.6,
    "max_tokens": 150
  }'
```

---

## 3 – Alternative: LM Studio

* **GUI download/management** of models  
* Spins up **`http://localhost:1234/v1`** by default  
* Interact with exactly the same code/snippets—just change the base URL.

---

