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
### Using Docker Compose (`docker-compose.yml`)

```yaml
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
      # Comma-separated list of GPUs you want this container to *see*.
      # vLLM will only be able to use these GPUs.
      - NVIDIA_VISIBLE_DEVICES=<cuda-visible-devices>   # e.g. 0,1,2
    command: [
      "--model", "/mnt/model/",

      # ─────────────────── Required arguments ───────────────────
      "--max-model-len", "<max-model-len>",                # required

      # ─────────────────── Optional arguments ───────────────────
      "--tensor-parallel-size", "<tp-size>",               # optional: shards model across GPUs
      "--gpu-memory-utilization", "<gpu-mem-utilization>", # optional: 0 < value ≤ 1
    ]
```

| Element (⚙ = *required*, 🛈 = *optional*) | Purpose / Notes |
|-------------------------------------------|-----------------|
| `<container-name>` ⚙ | Name of the Docker container. |
| `<model-directory-path-on-dgx>` ⚙ | Host path containing the **model weights**. |
| `<host-port>` ⚙ | Port on the host machine that will be forwarded to **8000** in the container. |
| `<cuda-visible-devices>` ⚙ | Sets **which GPUs the container can access** via `NVIDIA_VISIBLE_DEVICES`. Example: `0,1,2`. |
| `<max-model-len>` ⚙ | *Maximum* number of tokens (prompt + generation) allowed per request. |
| `<tp-size>` 🛈 | **Tensor‑parallel size**—how many *visible* GPUs vLLM should split the model across. Must be ≤ the number of GPUs in `<cuda-visible-devices>`. |
| `<gpu-mem-utilization>` 🛈 | Fraction (0–1) of each GPU’s memory vLLM will use for *KV cache* + weights. |

---

#### CUDA‑visible devices vs. Tensor Parallel size

* **`NVIDIA_VISIBLE_DEVICES` (CUDA‑visible devices)** simply tells the container “these are the GPUs you may use.”  
* **`--tensor-parallel-size`** tells **vLLM** how many of the *visible* GPUs to shard the model across.  
  *Example*: If `NVIDIA_VISIBLE_DEVICES=0,1,2,3` but `--tensor-parallel-size 2`, vLLM will load the model on GPUs **0 & 1 only**.

---

#### `--gpu-memory-utilization` and the KV cache

* **KV cache** holds each layer’s *key* and *value* tensors for tokens that have already been processed, enabling fast autoregressive generation.  
* vLLM estimates **how many “cache slots” (tokens × layers) can fit** based on:
  1. **Model weights** (fixed memory cost).  
  2. The memory budget you allow for everything else—primarily the KV cache.
* `--gpu-memory-utilization <fraction>` sets the **upper bound** of total GPU memory vLLM may allocate.  
  * High value (e.g., `0.9`) ⇒ **larger KV cache** ⇒ longer contexts / more parallel requests, but less head‑room for other processes.  
  * Low value (e.g., `0.7`) ⇒ smaller cache, but frees memory for monitoring tools, other containers, etc.

Monitor memory usage with `docker logs -f <container-name>` and adjust accordingly.

---

### Concrete example

```yaml
services:
  vllm:
    container_name: vllm-qwq
    image: vllm/vllm-openai:latest
    runtime: nvidia
    ipc: host
    volumes:
      - /home/otb/Desktop/QwQ-32B:/mnt/model/
      - /home/otb/.cache/hugging-face:/root/.cache/huggingface
    ports:
      - "1999:8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=1,2,3              # use GPUs 1–3 on the host
    command: [
      "--model", "/mnt/model/",
      "--max-model-len", "16000",                 # allow up to 16 k tokens per request
      "--tensor-parallel-size", "3",              # shard model across 3 GPUs
      "--gpu-memory-utilization", "0.9"           # use 90 % of each GPU’s memory
    ]
```

In this setup, a **32‑billion‑parameter model** stored at `/home/otb/Desktop/QwQ-32B` is served locally on **http://localhost:1999**.  
The model is tensor‑parallelised over **3 A100 GPUs** (IDs 1, 2, 3), each allowed to consume **90 %** of its memory, leaving ~10 % headroom for driver overhead and system processes.

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

