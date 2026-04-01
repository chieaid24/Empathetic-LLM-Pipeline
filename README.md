# Furhat Empathy LLM

Peer support empathy response system for the Furhat robot — SIRRL Lab, University of Waterloo.

**NOTE: This repo is for showcase purposes only. All sensitive data and benchmarking have been removed.**

## Overview

Generates empathetic responses using a RAG-augmented **Gemma-3-4B-IT** model. Retrieves similar scenarios from 216 human peer supporter examples via FAISS, then generates a contextually appropriate response. Supports multi-turn conversations (2 turns max) with child-safety guardrails.

## Architecture

- **Server:** FastAPI app serving a `POST /generate` and `POST /reset` endpoints
- **Model:** Google Gemma-3-4B-IT (HuggingFace Transformers, CUDA)
- **RAG:** Sentence-transformer embeddings + FAISS cosine similarity (top-3 retrieval)
- **Safety:** Prompt-level child-safety instructions + Base64-encoded profanity wordlist filter
- **Streaming:** Supports both standard and SSE streaming responses

## Quick Start

**Prerequisites:** Python 3.10+, CUDA GPU, [HuggingFace token](https://huggingface.co/settings/tokens) with Gemma model access

```bash
pip install -r requirements.txt

# Start server (loads model + FAISS index on startup)
python pipeline/runtime_server/server.py

# In a separate terminal, run the test client
python pipeline/runtime_client/client.py
```

The first run downloads the Gemma model (~5GB) from HuggingFace. Server starts at `http://localhost:8000`.

## API

### `POST /generate`

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *(required)* | The user's message |
| `max_new_tokens` | int | `1000` | Max tokens to generate |
| `stream` | bool | `false` | Enable SSE streaming |

Each conversation is two turns: the server remembers your first message by IP, so your second request continues the same conversation. On turn 2, an additional system prompt is appended instructing the model not to repeat itself, not to ask questions, and to close the conversation naturally and warmly. Sessions auto-expire after 5 minutes.

### Standard (non-streaming)

Client sends a message, waits for the model to finish generating, and gets the full response back at once.

```json
// Request
{ "text": "I failed my math test and I feel really sad" }

// Response
{
  "response": "It sounds like that test meant a lot to you...",
  "conversationComplete": false
}
```

> [!NOTE]                                                                                       
> The `conversationComplete` flag marks when turn 2 has been completed by the server (i.e. it has been completed once `response.conversationComplete == True`). Use this flag in the client-side logic to know when to move onto the next activity.  
> After this `conversationComplete` marker is returned, the server will delete the stored messages and RAG responses associated with the conversation, allowing you to start a completely fresh conversation on the next request. 

`conversationComplete` is `false` on turn 1 and `true` on turn 2 (session is then deleted).

### Streaming (SSE)

Instead of waiting for the full response, the server sends words back one at a time as they're generated. Uses Server-Sent Events (SSE), which is a long-lived HTTP connection that pushes data to the client.

```json
// Request
{ "text": "I failed my math test and I feel really sad", "stream": true }
```

The server returns a `text/event-stream` with three event types:

```
data: {"type": "token", "text": "It"}
data: {"type": "token", "text": " sounds"}
data: {"type": "token", "text": " like"}
...
data: {"type": "done", "response": "It sounds like...", "conversationComplete": false}
```

On error:
```
data: {"type": "error", "detail": "..."}
```

The `done` event contains the full response after profanity filtering. If the filter catches something, the `done` response will be a safe fallback, else it will be the full returned response.

### `POST /reset`

Clears the current session for the calling IP. No request body required. Use this to force the next `/generate` request to be treated as turn 1.

```json
// Response
{ "status": "ok" }
```

## Project Structure

```
pipeline/
├── runtime_server/
│   └── server.py              # Main server that you will be calling (POST /generate endpoint)
├── runtime_client/
│   └── client.py              # Interactive test client to interact with the server through the terminal
└── data/
    ├── scenarios_aggregated.json   # RAG data (216 aggregated scenarios)
    ├── responses.xlsx           # Raw human peer supporter study data
    ├── strong_examples.xlsx     # Strong example responses
    ├── preprocess_data.py          # Data preprocessing pipeline
    ├── profanity_wordlist.txt      # Base64-encoded profanity filter
    └── profanity_helper.py         # Encode/decode helper for wordlist editing
benchmarking/
└── final_script/
    └── benchmark.py           # Eval on empathetic_dialogues (BLEU, ROUGE, BERTScore)
```

## RAG

- At startup, scenario texts from `scenarios_aggregated.json` are embedded with `all-MiniLM-L6-v2` and indexed in a FAISS `IndexFlatIP` (cosine similarity on L2-normalized vectors). 
- At request time, the user's message is embedded and the top 3 scenarios are retrieved. 
- Each retrieved scenario's text, emotion, intensity, consensus summary, and strong example response are injected into the system prompt. 
- On turn 2, retrieval uses the turn 1 message to keep context consistent.

## Preprocessing

The preprocessing pipeline (`pipeline/data/preprocess_data.py`) transforms the survey data into the `scenarios_aggregated.json` file used by the RAG system at runtime. It does this by aggregating user responses for each scenario (vignette), then using those generated summaries as the RAG documents. This allows 3 _unique_ scenarios to be returned on each request.  

It uses Gemma-3-4B-IT to process the data in three stages:

1. **Stage 1 - Field Concatenation:** Each row in `responses.xlsx` is summarized into a natural language passage capturing the scenario emotion, intensity, selected strategies, reasoning, and response.
2. **Stage 2 - Thematic Clustering:** Passages for the same scenario are grouped and clustered into 2–4 themes with descriptions.
3. **Stage 3 - Consensus Summary:** A 3–5 sentence summary is generated from the themes, describing what most respondents felt and noting any minority viewpoints.

After the three stages, a strong example response is loaded from `strong_examples.xlsx` for each scenario. These "strong examples" have been hand picked to represent responses that correlate to the reponse generation framework highlighted in the study (i.e. include a varied combination of sub-themes in the response.) 
The final output is written to `pipeline/data/scenarios_aggregated.json`.

**Prerequisites:** CUDA GPU (loads Gemma-3-4B-IT)

```bash
python pipeline/data/preprocess_data.py
```

> [!NOTE]
> This only needs to be re-run if the source Excel files (`responses.xlsx` or `strong_examples.xlsx`) are updated. The output `scenarios_aggregated.json` is committed to the repo and used directly by the server.

## Data

The RAG system uses `scenarios_aggregated.json`, which is preprocessed from the human study Excel files. Each scenario contains the scenario text, emotion, intensity, consensus summary, and a strong example response with strategies used.

To edit the profanity wordlist:
```bash
python pipeline/data/profanity_helper.py decode   # creates plaintext file for editing
# edit profanity_wordlist_plaintext.txt
python pipeline/data/profanity_helper.py encode   # re-encodes back to Base64
```

## Dependencies

`fastapi`, `uvicorn`, `transformers`, `sentence-transformers`, `faiss-gpu`, `pandas`, `openpyxl`, `numpy`, `torch`, `requests`, `pydantic`
