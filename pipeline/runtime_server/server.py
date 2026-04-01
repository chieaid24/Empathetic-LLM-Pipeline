import base64
import json
import re
import time
import asyncio
import threading
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

app = FastAPI()

# Model and tokenizer are loaded once when the server starts 
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-4b-it')
model = AutoModelForCausalLM.from_pretrained('google/gemma-3-4b-it', device_map='cuda')

# RAG setup (runs at startup)
AGGREGATED_PATH = "pipeline/data/scenarios_aggregated.json"
with open(AGGREGATED_PATH) as f:
    scenarios = json.load(f)

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
scenario_embeddings = embed_model.encode([s["scenarioText"] for s in scenarios], convert_to_numpy=True)
scenario_embeddings = scenario_embeddings / np.linalg.norm(scenario_embeddings, axis=1, keepdims=True)

faiss_index = faiss.IndexFlatIP(scenario_embeddings.shape[1])
faiss_index.add(scenario_embeddings)

# --- Output filter --- 
SAFE_FALLBACK = "Thanks for telling me how you feel. I hear you, and your feelings really matter."

BLOCKLIST_PATH = "pipeline/data/profanity_wordlist.txt"
with open(BLOCKLIST_PATH) as f:
    _words = [base64.b64decode(line.strip()).decode().lower() for line in f if line.strip()]
_blocklist_pattern = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _words) + r")\b",
    re.IGNORECASE,
)

_emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero-width joiner
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


def check_response(text: str) -> str:
    if _blocklist_pattern.search(text):
        return SAFE_FALLBACK
    text = _emoji_pattern.sub("", text).strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return text


SESSION_TTL = 300  # seconds — abandon partial sessions after 5 minutes

sessions: dict = {}
# sessions[ip] = {
#     "turn": 1,
#     "history": [
#         {"role": "user", "content": "..."},      # turn 1 user text
#         {"role": "assistant", "content": "..."},  # turn 1 response text only
#     ],
#     "last_active": float  # time.time()
# }


@app.on_event("startup")
async def start_cleanup():
    asyncio.create_task(cleanup_sessions())


async def cleanup_sessions():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        expired = [ip for ip, s in list(sessions.items()) if now - s["last_active"] > SESSION_TTL]
        for ip in expired:
            del sessions[ip]


STRATEGY_DEFINITIONS = (
    "Empathy strategies and their meanings:\n"
    "- ptaking (Perspective Taking): Acknowledge the situation from the person's point of view\n"
    "- situation (Situation Awareness): Recognize and describe the specific circumstances\n"
    "- validation (Validation): Affirm that their feelings are understandable and justified\n"
    "- support (Support Offering): Express willingness to help or offer assistance\n"
    "- fq (Follow-up Question): Ask a question to show continued interest\n"
    "- sn (Say nothing): Allow a moment of silence or pause without filling it with words\n"
    "- ex (Exclamation): Use an exclamatory expression to convey emotional reaction\n"
    "- ga (Giving Advice): Offer practical advice or suggestions to help the person\n"
    "- ec (Expressing Condolences): Express sympathy or sorrow for the person's situation\n"
)


def retrieve_examples(query: str, k: int = 3) -> str:
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    _, indices = faiss_index.search(query_vec, k)

    parts = []
    print("\n--- Retrieved RAG scenarios ---")
    for i, idx in enumerate(indices[0], 1):
        scenario = scenarios[idx]
        ex = scenario["strong_example"]
        print(f"  [{i}] emotion={scenario['scenarioEmotion']} | intensity={scenario['scenarioIntensity']} | scenario={scenario['scenarioText'][:80]}...")
        block = (
            "--- Example Scenario ---\n"
            f"Scenario: {scenario['scenarioText']}\n"
            f"Emotion: {scenario['scenarioEmotion']} (intensity {scenario['scenarioIntensity']})\n"
            f"Summary of peer supporter responses:\n{scenario['consensus_summary']}\n"
            f"Strong example response: {ex['response']}\n"
            f"Strategies used: {', '.join(ex['strategies']) if ex['strategies'] else 'none'}\n"
            f"Reasoning: {ex['reasoning']}"
        )
        parts.append(block)
    print("-------------------------------\n")
    return "\n\n".join(parts)


def build_system_prompt(turn: int, retrieved: str) -> str:
    base = (
        "You are a peer supporter. Respond with ONLY your empathetic response in plaintext — "
        "no preamble, no labels, no extra formatting, no emojis. "
        "Your response must be between 50 and 150 words. "
        "Your audience is children. Use simple, age-appropriate language. "
        "Never include profanity, violence, self-harm, sexual content, or drug references.\n\n"
        + STRATEGY_DEFINITIONS
    )
    if turn == 2:
        base += (
            "\nThis is the FINAL turn of the conversation. "
            "You have already responded once — your prior response is in the conversation history. "
            "Do NOT repeat or restate what you already said. "
            "Acknowledge that the person has heard your first response, then go deeper or offer something new. "
            "Do NOT ask questions. Do NOT invite the user to continue the conversation. Close the conversation naturally and warmly.\n"
        )
    base += (
        "\n\nBelow are similar real examples from human peer supporters for reference only. "
        "Use them to inform your strategies and tone:\n\n"
        + retrieved
    )
    return base


@app.post("/reset")
def reset_session(req: Request):
    ip = req.client.host
    if ip in sessions:
        del sessions[ip]
    return {"status": "ok"}


class GenerateRequest(BaseModel):
    text: str
    max_new_tokens: int = 1000
    stream: bool = False


@app.post("/generate")
def generate(request: GenerateRequest, req: Request):
    ip = req.client.host
    session = sessions.get(ip)

    if session is None:
        turn = 1
        prior_history = []
        sessions[ip] = {"turn": 1, "history": [], "last_active": time.time()}
    else:
        turn = 2
        prior_history = session["history"]

    # RAG query should only be based on the user's original inference, not the follow up (this maintains the model's "view" of the situation)
    rag_query = prior_history[0]["content"] if turn == 2 else request.text
    retrieved = retrieve_examples(rag_query)

    system_content = build_system_prompt(turn, retrieved)

    messages = [{"role": "system", "content": system_content}]
    messages.extend(prior_history)
    messages.append({"role": "user", "content": f"Write a response to this text: {request.text}"})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    if request.stream:
        return _stream_response(inputs, request.max_new_tokens, ip, turn, request.text, messages)

    generated_ids = model.generate(**inputs, max_new_tokens=request.max_new_tokens)

    input_length = inputs["input_ids"].shape[1]
    response_text = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True).strip()
    response_text = check_response(response_text)

    if turn == 1:
        sessions[ip]["history"] = [
            {"role": "user", "content": request.text},
            {"role": "assistant", "content": response_text},
        ]
        sessions[ip]["last_active"] = time.time()
        conversation_complete = False
    else:
        del sessions[ip]
        conversation_complete = True
    print(messages)
    print(response_text)
    return {"response": response_text, "conversationComplete": conversation_complete}


def _stream_response(inputs, max_new_tokens: int, ip: str, turn: int, user_text: str, messages: list):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = {**inputs, "max_new_tokens": max_new_tokens, "streamer": streamer}
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def event_generator():
        full_response = []
        completed = False
        try:
            for token_text in streamer:
                if token_text:
                    full_response.append(token_text)
                    yield f"data: {json.dumps({'type': 'token', 'text': token_text})}\n\n"

            response_text = check_response("".join(full_response).strip())
            conversation_complete = (turn == 2)

            if turn == 1:
                sessions[ip]["history"] = [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": response_text},
                ]
                sessions[ip]["last_active"] = time.time()
            else:
                del sessions[ip]

            print(messages)
            print(response_text)

            yield f"data: {json.dumps({'type': 'done', 'response': response_text, 'conversationComplete': conversation_complete})}\n\n"
            completed = True
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
        finally:
            thread.join(timeout=5)
            if not completed and ip in sessions and not sessions[ip].get("history"):
                del sessions[ip]

    return StreamingResponse(event_generator(), media_type="text/event-stream")