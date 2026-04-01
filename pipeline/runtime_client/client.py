import json
import requests

SERVER_URL = "http://localhost:8000/generate"

# Calls the server with default settings (no streaming), in which the server responsds with a full JSON response 
def get_empathic_response(text, max_new_tokens=600):
    payload = {"text": text, "max_new_tokens": max_new_tokens}
    response = requests.post(SERVER_URL, json=payload)
    response.raise_for_status()
    return response.json()

# Calls the server with stream: True, in which the server sends Server Side Events, which returns the response in chunks 
def get_empathic_response_stream(text, max_new_tokens=600):
    payload = {"text": text, "max_new_tokens": max_new_tokens, "stream": True}
    response = requests.post(SERVER_URL, json=payload, stream=True)
    response.raise_for_status()

    full_response = ""
    conversation_complete = False

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = json.loads(line[len("data: "):])

        if data["type"] == "token":
            print(data["text"], end="", flush=True)
            full_response += data["text"]
        elif data["type"] == "done":
            full_response = data["response"]
            conversation_complete = data["conversationComplete"]
        elif data["type"] == "error":
            raise RuntimeError(f"Server error: {data['detail']}")

    print()  # newline after streamed tokens
    return {"response": full_response, "conversationComplete": conversation_complete}

if __name__ == "__main__":
    turn = 1
    print("=== Furhat Empathy Client (dev) ===")
    print("Type a message and press Enter to send. Empty input or Ctrl+C to exit.\n")

    while True:
        try:
            user_input = input(f"[Turn {turn}] You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            print("(empty input — exiting)\n")
            break

        # --- Send request (streaming) ---
        print(f"\n  >> Sending to {SERVER_URL} ...")
        print(f"\n  [Turn {turn} Response]")
        print("  ", end="")
        result = get_empathic_response_stream(user_input)

        # --- Check completion ---
        complete = result.get("conversationComplete", False)

        if complete:
            print("\n  --- Conversation complete. Session reset on server. ---\n")
            turn = 1
        else:
            turn = 2

        print()
