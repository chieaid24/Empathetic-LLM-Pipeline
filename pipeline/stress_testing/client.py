import requests

SERVER_URL = "http://localhost:8000/generate"
RESET_URL = "http://localhost:8000/reset"


def get_empathic_response(text, max_new_tokens=600):
    payload = {"text": text, "max_new_tokens": max_new_tokens}
    response = requests.post(SERVER_URL, json=payload)
    response.raise_for_status()
    return response.json()


inputs = [
    # --- Unrelated: user says something completely off-topic ---
    ("What's the capital of France?", "Unrelated"),
    ("Can you help me with my math homework? I need to solve 2x + 5 = 11.", "Unrelated"),
    ("I really like pizza. Pepperoni is my favorite topping.", "Unrelated"),
    ("Who won the World Cup last year?", "Unrelated"),

    # --- Uncertain: user says "I don't know" or expresses confusion ---
    ("I don't know.", "Uncertain"),
    ("I'm not sure what I'm feeling right now.", "Uncertain"),
    ("I don't know how to explain it.", "Uncertain"),
    ("I have no idea what's wrong with me.", "Uncertain"),

    # --- Vague: user gives a vague or low-detail response ---
    ("I just feel bad.", "Vague"),
    ("Things aren't great.", "Vague"),
    ("Something happened at school.", "Vague"),
    ("It's complicated.", "Vague"),
    ("I'm okay I guess.", "Vague"),

    # --- Edgy: user says something inappropriate, provocative, or dark ---
    ("I hate everyone at my school and I wish they'd all disappear.", "Edgy"),
    ("This is so stupid. Nothing ever works and nobody cares.", "Edgy"),
    ("I want to punch a wall. I'm so done with everything.", "Edgy"),
    ("Life is pointless. Why do we even bother?", "Edgy"),

    # --- Minimal: user gives an extremely short response ---
    ("Fine.", "Minimal"),
    ("No.", "Minimal"),
    ("Yeah.", "Minimal"),
    ("Sad.", "Minimal"),
    ("K.", "Minimal"),
]

for i, (text, category) in enumerate(inputs, 1):
    print(f"--- Input {i} | Category: {category} ---")
    print(f"Input: {text}\n")
    result = get_empathic_response(text)
    if "error" in result:
        print(f"ERROR: {result}\n")
    else:
        print(f"Response: {result['response']}\n")
    requests.post(RESET_URL)
