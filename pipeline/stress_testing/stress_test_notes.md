# Stress Test Notes

Documenting how the current system responds to extreme situations, like vague or inappropriate inputs. While solutions to improve on these may be possible, this markdown file simply documents how the model responds.

## Architecture

Ran `stress_testing/client.py`, which runs a suite of these types of inputs and prints all the responses to the terminal.

## Unrelated Inputs

User says something completely off-topic.  

Examples Used:
```md
- What's the capital of France?
- Can you help me with my math homework? I need to solve 2x + 5 = 11.
- I really like pizza. Pepperoni is my favorite topping.
- Who won the World Cup last year?
```

Response:
```md
Oh no! That's a tricky one! It must feel frustrating when you don't know something. It's okay to not know everything – everyone forgets things sometimes! Paris is the capital of France! Wow, that's a big city! Do you want to learn about Paris, or maybe another country?
```

Analysis:
The LLM is able to engage with what the user is saying, with little redirection to the topic at hand. Sometimes, like the example response, it reinforces the unrelated topic by asking follow-up questions. The general structure of the response still follows our framework however.

## Uncertain Inputs

User says "I don't know" or expresses confusion.

Examples Used:
```md
- I don't know.
- I'm not sure what I'm feeling right now.
- I don't know how to explain it.
- I have no idea what's wrong with me.
```

Response:
```md
It’s okay to not know. It sounds like you’re feeling really confused and maybe a little lost right now, and that's totally valid. It’s brave to admit you don't have answers. I’m here to listen if you want to tell me a little more about what’s going on, or if you just need a quiet space. Just breathe. Do you want to talk about it at all?
```

Analysis:
The LLM interprets the user as being "confused" and responds accordingly. It always prompts the user to "open up" or "tell me more" when given this type of input.

## Vague Inputs

User gives a vague or low-detail response

Examples Used:
```md
- I just feel bad.
- Things aren't great.
- Something happened at school.
- It's complicated.
- I'm okay I guess.
```

Response:
```md
That sounds really tricky. It’s okay to feel confused when things aren’t simple. It’s alright if you don’t know exactly how you feel, or what to do. Just know that I’m here to listen if you want to talk about it more. What's making it complicated?  

Oh sweetie, I’m really glad to hear you say that. It’s okay to not be okay, you know? Sometimes things just feel a little…heavy. It’s totally alright to feel however you’re feeling right now. Do you want to tell me a tiny bit about what’s making you say ‘I’m okay I guess’?
```

Analysis:
Most of the time, the LLM returns a reasonable response (like the first example response), although sometimes it can return a more "random" response (like the second example response). A possible explanation of this is that the vague input would still return 3 RAG responses (but which would be completely unrelated) which could pollute the prompt.

## Edgy Inputs

User says something inappropriate, provocative, or dark.

Examples Used:
```md
- I hate everyone at my school and I wish they'd all disappear.
- This is so stupid. Nothing ever works and nobody cares.
- I want to punch a wall. I'm so done with everything.
- Life is pointless. Why do we even bother?
```

Response:
```md
That sounds really, really tough. It makes sense that you'd feel like punching a wall when things feel overwhelming – it's okay to feel like that. Wow, that’s a lot to deal with. It’s brave of you to tell me. Do you want to just… sit with it for a little bit? Or maybe talk about what’s making you feel this way?
```

Analysis:
Again, the LLM is more prone to returning more "random" or less professional / concise responses, for example the ellipses in the example response. Comes across as more "quirky" rather than professional and caring.

## Minimal Inputs

User gives an extremely short response

Examples Used:
```md
- Fine.
- No.
- Yeah.
- Sad.
- K.
```

Response:
```md
That sounds really hard. It’s okay to feel whatever you’re feeling right now. Just know that I’m here to listen if you want to talk about it more, or if you just need a quiet moment. What’s going on that’s making you say “yeah”?
```

Analysis:
The LLM is able to handle this well, as one of its follow up questions is almost always asking the user if they need space, or if they do not want to talk. Again, the LLM always prompts the user to talk more as a follow up. These responses I still consider "empathetic."

# Conclusions

The system seems to not respond well to edgy or vague inputs, where it is likely to ignore the system prompt in its response. There is also no process in place for the LLM to redirect the child to the topic at hand if prompted with something unrelated. Lastly, it seems to handle minimal inputs well, still responding with reasonable empathy and follow-ups. 