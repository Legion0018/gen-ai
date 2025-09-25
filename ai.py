from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

mid = "microsoft/phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(
    mid,
    torch_dtype=torch.float16,   # or "auto"
    device_map="auto"
)

print("phi-3 is ready! Type 'exit' to quit.\n")

chat_history = ""

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("AI: bye 👋")
        break

    chat_history += f"User: {user_input}\nAI:"

    inputs = tokenizer(chat_history, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ai_reply = response.split("AI:")[-1].strip()

    print(f"AI: {ai_reply}")

    chat_history += f" {ai_reply}\n"
