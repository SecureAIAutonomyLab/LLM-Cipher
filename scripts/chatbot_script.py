import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from colorama import Fore, init

init(autoreset=True)

model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def chat_prompt(user_input):
    return (
        "You're an AI assistant, please politely respond to the Human's request.\n"
        "Please only respond once.\n"
        f"HUMAN: {user_input}\n"
        "AI: "
    )

def get_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generate_ids = model.generate(inputs, max_length=128)
    return tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

def main():
    while True:
        prompt = input(Fore.GREEN + "Human: ")
        if prompt.lower() in ["exit", "quit", "bye"]:
            print(Fore.BLUE + "AI: Goodbye!")
            break
        response = get_response(chat_prompt(prompt))
        print(Fore.BLUE + f"AI: {response.replace(chat_prompt(prompt), '')}")

if __name__ == "__main__":
    main()
