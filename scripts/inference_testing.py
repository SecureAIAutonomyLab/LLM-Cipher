from transformers import AutoTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

def rewrite_prompt(text):
    return (
        "Your task is to rewrite the following text.\n"
        f"Text: \n{text}\n"
        "New Text: "
    )

article_body = (
    "In a surprising turn of events, scientists at the fictitious Pangea University "
    "announced yesterday that they have discovered a new species of chameleon in the "
    "heart of New York City's Central Park. The 'Urban Chameleon', as it's being "
    "dubbed, is believed to have evolved over the past decade and possesses an "
    "uncanny ability to mimic common city objects like traffic cones and discarded "
    "coffee cups. Dr. Ima Pretend, the lead researcher on the project, mentioned, "
    "'It's truly a marvel of evolution. We've never seen anything quite like this. "
    "When in danger, the Urban Chameleon can seamlessly blend into its surroundings, "
    "making it nearly indistinguishable from everyday city trash.' The team believes "
    "that the chameleon's unique adaptation might be a result of the city's increasing "
    "litter issue. While fascinating, environmentalists are taking this discovery as "
    "a sign that the city needs to ramp up its waste management efforts. Studies on "
    "the Urban Chameleon will continue, with the hope of learning more about its "
    "origins and potential implications for urban ecosystems."
)

inputs = tokenizer(rewrite_prompt(article_body), return_tensors="pt")

# Generate okay how are you doing? blah blah  how are you doing today 
generate_ids = model.generate(inputs.input_ids.to(device=0), max_length=1024)

print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])






































































































































































































