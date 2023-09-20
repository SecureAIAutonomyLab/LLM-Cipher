from transformers import T5Config, T5EncoderModel, T5Tokenizer

config = T5Config()

encoder = T5EncoderModel(config)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

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
    "origins and potential implications for urban ecosystems. However, it's important to note that."
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
    "origins and potential implications for urban ecosystems. However, it's important to note that."
    "litter issue. While fascinating, environmentalists are taking this discovery as "
    "a sign that the city needs to ramp up its waste management efforts. Studies on "
    "the Urban Chameleon will continue, with the hope of learning more about its "
    "origins and potential implications for urban ecosystems. However, it's important to note that."
    "In a surprising turn of events, scientists at the fictitious Pangea University "
    "announced yesterday that they have discovered a new species of chameleon in the "
    "heart of New York City's Central Park. The 'Urban Chameleon', as it's being "
    "dubbed, is believed to have evolved over the past decade and possesses an "
    "uncanny ability to mimic common city objects like traffic cones and discarded "
    "coffee cups. Dr. Ima Pretend, the lead researcher on the project, mentioned, "
)

tokenized_text = tokenizer(article_body, return_tensors='pt', padding='max_length', max_length=512)

print(len(tokenized_text.input_ids[0]))

embeddings = encoder.forward(input_ids=tokenized_text.input_ids, attention_mask=tokenized_text.attention_mask)

print(embeddings.keys())

print(embeddings['last_hidden_state'].shape)

print(embeddings['last_hidden_state'][0][260].shape)

