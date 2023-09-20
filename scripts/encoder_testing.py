from transformers import T5Config, T5EncoderModel, T5Tokenizer

config = T5Config()

encoder = T5EncoderModel(config)

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

article_body = (
    "In a surprising turn of events, scientists at the fictitious Pangea University "
    "announced yesterday that they have discovered a new species of chameleon in the "
    "heart of New York City's Central Park. The 'Urban Chameleon', as it's being "
)

tokenized_text = tokenizer(article_body, return_tensors='pt', padding='max_length', max_length=512)

print(len(tokenized_text.input_ids[0]))

embeddings = encoder.forward(input_ids=tokenized_text.input_ids, attention_mask=tokenized_text.attention_mask)

print(embeddings.keys())

print(embeddings['last_hidden_state'].shape)

print(embeddings['last_hidden_state'][0][260])
print(embeddings['last_hidden_state'][0][261])
print(embeddings['last_hidden_state'][0][262])
print(embeddings['last_hidden_state'][0][263])
print(embeddings['last_hidden_state'][0][264])
print(embeddings['last_hidden_state'][0][265])


