from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer

encoder = T5EncoderModel.from_pretrained('google/flan-t5-xl').to(device=0)

print(encoder.config)
