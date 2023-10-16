For human, the filename is:
```<DOMAIN>_human.jsonl```

For machine-generated, filename is:
```<DOMAIN>_<MODEL>.jsonl```

DOMAIN refers to the dataset used, it can be one of the: ```wikipedia, wikihow, reddit, arxiv, ruATD, baike, urdu-news, id-newspaper```

MODEL refers to the model used, it can be one of the: ```davinci, chatGPT, cohere, dolly-v2, bloomz, flan-t5, llama```

The JSONL dictionary entry is:
```prompt, human_text, machine_text, model, source, source_ID```

# original
original M4 data


# original_cleaned
Truncated original data, up to 512 Tokens
Each row has `human_text` and `machine_text` keys