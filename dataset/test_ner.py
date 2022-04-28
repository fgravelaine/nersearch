from transformers import pipeline

model_checkpoint = "/Users/francois.gravelaine/.ankorsearchv1/models/hf/ankorsearchv1"
token_classifier = pipeline(
    "ner", model=model_checkpoint, aggregation_strategy="simple"
)
tokens = token_classifier("Yellow towel Cruelty free Eco-Friendly organic made in France")

end_tokens = []
prev=None

for token in tokens:
    if prev and token['start'] == prev['end']:
        if '##' in token['word']:
            token['word'] = token['word'][2:]
        prev['word'] += token['word']
        prev['end'] = token['end']
    else:
        if prev:
            end_tokens.append(prev)
        prev = token

if prev:
    end_tokens.append(prev)

print(end_tokens)
