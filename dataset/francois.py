from datasets import load_dataset_builder, load_dataset, get_dataset_config_names, list_metrics, load_metric
from transformers import DistilBertTokenizer, DistilBertModel, pipeline

dataset_builder = load_dataset_builder('./ankorsearch_dataset.py')

dataset = load_dataset('./ankorsearch_dataset.py', 'AnkorsearchDS', split='train')
# configs = get_dataset_config_names('./ankorsearch_dataset.py')
# print(configs)

print(dataset.info)

# distilbert-base-uncased

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# metric = load_metric('./ankorsearch_dataset.py', 'AnkorsearchDS')

ner_feature = dataset.features["ner_tags"]
label_names = ner_feature.feature.names

words = dataset["tokens"]
labels = dataset["ner_tags"]
line1 = ""
line2 = ""
for word, label in zip(words[0], labels[0]):
    full_label = label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)

old_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

training_corpus = (
    dataset[i : i + 1000]
    for i in range(0, len(dataset), 1000)
)

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
tokenizer = BertTokenizer.from_pretrained('./ankorbert-en')


from datasets import load_dataset_builder, load_dataset, get_dataset_config_names, list_metrics, load_metric
from transformers import DistilBertTokenizer, DistilBertModel, pipeline


# Replace this with your own checkpoint
model_checkpoint = "/Users/francois.gravelaine/.ankorsearchv1/models/hf/ankorsearchv1"
token_classifier = pipeline(
    "ner", model=model_checkpoint, aggregation_strategy="simple"
)
tokens = token_classifier("yellow towel vegan handmade Eco-Friendly organic made in france")
print(tokens)