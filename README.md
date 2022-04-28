# nersearch

[Http server](./dataset/ankor_http.py) : Serve the model through an HTTP server
[Model trainer](./dataset/ankor_model_train.py) : Train the model
[Tokenizer](./dataset/ankor_model_train.py) : Tokenize content/tokens (? not sure about this one)
[Dataset](./dataset/ankorsearch_dataset.py) : Generate dataset (tokens + train/valid/test data)
[Generator](./dataset/generate_dataset_files.py) : Generate files for dataset (using templates + ankorstore data)
[Test it](./dataset/test_ner.py) : Test NER without the http server