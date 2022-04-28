import random
import re

class AnkorDatasetGenerator:

    def generate_example(self):
        example = random.choice(self.examples)

        example_tokens = list(filter(None, example.split(' ')))
        splitted_and_replaced = []
        for token in example_tokens:
            result = re.search(r"\\#|([A-Z]+)|\\#", token)
            type='O'
            if result:
                type=result.group(0)
                subtokens = random.choice(self.values[type]).split(' ')
                first_tag = True
                for subtoken in subtokens:
                    if first_tag:
                        splitted_and_replaced.append(subtoken + ' B-' + type)
                        first_tag = False
                    else:
                        splitted_and_replaced.append(subtoken + ' I-' + type)
            else:
                splitted_and_replaced.append(token+' O')

        return splitted_and_replaced

    def __init__(
            self,
            vocab={
                'TAG': './dataset_vocab/tags_en.txt',
                'CAT': './dataset_vocab/categories_en.txt',
                'MADIN': './dataset_vocab/countries_en.txt',
                'COUNTRY': './dataset_vocab/countries_en.txt',
                'COLOR': './dataset_vocab/colors_en.txt',
            },
            example_file='./dataset_vocab/examples_en.txt'
    ):
        self._vocab = vocab
        self.example_file = example_file
        self.values = {}
        for vocab_key in vocab:
            self.values[vocab_key] = []
            vocab_file = vocab[vocab_key]
            with open(vocab_file) as f:
                self.values[vocab_key] = [line for line in f.read().splitlines()]
        with open(example_file) as f:
            self.examples = [line for line in f.read().splitlines()]

gen = AnkorDatasetGenerator()

f = open("./version2/train.txt", "a")
f.truncate(0)
for i in range(0, 100):
    example = gen.generate_example()
    for line in example:
        f.write(line + "\n")
    f.write('\n')
f.close()

f = open("./version2/test.txt", "a")
f.truncate(0)
for i in range(0, 10):
    example = gen.generate_example()
    for line in example:
        f.write(line + "\n")
    f.write('\n')
f.close()

f = open("./version2/valid.txt", "a")
f.truncate(0)
for i in range(0, 10):
    example = gen.generate_example()
    for line in example:
        f.write(line + "\n")
    f.write('\n')
f.close()
