import json
import pickle

import torch
from paddlenlp.data import JiebaTokenizer, Vocab
from paddlenlp.embeddings import TokenEmbedding
from torch.utils.data import Dataset
from setting import LAW_TXT_PATH
from setting import TEST_DICT, TEST_JSON_PATH, SELF_DICT, EMBEDDED_FACT_LAW_DATA


class Reader:
    def __init__(self, json_path, encoding='utf-8'):
        with open(json_path, encoding=encoding) as file:
            self.data = file.readlines()

    def __getitem__(self, item):
        return json.loads(self.data[item])


class SeqEmbeddedFactLawDataset(Dataset):
    def __init__(self, reader, tokenizer, embedding, law_path=LAW_TXT_PATH, pre_load=False):
        self.law2index = {}
        with open(law_path, encoding='utf-8') as file:
            for idx, law in enumerate(file, 0):
                self.law2index[law.strip()] = idx
        self.tokenizer = tokenizer
        self.embedding = embedding
        if pre_load:
            self.data = None
        else:
            self.data = self._process_data(reader)

    def _process_data(self, reader):
        def util(line: dict):
            fact, law = line['fact'], line['meta']['relevant_articles'][0]
            fact = self.embedding.search(self.tokenizer.cut(fact)).sum(axis=0)
            law = torch.tensor(self.law2index[law], dtype=torch.long)
            fact = torch.tensor(fact, dtype=torch.float)
            return fact, law

        return list(map(util, reader))

    def __getitem__(self, item):
        fact, law = self.data[item]
        return fact, law

    def __len__(self):
        return len(self.data)


def get_dataset(name: str):
    if name == 'test_embedded_fact_law':
        vocab = Vocab.load_vocabulary(TEST_DICT, unk_token='[UNK]', pad_token='[PAD]')
        tokenizer = JiebaTokenizer(vocab=vocab)
        embedding = TokenEmbedding(extended_vocab_path=SELF_DICT)
        reader = Reader(TEST_JSON_PATH)
        dataset = SeqEmbeddedFactLawDataset(reader, tokenizer, embedding, pre_load=True)
        with open(EMBEDDED_FACT_LAW_DATA,'rb') as file:
            dataset.data = pickle.load(file)
        return dataset
