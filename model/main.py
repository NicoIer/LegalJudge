from torch.utils.data import DataLoader

from dataset import call2018
from setting import TEST_JSON_PATH, TEST_DICT, LAW_TXT_PATH


def train_fact_classify_baseline():
    data = call2018.get_data(TEST_JSON_PATH)
    vocab = call2018.get_vocab(TEST_DICT)
    dataset = call2018.FactLawDataset(data, vocab, LAW_TXT_PATH, 400)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    padding_idx = vocab[dataset.pad_token]