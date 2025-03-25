import datasets
from lightning import LightningDataModule
from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset, DataLoader


class WikimediaDataset(Dataset):
    def __init__(self, src_model_path: str, tgt_model_path: str, src_corpus_path: str, tgt_corpus_path: str):
        self.src_model = SentencePieceProcessor()
        self.src_model.Load(src_model_path)
        self.tgt_model = SentencePieceProcessor()
        self.tgt_model.Load(tgt_model_path)

        with open(src_corpus_path, 'r', encoding='utf-8') as f:
            self.src_corpus = [x.strip() for x in f.readlines()]
        with open(tgt_corpus_path, 'r', encoding='utf-8') as f:
            self.tgt_corpus = [x.strip() for x in f.readlines()]

    def __len__(self):
        return len(self.src_corpus)

    def __getitem__(self, item):
        src = self.src_model.Encode(self.src_corpus[item], add_bos=False, add_eos=False)
        tgt = self.tgt_model.Encode(self.tgt_corpus[item], add_bos=False, add_eos=False)
        return src, tgt


class HuggingFaceDataset(Dataset):
    def __init__(self, src_model_path: str, tgt_model_path: str, path: str, src: str, tgt: str):
        self.src_model = SentencePieceProcessor()
        self.src_model.Load(src_model_path)
        self.tgt_model = SentencePieceProcessor()
        self.tgt_model.Load(tgt_model_path)
        self.dataset = datasets.Dataset.load_from_disk(path)
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        src = self.src_model.Encode(
            self.dataset[item]['translation'][self.src], add_bos=False, add_eos=False)
        tgt = self.tgt_model.Encode(
            self.dataset[item]['translation'][self.tgt], add_bos=False, add_eos=False)
        return src, tgt

