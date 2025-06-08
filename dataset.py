import os
import torch
from typing import Union, List, Tuple
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset
import random


class TextDataset(Dataset):
    TRAIN_VAL_RANDOM_SEED = 42
    VAL_RATIO = 0.05

    def __init__(self, data_file: str, train: bool = True, sp_model_prefix: str = None,
                 vocab_size: int = 2000, normalization_rule_name: str = 'nmt_nfkc_cf',
                 model_type: str = 'bpe', max_length: int = 128):
        """
        Dataset with texts, supporting BPE tokenizer
        :param data_file: txt file containing texts
        :param train: whether to use train or validation split
        :param sp_model_prefix: path prefix to save tokenizer model
        :param vocab_size: sentencepiece tokenizer vocabulary size
        :param normalization_rule_name: sentencepiece tokenizer normalization rule
        :param model_type: sentencepiece tokenizer model type
        :param max_length: maximal length of text in tokens
        """
        if not os.path.isfile(sp_model_prefix + '.model'):
            # train tokenizer if not trained yet
            SentencePieceTrainer.train(
                input=data_file, vocab_size=vocab_size,
                model_type=model_type, model_prefix=sp_model_prefix,
                normalization_rule_name=normalization_rule_name
            )
        # load tokenizer from file
        self.sp_model = SentencePieceProcessor(model_file=sp_model_prefix + '.model')

        with open(data_file, encoding='utf-8', errors='ignore') as file:
            texts = file.readlines()

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Split texts to train and validation fixing self.TRAIN_VAL_RANDOM_SEED
        The validation ratio is self.VAL_RATIO
        """
        random.seed(self.TRAIN_VAL_RANDOM_SEED)
        random.shuffle(texts)
        val_size=int(len(texts) * self.VAL_RATIO)
        val_texts=texts[:val_size]
        train_texts=texts[val_size:]
        self.texts=train_texts if train else val_texts
        self.indices =self.sp_model.encode(self.texts)
        raw_pad_id=self.sp_model.pad_id()
        raw_unk_id=self.sp_model.unk_id()
        raw_bos_id=self.sp_model.bos_id()
        raw_eos_id=self.sp_model.eos_id()
        self.vocab_size = self.sp_model.vocab_size()
        self.pad_id=0 if raw_pad_id < 0 else raw_pad_id
        self.unk_id=0 if raw_unk_id < 0 else raw_unk_id
        self.bos_id=1 if raw_bos_id < 0 else raw_bos_id
        self.eos_id=2 if raw_eos_id < 0 else raw_eos_id
        self.pad_id=min(self.pad_id, self.vocab_size - 1)
        self.unk_id=min(self.unk_id, self.vocab_size - 1)
        self.bos_id=min(self.bos_id, self.vocab_size - 1)
        self.eos_id=min(self.eos_id, self.vocab_size - 1)
        self.max_length max_length

    def text2ids(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        return self.sp_model.encode(texts)

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        return self.sp_model.decode(ids)

    def __len__(self):
        """
        Size of the dataset
        :return: number of texts in the dataset
        """
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        """
        Add specials to the index array and pad to maximal length
        :param item: text id
        :return: encoded text indices and its actual length (including BOS and EOS specials)
        """
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Take corresponding index array from self.indices,
        add special tokens (self.bos_id and self.eos_id) and 
        pad to self.max_length using self.pad_id.
        Return padded indices of size (max_length, ) and its actual length
        """
        text_indices=self.indices[item]
        text_indices=[idx for idx in text_indices if 0 <= idx < self.vocab_size]
        indices_with_specials =[self.bos_id] + text_indices + [self.eos_id]
        if len(indices_with_specials)> self.max_length:
            indices_with_specials=indices_with_specials[:self.max_length]
            indices_with_specials[-1]=self.eos_id
        length=len(indices_with_specials)
        while len(indices_with_specials) <self.max_length:
            indices_with_specials.append(self.pad_id)
        indices=torch.tensor(indices_with_specials, dtype=torch.long)
        indices=torch.clamp(indices, 0, self.vocab_size - 1)

        return indices, length