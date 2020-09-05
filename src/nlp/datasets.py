import os
from typing import Callable, List

import torchtext
from definitions import DEFAULT_DEVICE, PROJECT_DIR
from utils import *


def quora_loader(batch_size: int, dir: str, tokenizer: Callable[[str], List]='spacy', device: int = None, format: str = 'csv',
                 use_vocab: bool = False):
    device = DEFAULT_DEVICE if device is None else prepare_device(bool(device))
    id_field = torchtext.data.Field(sequential=False)
    text_field = torchtext.data.Field(lower=True, sequential=True, include_lengths=True, tokenize=tokenizer,
                                      use_vocab=use_vocab)
    label_field = torchtext.data.Field(sequential=False)
    train_path = os.path.join(dir, 'train.{}'.format(format))
    dev_path = os.path.join(dir, 'dev.{}'.format(format))
    test_path = os.path.join(dir, 'test.{}'.format(format))
    train_dataset = torchtext.data.TabularDataset(path=train_path, format=format,
                                                  fields=[
                                                      ('id', None), ('qid1', id_field), ('qid2', id_field),
                                                      ('question1', text_field),
                                                      ('question2', text_field),
                                                      ('label', label_field)],
                                                  skip_header=True)
    valid_dataset = torchtext.data.TabularDataset(path=dev_path, format=format,
                                                  fields=[
                                                      ('id', None), ('qid1', id_field), ('qid2', id_field),
                                                      ('question1', text_field),
                                                      ('question2', text_field),
                                                      ('label', label_field)],
                                                  skip_header=True)
    test_dataset = torchtext.data.TabularDataset(path=test_path, format=format,
                                                 fields=[
                                                     ('id', None), ('qid1', id_field), ('qid2', id_field),
                                                     ('question1', text_field),
                                                     ('question2', text_field),
                                                     ('label', label_field)],
                                                 skip_header=True)
    if use_vocab:
        text_field.build_vocab(train_dataset, valid_dataset)

    train_iterator = torchtext.data.BucketIterator(
        train_dataset, batch_size=batch_size, device=device, shuffle=True)
    valid_iterator = torchtext.data.BucketIterator(
        valid_dataset, batch_size=batch_size, device=device, shuffle=False)
    test_iterator = torchtext.data.BucketIterator(
        test_dataset, batch_size=batch_size, device=device, shuffle=False)
    return text_field, label_field, train_dataset, valid_dataset, test_dataset, train_iterator, valid_iterator, test_iterator


def quora_loader_from_file(batch_size: int, filename: str, tokenizer: Callable[[str], List]='spacy', device: int = None,
                           format: str = 'csv'):
    device = DEFAULT_DEVICE if device is None else prepare_device(bool(device))
    id_field = torchtext.data.Field(sequential=False)
    text_field = torchtext.data.Field(lower=True, sequential=True, include_lengths=True, tokenize=tokenizer, use_vocab=True)
    label_field = torchtext.data.Field(sequential=False)
    dataset = torchtext.data.TabularDataset(path=filename, format=format,
                                            fields=[
                                                ('id', None), ('qid1', id_field), ('qid2', id_field),
                                                ('question1', text_field),
                                                ('question2', text_field),
                                                ('label', label_field)],
                                            skip_header=True)
    text_field.build_vocab(dataset)
    ds_iterator = torchtext.data.BucketIterator(
        dataset, batch_size=batch_size, device=device, shuffle=True)
    return text_field, label_field, dataset, ds_iterator


if __name__ == '__main__':
    # text_field, label_field, dataset, ds_iterator = quora_loader_from_file(batch_size=1, filename=os.path.join(PROJECT_DIR, "data", "raw", "train.csv"))
    # for batch in ds_iterator:
    #     print(batch)
    #     break
    import fire
    #
    # fire.Fire(quora_loader)
