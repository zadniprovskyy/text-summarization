import torchtext
from torchtext import data
import pandas as pd
import random
import spacy

# create a tokenizer function
spacy_en = spacy.load('en_core_web_sm')

def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def load_data(batch_size=32):
    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=True)
    SUMMARY = data.Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=True)

    revs = pd.read_csv("Reviews.csv", nrows=100)
    revs[['Text', 'Summary']][1:].to_csv("reviews_small.csv", index=False)

    tab_data = torchtext.data.TabularDataset(
        path='/Users/Yegor/PycharmProjects/text-summarization/reviews_small.csv', format='csv',
        fields=[('Text', TEXT), ('Summary', SUMMARY)]
    )

    TEXT.build_vocab(tab_data, min_freq=3)
    SUMMARY.build_vocab(tab_data, min_freq=3)

    return tab_data