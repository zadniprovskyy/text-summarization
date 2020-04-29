import spacy

from utils import to_var, AttrDict
import torch
import os
import numpy as np
from encoders import GRUEncoder
from decoders import RNNDecoder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchtext
from torchtext import data
import pandas as pd
import random
import spacy
from amazon_reviews_loader import load_data

# create a tokenizer function
spacy_en = spacy.load('en_core_web_sm')

def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

TEST_SENTENCE = "Did not enjoy the product. Really bad cheese."


teacher_forcing_ratio = 0.5
MAX_LENGTH = 100
SOS_token = 0
EOS_token = 1

def train_step(input_tensors, target_tensors, encoder, decoder, optimizer, criterion, opts):
    batch_size = input_tensors.shape[0]
    encoder_hidden = encoder.init_hidden(batch_size)

    optimizer.zero_grad()

    # forward pass through encoder
    encoder_annotations, encoder_hidden = encoder(input_tensors)
    # generate sos vector for the first decoder input
    decoder_start_vector = torch.ones(batch_size).long().unsqueeze(1) * SOS_token  # BS x 1 --> 16x1  CHECKED
    decoder_input = to_var(decoder_start_vector, opts.cuda)  # BS x 1 --> 16x1  CHECKED

    seq_len = target_tensors.size(1)  # Gets seq_len from BS x seq_len

    decoder_inputs = torch.cat([decoder_input, target_tensors[:, 0:-1]],
                               dim=1)  # Gets decoder inputs by shifting the targets to the right

    decoder_outputs, attention_weights = decoder(decoder_inputs, encoder_annotations, encoder_hidden)
    decoder_outputs_flatten = decoder_outputs.view(-1, decoder_outputs.size(2))
    targets_flatten = target_tensors.view(-1)
    loss = criterion(decoder_outputs_flatten, targets_flatten)


    # Zero gradients
    optimizer.zero_grad()
    # Compute gradients
    loss.backward()
    # Update the parameters of the encoder and decoder
    optimizer.step()

    return loss

if __name__=="__main__":
    args = AttrDict()
    args_dict = {
        'cuda': False,
        'nepochs': 100,
        'checkpoint_dir': "checkpoints",
        'learning_rate': 0.005,
        'lr_decay': 0.99,
        'batch_size': 10,
        'hidden_size': 20,
        'encoder_type': 'rnn',  # options: rnn / transformer
        'decoder_type': 'rnn',  # options: rnn / rnn_attention / transformer
        'attention_type': '',  # options: additive / scaled_dot
    }
    args.update(args_dict)

    tab_data = load_data(batch_size=args.batch_size)
    TEXT_VOCAB_LEN = len(tab_data.fields['Text'].vocab.itos)
    SUMMARY_VOCAB_LEN = len(tab_data.fields['Summary'].vocab.itos)

    train_data, val_data = tab_data.split(split_ratio=0.9, random_state=random.getstate())

    train_iter = torchtext.data.BucketIterator(dataset=train_data,
                                               batch_size=args.batch_size,
                                               sort_key=lambda x: x.TEXT.__len__(),
                                               shuffle=True,
                                               sort=False)

    valid_iter = torchtext.data.BucketIterator(dataset=val_data,
                                               batch_size=args.batch_size,
                                               sort_key=lambda x: x.TEXT.__len__(),
                                               shuffle=True,
                                               sort=False)

    encoder = GRUEncoder(vocab_size=TEXT_VOCAB_LEN,
                         hidden_size=args.hidden_size, opts=args)
    decoder = RNNDecoder(vocab_size=SUMMARY_VOCAB_LEN,
                         hidden_size=args.hidden_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)

    for batch in train_iter:
        loss = train_step(batch.Text, batch.Summary, encoder, decoder, optimizer, criterion, args)
        print("Training loss: {0}".format(loss))