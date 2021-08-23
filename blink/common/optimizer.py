# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import os
import numpy as np

from pytorch_transformers.modeling_bert import (
    BertPreTrainedModel,
    BertConfig,
    BertModel,
)
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch import nn

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
# from pytorch_transformers.optimization import AdamW
from transformers import Adafactor
from transformers.optimization import AdamW, get_constant_schedule_with_warmup


patterns_optimizer = {
    'additional_layers': ['additional'],
    'top_layer': ['additional', 'bert_model.encoder.layer.11.'],
    'top4_layers': [
        'additional',
        'bert_model.encoder.layer.11.',
        'encoder.layer.10.',
        'encoder.layer.9.',
        'encoder.layer.8',
    ],
    'all_encoder_layers': ['additional', 'bert_model.encoder.layer'],
    'all': ['additional', 'bert_model.encoder.layer', 'bert_model.embeddings'],
}

def get_bert_optimizer(models, type_optimization, learning_rate, fp16=False):
    """ Optimizes the network with AdamWithDecay
    """
    if type_optimization not in patterns_optimizer:
        print(
            'Error. Type optimizer must be one of %s' % (str(patterns_optimizer.keys()))
        )
    parameters_with_decay = []
    parameters_with_decay_names = []
    parameters_without_decay = []
    parameters_without_decay_names = []
    no_decay = ['bias', 'gamma', 'beta']
    patterns = patterns_optimizer[type_optimization]

    for model in models:
        for n, p in model.named_parameters():
            if any(t in n for t in patterns):
                if any(t in n for t in no_decay):
                    parameters_without_decay.append(p)
                    parameters_without_decay_names.append(n)
                else:
                    parameters_with_decay.append(p)
                    parameters_with_decay_names.append(n)

    print('The following parameters will be optimized WITH decay:')
    print(ellipse(parameters_with_decay_names, 5, ' , '))
    print('The following parameters will be optimized WITHOUT decay:')
    print(ellipse(parameters_without_decay_names, 5, ' , '))

    optimizer_grouped_parameters = [
        {'params': parameters_with_decay, 'weight_decay': 0.01},
        {'params': parameters_without_decay, 'weight_decay': 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=learning_rate, 
        correct_bias=False
    )

    if fp16:
        optimizer = fp16_optimizer_wrapper(optimizer)

    return optimizer


def get_cpt_optimizer(model, params):
    """ Optimizes the network with AdamWithDecay
    """
    if params["optimizer"] == "adam":
        "Prepare optimizer and schedule (linear warmup and decay)"
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": params["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=params["learning_rate"],
            eps=params["adam_epsilon"],
        )
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=params["warmup_steps"])
    elif params["optimizer"] == "adafactor":
        # recommended hyperparams from https://huggingface.co/transformers/main_classes/optimizer_schedules.html
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=1e-3,
            clip_threshold=1.0,
        )
        # recommended to use constant schedule with warmup https://huggingface.co/transformers/main_classes/optimizer_schedules.html
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=params["warmup_steps"])
    else:
        raise NotImplementedError

    scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    return optimizer, scheduler["scheduler"]


def ellipse(lst, max_display=5, sep='|'):
    """
    Like join, but possibly inserts an ellipsis.
    :param lst: The list to join on
    :param int max_display: the number of items to display for ellipsing.
        If -1, shows all items
    :param string sep: the delimiter to join on
    """
    # copy the list (or force it to a list if it's a set)
    choices = list(lst)
    # insert the ellipsis if necessary
    if max_display > 0 and len(choices) > max_display:
        ellipsis = '...and {} more'.format(len(choices) - max_display)
        choices = choices[:max_display] + [ellipsis]
    return sep.join(str(c) for c in choices)
