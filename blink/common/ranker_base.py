# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn
from transformers import AutoTokenizer

from blink.cpt.model import gather_indexes
from blink.cpt.utils import shift_right


def get_model_obj(model):
    model = model.module if hasattr(model, "module") else model
    return model

class BertEncoder(nn.Module):
    def __init__(
        self, bert_model, output_dim, layer_pulled=-1, add_linear=None):
        super(BertEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = bert_model.embeddings.word_embeddings.weight.size(1)

        self.bert_model = bert_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask):
        output_bert, output_pooler = self.bert_model(
            token_ids, segment_ids, attention_mask
        )
        # get embedding of [CLS] token
        if self.additional_linear is not None:
            embeddings = output_pooler
        else:
            embeddings = output_bert[:, 0, :]

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result

class CPTEncoder(nn.Module):
    def __init__(
            self, cpt_model, output_dim, is_cpt, tokenizer_name, layer_pulled=-1, add_linear=None):
        super(CPTEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        if is_cpt:
            self.cpt_model = cpt_model.model
        else:
            self.cpt_model = cpt_model
        #bert_output_dim = cpt_model.embeddings.word_embeddings.weight.size(1)
        bert_output_dim = self.cpt_model.encoder.block[-1].layer[-1].DenseReluDense.wo.out_features

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def get_cls_token_representations(self, token_ids, apply_projection_and_normalize):
        cls_token = self.tokenizer.encode("<extra_id_98>")[0]
        device = token_ids.device
        dtype = token_ids.dtype
        cls_tokens = torch.full(
            size=(token_ids.shape[0], 1),
            fill_value=cls_token,
            device=device,
            dtype=dtype,
        )
        pad_decoder_tokens = torch.full(
            size=(token_ids.shape[0], 1),
            fill_value=self.tokenizer.pad_token_id,
            device=device,
            dtype=dtype,
        )
        # if self.hparams.cls_before_pad:
        decoder_input_ids = torch.cat([cls_tokens, pad_decoder_tokens], dim=1)
        # else:
        #     decoder_input_ids = torch.cat([pad_decoder_tokens, cls_tokens], dim=1)

        cls_at_encoder_side = False
        if not cls_at_encoder_side:  # decoder side cls
            input_ids = token_ids
        else:
            input_ids = torch.cat([cls_tokens, token_ids], dim=1)
            # at decoder side we just use a single pad token
            decoder_input_ids = pad_decoder_tokens

        model_output = self.cpt_model(
            input_ids=input_ids,
            attention_mask=(input_ids != self.tokenizer.pad_token_id),
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
        )

        # first select either encoder or decoder hidden states depending on model config
        full_hidden_states = (
            model_output.encoder_last_hidden_state
            if cls_at_encoder_side
            else model_output.decoder_hidden_states[-1]
        )

        # Now select just the index of the CLS token
        cls_token_representations = (
            full_hidden_states[:, 0, :]
            if True # self.hparams.cls_before_pad or self.hparams.cls_at_encoder_side
            else full_hidden_states[:, 1, :]
        )

        # if apply_projection_and_normalize:
        #     cls_token_representations = self.projection_tgt(cls_token_representations)

        return cls_token_representations

    def forward(self, token_ids, target=None, target_labels=None, link_decoder_indices=None, get_cls_rep=False):
        if get_cls_rep:
            target_rep = self.get_cls_token_representations(
                token_ids=token_ids,
                apply_projection_and_normalize=False,
            )

        else:
            decoder_input_ids = shift_right(
                token_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                decoder_start_token_id=self.tokenizer.pad_token_id,
            )

            # target_labels has masks for first prefix tokens
            output = self.cpt_model(
                input_ids=token_ids,
                decoder_input_ids=decoder_input_ids,
                labels=target_labels,
                output_hidden_states=True,
            )
            # get representation of target entities at decoder side, positions of which are specified at batch['link_decoder_indices']
            target_entity_repr, invalid_positions_mask = gather_indexes(
                output.decoder_hidden_states[-1], link_decoder_indices
            )
            target_rep = target_entity_repr
            # target_rep = self.projection_src(target_entity_repr)

        # get embedding of [CLS] token
        embeddings = target_rep

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result

