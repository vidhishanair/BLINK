# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import nn


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
            self, cpt_model, output_dim, layer_pulled=-1, add_linear=None):
        super(CPTEncoder, self).__init__()
        self.layer_pulled = layer_pulled
        bert_output_dim = cpt_model.embeddings.word_embeddings.weight.size(1)

        self.cpt_model = cpt_model
        if add_linear:
            self.additional_linear = nn.Linear(bert_output_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask):
        cls_rep = self.model.get_cls_token_representations(
            token_ids=token_ids,
            apply_projection_and_normalize=self.hparams.use_projection,
        )

        # decoder_input_ids = shift_right(
        #     batch["target"],
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     decoder_start_token_id=self.tokenizer.pad_token_id,
        # )
        #
        # # target_labels has masks for first prefix tokens
        #
        # output = self.model(
        #     input_ids=batch["source"],
        #     decoder_input_ids=decoder_input_ids,
        #     labels=batch["target_labels"],
        #     output_hidden_states=True,
        # )
        # # get representation of target entities at decoder side, positions of which are specified at batch['link_decoder_indices']
        # target_entity_repr, invalid_positions_mask = gather_indexes(
        #     output.decoder_hidden_states[-1], batch["link_decoder_indices"]
        # )
        # target_entity_repr = self.projection_src(target_entity_repr)

        # get embedding of [CLS] token
        embeddings = cls_rep

        # in case of dimensionality reduction
        if self.additional_linear is not None:
            result = self.additional_linear(self.dropout(embeddings))
        else:
            result = embeddings

        return result

