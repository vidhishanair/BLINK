# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.biencoder.zeshel_utils import world_to_id
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG
from blink.cpt.model import CPTModel
from blink.cpt.data import split_token_ids_with_span_mask_into_source_target


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):

    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:
        mention_tokens = tokenizer.tokenize(sample[mention_key])
        # mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]
        mention_tokens = mention_tokens

    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (
        context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )

    #context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]
    context_tokens = ["<extra_id_98>"] + context_tokens

    context_left_ids = tokenizer.convert_tokens_to_ids(["<extra_id_98>"]+context_left[-left_quota:])
    mention_ids = tokenizer.convert_tokens_to_ids(mention_tokens)
    context_right_ids = tokenizer.convert_tokens_to_ids(context_right[:right_quota]) 
    input_ids = context_left_ids + mention_ids + context_right_ids
    input_mention_mask = [False]*len(context_left_ids) + [True]*len(mention_ids) + [False]*len(context_right_ids)

    mention_start_index = [len(context_left_ids)]
    mention_indices = [0]

    sentinel_token_ids = [
        tokenizer.encode(f"<extra_id_{i}>", add_special_tokens=False)[0] for i in range(49)
    ]
    target_token_ids = [tokenizer.encode(f"<extra_id_{k}>", add_special_tokens=False)[0] for k in range(49, 98)]  # for contrast loss
    cls_token_id = tokenizer.encode(f"<extra_id_{98}>", add_special_tokens=False)[0]  # for contrast loss / document embedding

    source_target = split_token_ids_with_span_mask_into_source_target(input_ids, input_mention_mask, sentinel_token_ids)

    prefix_ids = [cls_token_id] + [
        target_token_ids[k] for k in range(source_target["num_masked_spans"])
    ]
    if True: # cls_at_encoder_side:
        source_target["source"] = [cls_token_id] + source_target["source"]
        # add one to mappings of original tokens to the source tokens (because we now have one additional CLS at the start)
        source_target["original_token_idx_to_source_idx"] = [
            el + 1 for el in source_target["original_token_idx_to_source_idx"]
        ]
    # ignore the prefix tokens for lm_loss except optionally for the first cls token
    lm_loss_on_cls_token = True
    masked_prefix_ids = [
        -100 if ((not lm_loss_on_cls_token) or idx != 0) else el for idx, el in enumerate(prefix_ids)
    ]
    # instance = {
    #     "source": torch.tensor(source_target["source"]),
    #     "tokens": torch.tensor(source_target["source"]),
    #     "target": torch.tensor(prefix_ids + source_target["target"]),
    #     "prefix_length": torch.tensor(len(prefix_ids)),
    #     "target_labels": torch.tensor(masked_prefix_ids + source_target["target"]),
    #     # TODO: Are we sure we should mask out prefix ids?
    # }
    instance = {
        "source": [0] * (max_seq_length - len(source_target["source"])) + source_target["source"],
        "tokens": context_tokens,
        "target": [0] * (max_seq_length - len(prefix_ids + source_target["target"])) + prefix_ids + source_target["target"],
        "prefix_length": len(prefix_ids),
        "target_labels": [0] * (max_seq_length - len(masked_prefix_ids + source_target["target"])) + masked_prefix_ids + source_target["target"],
        # TODO: Are we sure we should mask out prefix ids?
    }
    mention_mask_index = [
        source_target["mask_start_indices"].index(mask_start_index)
        for mask_start_index in mention_start_index
    ]
    original_token_idx_to_source_idx = source_target["original_token_idx_to_source_idx"]
    link_encoder_indices = [original_token_idx_to_source_idx[idx] for idx in mention_start_index]
    instance["link_encoder_indices"] = link_encoder_indices
    instance["link_decoder_indices"] = [x+1 for x in mention_mask_index]  # 1 + for the CLS token


    # input_ids = tokenizer.convert_tokens_to_ids(context_tokens)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mention_mask += padding
    assert len(input_ids) == max_seq_length
    assert len(input_mention_mask) == max_seq_length

    # return {
    #     "tokens": context_tokens,
    #     "ids": input_ids,
    # }
    return instance


def get_candidate_representation(
    candidate_desc, 
    tokenizer, 
    max_seq_length, 
    candidate_title=None,
    title_tag=ENT_TITLE_TAG,
):
    #cls_token = tokenizer.cls_token
    #sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)
        cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    #cand_tokens = [cls_token] + cand_tokens + [sep_token]
    cand_tokens = ["<extra_id_98>"] + cand_tokens
    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    input_mention_mask = [True]*len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


def process_mention_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    mention_key="mention",
    context_key="context",
    label_key="label",
    title_key='label_title',
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    use_world = True

    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )

        label = sample[label_key]
        title = sample.get(title_key, None)
        label_tokens = get_candidate_representation(
            label, tokenizer, max_cand_length, title,
        )
        label_idx = int(sample["label_id"])

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
        }

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    if debug and logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context source : " + " ".join([str(v) for v in sample["context"]["source"]])
            )
            logger.info(
                "Context target : " + " ".join([str(v) for v in sample["context"]["target"]])
            )
            logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
            logger.info(
                "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
            )
            logger.info("Src : %d" % sample["src"][0])
            logger.info("Label_id : %d" % sample["label_idx"][0])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "source"), dtype=torch.long,
    )
    context_vecs_target = torch.tensor(
        select_field(processed_samples, "context", "target"), dtype=torch.long,
    )
    context_vecs_target_labels = torch.tensor(
        select_field(processed_samples, "context", "target_labels"), dtype=torch.long,
    )
    context_vecs_link_decoder_indices = torch.tensor(
        select_field(processed_samples, "context", "link_decoder_indices"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label", "ids"), dtype=torch.long,
    )
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )
    data = {
        "context_vecs": context_vecs,
        "context_vecs_target": context_vecs_target,
        "context_vecs_target_labels": context_vecs_target_labels,
        "context_vecs_link_decoder_indices": context_vecs_link_decoder_indices,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
    }

    if use_world:
        data["src"] = src_vecs
        tensor_data = TensorDataset(context_vecs, context_vecs_target, context_vecs_target_labels,
                                    context_vecs_link_decoder_indices, cand_vecs, src_vecs, label_idx)
    else:
        tensor_data = TensorDataset(context_vecs, context_vecs_target, context_vecs_target_labels,
                                    context_vecs_link_decoder_indices, cand_vecs, label_idx)
    return data, tensor_data
