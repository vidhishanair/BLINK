import numpy as np
import itertools
from blink.cpt.linked_document import HyperlinkedDocument, HyperlinkedWikipediaMMap
from blink.cpt.utils import pad_sequence
from typing import List
from bisect import bisect_left

import torch
import transformers
import json

import tempfile
import os
import shutil


def _random_segmentation(num_items, num_segments, rng=None):
    """Partition a sequence of items randomly into non-empty segments.
    Args:
      num_items: an integer scalar > 0
      num_segments: an integer scalar in [1, num_items]
      rng = a np.random.default_rng() instance or None
    Returns:
      a list with shape [num_segments] containing positive integers that add
      up to num_items
    """
    first_in_segment = np.arange(num_items - 1) < num_segments - 1
    rng = rng or np.random.default_rng()
    rng.shuffle(first_in_segment)
    # The first position always starts a segment.
    # first_in_segment is boolean array for every position after the first that signals whether this location is the start of a new segment.
    segment_id = np.cumsum(first_in_segment)
    segment_length = [0] * num_segments
    segment_length[0] = 1  # first the first missing first in segment
    for k in range(num_items - 1):
        segment_length[segment_id[k]] += 1
    return segment_length


def t5_random_spans_mask(length, noise_density, mean_noise_span_length=3.0, rng=None):
    """Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
      num_noise_tokens = round(length * noise_density)
      num_nonnoise_spans = num_noise_spans = round(
         num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
      length: an int32 scalar (length of the incoming token sequence)
      noise_density: a float - approximate density of output mask
      mean_noise_span_length: a number
      rng = a np.random.default_rng() instance or None
    Returns:
      a boolean list with shape [length]

    adapted from https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py#L2704
    """
    orig_length = length
    # increase length to avoid degeneracy
    length = max(length, 2)

    # compute number of noise tokens and noise spans
    num_noise_tokens = int(length * noise_density)
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(num_noise_tokens / mean_noise_span_length)
    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans, rng=rng)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans, rng=rng)

    return list(
        itertools.chain.from_iterable(
            [[False] * nonnoise_span_lengths[k] + [True] * noise_span_lengths[k] for k in range(num_noise_spans)]
        )
    )[:orig_length]


def _sample_non_consecutive_links(doc: HyperlinkedDocument, num_links_to_sample: int, rng=None) -> List[int]:
    """
    Given a hyperlinked document, randomly sample a number of non-consective links.
    Returns a list of indices that index doc.link_* with the sampled links.
    """
    rng = rng or np.random.default_rng()
    indices = np.arange(len(doc.link_start_index))
    rng.shuffle(indices)

    kept_indices = []
    kept_start_indices = set()
    kept_end_indices = set()

    for k in range(len(doc.link_start_index)):
        start_index = doc.link_start_index[indices[k]]
        end_index = doc.link_end_index[indices[k]]
        # anchor text is non-overlapping, but could border another masked span
        # One span will border another if:
        #   - the span start index is the same as an existing end index
        #   - the span end index is the same as existing start index
        if start_index not in kept_end_indices and end_index not in kept_start_indices:
            kept_indices.append(indices[k])
            if len(kept_indices) == num_links_to_sample:
                break
            kept_start_indices.add(start_index)
            kept_end_indices.add(end_index)

    return kept_indices


def get_span_start_indices(span_mask: List[bool]):
    return list(range(int(span_mask[0]))) + [k for k in range(1, len(span_mask)) if span_mask[k] and not span_mask[k - 1]]


def count_masked_spans(span_mask: List[bool]):
    # sum number of start locations
    return len(get_span_start_indices(span_mask))


def adjust_span_mask_with_anchor_text(
    span_mask: List[bool],
    document: HyperlinkedDocument,
    num_links_to_include_in_mask: int,
    preserve_number_mask_spans: bool = False,
    rng=None,
):
    """
    Adjust a span mask to include some masked spans that exactly match anchor text.

    span_mask = [num_tokens] length boolean list with True indicating the masked locations.
    document = a HyperlinkedDocument with the link information.
    num_links_to_include_in_mask = the number of links to include in the mask (if possible, otherwise include all links in the document)
    if preserve_number_mask_spans is True, then the number of masked spans is held constant, otherwise additional spans are added.

    Returns:
        updated_span_mask, indices_of_links_in_span_mask

    After return we are guaranteed that

    all([all(span_mask[document.link_start_index[k]:document.link_end_index[k]]) for k in indices_of_links_in_span_mask])  (all the anchor text are included in masks)

    and

    not any[any([span_mask[document.link_start_index[k]-1], span_mask[document.link_end_index[k]]]) for k in indices_of_links_in_span_mask]   (the masked span matches exactly the anchor text without running into other masks)
    """
    if preserve_number_mask_spans:
        num_original_masked_spans = count_masked_spans(span_mask)

    indices = _sample_non_consecutive_links(document, num_links_to_include_in_mask, rng=rng)

    # Now update the mask.
    updated_span_mask = list(span_mask)
    for k in indices:
        anchor_text_start_index = document.link_start_index[k]
        anchor_text_end_index = document.link_end_index[k]
        for i in range(anchor_text_start_index, anchor_text_end_index):
            updated_span_mask[i] = True
        if anchor_text_start_index > 0:
            updated_span_mask[anchor_text_start_index - 1] = False
        if anchor_text_end_index < len(span_mask):
            updated_span_mask[anchor_text_end_index] = False

    if preserve_number_mask_spans:
        span_start_indices = get_span_start_indices(updated_span_mask)
        num_final_masked_spans = len(span_start_indices)
        if num_final_masked_spans > num_original_masked_spans:
            # randomly remove num_final_masked_spans - num_original_masked_spans
            num_spans_to_remove = num_final_masked_spans - num_original_masked_spans
            # (1) first remove the anchor text start indices from the list
            # (2) then randomly choose some anchor text to remove
            # (3) finally remove it from the mask
            candidate_span_start_indices_to_remove = set(span_start_indices)
            for k in indices:
                candidate_span_start_indices_to_remove.remove(document.link_start_index[k])
            final_candidates = list(candidate_span_start_indices_to_remove)
            _rng = rng or np.random.default_rng()
            if num_spans_to_remove >= len(final_candidates):
                start_indices_to_remove = final_candidates
            else:
                start_indices_to_remove = _rng.choice(final_candidates, size=num_spans_to_remove, replace=False)
            N = len(span_mask)
            for index in start_indices_to_remove:
                i = index
                while i < N and updated_span_mask[i]:
                    updated_span_mask[i] = False
                    i += 1

    return updated_span_mask, indices


def split_token_ids_with_span_mask_into_source_target(
    token_ids: List[int], span_mask: List[bool], sentinel_token_ids: List[int]
):
    """
    Given a list of token ids and a boolean span mask (True == mask, False == don't mask),
    create the source/target lists of token ids by replacing masked spans with the sentinel tokens.

    Example:
        token_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        span_mask = [F, F, T, T, F, F, F, T, T, T]
        sentinel_token_ids = [30000, 29999, 29998, 29997]

    Returns:
        source = [0, 1, 30000, 4, 5, 6, 29999]
        target = [30000, 2, 3, 29999, 7, 8, 9, 29998]
        num_masked_spans = 2
        mask_start_indices = [2, 7]
    """
    num_masked_spans = 0
    in_span = False
    source = []
    target = []
    mask_start_indices = []

    # for each token in the original document what is the start index in source
    # because of masking (could be multiple tokens), the mapping between source and original document tokens is not 1 to 1
    original_token_idx_to_source_idx = []

    for k, (token_id, is_mask) in enumerate(zip(token_ids, span_mask)):
        if not is_mask:
            source.append(token_id)
            in_span = False
        else:
            # if we are starting the span, add the sentinel to the source and the target
            if not in_span:
                in_span = True
                source.append(sentinel_token_ids[num_masked_spans])
                target.append(sentinel_token_ids[num_masked_spans])
                num_masked_spans += 1
                mask_start_indices.append(k)
            target.append(token_id)
        original_token_idx_to_source_idx.append(len(source) - 1)

    # add the final sentinel to the target
    if num_masked_spans > 0:
        target.append(sentinel_token_ids[num_masked_spans])

    return {
        "source": source,
        "target": target,
        "num_masked_spans": num_masked_spans,
        "mask_start_indices": mask_start_indices,
        "original_token_idx_to_source_idx": original_token_idx_to_source_idx,
    }


class HyperlinkedDocumentCorpusIndex:
    """
    An index to retrieve chunks of a collection of HyperlinkedDocument.

    Each document D_k has a variable length number of tokens t_k.
    The documents are ordered with some order of indices (e.g. by random shuffle for a training epoch), making a corpus of tokens with total size sum_k t_k.
    This class provides an interface logically treat the entire corpus as a single list for slicing.
    The main API is __getitem__ which only supports slices. For any slice, it returns a synthetic HyperlinkedDocument that corresponds to the slice's chuck of the full dataset.

    Intended use:
        - create index
        - shuffle once per epoch
        - iterate over corpus

    In addition to holding an entire corpus, can also provide information about training splits.
    In this case, we split on the source document.  If provided, split_file contains the doc ids in the
        underlying mmap for validation and test sets (everything else is in train).
    """

    def __init__(self, hyperlinked_mmap, shuffle=True, rng=None, split_file=None, split=None):
        # Create an array of index in mmap --> document lengths
        self._mmap = hyperlinked_mmap

        # The current ordering is given by a list of doc ids in the underlying mmap.
        # To enable restricting to a subset of documents for a particular split, we'll just restrict
        # the document order to only contain those documents.
        if split is not None:
            assert split in ("train", "validation", "test")
            with open(split_file, "r") as fin:
                split_info = json.load(fin)

            all_document_ids = np.arange(len(self._mmap))
            if split in ("validation", "test"):
                ids_to_include = set(split_info[split])
                self._document_order = np.array([i for i in range(len(self._mmap)) if i in ids_to_include])
            else:
                ids_to_exclude = set(split_info["validation"])
                ids_to_exclude.update(split_info["test"])
                self._document_order = np.array([i for i in range(len(self._mmap)) if i not in ids_to_exclude])
        else:
            self._document_order = np.arange(len(self._mmap))

        # We store the document lengths for all docs in the underlying mmap, and create an index
        # of the end lengths for just those included in the split when building the index.
        self._document_lengths = self._mmap.get_document_lengths()

        if shuffle:
            self.shuffle(rng)
        else:
            # just build the index without shuffling
            self._build_index()

    def __len__(self):
        return self._ordered_document_end_indices[-1]

    def _build_index(self):
        # For slicing, we need a list of the document end indices in the full dataset, and we'll use bisect_left
        ordered_document_lengths = self._document_lengths[self._document_order].astype(np.int64)
        self._ordered_document_end_indices = np.cumsum(ordered_document_lengths)

    def shuffle(self, rng=None):
        """
        Shuffle the document order and rebuild the index.  Call once per training epoch.
        """
        rng = rng or np.random.default_rng()
        rng.shuffle(self._document_order)
        self._build_index()

    def __getitem__(self, indices: slice) -> HyperlinkedDocument:
        assert isinstance(indices, slice) and indices.step is None and indices.stop > indices.start
        # Find the location of the start/end indices with bisect
        doc_start_index = bisect_left(self._ordered_document_end_indices, indices.start)
        doc_end_index = bisect_left(self._ordered_document_end_indices, indices.stop)

        # Read those documents from the underlying data store
        docs = [self._mmap[self._document_order[k]] for k in range(doc_start_index, doc_end_index + 1)]

        # Make the new hyperlinked document
        # We need to concatenate all of the docs, but need to adjust the start/end indices
        # for the first/last doc, and adjust the start/end indices of the anchor text.
        num_docs = len(docs)
        token_ids = []
        link_start = []
        link_end = []
        link_target = []
        for k, doc in enumerate(docs):
            if k == 0:
                if doc_start_index == 0:
                    first_doc_index = 0
                else:
                    first_doc_index = self._ordered_document_end_indices[
                        doc_start_index - 1
                    ]  # the index in the full corpus where this document starts
                start_index = indices.start - first_doc_index
            else:
                start_index = 0

            if k == num_docs - 1:
                if doc_end_index == 0:
                    last_doc_index = 0
                else:
                    last_doc_index = self._ordered_document_end_indices[doc_end_index - 1]
                end_index = indices.stop - last_doc_index
            else:
                end_index = len(doc.token_ids)

            # We need to adjust the link start/end indices from the subdocument to the full sequence
            offset = len(token_ids) - start_index
            token_ids.extend(doc.token_ids[start_index:end_index])
            for lstart, lend, ltarget in zip(doc.link_start_index, doc.link_end_index, doc.link_target):
                # only keep the link if it is fully in the truncated doc
                if start_index <= lstart and end_index >= lend:
                    link_start.append(lstart + offset)
                    link_end.append(lend + offset)
                    link_target.append(ltarget)

        return HyperlinkedDocument(token_ids, link_start, link_end, link_target)


class T5ContrastivePretrainingDataset(torch.utils.data.Dataset):
    """
    Dataset for contrastive pretraining with link annotations that combines
        - T5 masked span infilling
        - contrastive loss to match contextual representation of anchor text with page text
    """

    def __init__(
        self,
        mmap_prefix,
        tokenizer_name,
        split=None,
        chunk_size=512,
        seed=42,
        mask_percent=0.15,
        num_links_to_include_positive_samples=4,
        num_links_to_sample_for_mask=None,
        link_target_truncation_method="first",
        link_target_sequence_length=128,
        add_eos_token_link_target=False,
        add_eos_token_source=False,  # True -> add </s> to the source before masking
        span_infilling_only=False,  # True -> reduce to just T5 span infilling
        preserve_number_mask_spans=False,
        encoder_side_contrastive_loss=False,
        cls_at_encoder_side=False,  # True -> cls is at beginning of encoder, decoder doesn't have cls
        lm_loss_on_cls_token=True,
    ):
        """
        num_links_to_include_positive_samples: number of links to sample for the contrastive loss
        num_links_to_sample_for_mask: number of anchor text/links that we will sample for the T5 span infilling mask.  If this value is > num_links_to_include_positive_samples then a random sample of them will be chosen for the contrastive loss.  Default (None) will sample 2 * num_links_to_include_positive_samples anchor text spans.

        """
        self._mmap = HyperlinkedWikipediaMMap(mmap_prefix)
        # build the index
        self._rng = np.random.default_rng(seed)
        if split is not None:
            split_file = mmap_prefix + "_splits.json"
            shuffle = split == "train"
            self._index = HyperlinkedDocumentCorpusIndex(
                self._mmap,
                rng=self._rng,
                shuffle=shuffle,
                split_file=split_file,
                split=split,
            )
        else:
            self._index = HyperlinkedDocumentCorpusIndex(self._mmap, rng=self._rng)
        self._add_eos_token_source = add_eos_token_source
        self._chunk_size = chunk_size
        if self._add_eos_token_source:
            self._adjusted_chunk_size = chunk_size - 1
        else:
            self._adjusted_chunk_size = chunk_size
        num_tokens_full_data = len(self._index)
        self._num_chunks = num_tokens_full_data // self._adjusted_chunk_size

        self._mask_percent = mask_percent
        self._num_links_to_include_positive_samples = num_links_to_include_positive_samples
        if num_links_to_sample_for_mask is None:
            self._num_links_to_sample_for_mask = 2 * self._num_links_to_include_positive_samples
        else:
            # We need to sample at least as many as we will chose for the contrastive loss.
            assert num_links_to_sample_for_mask >= self._num_links_to_include_positive_samples
            self._num_links_to_sample_for_mask = num_links_to_sample_for_mask
        self._span_infilling_only = span_infilling_only
        self._preserve_number_mask_spans = preserve_number_mask_spans

        # We need additional token ids for the decoder <cls> and <target_*>.
        # T5 has 100 extra token ids used to mark each span.
        # If we impose a hard limit of 49 possible spans per sequence, then we can use
        # the first 49 as the existing sentinel tokens, the next 49 for the <target_*> and
        # one for <cls> (leaving 1 unused).  With 0.15 masking percent and 3.0 average
        # span length, this limits sequence length to about 980. If we use longer sequences
        # we'll need to add extra tokens / embeddings to the model.
        #
        # In transformers "<extra_id_k>" for k in range(100) are the special tokens.
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer = tokenizer

        def _tid(i):
            return tokenizer.encode(f"<extra_id_{i}>", add_special_tokens=False)[0]

        self._sentinel_token_ids = [
            _tid(k) for k in range(49)
        ]  # used the same as existing T5 sentinel tokens in encoder/decoder
        self._target_token_ids = [_tid(k) for k in range(49, 98)]  # for contrast loss
        self._cls_token_id = _tid(98)  # for contrast loss / document embedding

        assert link_target_truncation_method in ("first")
        self._link_target_truncation_method = link_target_truncation_method
        self._link_target_sequence_length = link_target_sequence_length

        self._pad_token_id = tokenizer.pad_token_id
        self._eos_token_id = tokenizer.eos_token_id
        self._add_eos_token_link_target = add_eos_token_link_target  # adds eos after each input doc
        self._encoder_side_contrastive_loss = encoder_side_contrastive_loss
        self._cls_at_encoder_side = cls_at_encoder_side
        self.cls_token_id = self._cls_token_id
        self.lm_loss_on_cls_token = lm_loss_on_cls_token

    def truncate_document(self, document):
        # truncate the document according to the truncate method - used for link targets
        if self._link_target_truncation_method == "first":
            if self._add_eos_token_link_target:
                tids = document.token_ids[: self._link_target_sequence_length - 1]
                tids.append(self._eos_token_id)
            else:
                tids = document.token_ids[: self._link_target_sequence_length]
            return tids
        else:
            NotImplementedError

    def __len__(self):
        return self._num_chunks

    @staticmethod
    def collate_fn(instances, pad_token_id, num_links_to_include_positive_samples):
        batch = {}
        for key in ["source", "target"]:
            batch[key] = torch.nn.utils.rnn.pad_sequence(
                [instance[key] for instance in instances],
                batch_first=True,
                padding_value=pad_token_id,
            )
        # NOTE: -100 will be ignored by cross-entropy loss for lm objective
        batch["target_labels"] = torch.nn.utils.rnn.pad_sequence(
            [instance["target_labels"] for instance in instances],
            batch_first=True,
            padding_value=-100,
        )

        if "prefix_length" not in instances[0]:
            # Just running T5 span masking, early exit
            return batch

        batch["prefix_length"] = torch.cat(
            [instance["prefix_length"].unsqueeze(0) for instance in instances], dim=0
        )  # (batch_size, )

        # link_target_token_ids is shape (num_links, seq_len) for each instance. need to pad
        # both dimensions.
        max_target_id_len = max([instance["link_target_token_ids"].shape[1] for instance in instances])
        link_target_token_ids_len_padded = []
        for instance in instances:
            i_target_token_ids = instance["link_target_token_ids"]
            i_pad = torch.empty(
                (i_target_token_ids.shape[0], max_target_id_len),
                dtype=i_target_token_ids.dtype,
            )
            i_pad.fill_(pad_token_id)
            i_pad[:, : i_target_token_ids.shape[1]] = i_target_token_ids
            link_target_token_ids_len_padded.append(i_pad)

        batch["link_target_token_ids"] = torch.nn.utils.rnn.pad_sequence(
            link_target_token_ids_len_padded,
            batch_first=True,
            padding_value=pad_token_id,
        )

        # For link_decoder_indices we pad with -999 to signal this index doesn't exist
        for key in ["link_decoder_indices", "link_encoder_indices"]:
            batch[key] = pad_sequence(
                [instance[key] for instance in instances],
                batch_first=True,
                padding_value=-999,
                max_len=num_links_to_include_positive_samples,
            )
        return batch

    def __getitem__(self, instance_index: int):
        """
        {"source": (input_seq_len, ), "target": (output_seq_len, ),
         "prefix_length": int   # length of the prefix for <cls> <target_*> tokens in target
         "link_decoder_indices" (num_links, )  # the indices in target corresponding to the <target_*> tokens sampled for contrastive loss
         "link_target_token_ids": (num_links, target_seq_len)  # the token ids of the link targets
        """
        # get the start/end indices and read the slice from the index
        start_index = instance_index * self._adjusted_chunk_size  # int
        end_index = start_index + self._adjusted_chunk_size  # int
        doc = self._index[start_index:end_index]  # hyperlinked doc instance
        if self._add_eos_token_source:
            doc.token_ids.append(self._eos_token_id)
        # sample the T5 mask
        initial_mask = t5_random_spans_mask(
            self._chunk_size,
            self._mask_percent,
            mean_noise_span_length=3.0,
            rng=self._rng,
        )  # List[bool] with doc length

        # Adjust the T5 mask to include some positively sampled anchor texts.
        # We sample up to 2x the number of links to include so that the masks are biased to include entire entity mentions,
        # but then just select self._num_links_to_include_positive_samples to include in the final data
        # final span mask: List[bool], selected_link_indices = List[int] with indices of all the links that were adjusted
        # positive_sampled_link_indices = the first self._num_links_to_include_positive_samples links in selected_link_indices
        final_span_mask, selected_link_indices = adjust_span_mask_with_anchor_text(
            initial_mask,
            doc,
            self._num_links_to_sample_for_mask,
            preserve_number_mask_spans=self._preserve_number_mask_spans,
            rng=self._rng,
        )
        positive_sampled_link_indices = selected_link_indices[: self._num_links_to_include_positive_samples]

        # Make the source and initial decoder tokens.
        source_target = split_token_ids_with_span_mask_into_source_target(
            doc.token_ids, final_span_mask, self._sentinel_token_ids
        )

        if self._span_infilling_only:
            # Just return the source and target from T5
            instance = {
                "source": torch.tensor(source_target["source"]),
                "target": torch.tensor(source_target["target"]),
                "target_labels": torch.tensor(source_target["target"]),
            }
        else:
            # construct the actual target by pre-pending the <CLS> <target_*> tokens
            if self._encoder_side_contrastive_loss:
                prefix_ids = [self._cls_token_id] if not self._cls_at_encoder_side else []
            else:
                prefix_ids = [self._cls_token_id] + [
                    self._target_token_ids[k] for k in range(source_target["num_masked_spans"])
                ]
            if self._cls_at_encoder_side:
                source_target["source"] = [self._cls_token_id] + source_target["source"]
                # add one to mappings of original tokens to the source tokens (because we now have one additional CLS at the start)
                source_target["original_token_idx_to_source_idx"] = [
                    el + 1 for el in source_target["original_token_idx_to_source_idx"]
                ]

            # ignore the prefix tokens for lm_loss except optionally for the first cls token
            masked_prefix_ids = [
                -100 if ((not self.lm_loss_on_cls_token) or idx != 0) else el for idx, el in enumerate(prefix_ids)
            ]
            instance = {
                "source": torch.tensor(source_target["source"]),
                "target": torch.tensor(prefix_ids + source_target["target"]),
                "prefix_length": torch.tensor(len(prefix_ids)),
                "target_labels": torch.tensor(masked_prefix_ids + source_target["target"]),
                # TODO: Are we sure we should mask out prefix ids?
            }

            # For the contrast loss, we need:
            #   - indices in "target" where the contrast loss is attached which is the index in the masked spans corresponding to the links
            #   - the gold token ids for the link targets for each of these locations

            # (a) get the index in the masked spans
            sampled_link_start_indices = [
                doc.link_start_index[sampled_index] for sampled_index in positive_sampled_link_indices
            ]
            sampled_link_mask_index = [
                source_target["mask_start_indices"].index(mask_start_index)
                for mask_start_index in sampled_link_start_indices
            ]

            original_token_idx_to_source_idx = source_target["original_token_idx_to_source_idx"]
            link_encoder_indices = [original_token_idx_to_source_idx[idx] for idx in sampled_link_start_indices]
            instance["link_encoder_indices"] = torch.tensor(link_encoder_indices)

            instance["link_decoder_indices"] = 1 + torch.tensor(sampled_link_mask_index)  # 1 + for the CLS token

            # (b) get the token ids for each of the link targets
            sampled_link_target_ids = [doc.link_target[sampled_index] for sampled_index in positive_sampled_link_indices]

            # For each positively sampled anchor text, get the link target and retrive the raw truncated document from the original self._mmap.
            link_target_docs = [self._mmap[target_id] for target_id in sampled_link_target_ids]
            link_target_token_ids = [self.truncate_document(d) for d in link_target_docs]

            # pad
            instance["link_target_token_ids"] = pad_sequence(
                [torch.tensor(t) for t in link_target_token_ids],
                batch_first=True,
                padding_value=self._pad_token_id,
                batch_size=self._num_links_to_include_positive_samples,
            )

        return instance

    def inspect_dataset_instance(self, instance_index: int, tokenizer):
        """
        Get the instance and then decode it to string
        """
        # a map from decoder <target_id_*> to encoder <extra_id_*>
        decoder_id_to_encoder_id = {tid: f"<extra_id_{k}>" for k, tid in enumerate(self._target_token_ids)}

        instance = self[instance_index]

        print("Source: ")
        print(tokenizer.decode(instance["source"]))
        print("\nTarget: ")
        print(tokenizer.decode(instance["target"]))

        # each extra ID and the target page
        decoder_link_targets = [
            decoder_id_to_encoder_id[i.item()] for i in instance["target"][instance["link_decoder_indices"]]
        ]
        for i, tid in enumerate(decoder_link_targets):
            print("\n", tid, tokenizer.decode(instance["link_target_token_ids"][i]))


def make_wikipedia_t5_splits(mmap_prefix, num_docs_dev=10000, num_docs_test=10000):
    # randomly sample some documents for dev/test
    mmap_prefix = "/net/nfs2.corp/allennlp/matthewp/cpt/wikipedia_t5"
    dataset = T5ContrastivePretrainingDataset(mmap_prefix, "t5-base")

    doc_ids = np.arange(len(dataset._mmap))
    rng = np.random.default_rng(seed=1234567890)
    rng.shuffle(doc_ids)
    splits = {
        "validation": [int(i) for i in doc_ids[:num_docs_dev]],
        "test": [int(i) for i in doc_ids[num_docs_dev:num_docs_test]],
    }

    split_file = mmap_prefix + "_splits.json"
    with open(split_file, "w") as fout:
        fout.write(json.dumps(splits))


def make_testing_mmap(output_prefix):
    # create a synthetic mmap for testing
    rng = np.random.default_rng(seed=42)
    num_docs = 10
    doc_lengths = rng.integers(3, 8, 10)

    # We'll construct the token ids and links so that they are easy to check after shuffling with a fixed seed
    rng = np.random.default_rng(seed=42)
    shuffled_indices = np.arange(num_docs)
    rng.shuffle(shuffled_indices)

    token_ids = [None] * num_docs
    total_len = 0
    for shuffled_index in range(num_docs):
        original_index = shuffled_indices[shuffled_index]
        doc_len = doc_lengths[original_index]
        token_ids[original_index] = np.arange(total_len, total_len + doc_len).tolist()
        total_len += doc_len

    # put links every 5 tokens in the shuffled corpus
    link_start_indices = []
    link_end_indices = []
    link_targets = []

    for doc_tokens in token_ids:
        start = []
        end = []
        targets = []
        for k, tid in enumerate(doc_tokens):
            if tid % 5 == 0:
                start.append(k)
                end.append(min(k + 3, len(doc_tokens)))
                targets.append(tid // 5)
        link_start_indices.append(start)
        link_end_indices.append(end)
        link_targets.append(targets)

    # make the mmap
    lengths = []
    data = []

    dd = []
    for k in range(num_docs):
        doc = HyperlinkedDocument(token_ids[k], link_start_indices[k], link_end_indices[k], link_targets[k])
        doc_lengths, doc_data = doc.serialize()
        doc_start_index = len(data)
        lengths.append(doc_start_index)
        lengths.extend(doc_lengths)
        data.extend(doc_data)

    # write out to mmap
    fp = np.memmap(output_prefix + "_lengths.npy", mode="w+", dtype="uint32", shape=(len(lengths),))
    fp[:] = np.array(lengths)[:]
    fp.flush()

    fp = np.memmap(output_prefix + "_data.npy", mode="w+", dtype="uint32", shape=(len(data),))
    fp[:] = np.array(data)[:]
    fp.flush()

    # write out a split file
    splits = {"validation": [0, 7], "test": [6, 9]}
    with open(output_prefix + "_splits.json", "w") as fout:
        fout.write(json.dumps(splits))


class MmapForTesting:
    """
    with MmapForTesting() as mmap_prefix:
        # mmap_prefix is a string pointing to temporary location
        dataset = Dataset(mmap_prefix ...)

    # on exit, the temporary files are cleaned up and mmap_prefix is no longer a valid
    # location to load dataset. Any references to it in the dataset object are broken.
    """

    def __init__(self):
        pass

    def __enter__(self):
        self.tmp_dir = tempfile.mkdtemp()
        mmap_prefix = os.path.join(self.tmp_dir, "testing")
        make_testing_mmap(mmap_prefix)
        return mmap_prefix

    def __exit__(self, exc_type, exc_value, exc_traceback):
        shutil.rmtree(self.tmp_dir)


def _t():
    mmap_prefix = "/net/nfs2.corp/allennlp/matthewp/cpt/wikipedia_t5"
    dataset = T5ContrastivePretrainingDataset(
        mmap_prefix,
        "t5-base",
        split="validation",
        num_links_to_include_positive_samples=4,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained("t5-base")
    dataset.inspect_dataset_instance(1234, tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    for batch in loader:
        break
