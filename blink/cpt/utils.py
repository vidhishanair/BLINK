import torch


def pad_sequence(sequences, batch_first=False, padding_value=0.0, batch_size=None, max_len=None):
    # type: (List[Tensor], bool, float) -> Tensor
    r"""extention of torch.nn.utils.rnn.pad_sequence to deal with empty sequence

    batch_size: if sequences was empty, then return a batch with batch_size filled with padding values
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    if sequences is None or len(sequences) == 0:
        sequences = [torch.tensor([padding_value]) for _ in range(batch_size)]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_sequence_len = max([s.size(0) for s in sequences])
    if max_len is None or max_len < max_sequence_len:
        max_len = max_sequence_len
    if batch_size is not None:
        while len(sequences) < batch_size:
            # add another tensor with paddings to the sequence to make sure batch size is met
            sequences.append(sequences[0].new_full(size=max_size, fill_value=padding_value))
    # the rest is just copied from torch.nn.utils.rnn.pad_sequence
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def shift_right(input_ids, pad_token_id, decoder_start_token_id):
    """adopted from https://github.com/huggingface/transformers/blob/a26f4d620874b32d898a5b712006a4c856d07de1/src/transformers/models/t5/modeling_t5.py#L771"""
    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id
    return shifted_input_ids
