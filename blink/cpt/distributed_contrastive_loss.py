# Large batch distributed contrastive loss with in batch negative sampling.
# Given two tensors:
#   x = (batch_size, embed_dim)
#   y = (batch_size, embed_dim)
# For each k, we compute <x_k, y_j> for all j, take the softmax over j, and use index k as the "correct" class with cross entropy loss.
#
# This file implements this loss where the x and y are chunked across devices in the batch_size dimension, and handles
# passing tensors and gradients between devices.

import torch
import numpy as np

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.debug.metrics as met
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


def all_reduce(tensor, mean=True):
    # in place all_reduce. If mean is True then divide by world size
    if XLA_AVAILABLE:
        tensor = xm.all_reduce(xm.REDUCE_SUM, tensor)
        if mean:
            tensor /= xm.xrt_world_size()
    else:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        if mean:
            tensor /= torch.distributed.get_world_size()
    return tensor


def get_rank():
    """returns the global rank, and -1 if not distributed"""
    if XLA_AVAILABLE:
        global_rank = xm.get_ordinal()
    else:
        try:
            global_rank = torch.distributed.get_rank()
        except:
            # not distributed
            global_rank = -1

    return global_rank


def get_world_size():
    if XLA_AVAILABLE:
        return xm.xrt_world_size()
    else:
        return torch.distributed.get_world_size()


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if XLA_AVAILABLE:
        return _concat_all_gather_xla(tensor)
    else:
        return _concat_all_gather_ddp(tensor)


def _concat_all_gather_xla(tensor, dim=0):
    return xm.all_gather(tensor, dim=dim)


def _concat_all_gather_ddp(tensor, dim=0):
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=dim)
    return output


def barrier(tag):
    if XLA_AVAILABLE:
        xm.rendezvous(tag)
    else:
        torch.distributed.barrier()


class AllGatherAutoGrad(torch.autograd.Function):
    """
    Runs all gather with input tensor and concats across dim,
    while handling automatic differentiation.
    """

    @staticmethod
    def forward(ctx, tensor, dim):
        ctx.dim = dim
        ctx.rank = get_rank()
        ctx.world_size = get_world_size()
        # concat_all_gather has @torch.no_grad() so call the lower level
        # functions here
        if XLA_AVAILABLE:
            return _concat_all_gather_xla(tensor, dim=dim)
        else:
            return _concat_all_gather_ddp(tensor, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        slice_size = grad_output.size(ctx.dim) // ctx.world_size
        return (
            torch.narrow(grad_output.clone(), ctx.dim, ctx.rank * slice_size, slice_size),
            None,
        )


class ScaleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, scale):
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def negative_cosine_similarity_loss_with_stop_gradient(p, z):
    # p, z = (batch_size, dim), returns (batch_size with cosine similarity
    # from https://arxiv.org/pdf/2011.10566.pdf, Algorithm 1
    p_normalized = torch.nn.functional.normalize(p, dim=1)  # l2-normalize
    z_normalized = torch.nn.functional.normalize(z.detach(), dim=1)  # l2-normalize
    return -(p_normalized * z_normalized).sum(dim=1).mean()


class SimSiam(torch.nn.Module):
    """
    Method from "Exploring Simple Siamese Representation Learning", Chen and He, 2020,
    https://arxiv.org/pdf/2011.10566.pdf
    """

    def __init__(self, prediction_mlp, distributed=False):
        """
        Prediction MLP should take the input and use 2-3 hidden layers.
        The final layer should just be the output of a linear layer without an activation, layer norm,
        or l2 normalization.
        """
        super().__init__()
        self.prediction_mlp = prediction_mlp
        self.distributed = distributed

    def forward(self, x, y, ignore_mask=None, compute_accuracy=True):
        # x = (batch_size, embed_dim)
        # y = (batch_size, embed_dim)
        # ignore_mask = (batch_size, )

        # check the shapes on ignore_mask
        if ignore_mask is not None:
            assert len(ignore_mask.shape) == 1 and ignore_mask.shape[0] == x.shape[0]

        # Compute the loss with the local batch as it just uses positive samples only.
        if ignore_mask is not None:
            x_mask = x[~ignore_mask]
            y_mask = y[~ignore_mask]
        else:
            x_mask = x
            y_mask = y

        p_x = self.prediction_mlp(x_mask)
        p_y = self.prediction_mlp(y_mask)

        # get loss
        ret = {
            "loss": 0.5
            * (
                negative_cosine_similarity_loss_with_stop_gradient(p_x, y_mask)
                + negative_cosine_similarity_loss_with_stop_gradient(p_y, x_mask)
            )
        }

        # Now compute the accuracy.
        if compute_accuracy:
            p_x_normalized = torch.nn.functional.normalize(self.prediction_mlp(x), dim=-1)
            p_y_normalized = torch.nn.functional.normalize(self.prediction_mlp(y), dim=-1)
            x_normalized = torch.nn.functional.normalize(x, dim=-1)
            y_normalized = torch.nn.functional.normalize(y, dim=-1)

            if self.distributed:
                x_all = concat_all_gather(x_normalized)
                y_all = concat_all_gather(y_normalized)
                p_x_all = concat_all_gather(p_x_normalized)
                p_y_all = concat_all_gather(p_y_normalized)
                ignore_mask_all = concat_all_gather(ignore_mask) if ignore_mask is not None else None
            else:
                x_all = x_normalized
                y_all = y_normalized
                p_x_all = p_x_normalized
                p_y_all = p_y_normalized
                ignore_mask_all = ignore_mask

            # Apply the mask.
            if ignore_mask_all is not None:
                x_all = x_all[~ignore_mask_all]
                y_all = y_all[~ignore_mask_all]
                p_x_all = p_x_all[~ignore_mask_all]
                p_y_all = p_y_all[~ignore_mask_all]

            # Now compute pairwise similarity between all x and y.
            similarities_x = (p_x_all.unsqueeze(2) * y_all.transpose(0, 1).unsqueeze(0)).sum(
                dim=1
            )  # (all_batch_size, all_batch_size)
            similarities_y = (p_y_all.unsqueeze(2) * x_all.transpose(0, 1).unsqueeze(0)).sum(
                dim=1
            )  # (all_batch_size, all_batch_size)

            batch_size = x_all.shape[0]
            gold_labels = torch.arange(batch_size, device=x_all.device)

            # compute accuracy
            accuracy_x = (
                0.5 * (similarities_x.argmax(dim=0) == gold_labels).float().mean()
                + (similarities_x.argmax(dim=1) == gold_labels).float().mean()
            ).detach()
            accuracy_y = (
                0.5 * (similarities_y.argmax(dim=0) == gold_labels).float().mean()
                + (similarities_y.argmax(dim=1) == gold_labels).float().mean()
            ).detach()

            ret["accuracy_x"] = accuracy_x
            ret["accuracy_y"] = accuracy_y

        return ret


class BarlowTwins(torch.nn.Module):
    """
    See https://arxiv.org/pdf/2103.03230.pdf
    Barlow Twins: Self-Supervised Learning via Redundancy Reduction, Zbontar et al 2021
    """

    def __init__(self, lam=5e-3):
        super().__init__()
        self._lam = lam
        self._eps = 1e-5  # for dividing by std

    def forward(self, x, y):
        # x, y = (batch_size, embedding_size)
        # see algorithm 1

        x_norm = (x - x.mean(dim=0)) / (x.std(dim=0) + self._eps)
        y_norm = (y - y.mean(dim=0)) / (y.std(dim=0) + self._eps)

        # cross correlation matrix
        batch_size = x.shape[0]
        c = torch.matmul(x_norm.transpose(0, 1), y_norm) / float(batch_size)

        # the loss
        # loss = sum_diagonal((c_ii - 1.0)**2) + sum_offdiagonal(c_ij**2)
        diag_mask = torch.eye(c.shape[0], device=c.device, dtype=c.dtype)  # 1.0 on diagonal
        diag_loss = (torch.diag(c) - 1.0).pow(2).sum()
        off_diag_loss = (c * (1.0 - diag_mask)).pow(2).sum() * self._lam
        loss = diag_loss + off_diag_loss

        # compute the accuracy
        logits = torch.matmul(x_norm, y_norm.transpose(0, 1))  # (batch_size, batch_size)

        # the gold label is just the index. compute it for both x --> y and y --> x
        batch_size = x_norm.shape[0]
        gold_labels = torch.arange(batch_size, device=x_norm.device)

        accuracy_x = (logits.argmax(dim=1) == gold_labels).float().mean().detach()
        accuracy_y = (logits.argmax(dim=0) == gold_labels).float().mean().detach()

        return {
            "loss": loss / batch_size,
            "accuracy_x": accuracy_x,
            "accuracy_y": accuracy_y,
        }


class ContrastiveLoss(torch.nn.Module):
    """
    method == 'cross_entropy': compute the loss using cross entropy of positive sample vs. negative samples
        using in-batch negative sampling across all devices.
    method == 'alignment_uniformity': use the method "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere", Wang and Isola, 2020.
    method == 'simsiam': use method from "Exploring Simple Siamese Representation Learning", Chen and He, 2020.
    method == 'barlow_twins': use method from "Barlow Twins: Self-Supervised Learning via Redundancy Reduction", Zbontar et al. 2021
    """

    def __init__(
        self,
        tau,
        distributed=False,
        method="cross_entropy",
        prediction_mlp=None,
        barlow_lambda=5e-3,
    ):
        # if distributed is True, assumes running in distributed data parallel (GPU) or across multiple TPUs
        super().__init__()
        self.tau = tau
        self.distributed = distributed
        self.world_size = None

        assert method in (
            "cross_entropy",
            "alignment_uniformity",
            "simsiam",
            "barlow_twins",
        )
        self.method = method
        if self.method == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
        elif self.method == "alignment_uniformity":
            self.alignment_uniformity_alpha = 2  # TODO: maybe allow these to vary
            self.alignment_uniformity_t = 2
        elif self.method == "simsiam":
            self.simsiam = SimSiam(prediction_mlp, distributed=distributed)
        elif self.method == "barlow_twins":
            self.barlow_twins = BarlowTwins(lam=barlow_lambda)

    def forward(self, x, y, ignore_mask=None):
        # x = (batch_size, embed_dim)
        # y = (batch_size, embed_dim)
        # ignore_mask = (batch_size, )
        #     ignore some of the positions according to the mask, assumes x and y at those positions are both invalid
        if self.method == "simsiam":
            return self.simsiam(x, y, ignore_mask)

        # can't initialize self.world_size in __init__ as the module is created before spawning
        if self.world_size is None:
            if self.distributed:
                self.world_size = float(get_world_size())
            else:
                self.world_size = 1.0

        # check the shapes on ignore_mask
        if ignore_mask is not None:
            assert len(ignore_mask.shape) == 1 and ignore_mask.shape[0] == x.shape[0]

        if self.distributed:
            x_all = AllGatherAutoGrad.apply(x, 0)
            y_all = AllGatherAutoGrad.apply(y, 0)
            ignore_mask_all = concat_all_gather(ignore_mask) if ignore_mask is not None else None
        else:
            x_all = x
            y_all = y
            ignore_mask_all = ignore_mask

        # Apply the mask.
        if ignore_mask_all is not None:
            x_all = x_all[~ignore_mask_all]
            y_all = y_all[~ignore_mask_all]

        # if after masking there is nothing left to compute
        if x_all.shape[0] == 0:
            return {
                "loss": torch.tensor(0.0, device=x_all.device),
                "accuracy_x": torch.tensor(0.0, device=x_all.device),
                "accuracy_y": torch.tensor(0.0, device=x_all.device),
            }

        if self.method == "cross_entropy":
            return self._cross_entropy_method(x_all, y_all)
        elif self.method == "alignment_uniformity":
            return self._alignment_uniformity_method(x_all, y_all)
        elif self.method == "barlow_twins":
            return self._barlow_twins_method(x_all, y_all)

    def _barlow_twins_method(self, x_all, y_all):
        ret = self.barlow_twins(x_all, y_all)
        return {
            "loss": ScaleBackward.apply(ret["loss"], self.world_size).clone(),
            "accuracy_x": ret["accuracy_x"],
            "accuracy_y": ret["accuracy_y"],
        }

    # These next 3 methods are for the alignment_uniformity loss.
    # See https://github.com/SsnL/align_uniform#documentation
    def _align_loss(self, x, y):
        return (x - y).norm(p=2, dim=1).pow(self.alignment_uniformity_alpha).mean()

    def _uniform_loss(self, x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-self.alignment_uniformity_t).exp().mean().log()

    def _alignment_uniformity_method(self, x_all, y_all):
        align_loss = self._align_loss(x_all, y_all)
        # See https://github.com/allenai/contrastive_pretraining/pull/25#pullrequestreview-688949111
        # TLDR: It's not clear what the form of the loss is, and the following maybe preferred
        # uniform_loss = self._uniform_loss(torch.cat([x_all, y_all]))
        uniform_loss = 0.5 * (self._uniform_loss(x_all) + self._uniform_loss(y_all))
        total_loss = 0.5 * (align_loss + uniform_loss)

        # To compute accuracy_x, we need the distance from each x to every y,
        # and same for accuracy_y.
        # x_all = (batch_size, dim), y_all = (batch_size, dim)
        # pairwise_distances = (batch_size, batch_size)
        # pairwise_distances[i, j] = distance between x[i] and y[j]
        pairwise_distances = torch.norm(x_all.unsqueeze(2) - y_all.transpose(0, 1).unsqueeze(0), p=2, dim=1)

        batch_size = x_all.shape[0]
        gold_labels = torch.arange(batch_size, device=x_all.device)

        # compute accuracy
        accuracy_x = (pairwise_distances.argmin(dim=1) == gold_labels).float().mean().detach()
        accuracy_y = (pairwise_distances.argmin(dim=0) == gold_labels).float().mean().detach()

        return {
            "loss": ScaleBackward.apply(total_loss, self.world_size).clone(),
            "accuracy_x": accuracy_x,
            "accuracy_y": accuracy_y,
        }

    def _cross_entropy_method(self, x_all, y_all):
        logits = torch.matmul(x_all, y_all.transpose(0, 1)) / self.tau  # (batch_size, batch_size)

        # the gold label is just the index. compute it for both x --> y and y --> x
        batch_size = x_all.shape[0]
        gold_labels = torch.arange(batch_size, device=x_all.device)

        loss = self.criterion(logits, gold_labels) + self.criterion(logits.transpose(0, 1), gold_labels)

        # compute accuracy
        accuracy_x = (logits.argmax(dim=1) == gold_labels).float().mean().detach()
        accuracy_y = (logits.argmax(dim=0) == gold_labels).float().mean().detach()

        # we are reducing over all members of the batch here. The loss is correct, but needs
        # to be rescaled in backward pass (to undo the scaling that will happen elsewhere).
        # the clone is needed to avoid some warning about inplace+view returned from custom functions
        return {
            "loss": ScaleBackward.apply(loss, self.world_size).clone(),
            "accuracy_x": accuracy_x,
            "accuracy_y": accuracy_y,
        }
