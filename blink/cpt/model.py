import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional, Any, List
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    T5ForConditionalGeneration,
    Adafactor,
)
from transformers.models.t5.modeling_t5 import T5LayerNorm
from transformers.optimization import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from blink.cpt.data import T5ContrastivePretrainingDataset
from blink.cpt.utils import shift_right
from blink.cpt.distributed_contrastive_loss import ContrastiveLoss
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import argparse

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.debug.metrics as met
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


def get_world_size():
    if XLA_AVAILABLE:
        return xm.xrt_world_size()
    else:
        return torch.distributed.get_world_size()


import functools

DEFAULT_MMAP_PREFIX = "/home/armanc/cpt/wikipedia_t5"
DEFAULT_MODEL = "t5-base"
DEFAULT_TOKENIZER = "t5-base"


def gather_indexes(sequence_tensor, positions):
    """gathers vecotrs at specified positions

    Args:
        sequence_tensor: shape [batch_size, seq_length, dim]
        positions: shape [batch_size, max_positions]

    Returns:
        (sequence_tensor: shape [batch_size, max_positions, dim]
         invalid_positions_mask, shape [batch_size, max_positions]
        )
    invalid_positions_mask = True where positions is invalid (== -999), False otherwise
    """
    batch_size = sequence_tensor.shape[0]
    seq_length = sequence_tensor.shape[1]
    dim = sequence_tensor.shape[2]

    invalid_positions_mask = positions.eq(-999)
    # if its invalid position, just pull the first token representation, we ignore it in loss computation
    positions[invalid_positions_mask] = 0

    flat_offsets = (torch.arange(0, batch_size, dtype=torch.int32, device=positions.device) * seq_length).view(-1, 1)

    # flatten all positions inside the batrch into one sequence
    flat_positions = (positions + flat_offsets).view([-1])

    flat_sequence_tensor = sequence_tensor.reshape([batch_size * seq_length, dim])
    output_tensor = flat_sequence_tensor[flat_positions.long()]
    output_tensor = output_tensor.view(batch_size, positions.shape[1], dim)

    # fill invalid positions with -999.0
    # output_tensor = output_tensor.masked_fill(invalid_positions_mask.unsqueeze(2), -999.0)
    return output_tensor, invalid_positions_mask


def strbool(value):
    """for argument parsing"""
    assert value in ["true", "false"]
    return True if value == "true" else False


class Projection(nn.Module):
    def __init__(
        self,
        activation_fn,
        hidden_sizes=[768],
        input_dim=768,
        output_dim=768,
        add_activation_at_final_layer=False,
        dropout_rate=0.15,
        add_dropout_last_layer=False,
        t5_config=None,
        add_layer_norm=True,
    ):
        super().__init__()
        self.t5_config = t5_config  # used for initializing the model
        self.output_dim = output_dim
        self.input_dim = input_dim
        module_list = []
        if add_layer_norm:  # add layer norm right after T5 output
            encoder_layer_norm = T5LayerNorm(t5_config.d_model, eps=t5_config.layer_norm_epsilon)
            module_list.append(encoder_layer_norm)
        for i, h in enumerate(hidden_sizes):
            if i == 0:
                module_list.append(nn.Linear(self.input_dim, h))
            elif i < len(hidden_sizes) - 1:
                module_list.append(nn.Linear(hidden_sizes[i - 1], h))
            module_list.append(activation_fn)
            module_list.append(nn.Dropout(dropout_rate))  # add dropout
        module_list.append(nn.Linear(hidden_sizes[-1] if hidden_sizes else input_dim, self.output_dim))
        if add_activation_at_final_layer:
            module_list.append(activation_fn)
        if add_dropout_last_layer:
            module_list.append(nn.Dropout(dropout_rate))  # add dropout after final layer
        self.model = nn.Sequential(*module_list)

    def _init_weights(self, module):
        factor = self.t5_config.initializer_factor
        # Adapted from T5: https://github.com/huggingface/transformers/blob/c73e35323d5805bff27c5dbd9a5691008be1316a/src/transformers/models/t5/modeling_t5.py#L738
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor * ((self.t5_config.d_model) ** -0.5))
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, p=2, dim=-1)


class CPTModel(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Any,
        cls_token_id: int,
        tau: float,
        distributed: bool,
        span_infilling_only: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=self.hparams.cache_dir)
        self.tokenizer = tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name_or_path, config=self.config, cache_dir=self.hparams.cache_dir
        )
        # NOTE: just to make sure that we are passing both the decoder_input_ids and labels
        self.model._shift_right = None
        self.contrastive_loss = ContrastiveLoss(tau, distributed=distributed, method=self.hparams.contrastive_loss_method)
        self.cls_token_id = cls_token_id
        self.extra_id_0_id = self.tokenizer.encode("<extra_id_0>")[0]

        if self.hparams.proj_activation == "relu":
            activation_fn = nn.ReLU()
        elif self.hparams.proj_activation == "gelu":
            activation_fn = nn.GELU()
        else:
            raise NotImplementedError()

        if not span_infilling_only and self.hparams.contr_loss_mixing_ratio > 0:
            hidden_sizes = [self.config.d_model for _ in range(self.hparams.num_proj_hidden_layers)]
            self.projection_src = Projection(
                input_dim=self.config.d_model,
                activation_fn=activation_fn,
                hidden_sizes=hidden_sizes,
                output_dim=self.config.d_model,
                add_activation_at_final_layer=self.hparams.proj_add_activation_at_final_layer,
                dropout_rate=self.hparams.projection_dropout,
                add_dropout_last_layer=self.hparams.projection_add_dropout_last_layer,
                t5_config=self.config,
                add_layer_norm=self.hparams.add_layer_norm_encoder and self.hparams.encoder_side_contrastive_loss,
            )
            self.projection_src.apply(self.projection_src._init_weights)
            if getattr(self.hparams, "symmetric_projections", False):
                self.projection_tgt = self.projection_src
            else:
                self.projection_tgt = Projection(
                    input_dim=self.config.d_model,
                    activation_fn=activation_fn,
                    hidden_sizes=hidden_sizes,
                    output_dim=self.config.d_model,
                    add_activation_at_final_layer=self.hparams.proj_add_activation_at_final_layer,
                    dropout_rate=self.hparams.projection_dropout,
                    add_dropout_last_layer=self.hparams.projection_add_dropout_last_layer,
                    t5_config=self.config,
                    add_layer_norm=self.hparams.add_layer_norm_encoder and self.hparams.encoder_side_contrastive_loss,
                )
                self.projection_tgt.apply(self.projection_tgt._init_weights)

    def _get_linked_document_representations(self, batch):
        """Get representation of the the linked documents
        Args:
            batch: see training_step docstring

        Returns:
            tensor of shape [batch_size, num_links_per_doc, dim]
        """
        batch_size = batch["source"].shape[0]
        linked_docs_seq_length = batch["link_target_token_ids"].shape[-1]
        num_links = batch["link_target_token_ids"].shape[1]

        # reshape so that all links are in one batch
        linked_docs_input_ids = batch["link_target_token_ids"].view(batch_size * num_links, -1)

        linked_doc_output = self.get_cls_token_representations(
            linked_docs_input_ids,
            apply_projection_and_normalize=False,  # projection is applied at forward
        )

        # shape of linked_doc_output is [batch_size * num_links, dim]
        # reshape to [batch_size, num_links, dim]
        linked_doc_output = linked_doc_output.view(batch_size, num_links, -1)

        return linked_doc_output

    def get_cls_token_representations(self, token_ids, apply_projection_and_normalize=False):
        """Get CLS representation of the documents

        Passes them through T5 and selects the appropriate CLS index depending on model configuration.
        Note: the token_ids should not include the CLS token, these are added to the encoder/decoder as needed depending on model configuration.

        Args:
            token_ids: shape [batch_size, sequence_length]
            apply_projection_and_normalize: if True then also apply the projection layers and l2-normalization used for computing contrastive loss

        Returns:
            tensor of shape [batch_size, embedding_size]
        """
        # the target for linked documents is just <cls> </s>
        # T5 uses self.pad_token_id as the decoder start id, so
        # decoder_input_ids = [<pad> <cls>]
        cls_token = self.cls_token_id
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
        if self.hparams.cls_before_pad:
            decoder_input_ids = torch.cat([cls_tokens, pad_decoder_tokens], dim=1)
        else:
            decoder_input_ids = torch.cat([pad_decoder_tokens, cls_tokens], dim=1)

        if not self.hparams.cls_at_encoder_side:  # decoder side cls
            input_ids = token_ids
        else:
            input_ids = torch.cat([cls_tokens, token_ids], dim=1)
            # at decoder side we just use a single pad token
            decoder_input_ids = pad_decoder_tokens

        model_output = self.model(
            input_ids=input_ids,
            attention_mask=(input_ids != self.tokenizer.pad_token_id),
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
        )

        # first select either encoder or decoder hidden states depending on model config
        full_hidden_states = (
            model_output.encoder_last_hidden_state
            if self.hparams.cls_at_encoder_side
            else model_output.decoder_hidden_states[-1]
        )

        # Now select just the index of the CLS token
        cls_token_representations = (
            full_hidden_states[:, 0, :]
            if self.hparams.cls_before_pad or self.hparams.cls_at_encoder_side
            else full_hidden_states[:, 1, :]
        )

        if apply_projection_and_normalize:
            cls_token_representations = self.projection_tgt(cls_token_representations)

        return cls_token_representations

    def get_cloze_token_representations(self, token_ids, apply_projection_and_normalize):
        """Get embedding of the <extra_id_0> token used in the cloze-style experiments.
        This function assumes token_ids has one and only one `<extra_id_0>` token

        Args:
            token_ids: shape [batch_size, sequence_length]
            apply_projection_and_normalize: if True then also apply the projection layers and
                                            l2-normalization used for computing contrastive loss

        Returns:
            tensor of shape [batch_size, embedding_size]
        """

        assert self.hparams.encoder_side_contrastive_loss  # TODO: support other settings

        token_ids = self.prepare_input_for_encoder_decoder(token_ids)

        model_output = self.model(
            input_ids=token_ids,
            attention_mask=(token_ids != self.tokenizer.pad_token_id),
            decoder_input_ids=token_ids[:, :1],  # FIXME: a dummy input. We only use the encoder
            output_hidden_states=True,
        )

        non_zeros = (token_ids == self.extra_id_0_id).nonzero(as_tuple=True)
        row_indices, cloze_indices = non_zeros

        # each example should have one and exactly one "<extra_id_0>"
        assert cloze_indices.size(0) == token_ids.size(0)

        cloze_token_representations = model_output.encoder_last_hidden_state[row_indices, cloze_indices]

        if apply_projection_and_normalize:
            cloze_token_representations = self.projection_src(cloze_token_representations)

        return cloze_token_representations

    def prepare_input_for_encoder_decoder(self, token_ids):
        """Add special tokens to input_ids needed for by the CPT model

        Args:
            token_ids: shape [batch_size, sequence_length]

        Returns:
            tensor of shape [batch_size, embedding_size]
        """
        if not self.hparams.encoder_side_contrastive_loss:
            return token_ids

        cls_token_tensor = torch.tensor([self.cls_token_id]).to(token_ids.device).expand(token_ids.size(0), 1)

        # assert that the input doesn't already have `self.cls_token_id`
        assert not torch.any(token_ids[:, 0].view(-1) == cls_token_tensor.view(-1)).item()

        token_ids = torch.cat((cls_token_tensor, token_ids), 1)
        return token_ids

    def forward(self, batch):
        # TODO: deal with mask, Currently the <target*> tokens are masked autoregressively

        # shift target labels as decoder input ids
        # NOTE: Maybe move shifting to data side
        # t5 uses pad_token_id for bos
        # NOTE/TODO: We should always pass both the decoder_input_ids and labels to `self.model()` as we have changed the default self.model._shitf_right
        # The original self.model._shift_right also changes labels of -100 to <pad_token_id> but we keep ignored labels as -100
        # as the loss function of T5 ignores them (https://github.com/huggingface/transformers/blob/a26f4d620874b32d898a5b712006a4c856d07de1/src/transformers/models/t5/modeling_t5.py#L791)
        decoder_input_ids = shift_right(
            batch["target"],
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.pad_token_id,
        )

        # target_labels has masks for first prefix tokens

        output = self.model(
            input_ids=batch["source"],
            decoder_input_ids=decoder_input_ids,
            labels=batch["target_labels"],
            output_hidden_states=True,
        )

        if not self.hparams.span_infilling_only and self.hparams.contr_loss_mixing_ratio > 0:
            if not self.hparams.encoder_side_contrastive_loss:
                # get representation of target entities at decoder side, positions of which are specified at batch['link_decoder_indices']
                target_entity_repr, invalid_positions_mask = gather_indexes(
                    output.decoder_hidden_states[-1], batch["link_decoder_indices"]
                )
            else:
                forwarded_states = output.encoder_last_hidden_state
                target_entity_repr, invalid_positions_mask = gather_indexes(forwarded_states, batch["link_encoder_indices"])

            # get representation of all linked documents (cls token output at decoder)
            linked_doc_repr = self._get_linked_document_representations(batch)

            # project and normalize
            target_entity_repr = self.projection_src(target_entity_repr)
            linked_doc_repr = self.projection_tgt(linked_doc_repr)

            dim = target_entity_repr.shape[-1]
            contr_loss = self.contrastive_loss(
                target_entity_repr.view(-1, dim),
                linked_doc_repr.view(-1, dim),
                invalid_positions_mask.view(-1),
            )
        else:
            contr_loss = {
                "loss": torch.tensor(0.0, device=output.loss.device),
                "accuracy_x": torch.tensor(0.0, device=output.loss.device),
                "accuracy_y": torch.tensor(0.0, device=output.loss.device),
            }

        return {"lm_loss": output.loss, "contr_loss": contr_loss}

    def on_after_backward(self):
        if self.hparams.gradient_log_interval is None:
            return super().on_after_backward()
        global_step = self.global_step
        if global_step % self.hparams.gradient_log_interval == 0:
            for name, param in self.named_parameters():
                # NOTE: only logging projection related weights
                if "proj" in name:
                    self.logger.experiment.add_histogram(name, param, global_step)
                    if param.requires_grad:
                        self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)

    def shared_step(self, batch, batch_idx):
        output = self(batch)
        loss = (
            self.hparams.contr_loss_mixing_ratio * output["contr_loss"]["loss"]
            + (1 - self.hparams.contr_loss_mixing_ratio) * output["lm_loss"]
        )
        return {
            "loss": loss,
            "lm_loss": output["lm_loss"],
            "contr_loss": output["contr_loss"]["loss"],
            "contr_acc_x": output["contr_loss"]["accuracy_x"],
            "contr_acc_y": output["contr_loss"]["accuracy_y"],
        }

    def training_step(self, batch, batch_idx):
        """

        Args:
            batch (dict): has the following keys:
                'prefix_length': length of the additional tokens (<CLS> and <target_*>)
                'source': batched token ids of the source doc
                'target': actual target of a source document with T5 sentinels and our special sentinel tokens
                'target_labels': same as target, with prefix tokens masked to -100
                'link_target_token_ids': token ids for the link targets, shape: [batch, num_linked_docs, max_seq_len]
                'link_decoder_indices': indices in "target" where the contrast loss is attached which
                    is the index in the masked spans corresponding to the links
            batch_idx ([type]): [description]

        Returns:
            loss
        """
        output = self.shared_step(batch, batch_idx)
        for k, v in output.items():
            self.log(k, v, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return output["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        output = self.shared_step(batch, batch_idx)
        for k, v in output.items():
            self.log("val_" + k, v, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {
            "val_lm_loss": output["lm_loss"],
            "val_contr_loss": output["contr_loss"],
            "val_total_loss": output["loss"],
        }

    def validation_epoch_end(self, outputs):
        lm_loss = torch.stack([x["val_lm_loss"] for x in outputs]).mean()
        total_loss = torch.stack([x["val_total_loss"] for x in outputs]).mean()
        contr_loss = torch.stack([x["val_contr_loss"] for x in outputs]).mean()
        # TODO: add distributed loss sync
        self.log("val_total_lm_loss", lm_loss, prog_bar=True, sync_dist=True)
        self.log("val_total_contr_loss", contr_loss, prog_bar=True, sync_dist=True)
        self.log("val_total_loss", total_loss, prog_bar=True, sync_dist=True)

    def setup(self, stage):
        if stage == "fit":
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()
            if getattr(self.hparams, "max_steps", None):
                self.total_steps = self.hparams.max_steps
            else:
                num_devices = get_world_size() if self.hparams.gpus > 1 or self.hparams.tpus != -1 else 1
                # used in scheduler
                self.total_steps = (
                    self.hparams.num_epochs * len(train_loader) / self.hparams.accumulate_grad_batches / num_devices
                )

    def configure_optimizers(self):
        # model = self.model
        model = self  # self has some parameters that are not in self.model
        if self.hparams.optimizer == "adam":
            "Prepare optimizer and schedule (linear warmup and decay)"
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon,
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif self.hparams.optimizer == "adafactor":
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
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        else:
            raise NotImplementedError
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    @staticmethod
    def add_training_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Training")
        parser.add_argument("--learning_rate", default=1e-3, type=float)
        parser.add_argument("--optimizer", default="adafactor")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=1000, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--gpus", default=1, type=int)
        parser.add_argument("--tpus", default=-1, type=int)
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--max_steps", default=None, type=int)
        parser.add_argument("--num_epochs", default=1, type=int)
        parser.add_argument("--run_dir", default="runs/")
        parser.add_argument("--limit_val_batches", default=1.0, type=float)
        parser.add_argument("--val_check_interval", default=1.0, type=float)
        parser.add_argument("--accumulate_grad_batches", default=1, type=int)
        parser.add_argument("--gradient_log_interval", default=None, type=int)
        parser.add_argument("--track_grad_norm", default=-1, type=int)
        parser.add_argument("--accelerator", default="ddp")
        parser.add_argument("--save_top_k", default=3, type=int)
        parser.add_argument(
            "--resume_from_checkpoint",
            default=None,
            help="path to checkpoint to resume from",
        )
        return parent_parser

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--tokenizer_name_or_path", default=DEFAULT_TOKENIZER, type=str)
        parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL, type=str)
        parser.add_argument("--proj_add_activation_at_final_layer", default="false", type=strbool)
        parser.add_argument("--add_layer_norm_encoder", default="true", type=strbool)
        parser.add_argument("--num_proj_hidden_layers", default="1", type=int)
        parser.add_argument("--proj_activation", default="relu", type=str)
        parser.add_argument("--projection_dropout", default=0.15, type=float)
        parser.add_argument("--projection_add_dropout_last_layer", default="false", type=strbool)
        parser.add_argument("--tau", default=0.20, type=float, help="temprature of contrastive loss")
        parser.add_argument(
            "--contr_loss_mixing_ratio",
            default=0.5,
            type=float,
            help="mixing ratio for the lm_loss and contrastive_loss. If 1.0 lm_loss will be ignored",
        )
        parser.add_argument(
            "--encoder_side_contrastive_loss",
            default="false",
            type=strbool,
            help="get document representations at the encoder side",
        )
        parser.add_argument(
            "--cls_before_pad",
            default="false",
            type=strbool,
            help="Start with cls token then get to pad token",
        )
        parser.add_argument("--contrastive_loss_method", default="cross_entropy", type=str)
        parser.add_argument(
            "--cls_at_encoder_side",
            default="false",
            type=strbool,
            help="whether to add cls at encoder side or decoder side",
        )
        parser.add_argument("--symmetric_projections", default="false", type=strbool)
        return parent_parser

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument("--mmap_prefix", default=DEFAULT_MMAP_PREFIX, type=str)
        parser.add_argument(
            "--add_eos_token_link_target",
            choices={True, False},
            default=True,
            type=strbool,
        )
        parser.add_argument("--add_eos_token_source", choices={True, False}, default=False, type=strbool)
        parser.add_argument(
            "--preserve_number_mask_spans",
            choices={True, False},
            default=True,
            type=strbool,
        )
        parser.add_argument("--span_infilling_only", choices={True, False}, default=False, type=strbool)
        parser.add_argument("--lm_loss_on_cls_token", choices={True, False}, default=True, type=strbool)
        parser.add_argument("--workers", default=32, type=int)
        parser.add_argument("--link_target_sequence_length", default=128, type=int)
        return parent_parser

    @staticmethod
    def add_misc_args(parent_parser):
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument("--seed", type=int, default=4201)
        parser.add_argument("--num_links_to_include_positive_samples", default=4, type=int)
        parser.add_argument(
            "--num_links_to_sample_for_mask", default=None, type=int
        )  # None default = 2 * num_links_to_include_positive_samples
        parser.add_argument("--cache_dir", default=None)
        return parent_parser

    @staticmethod
    def refine_args(args):
        # TODO: add a better check
        args.distributed = True if (args.gpus > 1 or args.tpus > 1) else False
        if not args.distributed:
            args.accelerator = None
        if args.cls_at_encoder_side:
            assert (
                args.encoder_side_contrastive_loss
            ), "When setting cls_at_encoder_side, make sure to also set encode_side_contrastive_loss to True"
        return args


def get_args():
    parser = argparse.ArgumentParser()
    parser = CPTModel.add_data_specific_args(parser)
    parser = CPTModel.add_model_specific_args(parser)
    parser = CPTModel.add_training_specific_args(parser)
    parser = CPTModel.add_misc_args(parser)
    args = parser.parse_args()
    args = CPTModel.refine_args(args)
    return args


def get_dataloader(split, args):
    dataset = T5ContrastivePretrainingDataset(
        args.mmap_prefix,
        tokenizer_name=args.tokenizer_name_or_path,
        num_links_to_include_positive_samples=args.num_links_to_include_positive_samples,
        num_links_to_sample_for_mask=args.num_links_to_sample_for_mask,
        add_eos_token_link_target=args.add_eos_token_link_target,
        add_eos_token_source=args.add_eos_token_source,
        split=split,
        preserve_number_mask_spans=args.preserve_number_mask_spans,
        encoder_side_contrastive_loss=args.encoder_side_contrastive_loss,
        span_infilling_only=args.span_infilling_only,
        cls_at_encoder_side=args.cls_at_encoder_side,
        lm_loss_on_cls_token=args.lm_loss_on_cls_token,
        link_target_sequence_length=args.link_target_sequence_length,
    )
    custom_collate = functools.partial(
        T5ContrastivePretrainingDataset.collate_fn,
        pad_token_id=dataset.tokenizer.pad_token_id,
        num_links_to_include_positive_samples=args.num_links_to_include_positive_samples,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate,
        num_workers=args.workers,
    )
    return dataset, data_loader


def main():
    args = get_args()

    pl.seed_everything(args.seed)

    train_dataset, train_dataloader = get_dataloader("train", args)
    _, val_dataloader = get_dataloader("validation", args)

    model = CPTModel(
        tokenizer=train_dataset.tokenizer,
        cls_token_id=train_dataset.cls_token_id,
        **vars(args),
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_total_loss",
        filename="model-{val_total_loss:.3f}-{val_total_lm_loss:.3f}-{val_total_contr_loss:.3f}",
        save_top_k=args.save_top_k,
    )
    logger = pl_loggers.TensorBoardLogger(args.run_dir, name="cpt_model")
    # lightning recommends adding a plugin to find_unused_parameters=False when using ddp
    # https://pytorch-lightning.readthedocs.io/en/1.3.3/benchmarking/performance.html#when-using-ddp-set-find-unused-parameters-false
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback],
        plugins=DDPPlugin(find_unused_parameters=False) if args.gpus > 1 else None,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
