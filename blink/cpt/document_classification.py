import argparse
import random
import numpy as np
import time
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import datasets
from transformers import T5ForConditionalGeneration, Adafactor, AutoTokenizer
import transformers
from transformers.optimization import AdamW, get_constant_schedule_with_warmup
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from blink.cpt.linked_document import HyperlinkedWikipediaMMap
from blink.cpt.model import CPTModel
from blink.cpt.utils import shift_right


class ClassificationDataset(Dataset):

    # Values will be set in subclasses
    label_to_text_mapping = None
    label_to_wikipedia_pageid = None
    class_hf_dataset_splits = None

    def _load_hf_dataset_splits(self, split):
        # implemented in subclasses
        raise NotImplementedError()

    # HyperlinkedWikipediaMMap memmap path
    wikipedia_memmap_path = "/net/nfs2.corp/allennlp/matthewp/cpt/wikipedia_t5"

    def __init__(
        self,
        tokenizer_name,
        split,
        num_shots,
        prompt,
        label_description_seqlen,
        use_descriptions,
        embed,
        is_label_warmup,
    ):
        """
        Args:
            tokenizer_name: str
                name of the tokenizer to load from huggingface
            split: str
                which dataset split to load
            num_shots: int
                how many examples to include from each class in fewshot training
            prompt: str
                which prompt style to use (t5, t5-cloze, unifiedqa)
            label_description_seqlen: int
                what length to clip the label descriptions to
            use_descriptions: bool
                whether to use the descriptions of the labels loaded from wikipedia, or just
                the label names
            embed: bool
                whether to use the integer class labels
            is_label_warmup: bool
                we use this argument to determine whether the dataset should produce real data, or
                fake data (with real labels). The fake data is used to adapt the T5 decoder to the label space
        """

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        ClassificationDataset.pad_token_id = self.tokenizer.pad_token_id
        assert split in ["train", "test", "validation"]
        self._load_hf_dataset_splits(split)

        # sample num_shots instanes per class
        dataset = self.class_hf_dataset_splits[split]
        data_subsets = []  # list of HF dataset objects, one per label
        for label_id in self.label_to_text_mapping.keys():
            # avoid using HF datasets.shuffle because it loads from the cache instead of reshuffling
            data_subset_of_label = dataset.filter(lambda example: example["label"] == label_id)
            indices = range(len(data_subset_of_label))
            sampled_indices = random.sample(indices, num_shots)
            data_subsets.append(data_subset_of_label.select(sampled_indices))
        self.hf_dataset = datasets.concatenate_datasets(data_subsets)  # size = num_shots * len(label_to_text_mapping)

        self.prompt = prompt
        self.use_descriptions = use_descriptions
        self.embed = embed

        self.label_token_ids = []
        if self.use_descriptions:
            wikiepdia_memmap = HyperlinkedWikipediaMMap(self.wikipedia_memmap_path)
            for label, pageid in self.label_to_wikipedia_pageid.items():
                token_ids = wikiepdia_memmap[pageid].token_ids[:label_description_seqlen]
                self.label_token_ids.append(token_ids)
        else:
            for label, label_text in self.label_to_text_mapping.items():
                self.label_token_ids.append(self.tokenizer.encode(label_text))
        self.label_token_ids = [torch.tensor(l) for l in self.label_token_ids]
        self.label_token_ids = torch.nn.utils.rnn.pad_sequence(
            self.label_token_ids,
            batch_first=True,
            padding_value=ClassificationDataset.pad_token_id,
        )

        # we optionally adapt the model to the label space by training for a bit with random english sentences as input,
        # but with actual labels. The random sentences from from the DocumentGenerator
        self.is_label_warmup = is_label_warmup
        if self.is_label_warmup:
            from essential_generators import DocumentGenerator

            self.text_generator = DocumentGenerator()

    def __len__(self):
        return len(self.hf_dataset)

    def _get_unifiedqa_prompt(self, instance):
        choice_ids = ["(A)", "(B)", "(C)", "(D)", "(E)"]
        choices = self.label_to_text_mapping.values()
        choices_str = " ".join([f"{choice_id} {choice}" for choice_id, choice in zip(choice_ids, choices)])
        return f'Topic?\\n {instance["text"]}\\n {choices_str}'

    def __getitem__(self, instance_index: int):
        """Gets one item from the dataset corresponding to index `instance_index`
        Returns:
            input_ids: torch.tensor of input token ids following the prompt of `self.prompt`
            output_ids: torch.tensor of input token ids
                If embed, returns label id
                else
                    if use_descriptions, returns token ids for the wikipedia page corresponding to the label
                    else, return token ids for the label text
        """
        instance = self.hf_dataset[instance_index]

        if self.embed:
            output_ids = torch.tensor([instance["label"]])  # 0, 1, 2, ..
        else:
            output_ids = self.label_token_ids[instance["label"]]  # label text or label description

        if self.prompt == "t5":
            if not self.is_label_warmup:
                input_text = instance["text"]
            else:
                input_text = self.text_generator.sentence()
        elif self.prompt == "t5-cloze":
            input_text = "This article is about <extra_id_0>. " + instance["text"]
            if not self.embed:  # if embed, the output_ids is just the label number, so don't add sentinal token
                sentinal_token = torch.tensor(self.tokenizer.encode("<extra_id_0>", add_special_tokens=False))
                output_ids = torch.cat((sentinal_token, output_ids))  # The output format is "<extra_id_0> LABEL"
        elif self.prompt == "unifiedqa":
            input_text = self._get_unifiedqa_prompt(instance)
        else:
            assert False
        input_ids = self.tokenizer.encode(input_text, truncation=True, max_length=511)  # arbitrary limit to the input seqlen

        return torch.tensor(input_ids), output_ids

    @staticmethod
    def collate_fn(batch):
        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=ClassificationDataset.pad_token_id,
        )
        output_ids = torch.nn.utils.rnn.pad_sequence(
            output_ids,
            batch_first=True,
            padding_value=ClassificationDataset.pad_token_id,
        )
        return input_ids, output_ids


class AGNewsClassificationDataset(ClassificationDataset):
    label_to_text_mapping = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech",
    }  # 'Science'

    # From /net/nfs2.corp/allennlp/matthewp/cpt/wikipedia_index.json, find the pageid
    # that corresponds to each of the labels, "World", "Sports", "Business", "Science and technology".
    # Given the pageid, get the page text using `HyperlinkedWikipediaMMap`
    label_to_wikipedia_pageid = {0: 3340418, 1: 471734, 2: 3361026, 3: 2905586}

    class_hf_dataset_splits = (
        {}
    )  # save the HF dataset splits here for repeated episode sampling without reloading the dataset

    def _load_hf_dataset_splits(self, split):
        if split == "train":  # split train into train and validation
            split_string = "train[:50%]"
        elif split == "validation":
            split_string = "train[50%:]"
        elif split == "test":
            split_string = "test"
        if split not in self.class_hf_dataset_splits:
            self.class_hf_dataset_splits[split] = datasets.load_dataset("ag_news", split=split_string)


class SNLIClassificationDataset(ClassificationDataset):
    label_to_text_mapping = {0: "Yes", 1: "Maybe", 2: "No"}
    label_to_wikipedia_pageid = {}
    class_hf_dataset_splits = {}

    def _load_hf_dataset_splits(self, split):
        d = datasets.load_dataset("snli", split=split).filter(lambda x: x["label"] != -1)
        d = d.map(lambda x: {"text": f'{x["premise"]} Is {x["hypothesis"]}?'})
        self.class_hf_dataset_splits[split] = d

    def _get_unifiedqa_prompt(self, instance):
        choice_ids = ["(A)", "(B)", "(C)", "(D)", "(E)"]
        choices = self.label_to_text_mapping.values()
        choices_str = " ".join([f"{choice_id} {choice}" for choice_id, choice in zip(choice_ids, choices)])
        return f'{instance["premise"]} Is {instance["hypothesis"].lower().replace(".", "")}?\\n {choices_str}'


class DocumentClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        if self.hparams.cpt:
            self.model = CPTModel.load_from_checkpoint(
                self.hparams.model_name_or_path, contrastive_loss_method="cross_entropy"
            )
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        if self.hparams.tokenizer is None:
            self.hparams.tokenizer = self.hparams.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def forward(self, batch):
        # prompt: cloze-style or not
        # cpt: CPTModel or T5
        # embed: compare embeddings or text
        # projections: with/without
        # label description: with description or just label text (already handeled in the Dataset)

        input_ids, output_ids = batch
        if self.hparams.embed:
            if self.hparams.cpt:
                if "cloze" in self.hparams.prompt:
                    input_embeddings = self.model.get_cloze_token_representations(
                        token_ids=input_ids,
                        apply_projection_and_normalize=self.hparams.use_projection,
                    )
                else:
                    input_embeddings = self.model.get_cls_token_representations(
                        token_ids=input_ids,
                        apply_projection_and_normalize=self.hparams.use_projection,
                    )

                label_token_ids = self.trainer.val_dataloaders[0].dataset.label_token_ids.to(input_ids.device)
                label_embeddings = self.model.get_cls_token_representations(
                    label_token_ids,
                    apply_projection_and_normalize=self.hparams.use_projection,
                )

                logits = input_embeddings.matmul(label_embeddings.T)

                # compute accuracy
                accuracy = (logits.argmax(dim=1) == output_ids.squeeze()).float().mean().item()
                self.log(
                    "accuracy",
                    accuracy,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

                # compute loss
                assert output_ids.shape[0] == logits.shape[0]
                assert output_ids.shape[1] == 1

                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits, output_ids.squeeze())
                return loss, logits
            else:
                # TODO: T5 compare embeddings
                raise NotImplementedError
        else:
            if self.hparams.cpt:
                # get the T5 model inside the CPT model to use for text generation
                model = self.model.model
                input_ids = self.model.prepare_input_for_encoder_decoder(input_ids)
            else:
                model = self.model
            decoder_input_ids = shift_right(
                output_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                decoder_start_token_id=self.tokenizer.pad_token_id,
            )
            output_ids[output_ids == self.tokenizer.pad_token_id] = -100
            output = model(
                input_ids=input_ids,
                attention_mask=(input_ids != self.tokenizer.pad_token_id),
                labels=output_ids,
                decoder_input_ids=decoder_input_ids if self.hparams.cpt else None,
            )
            return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        loss, logits = self.forward(batch)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _eval_step(self, split, batch, batch_idx, dataloader_idx=0):
        input_ids, output_ids = batch
        loss, logits = self.forward(batch)
        self.log(
            f"{split}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        if self.hparams.embed:
            # accuracy is logged in the forward function
            pass
        else:
            model = (
                self.model.model if self.hparams.cpt else self.model
            )  # get the T5ForConditioalGeneration model from the CPT model
            # FIXME: we might be using a custom shift_right function for the CPTModel
            # If so, the following model.generate is going to fail
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=(input_ids != self.tokenizer.pad_token_id),
                use_cache=True,
                num_beams=1,
                # max_length=self.args.max_output_len,
            )
            pred_texts = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
            output_ids[output_ids == -100] = self.tokenizer.pad_token_id
            gold_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            correct_predictions_count = 0
            for pred, gold in zip(pred_texts, gold_texts):
                if pred == gold:
                    correct_predictions_count += 1
            self.log(
                "accuracy",
                correct_predictions_count / len(pred_texts),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step("val", batch, batch_idx, dataloader_idx)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step("test", batch, batch_idx, dataloader_idx)

    def setup(self, stage):
        if stage == "fit":
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()
            if getattr(self.hparams, "max_steps", None):
                self.total_steps = self.hparams.max_steps
            else:
                num_devices = torch.distributed.get_world_size() if self.hparams.gpus != -1 else self.hparams.gpus
                # used in scheduler
                self.total_steps = self.hparams.num_epochs * len(train_loader) / self.hparams.grad_accum / num_devices

    def configure_optimizers(self):
        model = self.model
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
            # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps)
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps)
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

    def get_dataloader(self, split):
        if split == "train":
            num_shots = self.hparams.train_shots  # number of shots per class
        elif split in ("validation", "test"):
            # number of shots per class = number of instances / number of classes
            num_shots = self.hparams.test_instances // len(self.hparams.dataset_class.label_to_text_mapping)
        else:
            assert False

        dataset = self.hparams.dataset_class(
            tokenizer_name=self.hparams.tokenizer,
            split=split,
            num_shots=num_shots if not self.hparams.is_label_warmup else self.hparams.num_label_warmup_steps,
            prompt=self.hparams.prompt,
            label_description_seqlen=self.hparams.label_description_seqlen,
            use_descriptions=self.hparams.use_descriptions,
            embed=self.hparams.embed,
            is_label_warmup=self.hparams.is_label_warmup,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.workers,
            collate_fn=self.hparams.dataset_class.collate_fn,
        )
        return dataset, data_loader

    def train_dataloader(self):
        train_dataset, train_dataloader = self.get_dataloader(split="train")
        return train_dataloader

    def val_dataloader(self):
        _, val_dataloader = self.get_dataloader(split="validation")
        return val_dataloader

    def test_dataloader(self):
        _, test_dataloader = self.get_dataloader(split="test")
        return test_dataloader

    @staticmethod
    def add_training_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Training")
        parser.add_argument("--learning_rate", default=3e-5, type=float)  # 1e-3
        parser.add_argument("--optimizer", default="adam")  # adafactor
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)  # 1000
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--gpus", default=-1, type=int)
        parser.add_argument("--tpus", default=-1, type=int)
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--max_steps", default=300, type=int)
        parser.add_argument("--max_epochs", default=100000, type=int)
        parser.add_argument("--run_dir", default="runs/")
        parser.add_argument("--limit_val_batches", default=0.0, type=float)
        parser.add_argument("--val_check_interval", default=1.0, type=int)
        parser.add_argument("--grad_accum", default=1, type=float)
        parser.add_argument("--num_sanity_val_steps", default=0, type=int)
        parser.add_argument("--checkpoint_callback", default=False, type=bool)
        parser.add_argument("--num_label_warmup_steps", default=0, type=int)

        return parent_parser

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--model_name_or_path", type=str, required=True)

        # Read the followoing params from the pretrained CPTModel checkpoint if needed
        # parser.add_argument('--proj_add_activation_at_final_layer', dest='proj_add_activation_at_final_layer', action='store_true')
        # parser.add_argument('--dont_proj_add_activation_at_final_layer', dest='proj_add_activation_at_final_layer', action='store_false')
        # parser.set_defaults(proj_add_activation_at_final_layer=True)
        # parser.add_argument('--num_proj_hidden_layers', default='1', type=int)
        # parser.add_argument('--proj_activation', default='t5-relu', type=str)
        # parser.add_argument('--tau', default=0.20, type=float, help='temprature of contrastive loss')
        # parser.add_argument('--symmetric_projections', default='true', type=strbool)
        # parser.add_argument('--cls_at_encoder_side', default='true', type=strbool,
        #                     help='whether to add cls at encoder side or decoder side')
        # parser.add_argument('--encoder_side_contrastive_loss',  default='true', type=strbool,
        #                     help='get document representations at the encoder side')

        return parent_parser

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument("--workers", default=0, type=int)
        return parent_parser

    @staticmethod
    def add_misc_args(parent_parser):
        parser = parent_parser.add_argument_group("Data")
        parser.add_argument("--seed", type=int, default=4201)
        parser.add_argument("--prompt", default="t5", choices=["t5", "unifiedqa", "t5-cloze", "bert"])
        parser.add_argument(
            "--embed",
            action="store_true",
            help="""If false, compare text, if true, compare embeddings""",
        )
        parser.add_argument(
            "--use_descriptions",
            action="store_true",
            help="""If false, embed label text. If true, embed label description."""
            """This arg is ignored if --embed is false""",
        )
        parser.add_argument("--use_projection", action="store_true")
        parser.add_argument("--label_description_seqlen", default=128, type=int)
        parser.add_argument(
            "--episodes",
            default=65,
            type=int,
            help="Episodic sampling. Default value is copied from FLEX paper.",
        )
        parser.add_argument(
            "--train_shots",
            default=16,
            type=int,
            help="Number of training shots per class per episode",
        )
        parser.add_argument(
            "--test_instances",
            default=350,
            type=int,
            help="Total number of testing instances per episode.  Default value is copied from FLEX paper.",
        )
        parser.add_argument("--dataset", default="agnews", type=str, choices=["snli", "agnews"])
        parser.add_argument(
            "--cpt",
            action="store_true",
            help="If true, use --model_name_or_path as a path to PL ckpt of the CPT model",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            help="Name of the tokenizer. If none, use --model_name_or_path instead",
        )
        return parent_parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = DocumentClassifier.add_data_specific_args(parser)
    parser = DocumentClassifier.add_model_specific_args(parser)
    parser = DocumentClassifier.add_training_specific_args(parser)
    parser = DocumentClassifier.add_misc_args(parser)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # TODO: enumerate valid combinations of --prompt and --embed
    # TODO: implement and test other combinations
    # UnifiedQA     compare labels                          DONE
    # T5            compare labels                          DONE
    # CPT           compare labels                          DONE but bad result (check label warmup for a fix)
    # T5-cloze      compare labels                          DONE but need to find better prompt
    # CPT-cloze     compare labels                          DONE but need to find better prompt

    # T5            CLS embedding      compare embeddings   Won't implement
    # CPT           CLS embedding      compare embeddings   DONE
    # T5-cloze      Span embedding     compare embeddings   Won't implement
    # CPT-cloze     Span embedding     compare embeddings   DONE

    # TODO: run on multiple GPUs

    # DON'T USE `pytorch_lightning.seed_everything(args.seed)`. IT SETS A FIXED SEED TO ALL SUBPROCESSES.
    # We sample the fewshots in the dataset subprocess. With a fixed seed for all workers, all episodes become identical.
    transformers.set_seed(args.seed)

    start = time.time()

    dataset_name_to_class = {
        "snli": SNLIClassificationDataset,
        "agnews": AGNewsClassificationDataset,
    }
    args.dataset_class = dataset_name_to_class[args.dataset]

    # load the model and dataset in warmup mode if warmup steps is greater than 0
    args.is_label_warmup = args.num_label_warmup_steps > 0
    model = DocumentClassifier(**vars(args))
    model_original_state_dict = deepcopy(model.state_dict())

    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = pl_loggers.TensorBoardLogger(args.run_dir, name="cls_model")

    # The model is very biased to produce <extra_id_0> first, and we can optionally adapt the decoder to the label space
    # before performing the few shot evaluation. Setting num_label_warmup_steps > 0 controls whether this happens
    if args.num_label_warmup_steps > 0:
        tmp_max_steps = args.max_steps
        args.max_steps = args.num_label_warmup_steps
        warmup_trainer = pl.Trainer.from_argparse_args(
            args,
            accelerator="ddp",
            logger=logger,
            callbacks=[
                lr_monitor,
            ],
        )
        encoder = model.model.model.encoder if args.cpt else model.model.encoder
        for param in encoder.parameters():
            param.requires_grad = False
        warmup_trainer.fit(model)
        for param in encoder.parameters():
            param.requires_grad = True
        args.max_steps = tmp_max_steps

        # save warmed up weights as the base weights for each episode
        model_original_state_dict = deepcopy(model.state_dict())

        # reload model and dataset not in warmup state
        args.is_label_warmup = False
        model = DocumentClassifier(**vars(args))

        # set up the model for the first episode
        model.load_state_dict(model_original_state_dict)

    results = []
    print("Starting episodes...")
    for episode in range(args.episodes):
        model.load_state_dict(model_original_state_dict)
        trainer = pl.Trainer.from_argparse_args(
            args,
            accelerator="ddp",
            logger=logger,
            callbacks=[
                lr_monitor,
            ],
        )
        trainer.fit(model)
        result = trainer.test(model)
        results.append(result)

        print(f"At episode: {episode}")
        print(f"Elapsed Time: {(time.time() - start) / 60:.2f}m")
        print(f"Dataset: {args.dataset}")
        for key in results[0][0].keys():
            if "epoch" not in key:
                continue
            vals = [x[0][key] for x in results]
            print(f"{np.mean(vals):.4f} +- {np.std(vals):.4f} --> {key}")

    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    main()


"""
# To reproduce the t5-base baseline result in
# https://docs.google.com/spreadsheets/d/1SxTzxP-g6zpz_aEo4tQ2n02vUs9uJPQ0BTKK4KdHl-c/edit#gid=0

CUDA_VISIBLE_DEVICES=0 python cpt/document_classification.py --model_name_or_path t5-base  --prompt t5  --dataset agnews  --episodes 10

# To run the pretrained CPTModel and generate label text
CUDA_VISIBLE_DEVICES=2 python cpt/document_classification.py --model_name_or_path  \
    /net/s3/s2-research/armanc/cpt/checkpoints/1-sampled-target-0.5mix-encoder-cls-symmetric/model-val_total_loss\=1.383-val_total_lm_loss\=1.893-val_total_contr_loss\=0.873.ckpt  \
    --prompt t5  --dataset agnews  --cpt --tokenizer t5-base
"""
