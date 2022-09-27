import argparse
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.utils.data
import transformers

from health_bert import optuna_helpers


class HealthBert(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.hparams["bert_base_name"]
        )
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.hparams["bert_base_name"], num_labels=2
        )

    def forward(self, inp: Dict[str, Any]):
        encoded = self.tokenizer(
            inp["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        )
        encoded = encoded.to(self.device)
        labels = None
        if "labels" in inp:
            labels = inp["labels"].to(self.device).long()
        out = self.model(**encoded, labels=labels)
        return out

    def training_step(self, data_batch, batch_i):
        out = self(data_batch)
        self.log("loss", out["loss"])
        return out["loss"]

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.hparams.lr)]

    @classmethod
    def add_model_specific_args(cls, parser: argparse.ArgumentParser):

        group = parser.add_argument_group(f"{cls.__module__}.{cls.__qualname__}")

        group.add_argument(
            "--bert_base_name", type=str, default="allenai/scibert_scivocab_uncased"
        )

        group.add_argument(
            "--lr", type=optuna_helpers.OptunaArg.parse, nargs="+", default=1e-5
        )

        return parser
