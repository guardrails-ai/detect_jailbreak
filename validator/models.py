import os
from typing import Optional, Union

import numpy
import torch
import torch.nn as nn


def string_to_one_hot_tensor(
        text: Union[str, list[str]],
        max_length: int = 2048,
        left_truncate: bool = True,
) -> torch.Tensor:
    if isinstance(text, str):
        out = torch.zeros((1, min(max_length, len(text)), 256), dtype=torch.float32)
        if left_truncate:
            text = text[-max_length:]
        else:
            text = text[:max_length]
        for idx, c in enumerate(text):
            one_hot = ord(c) if c.isascii() else 255
            out[0, idx, one_hot] = 1.0
    elif isinstance(text, list) or isinstance(text, tuple):
        out = torch.zeros(
            (len(text), max(min(max_length, len(t)) for t in text), 256),
            dtype=torch.float32
        )
        for idx, t in enumerate(text):
            if left_truncate:
                t = t[-max_length:]
                out[idx, -len(t):, :] = string_to_one_hot_tensor(t, max_length, left_truncate)[0, :, :]
            else:
                t = t[:max_length]
                out[idx, :len(t), :] = string_to_one_hot_tensor(t, max_length, left_truncate)[0, :, :]
    else:
        raise Exception("Input was neither a string nor a list of strings.")
    return out


class PromptSaturationDetectorV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.output_activation = nn.Sigmoid()
        self.lstm0 = nn.LSTM(
            input_size=256,
            hidden_size=256,
            batch_first=True,
            num_layers=4,
        )
        self.fan_out = nn.Linear(256, 1024)
        self.fan_in = nn.Linear(1024, 256)
        self.lstm1 = nn.LSTM(
            input_size=256,
            hidden_size=256,
            batch_first=True
        )
        self.output_head = nn.Linear(256, 1)
        # This is a silly hack to allow us to get the current device.
        # If the model gets moved to the GPU or CPU or MPS, we'll be able to tell.
        # If this tiny model gets scattered across a bunch of GPUs, this won't work.
        self.dummy_param = nn.Parameter(torch.empty(0))

    def get_current_device(self):
        return self.dummy_param.device

    def forward(
            self,
            x: Union[str, list[str], numpy.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(x, str) or isinstance(x, list) or isinstance(x, tuple):
            x = string_to_one_hot_tensor(x).to(self.get_current_device())
        x = self.lstm0(x)[0]
        x = self.fan_out(x)
        x = self.hidden_activation(x)
        x = self.fan_in(x)
        x = self.lstm1(x)[0]
        x = self.output_head(x)
        x = x[:,-1,0]
        x = self.output_activation(x)
        return x


class PromptSaturationDetectorV2(nn.Module):
    @staticmethod
    def initialize_from_pretrained():
        transformer = torch.hub.load(
            'huggingface/pytorch-transformers',
            'modelForSequenceClassification',
            'bert-base-uncased'
        )
        tokenizer = torch.hub.load(
            'huggingface/pytorch-transformers',
            'tokenizer',
            'bert-base-cased'
        )
        return PromptSaturationDetectorV2(tokenizer, transformer)

    def __init__(self, tokenizer=None, model=None):
        super().__init__()
        self.pad_token = 0
        self.transformer = model
        self.tokenizer = tokenizer
        self.dummy_param = nn.Parameter(torch.empty(0))

    def get_current_device(self):
        return self.dummy_param.device

    def forward(
            self,
            x: Union[str, list[str], numpy.ndarray, torch.Tensor],
            y: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Presumably we'd get a pre-tokenized array in here, but if we get text...
        max_size = 512  # For BERT.
        longest_sequence = 0
        if isinstance(x, str):
            x = [self.tokenizer.encode(x, add_special_tokens=True)[-max_size:]]
            longest_sequence = len(x[0])
            x = torch.LongTensor(x).to(self.get_current_device())
            # TODO: is 1 masked or unmasked?
            attention_mask = torch.LongTensor([1] * longest_sequence).to(self.get_current_device())
        elif isinstance(x, list) or isinstance(x, tuple):
            sequences = [self.tokenizer.encode(text, add_special_tokens=True)[-max_size:] for text in x]
            for token_list in sequences:
                longest_sequence = max(longest_sequence, len(token_list))
            x = list()
            attention_mask = list()
            for sequence in sequences:
                x.append(
                    ([self.pad_token] * (longest_sequence - len(sequence))) + sequence
                )
                attention_mask.append([0] * (longest_sequence - len(sequence)) + [1] * len(sequence))
            x = torch.LongTensor(x).to(self.get_current_device())
            attention_mask = torch.tensor(attention_mask).to(self.get_current_device())

        #segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        segments_tensors = torch.zeros(x.shape, dtype=torch.int).to(self.get_current_device())
        if y is not None:
            return self.transformer(x, token_type_ids=segments_tensors, attention_mask=attention_mask, labels=y)
        else:
            return self.transformer(x, token_type_ids=segments_tensors, attention_mask=attention_mask).logits


class PromptSaturationDetectorV3:  # Note: Not nn.Module.
    # This is a dumb convenience wrapper for a pipeline.  It sets up a bunch of
    # tokenizer settings that we need and turns this into something like a pipeline.
    def __init__(self, device: torch.device = torch.device('cpu')):
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        #self.model = torch.load(model_pth, map_location=device)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "GuardrailsAI/prompt-saturation-attack-detector",
        )
        self.model.config.id2label = {0: 'safe', 1: 'jailbreak'}
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google-bert/bert-base-cased",
            truncation_side='left',
            max_length=512,
            truncation=True,
            padding=True,
        )
        self.pipe = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            padding=True,
            max_length=512,
            device=device,
        )

    def __call__(self, text: Union[str, list[str]]) -> list[dict]:
        return self.pipe(text)
