# MIT License
#
# Copyright (c) 2023- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import lru_cache
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, LoraModel
from pyannote.core.utils.generators import pairwise
from rich.console import Console
from transformers import AutoModel

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)

console = Console()


class SSeRiouSS(Model):
    """Self-Supervised Representation for Speaker Segmentation

    wav2vec > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    wav2vec: dict or str, optional
        Defaults to "WAVLM_BASE".
    finetune: dict, optional
        Select a way of using the self supervised extractor
        contains a key "type" and other optional keys depending on the type
        Existing types for now:
            - average (default): weighted average of every layers with learnable weights
            - last: get the last output of a frozen extractor
            - lora: adapt the extractor with LoRA (https://github.com/microsoft/LoRA) and get the last output
                - adapter: custom layers to adapt to LoRA, default to ["q_proj", "v_proj"]
                - r: Rank of Decomposition, default to 32
                - lora_alpha: alpha for lora scaling, default to 32.0
                - lora_dropout: probability of dropout for adapters, default to 0.05
                - init_lora_weight: type of lora to use, default to gaussian

    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 4, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
        Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
        This may proove useful for probing LSTM internals.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    wav2vec_layer: int, optional
        Deprecated in favor of finetune
        Index of layer to use as input to the LSTM.
        Defaults (-1) to use average of all layers (with learnable weights).
    finetune_wavlm: dict
        changing name to finetune
    """

    WAV2VEC_DEFAULTS = "WAVLM_BASE"

    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 4,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        ssl: Union[dict, str] = None,
        finetune={"type": "average"},
        lstm: Optional[dict] = None,
        linear: Optional[dict] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
        gradient_clip_val: float = 5.0,
        wav2vec_layer: int = -1,  # Deprecated
        finetune_wavlm=None,
    ):

        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)
        self.gradient_clip_val = gradient_clip_val

        self.finetune = finetune if finetune_wavlm is None else finetune_wavlm

        if isinstance(ssl, str):
            # `ssl` is one of the pretrained model in huggingface
            self.ssl = AutoModel.from_pretrained(ssl, local_files_only=True)
            model_dim = self.ssl.feature_projection.projection.out_features
            model_num_layers = self.ssl.config.num_hidden_layers
        self.use_last = True
        if "dual_optimizer" in self.finetune:
            self.automatic_optimization = False
        # the ssl should be freeze there for an accurate parameter counting
        if self.finetune["type"] == "last":
            for param in self.ssl.parameters():
                param.requires_grad = False
        elif self.finetune["type"] == "lora":

            if "adapter" in self.finetune:
                adapter = self.finetune["adapter"]
            else:
                adapter = ["q_proj", "v_proj"]
            if "r" in self.finetune:
                r = self.finetune["r"]
            else:
                r = 32
            if "lora_alpha" in self.finetune:
                lora_alpha = self.finetune["lora_alpha"]
            else:
                lora_alpha = 32.0
            if "lora_dropout" in self.finetune:
                lora_alpha = self.finetune["lora_dropout"]
            else:
                lora_dropout = 0.05
            if "init_lora_weights" in self.finetune:
                init_lora_weights = self.finetune["init_lora_weights"]
            else:
                init_lora_weights = "gaussian"
            config = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                target_modules=adapter,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
            )
            self.ssl = LoraModel(self.ssl, config, "lora-adapter")
        elif self.finetune["type"] == "average":
            self.use_last = False
            for param in self.ssl.parameters():
                param.requires_grad = False
            self.weights = nn.Parameter(
                data=torch.ones(model_num_layers), requires_grad=True
            )

        else:
            self.automatic_optimization = False

        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)

        self.save_hyperparameters("ssl", "wav2vec_layer", "lstm", "linear")

        monolithic = lstm["monolithic"]
        if monolithic:
            multi_layer_lstm = dict(lstm)
            del multi_layer_lstm["monolithic"]
            self.lstm = nn.LSTM(model_dim, **multi_layer_lstm)

        else:
            num_layers = lstm["num_layers"]
            if num_layers > 1:
                self.dropout = nn.Dropout(p=lstm["dropout"])

            one_layer_lstm = dict(lstm)
            one_layer_lstm["num_layers"] = 1
            one_layer_lstm["dropout"] = 0.0
            del one_layer_lstm["monolithic"]

            self.lstm = nn.ModuleList(
                [
                    nn.LSTM(
                        (
                            model_dim
                            if i == 0
                            else lstm["hidden_size"]
                            * (2 if lstm["bidirectional"] else 1)
                        ),
                        **one_layer_lstm,
                    )
                    for i in range(num_layers)
                ]
            )

        if linear["num_layers"] < 1:
            return

        lstm_out_features: int = self.hparams.lstm["hidden_size"] * (
            2 if self.hparams.lstm["bidirectional"] else 1
        )
        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        lstm_out_features,
                    ]
                    + [self.hparams.linear["hidden_size"]]
                    * self.hparams.linear["num_layers"]
                )
            ]
        )

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("SSeRiouSS does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        self.classifier = nn.Linear(in_features, self.dimension)
        self.activation = self.default_activation()

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """

        num_frames = num_samples
        try:
            for conv_layer in self.ssl.feature_extractor.conv_layers:
                num_frames = conv1d_num_frames(
                    num_frames,
                    kernel_size=conv_layer.conv.kernel_size[0],
                    stride=conv_layer.conv.stride[0],
                    padding=conv_layer.conv.padding[0],
                    dilation=conv_layer.conv.dilation[0],
                )
        except Exception as e:
            print(e)
        return num_frames

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """

        receptive_field_size = num_frames
        try:
            for conv_layer in reversed(self.ssl.feature_extractor.conv_layers):

                receptive_field_size = conv1d_receptive_field_size(
                    num_frames=receptive_field_size,
                    kernel_size=conv_layer.conv.kernel_size[0],
                    stride=conv_layer.conv.stride[0],
                    dilation=conv_layer.conv.dilation[0],
                )
        except Exception as e:
            print(e)
        return receptive_field_size

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """
        receptive_field_center = frame
        try:
            for conv_layer in reversed(self.ssl.feature_extractor.conv_layers):
                # print(f"{conv_layer.conv.kernel_size=},{conv_layer.conv.stride=},{conv_layer.conv.padding=},{conv_layer.conv.dilation=}")
                receptive_field_center = conv1d_receptive_field_center(
                    receptive_field_center,
                    kernel_size=conv_layer.conv.kernel_size[0],
                    stride=conv_layer.conv.stride[0],
                    padding=conv_layer.conv.padding[0],
                    dilation=conv_layer.conv.dilation[0],
                )
        except Exception as e:
            print(e)
        return receptive_field_center

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        # num_layers = (
        #     None if self.hparams.wav2vec_layer < 0 else self.hparams.wav2vec_layer
        # )
        if self.finetune["type"] == "last":
            with torch.no_grad():
                outputs = self.ssl(waveforms.squeeze(1)).last_hidden_state
        else:
            with torch.no_grad():
                outputs = self.ssl(waveforms.squeeze(1)).hidden_states
            outputs = torch.stack(outputs, dim=-1) @ F.softmax(self.weights, dim=0)

        # if num_layers is None:
        #     outputs = torch.stack(outputs, dim=-1) @ F.softmax(
        #         self.wav2vec_weights, dim=0
        #     )
        # else:
        #     outputs = outputs[-1]

        if self.hparams.lstm["monolithic"]:
            outputs, _ = self.lstm(outputs)
        else:
            for i, lstm in enumerate(self.lstm):
                outputs, _ = lstm(outputs)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    outputs = self.dropout(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))
        return self.activation(self.classifier(outputs))
