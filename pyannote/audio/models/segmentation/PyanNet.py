# MIT License
#
# Copyright (c) 2020 CNRS
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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyannote.core.utils.generators import pairwise

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.models.blocks.tcn import TCN
from pyannote.audio.utils.params import merge_dict


class PyanNet(Model):
    """PyanNet segmentation model

    SincNet > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    sincnet : dict, optional
        Keyword arugments passed to the SincNet block.
        Defaults to {"stride": 1}.
    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
        Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
        This may proove useful for probing LSTM internals.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    SINCNET_DEFAULTS = {"stride": 10}
    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "monolithic": True,
        "dropout": 0.0,
    }
    TCN_DEFAULTS = {
        "in_chan": 60,
        "n_src": 1,
        "out_chan": 1,
        "n_blocks": 3,
        "n_repeats": 5,
        "bn_chan": 128,
        "hid_chan": 512,
        "kernel_size": 3,
        "norm_type": "gLN",
        "representation": False,
        # "dropout": 0.0,
    }
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sincnet: Optional[dict] = None,
        lstm: Optional[dict] = None,
        linear: Optional[dict] = None,
        tcn: Optional[dict] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
        use_tcn: bool = False,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)
        self.use_tcn = use_tcn
        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate

        linear = merge_dict(self.LINEAR_DEFAULTS, linear)

        if not use_tcn:

            lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
            lstm["batch_first"] = True
            self.save_hyperparameters("sincnet", "lstm", "linear")
        else:

            tcn = merge_dict(self.TCN_DEFAULTS, tcn)
            self.save_hyperparameters("sincnet", "tcn", "linear")

        self.sincnet = SincNet(**self.hparams.sincnet)

        if not use_tcn:
            monolithic = lstm["monolithic"]
            if monolithic:
                multi_layer_lstm = dict(lstm)
                del multi_layer_lstm["monolithic"]
                self.lstm = nn.LSTM(60, **multi_layer_lstm)

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
                                60
                                if i == 0
                                else lstm["hidden_size"]
                                * (2 if lstm["bidirectional"] else 1)
                            ),
                            **one_layer_lstm,
                        )
                        for i in range(num_layers)
                    ]
                )
        else:
            print("No need to load LSTM")
            pass

        if linear["num_layers"] < 1:
            return
        if not use_tcn:
            out_features: int = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )
        else:
            out_features = tcn["out_chan"]

        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        out_features,
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
            raise ValueError("PyanNet does not support multi-tasking.")

        if self.specifications.powerset:

            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    def build(self):

        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            if not self.use_tcn:
                in_features = self.hparams.lstm["hidden_size"] * (
                    2 if self.hparams.lstm["bidirectional"] else 1
                )
                self.classifier = nn.Linear(in_features, self.dimension)
            else:
                in_features = 0

        if self.use_tcn:

            self.hparams.tcn["out_chan"] = self.dimension
            # self.dropout = nn.Dropout(p=self.hparams.tcn["dropout"])
            self.classifier = TCN(**self.hparams.tcn)

        self.activation = self.default_activation()

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames for a given number of input samples

        Parameters
        ----------
        num_samples : int
            Number of input samples

        Returns
        -------
        num_frames : int
            Number of output frames
        """

        return self.sincnet.num_frames(num_samples)

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
        return self.sincnet.receptive_field_size(num_frames=num_frames)

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

        return self.sincnet.receptive_field_center(frame=frame)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        outputs = self.sincnet(waveforms)

        if not self.use_tcn:
            if self.hparams.lstm["monolithic"]:
                outputs, _ = self.lstm(
                    rearrange(outputs, "batch feature frame -> batch frame feature")
                )
            else:
                outputs = rearrange(
                    outputs, "batch feature frame -> batch frame feature"
                )
                for i, lstm in enumerate(self.lstm):
                    outputs, _ = lstm(outputs)
                    if i + 1 < self.hparams.lstm["num_layers"]:
                        outputs = self.dropout(outputs)
        else:

            outputs = self.classifier(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))
            out = self.classifier(outputs)
        else:

            out = rearrange(outputs, "batch classes frame -> batch frame classes")
        return self.activation(out)
