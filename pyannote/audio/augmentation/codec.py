from typing import Optional

import torch
import torchaudio
from torch import Tensor
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict


class PhoneCodec:
    """
    Simulation of a phone codec, using a sinc bandpass and a mu-law encoding
    """

    def __init__(self, quantization=256):
        self.quant = quantization

        self.encode = torchaudio.transforms.MuLawEncoding(self.quant)
        self.decode = torchaudio.transforms.MuLawDecoding(self.quant)

    def apply(self, samples):

        bp_sample = torchaudio.functional.highpass_biquad(
            torchaudio.functional.lowpass_biquad(samples, 16000, 3400), 16000, 300
        )
        return self.decode(self.encode(bp_sample))


class IdentityCodec:
    """
    Blank codec, does nothing
    """

    def __init__(self, quantization=256):
        self.quant = quantization

    def apply(self, samples):

        print(f"{samples.shape=}")
        bp_sample = torchaudio.functional.highpass_biquad(
            torchaudio.functional.lowpass_biquad(samples, 16000, 3400), 16000, 300
        )

        print(f"{bp_sample.shape=}")
        return bp_sample


class CodecAugmentation(BaseWaveformTransform):
    """
    Add reverb to a sample using RiR augmentation

    Signal-to-noise ratio (where "noise" is the second random sample) is selected
    randomly between `min_snr_in_db` and `max_snr_in_db`.

    Parameters
    ----------
    width_range : float, optional
        Width in of the room in meter
    depth_range : float, optional
        Depth in of the room in meter
    height_range: int, optional
        Height in of the room in meter
    max_order:
        max number of reflexion
    absorption:
        Coefficient of the walls

    """

    supported_modes = {"per_example", "per_channel"}

    supports_multichannel = False
    requires_sample_rate = False

    supports_target = True
    requires_target = False

    def __init__(
        self,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: Optional[str] = None,
        sample_rate: Optional[int] = None,
        target_rate: Optional[int] = None,
        output_type: str = "dict",
    ):
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self.configs = [
            PhoneCodec(),
            # IdentityCodec()
        ]

    def randomize_parameters(
        self,
        samples: Optional[Tensor] = None,
        targets=None,
        target_rate: Optional[int] = None,
        sample_rate: Optional[int] = None,
    ):

        self.codec = self.configs[torch.randint(len(self.configs), (1,))]

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        batch_size, num_channels, num_samples = samples.shape
        # snr = self.transform_parameters["snr_in_db"]
        augmented = self.codec.apply(samples)

        return ObjectDict(
            samples=augmented,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
