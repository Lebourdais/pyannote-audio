from typing import Optional

from torch import Tensor
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict


class CropSignal(BaseWaveformTransform):
    """
    Cropping augmentation results to the desired length
    Imply that original signal is longer than wanted length
    Should always be applied if an augmentation manipulate file length

    Parameters
    ----------
    duration: desired duration

    """

    supported_modes = {"per_example", "per_channel"}

    supports_multichannel = False
    requires_sample_rate = False

    supports_target = True
    requires_target = True

    def __init__(
        self,
        mode: str = "per_example",
        p: float = 1,
        duration=5,
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
        self.num_samples = int(duration * sample_rate)
        self.num_targets = int(duration * target_rate)

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        batch_size, num_channels, num_samples = samples.shape
        # snr = self.transform_parameters["snr_in_db"]
        # target [Batch,1, num_speakers, num_samples]
        cropped_sample = samples[:, :, : self.num_samples]
        cropped_target = targets[:, :, :, : self.num_targets]

        return ObjectDict(
            samples=cropped_sample,
            sample_rate=sample_rate,
            targets=cropped_target,
            target_rate=target_rate,
        )
