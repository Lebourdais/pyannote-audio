from typing import Optional

import torch
import torchaudio
from torch import Tensor
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.io import Audio
from torch_audiomentations.utils.object_dict import ObjectDict
from torchaudio.prototype.functional import simulate_rir_ism
from torchaudio.transforms import FFTConvolve


class ReverbAugmentation(BaseWaveformTransform):
    """
    Create a new sample by mixing it with another random sample from the same batch

    Signal-to-noise ratio (where "noise" is the second random sample) is selected
    randomly between `min_snr_in_db` and `max_snr_in_db`.

    Parameters
    ----------
    min_snr_in_db : float, optional
        Defaults to 0.0
    max_snr_in_db : float, optional
        Defaults to 5.0
    max_num_speakers: int, optional
        Maximum number of speakers in mixtures.  Defaults to actual maximum number
        of speakers in each batch.
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
        output_type: str = "tensor",
        width_range=(3, 9),
        depth_range=(3, 9),
        height_range=(2, 4),
        max_order=3,
        absorption=0.2,
    ):
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self.convolve = FFTConvolve(mode="same")
        self.width_range = width_range
        self.depth_range = depth_range
        self.height_range = height_range
        self.max_order = max_order
        self.absorption = absorption

    def rand_pos_in_room(self, room, device):
        arr = []
        for dim in room:
            rand = torch.distributions.Uniform(
                low=0,
                high=dim,
                validate_args=True,
            )
            arr.append(rand.sample_n(1))
        return torch.tensor(arr, dtype=torch.float32, device=device)

    def randomize_parameters(
        self,
        samples: Optional[Tensor] = None,
        targets=None,
        target_rate: Optional[int] = None,
        sample_rate: Optional[int] = None,
    ):

        width_rand = torch.distributions.Uniform(
            low=torch.tensor(
                self.width_range[0],
                dtype=torch.float32,
                device=samples.device,
            ),
            high=torch.tensor(
                self.width_range[1],
                dtype=torch.float32,
                device=samples.device,
            ),
            validate_args=True,
        )
        depth_rand = torch.distributions.Uniform(
            low=torch.tensor(
                self.depth_range[0],
                dtype=torch.float32,
                device=samples.device,
            ),
            high=torch.tensor(
                self.depth_range[1],
                dtype=torch.float32,
                device=samples.device,
            ),
            validate_args=True,
        )
        height_rand = torch.distributions.Uniform(
            low=torch.tensor(
                self.height_range[0],
                dtype=torch.float32,
                device=samples.device,
            ),
            high=torch.tensor(
                self.height_range[1],
                dtype=torch.float32,
                device=samples.device,
            ),
            validate_args=True,
        )
        room = torch.tensor(
            [width_rand.sample_n(1), depth_rand.sample_n(1), height_rand.sample_n(1)],
            dtype=torch.float32,
            device=samples.device,
        )
        source = self.rand_pos_in_room(room, samples.device)
        mic_pos = self.rand_pos_in_room(room, samples.device)
        mic = mic_pos.unsqueeze(0)
        self.transform_parameters["room"] = room
        self.transform_parameters["source"] = source
        self.transform_parameters["mic"] = mic
        self.transform_parameters["max_order"] = self.max_order

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:

        batch_size, num_channels, num_samples = samples.shape
        # snr = self.transform_parameters["snr_in_db"]
        rir = simulate_rir_ism(
            room=self.transform_parameters["room"],
            source=self.transform_parameters["source"],
            mic_array=self.transform_parameters["mic"],
            max_order=self.transform_parameters["max_order"],
            absorption=self.absorption,
        )
        reverbed_sample = samples.clone()
        for ii in range(samples.shape[0]):
            reverbed_sample[ii] = self.convolve(samples[ii], rir)

        return ObjectDict(
            samples=reverbed_sample,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


if __name__ == "__main__":
    audio = Audio(sample_rate=16000)
    wav = audio("/gpfswork/rech/lpv/uqq71lk/mix_gen/data/sample.wav")
    wav_cutted = torch.stack(
        [wav[:, 600 * 16000 : 605 * 16000], wav[:, 605 * 16000 : 610 * 16000]]
    )

    print(wav_cutted.shape)
    rir = ReverbAugmentation()
    rir.randomize_parameters(wav_cutted, 16000)
    res = rir.apply_transform(wav_cutted)
    print(res["samples"].shape)

    torchaudio.save("test_reverb.wav", res["samples"][0], 16000)
