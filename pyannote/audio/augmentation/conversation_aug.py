from typing import Optional

import torch
from torch import Tensor
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.io import Audio
from torch_audiomentations.utils.object_dict import ObjectDict
from torchaudio.transforms import Fade


def expand_matrix(input_matrix):
    padded_matrix = torch.nn.functional.pad(
        input_matrix, (1, 1)
    )  # Pad with zeros on both sides
    expanded_matrix = torch.zeros_like(input_matrix)

    # Determine where to add ones based on the position of ones in the input matrix
    expanded_matrix += padded_matrix[:, :-2]  # Add left
    expanded_matrix += padded_matrix[:, 1:-1]  # Add center
    expanded_matrix += padded_matrix[:, 2:]  # Add right

    return expanded_matrix


class ConversationGeneration(BaseWaveformTransform):
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
    requires_target = True

    def __init__(
        self,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: Optional[str] = None,
        sample_rate: Optional[int] = None,
        target_rate: Optional[int] = None,
        max_num_speakers: Optional[int] = None,
        output_type: str = "tensor",
        cross_fade: Optional[float] = None,
        ignore_loss: bool = False,
        simple_concat: bool = True,
        range_seg=(0.2, 0.8),
        range_ov=(0, 0.2),
    ):
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )
        self.cross_fade = cross_fade
        self.range_seg = range_seg
        self.range_ov = range_ov
        self.ignore_loss = ignore_loss
        self.simple_concat = simple_concat

    def randomize_parameters(
        self,
        samples: Optional[Tensor] = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        batch_size, num_channels, num_samples = samples.shape

        batch_size_t, num_channels_t, num_frames, num_speakers_t = targets.shape
        assert (
            batch_size == batch_size_t
        ), "Different batch size between target and signal"
        assert (
            batch_size % 2 == 0
        ), "Need even number of chunk per batch (to keep same file origin policy)"
        # count number of active speakers per sample
        # num_speakers: torch.Tensor = torch.sum(torch.any(targets, dim=-2), dim=-1)

        # randomize index of second sample, constrained by the fact that the
        # resulting mixture should have less than max_num_speakers
        arr = torch.arange(batch_size, dtype=torch.int64)
        arr[::2], arr[1::2] = arr[1::2], arr[::2].clone()
        self.transform_parameters["sample_idx"] = arr

        seg_distribution = torch.distributions.Uniform(
            low=torch.tensor(
                self.range_seg[0],
                dtype=torch.float32,
                device=samples.device,
            ),
            high=torch.tensor(
                self.range_seg[1],
                dtype=torch.float32,
                device=samples.device,
            ),
            validate_args=True,
        )

        ov_distribution = torch.distributions.Uniform(
            low=torch.tensor(
                self.range_ov[0],
                dtype=torch.float32,
                device=samples.device,
            ),
            high=torch.tensor(
                self.range_ov[1],
                dtype=torch.float32,
                device=samples.device,
            ),
            validate_args=True,
        )
        percentage = seg_distribution.sample(
            self.transform_parameters["sample_idx"].shape
        )
        if not self.simple_concat:

            overlaps = ov_distribution.sample(
                self.transform_parameters["sample_idx"].shape
            )
            ov_samples = (overlaps * num_samples).int()
            ov_frames = (overlaps * num_frames).int()
            num_samples_seg1 = (num_samples * percentage).int()
            num_samples_seg2 = torch.clamp(
                num_samples - num_samples_seg1 + ov_samples, None, num_samples
            )
            num_frames_seg1 = (num_frames * percentage).int()
            num_frames_seg2 = torch.clamp(
                num_frames - num_frames_seg1 + ov_frames, None, num_frames
            )
        total = len(self.transform_parameters["sample_idx"])

        matrix_sample = torch.zeros(
            total,
            num_channels,
            num_samples,
            device=samples.device,
        )
        matrix_frame = torch.zeros(
            total,
            num_channels,
            num_frames,
            num_speakers_t,
            device=samples.device,
        )
        # print("matrix_frame", matrix_frame.shape)
        # torch.arange(matrix_sample.size(2)).expand_as(matrix_sample).float()
        # This create a matrix B*C*T with each row spanning from 0 to T
        arr_sample = torch.arange(num_samples).expand_as(matrix_sample).float()
        seg1_mask_sample = ~(arr_sample < num_samples_seg1.view(-1, 1, 1))
        if not self.simple_concat:
            seg2_mask_sample = arr_sample < num_samples_seg2.view(-1, 1, 1)

        trans_matrix_frame = matrix_frame.transpose(2, 3)
        arr_frame = torch.arange(num_frames).expand_as(trans_matrix_frame).float()
        seg1_mask_frame = ~(arr_frame < num_frames_seg1.view(-1, 1, 1, 1)).transpose(
            2, 3
        )
        seg2_mask_frame = (arr_frame < num_frames_seg2.view(-1, 1, 1, 1)).transpose(
            2, 3
        )
        if self.simple_concat:
            self.transform_parameters["sample1"] = seg1_mask_sample
            self.transform_parameters["frame1"] = seg1_mask_frame
            self.transform_parameters["sample2"] = ~seg1_mask_sample
            self.transform_parameters["frame2"] = ~seg1_mask_frame

            loss_mask = torch.zeros_like(matrix_frame)
            index1 = torch.argmax(arr_frame[:, :, ::-1, :][seg1_mask_frame], 2)
            loss_mask1 = loss_mask.clone()
            loss_mask1[index1] = 1
            loss_mask1 = expand_matrix(loss_mask1)
            self.transform_parameters["grey"] = loss_mask1.bool()
        else:
            # Concat with overlap
            self.transform_parameters["sample1"] = seg1_mask_sample
            self.transform_parameters["sample2"] = seg2_mask_sample
            self.transform_parameters["frame1"] = seg1_mask_frame
            self.transform_parameters["frame2"] = seg2_mask_frame

            loss_mask = torch.zeros_like(matrix_frame)
            index1 = torch.argmax(arr_frame[:, :, ::-1, :][seg1_mask_frame], 2)

            index2 = torch.argmax(arr_frame[seg2_mask_frame], 2)
            loss_mask1 = loss_mask.clone()
            loss_mask1[index1] = 1
            loss_mask1 = expand_matrix(loss_mask1)

            loss_mask2 = loss_mask.clone()
            loss_mask2[index2] = 1
            loss_mask2 = expand_matrix(loss_mask2)

            self.transform_parameters["grey1"] = loss_mask1.bool()
            self.transform_parameters["grey2"] = loss_mask2.bool()

    def concat_opti_signal(self, tensor1, tensor2):
        batch, _, total = tensor1.shape
        tensor1_zeroed = tensor1.detach().clone()
        tensor1_zeroed[self.transform_parameters["sample1"]] = 0
        tensor2_zeroed = tensor2.detach().clone()
        tensor2_zeroed[self.transform_parameters["sample2"]] = 0
        if self.cross_fade is not None:
            tensor1_zeroed[~self.transform_parameters["sample1"]] = Fade(
                tensor1_zeroed[~self.transform_parameters["sample1"]], self.cross_fade
            )
            tensor2_zeroed[~self.transform_parameters["sample2"]] = Fade(
                tensor2_zeroed[~self.transform_parameters["sample2"]], self.cross_fade
            )
        signal = (
            tensor1_zeroed
            + tensor2_zeroed
            / torch.clamp(
                ((tensor1_zeroed != 0) + (tensor2_zeroed != 0)).int(),
                1,
                2,  # Normalizing the values in overlap
            ).float()
        )
        return signal

    def concat_opti_targets(self, target1, target2):
        batch, _, total, speakers = target1.shape
        target1_zeroed = target1.detach().clone()
        target1_zeroed[self.transform_parameters["frame1"]] = 0
        target2_zeroed = target2.detach().clone()
        target2_zeroed[self.transform_parameters["frame2"]] = 0
        target = torch.clamp(
            target1_zeroed + target2_zeroed, None, 1
        )  # Case of same speaker occuring twice

        # remove the possible concatenation artefact from the loss
        if self.ignore_loss:
            if not self.simple_concat:
                target[self.transform_parameters["grey1"]] = -1
                target[self.transform_parameters["grey2"]] = -1
            else:
                target[self.transform_parameters["grey"]] = -1
            assert (target < 0).sum() > 0
        return target.byte()

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape
        batch_size, num_channels, num_frames, num_spk = targets.shape
        # snr = self.transform_parameters["snr_in_db"]
        idx = self.transform_parameters["sample_idx"]

        background_samples = Audio.rms_normalize(samples[idx])

        mixed_samples = self.concat_opti_signal(samples, background_samples)

        if targets is None:
            mixed_targets = None

        else:
            background_targets = targets[idx]
            mixed_targets = self.concat_opti_targets(targets, background_targets)
            # mixed_targets = self.concat_target_with_overlap(
            #     targets, background_targets, num_frames_seg1, num_frames_seg2
            # )

        return ObjectDict(
            samples=mixed_samples,
            sample_rate=sample_rate,
            targets=mixed_targets,
            target_rate=target_rate,
        )
