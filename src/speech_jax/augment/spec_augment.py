from typing import Tuple, Optional

import jax.numpy as jnp
import jax
from pydantic import BaseModel


class SpecAugment(BaseModel):
    mask_prob: float
    mask_span: int
    min_masks: int

    def compute_mask_indices(
        self, shape: Tuple[int, int], rng: jax.DeviceArray, attention_mask: Optional[jnp.DeviceArray] = None
    ):
        spec_augment_mask = jnp.zeros(shape, dtype=jnp.bool)
        bsz, seqlen = shape

        # -> (batch_size, seqlen)
        # [[0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 0]]
        num_spans = max(self.mask_prob * seqlen, self.min_masks)

        jax.random.randint(rng, (bsz, num_spans), 0, seqlen)

        return spec_augment_mask

    def __call__(
        self,
        hidden_states: jnp.DeviceArray,
        target_states: jnp.DeviceArray,
        attention_mask: Optional[jnp.DeviceArray] = None,
    ):
        masked_indices = self.compute_mask_indices(
            hidden_states.shape, attention_mask=attention_mask
        )
        return jnp.where(masked_indices, target_states, hidden_states)


spec_aug = SpecAugment(0.1, 4, 1)
rng = jax.random.PRNGKey(0)
spec_aug.compute_mask_indices((2, 10), rng)









def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[np.ndarray] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        mask_prob:
            probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_length: size of the mask
        min_masks: minimum number of masked spans

    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and"
            f" `sequence_length`: {sequence_length}`"
        )

    # compute number of masked spans in batch
    num_masked_spans = int(mask_prob * sequence_length / mask_length + np.random.rand(1).item())
    num_masked_spans = max(num_masked_spans, min_masks)

    # make sure num masked indices <= sequence_length
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=np.bool)

    # get random indices to mask
    spec_aug_mask_idxs = np.array(
        [
            np.random.choice(np.arange(sequence_length - (mask_length - 1)), num_masked_spans, replace=False)
            for _ in range(batch_size)
        ]
    )

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(spec_aug_mask_idxs[:, :, None], (batch_size, num_masked_spans, mask_length))
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, num_masked_spans * mask_length)

    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, num_masked_spans, mask_length)).reshape(
        batch_size, num_masked_spans * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    if attention_mask is not None:
        # make sure padded input ids cannot be masked
        spec_aug_mask = np.where(attention_mask, spec_aug_mask, False)

    return spec_aug_mask
