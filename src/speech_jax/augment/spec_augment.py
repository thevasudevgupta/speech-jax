from pydantic import BaseModel
from typing import Tuple
import jax.numpy as jnp

class SpecAugment(BaseModel):
    mask_prob: float
    mask_span: int
    min_masks: int

    def compute_mask_indices(self, shape: Tuple[int, int], attention_mask: Optional[jnp.DeviceArray] = None):
        spec_augment_mask = jnp.zeros(shape, dtype=jnp.bool)
        
        # batch_size, seqlen = shape
        return spec_augment_mask

    def __call__(
        self,
        hidden_states: jnp.DeviceArray,
        target_states: jnp.DeviceArray,
        attention_mask: Optional[jnp.DeviceArray] = None
    ):
        masked_indices = self.compute_mask_indices(hidden_states.shape, attention_mask=attention_mask)
        return jnp.where(masked_indices, target_states, hidden_states)
