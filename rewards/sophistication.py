import torch

from rewards.base import Model, Reward
from two_part_codes.base import TwoPartCode


class SophisticationReward(Reward):

    def __init__(
        self,
        two_part_code: TwoPartCode[Model],
        normalize_reward: bool = False,
        normalization_quantile: float = 0.95,
        obj_shape: tuple[int, ...] | None = None,
        obj_num_symbols: int | None = None,
    ) -> None:
        """Reward defined from my compositionality metric applied on a recursive decomposition of the object.

        Args:
            two_part_code (TwoPartCode[Model]): Two-part-code model fitting `m_x` from `x`.
            normalize_reward (bool, optional): Whether or not to subtract the reward of a typical random object. Defaults to False.
            normalization_quantile (float, optional): Which quantile of the random object reward to use for normalization. Defaults to 0.95.
            obj_shape (tuple[int, ...] | None, optional): The shape of the object (excluding batch dimension), necessary when using reward normalization. Defaults to None.
            obj_num_symbols (int | None, optional): Number of symbols in the object, necessary when using reward normalization. Defaults to None.
        """
        super().__init__()

        self.two_part_code = two_part_code

        self.random_obj_reward = 0.0
        if normalize_reward:
            assert (obj_shape is not None) and (
                obj_num_symbols is not None
            ), "To use centered rewards, the shape and number of symbols of the objects must be provided."
            self.random_obj_reward = (
                self(torch.randint(0, obj_num_symbols, (1000, *obj_shape)))
                .quantile(normalization_quantile)
                .item()
            )

    def __call__(
        self,
        obj: torch.Tensor,
        return_models: bool = False,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, Model]:
        models, complexities, _ = self.two_part_code.compress(obj)
        if return_models:
            return complexities, models
        return complexities
