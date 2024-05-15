import math
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.common.tensor import Tensor

def _get_warmup_steps(warmup_steps: int, warmup_ratio: float, total_steps: int):
    """check warmup args and get warmup steps."""
    if warmup_ratio is None:
        if not isinstance(warmup_steps, int):
            raise TypeError(f"The type of warmup_steps must be int, but got {type(warmup_steps)}")
        if warmup_steps < 0:
            raise ValueError(f"Warmup_steps must be >= 0, but got {warmup_steps}")
        return warmup_steps

    if not isinstance(warmup_ratio, float):
        raise TypeError(f"The type of warmup_ratio must be float, but got {type(warmup_ratio)}")

    if warmup_ratio > 1.0 or warmup_ratio < 0.0:
        raise ValueError(f"Warmup_ratio's value range must be in [0,1], but got {warmup_ratio}")

    if total_steps is None:
        raise ValueError(f"When warmup_ratio takes effect, total_steps must be set, but got {total_steps} ")
    if not isinstance(total_steps, int):
        raise TypeError(f"The type of total_steps must be int, but got {type(total_steps)}")

    warmup_steps = int(total_steps * warmup_ratio)
    return warmup_steps


class LinearWithWarmUpLR(LearningRateSchedule):
    """
    Linear with Warm Up Learning Rate.

    Args:
        learning_rate (`float`):
            Initial value of learning rate.
        total_steps (`int`):
            The number of total steps.
        warmup_steps (`int`, *optional*, defaults to None):
            The number of warm up steps.
        warmup_lr_init (`float`, *optional*, defaults to 0.):
            Initial learning rate in warm up steps.
        warmup_ratio (`float`, *optional*, defaults to None):
            Ratio of total training steps used for warmup.

    Returns:
        Class, LinearWithWarmUpLR
    """

    def __init__(self, learning_rate: float, total_steps: int, warmup_steps: int = None,
                 warmup_lr_init: float = 0., warmup_ratio: float = None,
                 **kwargs):
        super(LinearWithWarmUpLR, self).__init__()
        warmup_steps = _get_warmup_steps(warmup_steps, warmup_ratio, total_steps)
        linear_steps = max(1, total_steps - warmup_steps)
        self.kwargs = kwargs
        self.learning_rate = learning_rate
        self.warmup_lr_init = warmup_lr_init
        self.total_steps = Tensor(total_steps, mstype.float32)
        self.warmup_steps = Tensor(warmup_steps, mstype.float32)
        self.linear_steps = Tensor(linear_steps, mstype.float32)
        self.greater = P.Greater()
        self.max = P.Maximum()
        self.zero_constant = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()

    def construct(self, global_step):
        """compute current step lr."""
        global_step = self.cast(global_step, mstype.float32)
        if self.greater(self.warmup_steps, global_step):
            percent = global_step / self.warmup_steps
            learning_rate = self.warmup_lr_init + self.learning_rate * percent
        else:
            percent = self.max(self.zero_constant, (self.total_steps - global_step) / self.linear_steps)
            learning_rate = self.learning_rate * percent
        return learning_rate


if __name__=='__main__':
    lr_scheluer = LinearWithWarmUpLR(learning_rate=0.1, warmup_ratio=0.1, total_steps=20)
    for step in range(20):
        print(step, lr_scheluer(step))