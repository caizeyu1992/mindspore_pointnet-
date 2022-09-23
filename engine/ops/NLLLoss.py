"""NLLLoss"""

from mindspore.nn.loss.loss import LossBase
from mindspore import ops
from mindspore.ops import functional as F
from engine.class_factory import ClassFactory, ModuleType

__all__ = ['NLLLoss']


@ClassFactory.register(ModuleType.LOSS)
class NLLLoss(LossBase):
    """NLLLoss"""

    def __init__(self, reduction="mean"):
        super(NLLLoss, self).__init__(reduction)
        self.one_hot = ops.OneHot()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, logits, labels):
        """NLLLoss construct."""
        label_one_hot = self.one_hot(labels, F.shape(logits)[-1], F.scalar_to_array(1.0), F.scalar_to_array(0.0))
        loss = self.reduce_sum(-1.0 * logits * label_one_hot, (1,))
        return self.get_loss(loss)
