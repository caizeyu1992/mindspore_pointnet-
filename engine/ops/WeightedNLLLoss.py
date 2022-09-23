"""WeightedNLLLoss"""

from mindspore.nn.loss.loss import LossBase
import mindspore.ops as ops
from mindspore.ops import functional as F
import mindspore.nn as nn


class WeightedNLLLoss(LossBase):
    """WeightedNLLLoss"""

    def __init__(self, reduction="mean"):
        super(WeightedNLLLoss, self).__init__(reduction)
        self.one_hot = ops.OneHot()
        self.reduce_sum = ops.ReduceSum()
        self.net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, logits, labels, weights):
        """WeightedNLLLoss construct."""
        logits = ops.LogSoftmax(1)(logits)
        label_one_hot = self.one_hot(labels, F.shape(logits)[-1], F.scalar_to_array(1.0), F.scalar_to_array(0.0))
        loss = self.reduce_sum(-1.0 * logits * label_one_hot, (1,))
        return self.get_loss(loss, weights)
