"""BaseClassifier"""

from mindspore import nn

from mindvision.classification.models.builder import build_backbone, build_neck, build_head
from mindvision.engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.RECOGNIZER)
class BaseRecognizer(nn.Cell):
    """
    Generate recognizer for video recogniztion task.
    """

    def __init__(self, backbone, neck=None, head=None):
        super(BaseRecognizer, self).__init__()
        self.backbone = build_backbone(backbone) if isinstance(backbone, dict) else backbone
        if neck:
            self.neck = build_neck(neck) if isinstance(neck, dict) else neck
            self.with_neck = True
        else:
            self.with_neck = False
        if head:
            self.head = build_head(head) if isinstance(head, dict) else head
            self.with_head = True
        else:
            self.with_head = False

    def construct(self, x):
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        if self.with_head:
            x = self.head(x)
        return x
