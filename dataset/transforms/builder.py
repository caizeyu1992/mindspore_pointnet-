"""3D transforms functions."""

import inspect

import mindspore.dataset.vision.c_transforms as c_trans
import mindspore.dataset.vision.py_transforms as py_trans

from mindvision.engine.class_factory import ClassFactory, ModuleType


def register_builtin_transforms():
    """ register MindSpore builtin dataset class. """
    for module_name in set(dir(c_trans) + dir(py_trans)):
        if module_name.startswith('__'):
            continue
        transforms = getattr(c_trans, module_name) if getattr(c_trans, module_name) \
            else getattr(py_trans, module_name)

        if inspect.isclass(transforms):
            ClassFactory.register_cls(transforms, ModuleType.PIPELINE)
