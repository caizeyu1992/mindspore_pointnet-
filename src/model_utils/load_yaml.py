
"""      Load Yaml    """

import yaml
from collections import OrderedDict


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def load_yaml(yaml_path):
    """Load yaml.
    Input:
        yaml path.
    Returns:
        parameters from yaml.
    """
    with open(yaml_path, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    return opt
