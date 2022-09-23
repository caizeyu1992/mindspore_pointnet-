""" PointNet++ classfication eval script. """

import argparse
import os
import sys
import importlib

import mindspore
import mindspore.nn as nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.train import Model
import mindspore.dataset as ds

from dataset.ModelNet40v1 import ModelNet40Dataset
from engine.ops.NLLLoss import NLLLoss
from src.model_utils.load_yaml import load_yaml

set_seed(1)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # '/ms3d/example/pointnet2
ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'models')  # '/ms3d/models
sys.path.append(ROOT_DIR)


def pointnet2_eval(opt):
    """PointNet++ eval."""

    # device.
    device_id = int(os.getenv('DEVICE_ID', '0'))
    device_num = int(os.getenv('RANK_SIZE', '1'))

    if not opt['device_target'] in ("Ascend", "GPU"):
        raise ValueError("Unsupported platform {}".format(opt['device_target']))

    if opt['device_target'] == "Ascend":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="Ascend",
                            save_graphs=False,
                            device_id=device_id)
        context.set_context(max_call_depth=2048)

    else:
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="GPU",
                            save_graphs=False,
                            max_call_depth=2048)

    # run distribute.
    if opt['run_distribute']:
        if opt['device_target'] == "Ascend":
            if device_num > 1:
                init()
                context.set_auto_parallel_context(
                    parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        else:
            if device_num > 1:
                mindspore.dataset.config.set_enable_shared_mem(False)
                context.set_auto_parallel_context(
                    parallel_mode=context.ParallelMode.DATA_PARALLEL,
                    gradients_mean=True,
                    device_num=device_num)
                mindspore.common.set_seed(1234)
                init()
            else:
                context.set_context(device_id=device_id)

    # Data Pipeline.
    dataset = ModelNet40Dataset(root_path=opt['datasets']['val'].get('data_path'),
                                split="val",
                                num_points=opt['datasets']['val'].get('resize'),
                                use_norm=opt['datasets']['val'].get('use_norm'))

    dataset_val = ds.GeneratorDataset(dataset, ["data", "label"], shuffle=False)
    dataset_val = dataset_val.batch(batch_size=opt['datasets']['val'].get('batch_size'), drop_remainder=True)

    '''MODEL LOADING'''
    model_file = importlib.import_module("pointnet2")
    model = getattr(model_file, opt['model'])
    # Create model.
    network = model(normal_channel=opt['datasets']['train'].get('use_norm'))

    # Load param.
    param_dict = load_checkpoint(opt['val']['pretrained_ckpt'])
    load_param_into_net(network, param_dict)

    # Define loss function.
    network_loss = NLLLoss(reduction="mean")

    # Define metrics.
    metrics = {"Accuracy": nn.Accuracy()}

    # Init the model.
    model = Model(network, loss_fn=network_loss, metrics=metrics)

    # Begin to eval.
    result = model.eval(dataset_val, dataset_sink_mode=True)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointNet2 eval.')
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_known_args()[0]
    opt = load_yaml(args.opt)
    pointnet2_eval(opt)
