""" PointNet++ classfication training script."""

import argparse
import os
import importlib
import sys

import mindspore
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from engine.ops.NLLLoss import NLLLoss

from dataset.ModelNet40v1 import ModelNet40Dataset
from engine.callback import ValAccMonitor
from src.model_utils.load_yaml import load_yaml


set_seed(1)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # '/ms3d/example/pointnet2
ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'models')  # '/ms3d/models
sys.path.append(ROOT_DIR)


def pointnet2_train(opt):
    """PointNet++ train."""

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
    dataset = ModelNet40Dataset(root_path=opt['datasets']['train'].get('data_path'),
                                split="train",
                                num_points=opt['datasets']['train'].get('resize'),
                                use_norm=opt['datasets']['train'].get('use_norm'))

    dataset_train = ds.GeneratorDataset(dataset, ["data", "label"], shuffle=True)
    dataset_train = dataset_train.batch(batch_size=opt['datasets']['train'].get('batch_size'), drop_remainder=True)

    dataset = ModelNet40Dataset(root_path=opt['datasets']['val'].get('data_path'),
                                split="val",
                                num_points=opt['datasets']['val'].get('resize'),
                                use_norm=opt['datasets']['val'].get('use_norm'))

    dataset_val = ds.GeneratorDataset(dataset, ["data", "label"], shuffle=True)
    dataset_val = dataset_val.batch(batch_size=opt['datasets']['val'].get('batch_size'), drop_remainder=True)

    step_size = dataset_train.get_dataset_size()

    '''MODEL LOADING'''
    model_file = importlib.import_module("pointnet2")
    model = getattr(model_file, opt['model'])
    # Create model.
    network = model(normal_channel=opt['datasets']['train'].get('use_norm'))

    # load checkpoint
    if opt['train']['pretrained_ckpt'].endswith('.ckpt'):
        print("Load checkpoint: %s" % opt['train']['pretrained_ckpt'])
        param_dict = load_checkpoint(opt['train']['pretrained_ckpt'])
        load_param_into_net(network, param_dict)

    # Set learning rate scheduler.
    if opt['train']['lr_decay_mode'] == "cosine_decay_lr":
        lr = nn.cosine_decay_lr(min_lr=opt['train']['min_lr'],
                                max_lr=opt['train']['max_lr'],
                                total_step=opt['train']['epoch_size'] * step_size,
                                step_per_epoch=step_size,
                                decay_epoch=opt['train']['decay_epoch'])
    elif opt['train']['lr_decay_mode'] == "piecewise_constant_lr":
        lr = nn.piecewise_constant_lr(opt['train']['milestone'], opt['train']['learning_rates'])

    # Define optimizer.
    network_opt = nn.Adam(network.trainable_params(), lr, opt['train']['momentum'])

    # Define loss function.
    network_loss = NLLLoss(reduction="mean")

    # Define metrics.
    metrics = {"Accuracy": nn.Accuracy()}

    # Init the model.
    model = Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(save_checkpoint_steps=step_size,
                                   keep_checkpoint_max=opt['train']['keep_checkpoint_max'])
    ckpt_callback = ModelCheckpoint(prefix='pointnet2_cls',
                                    directory=opt['train']['ckpt_save_dir'],
                                    config=ckpt_config)

    # Begin to train.
    model.train(opt['train']['epoch_size'],
                dataset_train,
                callbacks=[ckpt_callback, ValAccMonitor(model, dataset_val, opt['train']['epoch_size'])],
                dataset_sink_mode=opt['train']['dataset_sink_mode'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointNet train.')
    #parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('-opt', type=str, default="/home/czy/HuaWei/final/ms3d/example/pointnet2/pointnet2_classfication.yaml", help='Path to option YAML file.')
    args = parser.parse_known_args()[0]
    opt = load_yaml(args.opt)
    pointnet2_train(opt)
