""" PointNet++ segmentation training script."""

import argparse
import os
import sys
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm
import importlib

import mindspore
import mindspore.nn as nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.common import set_seed
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import save_checkpoint

from dataset.scannet import ScannetDataset, ScannetDatasetWholeScene
from engine.ops.WeightedNLLLoss import WeightedNLLLoss
from example.pointnet2.pointnet2_scannet_eval import miou_eval
from src.model_utils.load_yaml import load_yaml


set_seed(1)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # '/ms3d/example/pointnet2
ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'models')  # '/ms3d/models
sys.path.append(ROOT_DIR)


class CustomWithLossCell(nn.Cell):
    """连接前向网络和损失函数"""

    def __init__(self, backbone, loss_fn):
        """输入有两个，前向网络backbone和损失函数loss_fn"""
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label, weights):
        "net construct"
        output = self._backbone(data)  # 前向计算得到网络输出
        return self._loss_fn(ops.Reshape()(output, (-1, 20)), label.view(-1), weights.view(-1))


def log_string(filename, verbosity=1, name=None):
    """log init"""
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def pointnet2_train(opt):
    """PointNet++ train."""
    # log
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if opt['tag']: stamp += "_" + opt['tag'].upper()
    output_root = os.path.join(opt['root'], "outputs")
    root = os.path.join(output_root, stamp)
    os.makedirs(root, exist_ok=True)
    logger = log_string(os.path.join(root, "exp.log"))


    # device.
    device_id = int(os.getenv('DEVICE_ID', '0'))
    device_num = int(os.getenv('RANK_SIZE', '1'))

    if not opt['device_target'] in ("Ascend", "GPU"):
        raise ValueError("Unsupported platform {}".format(opt['device_target']))

    if opt['device_target'] == "Ascend":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="Ascend",
                            device_id=device_id)
        context.set_context(max_call_depth=20480)

    else:
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="GPU",
                            max_call_depth=20480)

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
    # batch size
    batch_size = opt['batch']

    # Data Pipeline.
    logger.info("preparing data...")
    traindata = ScannetDataset(phase='train', path=opt['datasets']['train'].get('data_path'), is_weighting=not opt['datasets']['train'].get('use_no_weighting'),
                               use_color=opt['datasets']['train'].get('use_color'), use_normal=opt['datasets']['train'].get('use_normal'))
    train_ds = ds.GeneratorDataset(traindata, ["data", "label", "weights"], num_parallel_workers=1, shuffle=True)
    train_ds = train_ds.batch(batch_size=batch_size)

    testdata = ScannetDatasetWholeScene(path=opt['datasets']['val'].get('data_path'), phase='val', is_weighting=not opt['datasets']['val'].get('use_no_weighting'),
                                        use_color=opt['datasets']['val'].get('use_color'), use_normal=opt['datasets']['val'].get('use_normal'))
    test_ds = ds.GeneratorDataset(testdata, ["data", "label", "weights"], num_parallel_workers=1, shuffle=True)

    steps_per_epoch = train_ds.get_dataset_size()
    step_size = steps_per_epoch
    test_steps_per_epoch = test_ds.get_dataset_size()

    logger.info("initializing...")

    '''MODEL LOADING'''
    model_file = importlib.import_module("pointnet2")
    Pointnet2segModel = getattr(model_file, opt['model'])
    # Create model.
    model = Pointnet2segModel(num_classes=opt['datasets']['train'].get('num_classes'), use_color=opt['datasets']['train'].get('use_color'),
                              use_normal=opt['datasets']['train'].get('use_normal'))

    # load pretrained ckpt
    if opt['train']['pretrained_ckpt'].endswith('.ckpt'):
        try:
            param_dict = load_checkpoint(opt['train']['pretrained_ckpt'])
            load_param_into_net(model, param_dict)
        except RuntimeError:
            logger.info("error in loading pretrained ckpt")

    # Set learning rate scheduler.
    if opt['train']['lr_decay_mode'] == "cosine_decay_lr":
        lr = nn.cosine_decay_lr(min_lr=opt['train']['min_lr'],
                                max_lr=opt['train']['max_lr'],
                                total_step=opt['epoch'] * step_size,
                                step_per_epoch=step_size,
                                decay_epoch=opt['train']['decay_epoch'])
    elif opt['train']['lr_decay_mode'] == "piecewise_constant_lr":
        lr = nn.piecewise_constant_lr(opt['train']['milestone'], opt['train']['learning_rates'])

    # Define optimizer.
    network_opt = nn.Adam(model.trainable_params(), lr, opt['train']['momentum'])

    # Define loss function.
    criterion = WeightedNLLLoss(reduction="mean")

    net_with_loss = CustomWithLossCell(model, criterion)
    net_train = nn.TrainOneStepCell(net_with_loss, network_opt)
    net_train.set_train()

    best_iou = 0
    global_iter_id = 0
    epoch = opt['epoch']

    for epoch_id in range(epoch):
        logger.info("epoch %d starting...", (epoch_id + 1))
        loss_list = []
        # generate new chunks
        traindata.generate_new_datas()
        net_train.set_train(True)
        # train
        for iter_id, data in tqdm(enumerate(train_ds.create_dict_iterator(), 0), total=steps_per_epoch, smoothing=0.9):
            # unpack the data
            coords, label, weights = data["data"], data["label"], data["weights"]
            coords = ops.Transpose()(coords, (0, 2, 1)).astype("float32")

            # forward
            loss = net_train(coords, label, weights)
            loss_num = loss.asnumpy()
            loss_list.append(loss_num)

            if (iter_id + 1) % 75 == 0:
                # print report
                logger.info("train")
                logger.info("global_iter_id: %d", global_iter_id)
                logger.info("train_loss: %f", np.mean(np.array(loss_list)))
                loss_list = []

            # update the _global_iter_id
            global_iter_id += 1

        # eval
        avg_pointacc, avg_pointacc_per_class, avg_voxacc, avg_voxacc_per_class, avg_voxcaliacc, \
        avg_pointmiou_per_class, avg_pointmiou, avg_voxmiou_per_class, avg_voxmiou = miou_eval(model, test_ds,
                                                                                               test_steps_per_epoch,
                                                                                               batch_size, opt['datasets']['train'].get('num_classes'))

        # report
        logger.info('-------------------------------------------------------------------------------------------')
        logger.info("Point accuracy: %f", avg_pointacc)
        logger.info("Point accuracy per class: %f", np.mean(avg_pointacc_per_class))
        logger.info("Voxel accuracy: %f", avg_voxacc)
        logger.info("Voxel accuracy per class: %f", np.mean(avg_voxacc_per_class))
        logger.info("Calibrated voxel accuracy: %f", avg_voxcaliacc)
        logger.info("Point miou: %f", avg_pointmiou)
        logger.info("Voxel miou: %f", avg_voxmiou)
        logger.info('--------------------------------------')
        logger.info("Point acc/voxel acc/point miou/voxel miou per class:")
        for l in range(opt['datasets']['train'].get('num_classes')):
            logger.info(
                "Class %s: %f / %f / %f / %f", opt['NYUCLASSES'][l], avg_pointacc_per_class[l], avg_voxacc_per_class[l],
                avg_pointmiou_per_class[l], avg_voxmiou_per_class[l])

        logger.info('-------------------------------------------------------------------------------------------')
        # checkpoint
        if avg_voxmiou >= best_iou:
            best_iou = avg_voxmiou
            logger.info('Save model...')
            savepath = os.path.join(root, "best_iou_model.ckpt")
            save_checkpoint(model, savepath)  # 保存模型
            logger.info('Saving best iou model....')
        logger.info('Best iou: %f', best_iou)

    print("training completed...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointNet++ segmentation train.')
    #parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('-opt', type=str, default="/home/czy/HuaWei/final/ms3d/example/pointnet2/pointnet2_scannet.yaml", help='Path to option YAML file.')
    args = parser.parse_known_args()[0]
    opt = load_yaml(args.opt)
    pointnet2_train(opt)
