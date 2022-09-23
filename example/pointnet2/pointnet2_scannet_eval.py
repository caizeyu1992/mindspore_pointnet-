""" PointNet++ segmentation eval script."""

import argparse
import sys
import os
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

from engine.ops.WeightedNLLLoss import WeightedNLLLoss
from dataset.scannet import ScannetDatasetWholeScene
from src.model_utils.load_yaml import load_yaml



set_seed(1)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # '/ms3d/example/pointnet2
ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'models')  # '/ms3d/models
sys.path.append(ROOT_DIR)


def filter_points(coords, preds, targets, weights):
    """filter points"""
    assert coords.shape[0] == preds.shape[0] == targets.shape[0] == weights.shape[0]
    coord_hash = [hash(str(coords[point_idx][0]) + str(coords[point_idx][1]) + str(coords[point_idx][2])) for
                  point_idx in range(coords.shape[0])]
    _, coord_ids = np.unique(np.array(coord_hash), return_index=True)
    coord_filtered, pred_filtered, target_filtered, weight_filtered = coords[coord_ids], preds[coord_ids], targets[
        coord_ids], weights[coord_ids]

    return coord_filtered, pred_filtered, target_filtered, weight_filtered


def point_cloud_label_to_surface_voxel_label_fast(point_cloud, label, res=0.0484):
    """Convert point cloud label to surface voxel label"""
    coordmax = np.max(point_cloud, axis=0)
    coordmin = np.min(point_cloud, axis=0)
    nvox = np.ceil((coordmax - coordmin) / res)
    vidx = np.ceil((point_cloud - coordmin) / res)
    vidx = vidx[:, 0] + vidx[:, 1] * nvox[0] + vidx[:, 2] * nvox[0] * nvox[1]
    uvidx, vpidx = np.unique(vidx, return_index=True)
    if label.ndim == 1:
        uvlabel = label[vpidx]
    else:
        assert label.ndim == 2
    uvlabel = label[vpidx, :]
    return uvidx, uvlabel, nvox


def compute_acc(coords, preds, targets, weights, num_classes):
    """compute acc"""
    coords, preds, targets, weights = filter_points(coords, preds, targets, weights)

    seen_classes = np.unique(targets)
    mask = np.zeros(num_classes)
    mask[seen_classes] = 1

    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(num_classes)]
    total_correct_class = [0 for _ in range(num_classes)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(num_classes)]
    total_correct_class_vox = [0 for _ in range(num_classes)]

    labelweights = np.zeros(num_classes)
    labelweights_vox = np.zeros(num_classes)

    correct = np.sum(preds == targets)  # evaluate only on 20 categories but not unknown
    total_correct += correct
    total_seen += targets.shape[0]
    tmp, _ = np.histogram(targets, range(num_classes + 1))
    labelweights += tmp
    for l in seen_classes:
        total_seen_class[l] += np.sum(targets == l)
        total_correct_class[l] += np.sum((preds == l) & (targets == l))

    _, uvlabel, _ = point_cloud_label_to_surface_voxel_label_fast(coords, np.concatenate(
        (np.expand_dims(targets, 1), np.expand_dims(preds, 1)), axis=1), res=0.02)
    total_correct_vox += np.sum(uvlabel[:, 0] == uvlabel[:, 1])
    total_seen_vox += uvlabel[:, 0].shape[0]
    tmp, _ = np.histogram(uvlabel[:, 0], range(num_classes + 1))
    labelweights_vox += tmp
    for l in seen_classes:
        total_seen_class_vox[l] += np.sum(uvlabel[:, 0] == l)
        total_correct_class_vox[l] += np.sum((uvlabel[:, 0] == l) & (uvlabel[:, 1] == l))

    pointacc = total_correct / float(total_seen)
    voxacc = total_correct_vox / float(total_seen_vox)

    labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
    labelweights_vox = labelweights_vox.astype(np.float32) / np.sum(labelweights_vox.astype(np.float32))
    caliweights = labelweights_vox
    voxcaliacc = np.average(
        np.array(total_correct_class_vox) / (np.array(total_seen_class_vox, dtype=np.float64) + 1e-8),
        weights=caliweights)

    pointacc_per_class = np.zeros(num_classes)
    voxacc_per_class = np.zeros(num_classes)
    for l in seen_classes:
        pointacc_per_class[l] = total_correct_class[l] / (total_seen_class[l] + 1e-8)
        voxacc_per_class[l] = total_correct_class_vox[l] / (total_seen_class_vox[l] + 1e-8)

    return pointacc, pointacc_per_class, voxacc, voxacc_per_class, voxcaliacc, mask


def compute_miou(coords, preds, targets, weights, num_classes):
    """compute miou"""
    coords, preds, targets, weights = filter_points(coords, preds, targets, weights)
    seen_classes = np.unique(targets)
    mask = np.zeros(num_classes)
    mask[seen_classes] = 1

    pointmiou = np.zeros(num_classes)
    voxmiou = np.zeros(num_classes)

    uvidx, uvlabel, _ = point_cloud_label_to_surface_voxel_label_fast(coords, np.concatenate(
        (np.expand_dims(targets, 1), np.expand_dims(preds, 1)), axis=1), res=0.02)
    for l in seen_classes:
        target_label = np.arange(targets.shape[0])[targets == l]
        pred_label = np.arange(preds.shape[0])[preds == l]
        num_intersection_label = np.intersect1d(pred_label, target_label).shape[0]
        num_union_label = np.union1d(pred_label, target_label).shape[0]
        pointmiou[l] = num_intersection_label / (num_union_label + 1e-8)

        target_label_vox = uvidx[(uvlabel[:, 0] == l)]
        pred_label_vox = uvidx[(uvlabel[:, 1] == l)]
        num_intersection_label_vox = np.intersect1d(pred_label_vox, target_label_vox).shape[0]
        num_union_label_vox = np.union1d(pred_label_vox, target_label_vox).shape[0]
        voxmiou[l] = num_intersection_label_vox / (num_union_label_vox + 1e-8)

    return pointmiou, voxmiou, mask


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


def miou_eval(model, test_ds, test_steps_per_epoch, batch_size, num_classes):
    """eval"""
    print("evaluating...")
    pointacc_list = []
    pointacc_per_class_array = np.zeros((test_steps_per_epoch, num_classes))
    voxacc_list = []
    voxacc_per_class_array = np.zeros((test_steps_per_epoch, num_classes))
    voxcaliacc_list = []
    pointmiou_per_class_array = np.zeros((test_steps_per_epoch, num_classes))
    voxmiou_per_class_array = np.zeros((test_steps_per_epoch, num_classes))
    masks = np.zeros((test_steps_per_epoch, num_classes))

    # iter
    for load_idx, data in tqdm(enumerate(test_ds.create_dict_iterator(), 0), total=test_steps_per_epoch, smoothing=0.9):
        # feed
        coords, targets, weights = data["data"], data["label"], data["weights"]
        pred = []
        n, _, chan = coords.shape
        if n > batch_size:
            for i in range((n - 1) // batch_size + 1):
                if (i + 1) * batch_size <= n:
                    coord = coords[i * batch_size:(i + 1) * batch_size, :, :]
                else:
                    coord = coords[i * batch_size:, :, :]
                coord = ops.Transpose()(coord, (0, 2, 1)).astype("float32")
                output = model(coord)
                pred.append(output)
        else:
            coord = coords
            coord = ops.Transpose()(coord, (0, 2, 1)).astype("float32")
            output = model(coord)
            pred.append(output)

        x = ops.Concat(0)(pred)
        pred = ops.ExpandDims()(x, 0)  # (1, CK, N, C)
        preds = pred.argmax(3)

        # eval
        coords = coords.view(-1, chan).asnumpy()  # (CK*N, C)
        preds = preds.squeeze(0).view(-1).asnumpy()  # (CK*N, C)
        targets = targets.view(-1).asnumpy()  # (CK*N, C)
        weights = weights.view(-1).asnumpy()  # (CK*N, C)
        pointacc, pointacc_per_class, voxacc, voxacc_per_class, voxcaliacc, acc_mask = compute_acc(coords, preds,
                                                                                                   targets, weights,
                                                                                                   num_classes)
        pointmiou, voxmiou, miou_mask = compute_miou(coords, preds, targets, weights, num_classes)
        assert acc_mask.all() == miou_mask.all()
        mask = acc_mask

        # dump
        pointacc_list.append(pointacc)
        pointacc_per_class_array[load_idx] = pointacc_per_class
        voxacc_list.append(voxacc)
        voxacc_per_class_array[load_idx] = voxacc_per_class
        voxcaliacc_list.append(voxcaliacc)
        pointmiou_per_class_array[load_idx] = pointmiou
        voxmiou_per_class_array[load_idx] = voxmiou
        masks[load_idx] = mask

    avg_pointacc = np.mean(pointacc_list)
    avg_pointacc_per_class = np.sum(pointacc_per_class_array * masks, axis=0) / np.sum(masks, axis=0)

    avg_voxacc = np.mean(voxacc_list)
    avg_voxacc_per_class = np.sum(voxacc_per_class_array * masks, axis=0) / np.sum(masks, axis=0)

    avg_voxcaliacc = np.mean(voxcaliacc_list)

    avg_pointmiou_per_class = np.sum(pointmiou_per_class_array * masks, axis=0) / np.sum(masks, axis=0)
    avg_pointmiou = np.mean(avg_pointmiou_per_class)

    avg_voxmiou_per_class = np.sum(voxmiou_per_class_array * masks, axis=0) / np.sum(masks, axis=0)
    avg_voxmiou = np.mean(avg_voxmiou_per_class)

    return avg_pointacc, avg_pointacc_per_class, avg_voxacc, avg_voxacc_per_class, avg_voxcaliacc, \
           avg_pointmiou_per_class, avg_pointmiou, avg_voxmiou_per_class, avg_voxmiou


def pointnet2_eval(opt):
    """PointNet++ eval."""
    # log
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if opt['tag']: stamp += "_" + opt['tag'].upper()
    output_root = os.path.join(opt['root'], "outputs")
    root = os.path.join(output_root, stamp)
    os.makedirs(root, exist_ok=True)
    logger = log_string(os.path.join(root, "exp_eval.log"))

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

    testdata = ScannetDatasetWholeScene(path=opt['datasets']['val'].get('data_path'), phase='val', is_weighting=not opt['datasets']['val'].get('use_no_weighting'),
                                        use_color=opt['datasets']['val'].get('use_color'), use_normal=opt['datasets']['val'].get('use_normal'))
    test_ds = ds.GeneratorDataset(testdata, ["data", "label", "weights"], num_parallel_workers=1, shuffle=True)

    test_steps_per_epoch = test_ds.get_dataset_size()

    logger.info("initializing...")


    '''MODEL LOADING'''
    model_file = importlib.import_module("pointnet2")
    Pointnet2segModel = getattr(model_file, opt['model'])
    # Create model.
    model = Pointnet2segModel(num_classes=opt['datasets']['val'].get('num_classes'), use_color=opt['datasets']['val'].get('use_color'),
                              use_normal=opt['datasets']['val'].get('use_normal'))
    model.set_train()
    # load pretrained ckpt
    param_dict = load_checkpoint(opt['val']['pretrained_ckpt'])
    load_param_into_net(model, param_dict)


    # eval
    avg_pointacc, avg_pointacc_per_class, avg_voxacc, avg_voxacc_per_class, avg_voxcaliacc, avg_pointmiou_per_class, \
    avg_pointmiou, avg_voxmiou_per_class, avg_voxmiou = miou_eval(model, test_ds, test_steps_per_epoch, batch_size,
                                                                  opt['datasets']['val'].get('num_classes'))

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
    for l in range(opt['datasets']['val'].get('num_classes')):
        logger.info(
            "Class %s: %f / %f / %f / %f", opt['NYUCLASSES'][l], avg_pointacc_per_class[l], avg_voxacc_per_class[l],
            avg_pointmiou_per_class[l], avg_voxmiou_per_class[l])

    logger.info('-------------------------------------------------------------------------------------------')
    print("eval completed...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointNet++ segmentation train.')
    #parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('-opt', type=str, default="/home/czy/HuaWei/final/ms3d/example/pointnet2/pointnet2_scannet.yaml", help='Path to option YAML file.')
    args = parser.parse_known_args()[0]
    opt = load_yaml(args.opt)
    pointnet2_eval(opt)
