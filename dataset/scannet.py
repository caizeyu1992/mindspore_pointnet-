""" Load Scannet dataset."""

import os
import numpy as np
from tqdm import tqdm

__all__ = ["ScannetDataset", "ScannetDatasetWholeScene"]


class ScannetDataset():
    """A source dataset that reads, parses and augments the Scannet dataset.

        Args:
            path (str): The root directory of the Scannet dataset or inference pointcloud.
            phase (str): The dataset split, supports "train", "val", or "test". Default: "train".
            num_classes (int): Number of classes.Default: 21.
            npoints (int): Number of points selected from each scene.Default: 8192.
            is_weighting (bool): Whether or not to use the weights of the number of points distributed in
                        different classes. Default: True. if dataset choose 'val' ,the weights are all ones.

            use_color (bool): Whether or not to use the colors of all points. Default: False.
            use_normal (bool): Whether or not to use the normals of all points. Default: False.

        Raises:
            ValueError: If `split` is not 'train', 'test' or 'test'.

        Examples:

            >> import mindspore.dataset as ds
            >> from mindvision.ms3d.dataset.scannet import ScannetDataset
            >> root = './ScanNet'
            >> traindata = ScannetDataset(phase='train', path=root, num_classes=21, is_weighting=not False,
                                                                        use_color=False, use_normal=False)
            >> train_ds = ds.GeneratorDataset(traindata, ["data", "label", "weights"],
                                                                    num_parallel_workers=1, shuffle=True)
            >> for data in train_ds.create_dict_iterator():
                    print(data["data"].shape, data["label"].shape, data["weights"].shape)

        About Scannet dataset:
        ScanNet is an RGB-D video dataset containing 2.5 million views in more than 1500 scans, annotated with
                    3D camera poses, surface reconstructions, and instance-level semantic segmentations.

        Original dataset website: <a href="http://www.scan-net.org/">http://www.scan-net.org/</a>

        note: If you would like to download the ScanNet data, please fill out an agreement to the
        ScanNet Terms of Use<a href="http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf"> and send
        it to Scannet-org at scannet@googlegroups.com.


        .. code-block::

            ./ScanNet/
            ├── data
            │   ├── scannetv2.txt
            │   ├── scannetv2_train.txt
            │   ├── scannetv2_val.txt
            │   └── scannetv2_test.txt
            ├── preprocessing
            │   ├── points
            │   ├── scannet_scenes
            │       ├── scene0000_00.npy
            │       ├── scene0000_01.npy
            │       └── ....
            │   ├── collect_scannet_scenes.py
            │   └── scannetv2-labels.combined.tsv
            ├── scannet
            │   ├── scene0000_00
            │       ├── scene0000_00.sens
            │       ├── scene0000_00_vh_clean.ply
            │       ├── scene0000_00_vh_clean_2.ply
            │       ├── scene0000_00_vh_clean_2.0.010000.segs.json
            │       ├── scene0000_00.aggregation.json
            │       ├── scene0000_00_vh_clean.aggregation.json
            │       ├── scene0000_00_vh_clean_2.0.010000.segs.json
            │       ├── scene0000_00_vh_clean.segs.json
            │       ├── scene0000_00_vh_clean_2.labels.ply
            │       ├── scene0000_00_2d-label.zip
            │       ├── scene0000_00_2d-instance.zip
            │       ├── scene0000_00_2d-label-filt.zip
            │       └──scene0000_00_2d-instance-filt.zip
                ├── scene0000_01
                └── ....
                ├── train_test_split
                └── synsetoffset2category.txt

        Citation:

        .. code-block::

            @inproceedings{dai2017scannet,
                title={ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes},
                author={Dai, Angela and Chang, Angel X. and Savva, Manolis and Halber, Maciej and Funkhouser,
                                                                            Thomas and Nie{ss}ner, Matthias},
                booktitle = {Proc. Computer Vision and Pattern Recognition (CVPR), IEEE},
                year = {2017}
            }
        """

    def __init__(self, phase, path, num_classes=21, npoints=8192, is_weighting=True, use_color=False, use_normal=False):
        assert phase in ["train", "val", "test"]
        self.phase = phase
        self.root = path
        self.preprocess_path = os.path.join(self.root, "preprocessing")
        self.scannet_datapath = os.path.join(self.preprocess_path, "scannet_scenes")

        valpath = os.path.join(self.root, "data/scannetv2_val.txt")
        trainpath = os.path.join(self.root, "data/scannetv2_train.txt")
        scene_list = []
        if phase == 'train':
            list_path = trainpath
        else:
            list_path = valpath
        with open(list_path) as f:
            for scene_id in f.readlines():
                scene_list.append(scene_id.strip())

        self.scene_list = scene_list
        self.num_classes = num_classes
        self.npoints = npoints
        self.is_weighting = is_weighting
        self.use_color = use_color
        self.use_normal = use_normal
        self.new_data = {}  # init in generate_new_datas()
        self.prepare_weights()

    def prepare_weights(self):
        """Generate the weights of all classes"""
        self.points_data = {}
        points_list = []
        labels_list = []
        for scene_id in tqdm(self.scene_list):
            point_data = np.load(os.path.join(self.scannet_datapath, "{}.npy").format(scene_id))
            label = point_data[:, 10]

            points_list.append(point_data)
            labels_list.append(label)
            self.points_data[scene_id] = point_data

        if self.is_weighting:
            labelweights = np.zeros(self.num_classes)
            for seg in labels_list:
                tmp, _ = np.histogram(seg, range(self.num_classes + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.label_weights = 1 / np.log(1.2 + labelweights)
        else:
            self.label_weights = np.ones(self.num_classes)

    def __getitem__(self, index):
        """get item"""
        # load chunks
        scene_id = self.scene_list[index]
        point_data = self.new_data[scene_id]
        # unpack
        point_set = point_data[:, :3]  # include xyz by default
        rgb = point_data[:, 3:6] / 255.  # normalize the rgb values to [0, 1]
        normal = point_data[:, 6:9]
        label = point_data[:, 10].astype(np.int32)

        if self.use_color:
            point_set = np.concatenate([point_set, rgb], axis=1)

        if self.use_normal:
            point_set = np.concatenate([point_set, normal], axis=1)

        if self.phase == "train":
            point_set = self.default_transform(point_set)

        # prepare mask
        coord_min = np.min(point_set, axis=0)[:3]
        coord_max = np.max(point_set, axis=0)[:3]
        mask = np.sum((point_set[:, :3] >= (coord_min - 0.01)) * (point_set[:, :3] <= (coord_max + 0.01)), axis=1) == 3
        sample_weight = self.label_weights[label]
        sample_weight *= mask
        return point_set, label, sample_weight

    def __len__(self):
        """Get len"""
        return len(self.scene_list)

    def default_transform(self, point_set):
        """Data Augmentation"""
        center = np.mean(point_set[:, :3], axis=0)
        coords = point_set[:, :3] - center
        rand = np.random.choice(np.arange(0.01, 1.01, 0.01), size=1)[0]
        if rand < 1 / 8:
            # random translation
            coords = self.translate(coords)
        elif 1 / 8 <= rand < 2 / 8:
            # random rotation
            coords = self.rotate(coords)
        elif 2 / 8 <= rand < 3 / 8:
            # random scaling
            coords = self.scale(coords)
        elif 3 / 8 <= rand < 4 / 8:
            coords = self.translate(coords)
            coords = self.rotate(coords)
        elif 4 / 8 <= rand < 5 / 8:
            coords = self.translate(coords)
            coords = self.scale(coords)
        elif 5 / 8 <= rand < 6 / 8:
            coords = self.rotate(coords)
            coords = self.scale(coords)
        elif 6 / 8 <= rand < 7 / 8:
            coords = self.translate(coords)
            coords = self.rotate(coords)
            coords = self.scale(coords)
        else:
            # no augmentation
            pass
        coords += center
        point_set[:, :3] = coords
        return point_set

    def translate(self, point_set):
        """translation factors"""
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        coords = point_set[:, :3]
        coords += [x_factor, y_factor, z_factor]
        point_set[:, :3] = coords
        return point_set

    def rotate(self, point_set):
        """rotate translat"""
        coords = point_set[:, :3]
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180
        rx = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]]
        )

        
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180
        ry = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]]
        )


        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180
        rz = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]]
        )

        # rotate
        r = np.matmul(np.matmul(rz, ry), rx)
        coords = np.matmul(r, coords.T).T

        point_set[:, :3] = coords
        return point_set

    def scale(self, point_set):
        """scale translat"""
        factor = np.random.choice(np.arange(0.95, 1.051, 0.001), size=1)[0]
        coords = point_set[:, :3]
        coords *= [factor, factor, factor]
        point_set[:, :3] = coords
        return point_set

    def generate_new_datas(self):
        """ generate new chunks for a new epoch
            note: must be called before training every epoch.
        """
        print("generate new chunks for {}...".format(self.phase))
        for scene_id in tqdm(self.scene_list):
            points = self.points_data[scene_id]
            labels = points[:, 10].astype(np.int32)

            coordmax = np.max(points, axis=0)[:3]
            coordmin = np.min(points, axis=0)[:3]

            for _ in range(5):
                cur_center = points[np.random.choice(len(labels), 1)[0], :3]
                cur_min = cur_center - [0.75, 0.75, 1.5]
                cur_max = cur_center + [0.75, 0.75, 1.5]
                cur_min[2] = coordmin[2]
                cur_max[2] = coordmax[2]
                cur_choice = np.sum((points[:, :3] >= (cur_min - 0.2)) * (points[:, :3] <= (cur_max + 0.2)), axis=1) == 3
                cur_point_set = points[cur_choice]
                cur_semantic_seg = labels[cur_choice]

                if not cur_semantic_seg.any():
                    continue

                mask = np.sum((cur_point_set[:, :3] >= (cur_min - 0.01)) * (cur_point_set[:, :3] <= (cur_max + 0.01)),
                              axis=1) == 3
                vidx = np.ceil((cur_point_set[mask, :3] - cur_min) / (cur_max - cur_min) * [31.0, 31.0, 62.0])
                vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 + vidx[:, 2])
                isvalid = np.sum(cur_semantic_seg > 0) / len(cur_semantic_seg) >= 0.7 and len(
                    vidx) / 31.0 / 31.0 / 62.0 >= 0.02
                if isvalid:
                    break

            # store chunk
            new_data = cur_point_set
            choices = np.random.choice(new_data.shape[0], self.npoints, replace=True)
            new_data = new_data[choices]
            self.new_data[scene_id] = new_data

        print("done!\n")


class ScannetDatasetWholeScene():
    """
        A source dataset that reads, parses and augments the whole scene of Scannet dataset.

        Args:
            path (str): The root directory of the Scannet dataset or inference pointcloud.
            phase (str): The dataset split, supports "train", "val", or "test". Default: "val".
            npoints (int): Number of points selected from each scene.Default: 8192.
            is_weighting (bool): Whether or not to use the weights of the number of points distributed in different classes. Default: True.
                                if dataset choose 'val' ,the weights are all ones.

            use_color (bool): Whether or not to use the colors of all points. Default: False.
            use_normal (bool): Whether or not to use the normals of all points. Default: False.

        Examples:

            >> import mindspore.dataset as ds
            >> from mindvision.ms3d.dataset.scannet import ScannetDatasetWholeScene
            >> ROOT = './ScanNet'
            >> testdata = ScannetDatasetWholeScene(path=ROOT, phase='val', is_weighting=not False, use_color=False, use_normal=False)
            >> test_ds = ds.GeneratorDataset(testdata, ["data", "label", "weights"], num_parallel_workers=1, shuffle=True)
            >> for data in test_ds.create_dict_iterator():
                    print(data["data"].shape, data["label"].shape, data["weights"].shape)

        Note: Although Scannet has 21 classes, it only computes MIoU for the following 20 classes
        NYUCLASSES = [
                        'floor',
                        'wall',
                        'cabinet',
                        'bed',
                        'chair',
                        'sofa',
                        'table',
                        'door',
                        'window',
                        'bookshelf',
                        'picture',
                        'counter',
                        'desk',
                        'curtain',
                        'refrigerator',
                        'bathtub',
                        'shower curtain',
                        'toilet',
                        'sink',
                        'otherprop'
                    ]
        """

    def __init__(self, path, phase='val', npoints=8192, is_weighting=True, use_color=False, use_normal=False):
        self.ROOT = path
        self.PREP = os.path.join(self.ROOT, "preprocessing")
        self.PREP_SCANS = os.path.join(self.PREP, "scannet_scenes")

        valpath = os.path.join(self.ROOT, "data/scannetv2_val.txt")
        trainpath = os.path.join(self.ROOT, "data/scannetv2_train.txt")
        scene_list = []
        if phase == 'train':
            list_path = trainpath
        else:
            list_path = valpath
        with open(list_path) as f:
            for scene_id in f.readlines():
                scene_list.append(scene_id.strip())

        self.scene_list = scene_list
        self.npoints = npoints
        self.is_weighting = is_weighting
        self.use_color = use_color
        self.use_normal = use_normal

        self._load_scene_file()

    def _load_scene_file(self):
        """load scene file"""
        self.scene_points_list = []
        self.semantic_labels_list = []

        for scene_id in tqdm(self.scene_list):
            scene_data = np.load(os.path.join(self.PREP_SCANS, "{}.npy").format(scene_id))
            label = scene_data[:, 10].astype(np.int32)
            self.scene_points_list.append(scene_data)
            self.semantic_labels_list.append(label)

        if self.is_weighting:
            labelweights = np.zeros(20)
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg, range(21))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = 1 / np.log(1.2 + labelweights)
        else:
            self.labelweights = np.ones(20)

    def __getitem__(self, index):
        """get item"""
        scene_data = self.scene_points_list[index]

        # unpack
        point_set_ini = scene_data[:, :3]  # include xyz by default
        color = scene_data[:, 3:6] / 255.  # normalize the rgb values to [0, 1]
        normal = scene_data[:, 6:9]

        if self.use_color:
            point_set_ini = np.concatenate([point_set_ini, color], axis=1)

        if self.use_normal:
            point_set_ini = np.concatenate([point_set_ini, normal], axis=1)

        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = point_set_ini[:, :3].max(axis=0)
        coordmin = point_set_ini[:, :3].min(axis=0)
        xlength = 1.5
        ylength = 1.5
        nsubvolume_x = np.ceil((coordmax[0] - coordmin[0]) / xlength).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1] - coordmin[1]) / ylength).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()

        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin + [i * xlength, j * ylength, 0]
                curmax = coordmin + [(i + 1) * xlength, (j + 1) * ylength, coordmax[2] - coordmin[2]]
                mask = np.sum((point_set_ini[:, :3] >= (curmin - 0.01)) * (point_set_ini[:, :3] <= (curmax + 0.01)),
                              axis=1) == 3
                cur_point_set = point_set_ini[mask, :]
                cur_semantic_seg = semantic_seg_ini[mask]
                if len(cur_semantic_seg) == 0:
                    continue

                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                point_set = cur_point_set[choice, :]  # Nx3
                semantic_seg = cur_semantic_seg[choice]  # N
                mask = mask[choice]
                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask  # N
                point_sets.append(np.expand_dims(point_set, 0))  # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg, 0))  # 1xN
                sample_weights.append(np.expand_dims(sample_weight, 0))  # 1xN

        point_sets = np.concatenate(tuple(point_sets), axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs), axis=0)
        sample_weights = np.concatenate(tuple(sample_weights), axis=0)

        return point_sets, semantic_segs, sample_weights

    def __len__(self):
        """get len"""
        return len(self.scene_points_list)
