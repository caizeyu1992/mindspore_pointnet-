# PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space (NIPS 2017)
This work is based on Mindspore Frameworks.

Original code is created by <a href="http://charlesrqi.com" target="_blank">Charles R. Qi</a>, <a href="http://stanford.edu/~ericyi">Li (Eric) Yi</a>, <a href="http://ai.stanford.edu/~haosu/" target="_blank">Hao Su</a>, <a href="http://geometry.stanford.edu/member/guibas/" target="_blank">Leonidas J. Guibas</a> from Stanford University. [[PDF](https://arxiv.org/abs/1706.02413)]

<hr />

> **Abstract:** *Few prior works study deep learning on point sets. PointNet by Qi et al. is a pioneer in this direction. However, by design PointNet does not capture local structures induced by the metric space points live in, limiting its ability to recognize fine-grained patterns and generalizability to complex scenes. In this work, we introduce a hierarchical neural network that applies PointNet recursively on a nested partitioning of the input point set. By exploiting metric space distances, our network is able to learn local features with increasing contextual scales. With further observation that point sets are usually sampled with varying densities, which results in greatly decreased performance for networks trained on uniform densities, we propose novel set learning layers to adaptively combine features from multiple scales. Experiments show that our network called PointNet++ is able to learn deep point set features efficiently and robustly. In particular, results significantly better than state-of-the-art have been obtained on challenging benchmarks of 3D point clouds.* 
<hr />

## Introduction
This work is based on our NIPS'17 paper. You can find arXiv version of the paper <a href="https://arxiv.org/pdf/1706.02413.pdf">here</a> or check <a href="http://stanford.edu/~rqi/pointnet2">project webpage</a> for a quick overview. PointNet++ is a follow-up project that builds on and extends <a href="https://github.com/charlesq34/pointnet">PointNet</a>. It is version 2.0 of the PointNet architecture.

PointNet (the v1 model) either transforms features of *individual points* independently or process global features of the *entire point set*. However, in many cases there are well defined distance metrics such as Euclidean distance for 3D point clouds collected by 3D sensors or geodesic distance for manifolds like isometric shape surfaces. In PointNet++ authors want to respect *spatial localities* of those point sets. PointNet++ learns hierarchical features with increasing scales of contexts, just like that in convolutional neural networks. Besides, authors also observe one challenge that is not present in convnets (with images) -- non-uniform densities in natural point clouds. To deal with those non-uniform densities, we further propose special layers that are able to intelligently aggregate information from different scales.


## Diagram of PointNet++
![Illustration of PointNet++](./figure/teaser.jpg)



## Create Environment:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

```shell
pip install -r requirements.txt
```


## Prepare Dataset:

### ModelNet40
You can get our sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape) at this <a href="https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip">link</a>. 

```shell
|--./ModelNet40/
    |--airplane
    	|-- airplane_0001.txt
    	|-- airplane_0002.txt
        |-- .......
    |--bathtub
    	|-- bathtub_0001.txt
    	|-- bathtub_0002.txt
        |-- .......
    |--filelist.txt
    |-- modelnet40_shape_names.txt
    |-- modelnet40_test.txt
    |-- modelnet40_train.txt
```

### ScanNet
ScanNet is an RGB-D video dataset containing 2.5 million views in more than 1500 scans, annotated with 3D camera poses, surface reconstructions, and instance-level semantic segmentations. Original dataset website: <a href="http://www.scan-net.org/">http://www.scan-net.org/</a>

note: If you would like to download the ScanNet data, please fill out an agreement to the ScanNet Terms of Use<a href="http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf"> and send it to Scannet-org at scannet@googlegroups.com.

```shell
|--./ScanNet/
    |--data
    	|-- scannetv2.txt
    	|-- scannetv2_train.txt
        |-- scannetv2_val.txt
        |-- scannetv2_test.txt
    |--preprocessing
    	|-- points
    	|-- scannet_scenes
    	    	|-- scene0000_00.npy
                |-- scene0000_01.npy
                |-- .......
    	|-- collect_scannet_scenes.py
        |-- scannetv2-labels.combined.tsv
    |--scannet           
        |--scene0000_00
            |-- scene0000_00.sens
            |-- scene0000_00_vh_clean.ply
            |-- scene0000_00_vh_clean_2.ply
            |-- scene0000_00_vh_clean_2.0.010000.segs.json            
            |-- scene0000_00.aggregation.json           
            |-- scene0000_00_vh_clean.aggregation.json          
            |-- scene0000_00_vh_clean_2.0.010000.segs.json 
            |-- scene0000_00_vh_clean.segs.json              
            |-- scene0000_00_vh_clean_2.labels.ply             
            |-- scene0000_00_2d-label.zip  
            |-- scene0000_00_2d-instance.zip               
            |-- scene0000_00_2d-label-filt.zip               
            |-- scene0000_00_2d-instance-filt.zip    
        |--scene0000_01     
            |-- .......    
        |-- .......                                                         
        |-- train_test_split
        |-- synsetoffset2category.txt
```

1) Preprocess ScanNet scenes
Parse the ScanNet data into `*.npy` files and save them in `ms3d/dataset/prepare_data/preprocessing_scannet/scannet_scenes/`
```shell
cd ms3d/dataset/prepare_data/preprocessing_scannet
python preprocessing/collect_scannet_scenes.py
```
2) Sanity check
Don't forget to visualize the preprocessed scenes to check the consistency
```shell
python preprocessing/visualize_prep_scene.py --scene_id <scene_id>
```


## Train:

```shell
cd ms3d/example/pointnet2/

# PointNet++ classfication
python pointnet2_modelnet40_train.py --opt pointnet2_classfication.yaml

# PointNet++ segmentation_scannet
python pointnet2_scannet_train.py --opt pointnet2_scannet.yaml 

```

## Test:

```shell
cd ms3d/example/pointnet2/

# PointNet++ classfication
python pointnet2_modelnet40_eval.py --opt pointnet2_classfication.yaml

# PointNet++ segmentation_scannet
python pointnet2_scannet_eval.py --opt pointnet2_scannet.yaml 


```


## Results(ALL codes have been open, resuts will coming soon.)

### Classfication_Modelnet40

|     Model     |   npoints  | use_norm | Methods  | Accuracy |
|   PointNet++  |   1024     |  FALSE   |   SSG    |  0.918   |[Baidu Drive](https://pan.baidu.com/s/1OxY7wxQS8oyQoJ-AvUr5Ag?pwd=hmxh).
|   PointNet++  |   1024     |  TRUE    |   SSG    |  TODO    |
|   PointNet++  |   1024     |  FALSE   |   MSG    |  TODO    |
|   PointNet++  |   1024     |  TRUE    |   MSG    |  TODO    |



### segmengation_scannetv2

|   Model       |       use XYZ     |   use color       |   use normal      | use multiview     |   use MSG         |   mIoU    |
|   PointNet++  |:heavy_check_mark: |  -                |  -                |  -                | -                 |   34.0    |[Baidu Drive](https://pan.baidu.com/s/18dy-XM6_-2BiZCjJhM4NSQ?pwd=x3bm).
|   PointNet++  |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |  -                | -                 |   TODO    |
|   PointNet++  |:heavy_check_mark: |  -                |  -                |  -                |:heavy_check_mark: |   40.0    |[Baidu Drive](https://pan.baidu.com/s/1s02-jWFpx8sQamqun3CUdw?pwd=2t94).
|   PointNet++  |:heavy_check_mark: |:heavy_check_mark: |:heavy_check_mark: |  -                |:heavy_check_mark: |   TODO    |


## Citation
If you find the code helpful in your resarch or work, please cite the following paper:

```shell

# PointNet++
@article{qi2017pointnetplusplus,
  title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
  author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
  journal={arXiv preprint arXiv:1706.02413},
  year={2017}
}

```


## Acknowledgement
We thank the authors of [pointnet++](https://github.com/charlesq34/pointnet2) for sharing their codes and data.
We thank the authors of [Pointnet2.ScanNet](https://github.com/daveredrum/Pointnet2.ScanNet) for sharing their codes and data.
