# Point Cloud Semantic Segmentation Using Deformable Kernels

Self-Driving Cars course project, Alexandria University.

This project segments LiDAR point clouds from the [SemanticKITTI](http://www.semantic-kitti.org) dataset into 19 classes (road, car, building, vegetation, person, …). It uses **3D-GCN** ([Lin et al., CVPR 2020](https://github.com/zhihao-lin/3dgcn)), a graph convolution network with learnable *deformable 3D kernels* that runs directly on the raw, unordered points.

It also tests whether **fusing camera colour with the LiDAR points** ("point painting") helps. Each point is projected into the left colour camera (`image_2`) and given the RGB value of the pixel it lands on. These RGB values are concatenated with the network's per-point features just before the final classification layers (late fusion).

📄 Full write-up: [report/Self_Driving_Cars_Project.pdf](report/Self_Driving_Cars_Project.pdf)

## Results

Validation results on sequence 08 (mean IoU over all classes):

| Model | Input | Batch size | LR | mIoU |
|---|---|---|---|---|
| No painting | XYZ | 4 | 0.01 | 0.589 |
| Painting | XYZ + RGB | 4 | 0.01 | 0.593 |
| No painting | XYZ | 8 | 0.001 | 0.595 |
| **Painting** | **XYZ + RGB** | **8** | **0.001** | **0.621** |

On the test split (sequence 05), the painted model reached **62.8% mIoU**, compared with **57%** for the LiDAR-only model.

Each model was trained for 6 epochs (about 4 hours per epoch on a V100 GPU). The models were still improving when training stopped.

## Method

- **Network:** the 3D-GCN segmentation model: two blocks of deformable-kernel graph convolutions with max pooling, followed by 1×1 convolutions. It uses neighbourhood size 50 and kernel support size 1.
- **Deformable kernel:** each kernel has a centre and learnable support directions. For each point, the support directions are compared to the directions of its neighbours using cosine similarity. The best-matching neighbour features for each support are aggregated.
- **Painting:** LiDAR points are projected into camera 2 using the KITTI calibration (`P2 · Tr`). Points that fall outside the image are dropped, and 10,000 of the remaining points are sampled per scan.
- **Training:** Adam optimiser, cross-entropy loss, learning rate halved every 10 epochs.
- **Split:** train on sequences 00–04, 06, 07, 09, 10; validate on 08; test on 05.

## Repository structure

```
├── notebooks/
│   ├── 01_explore_semantickitti.ipynb        # dataset statistics and reading scans and labels
│   ├── 02_train_gcn3d_baseline_no_rgb.ipynb  # initial model: 3D-GCN on raw LiDAR XYZ (val mIoU 0.595)
│   └── 03_train_gcn3d_rgb_painting.ipynb     # 3D-GCN + RGB point painting, best model (val mIoU 0.621)
├── config/sem_kitti.yaml               # SemanticKITTI label names, colours and class mapping
├── weights/gcn3d_rgb_painting_lr0.001_b8.pt   # best painted model (val mIoU 0.621)
├── report/Self_Driving_Cars_Project.pdf
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

### Data

The dataset isn't included because it is very large and its licence doesn't allow redistribution. Download these from the [KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) and [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download):

1. Velodyne laser data (`data_odometry_velodyne`)
2. Colour images (`data_odometry_color`)
3. Calibration files (`data_odometry_calib`). These provide `Tr`, the LiDAR→camera transform, which the projection needs.
4. SemanticKITTI labels (`data_odometry_labels`)

Extract the files so that each sequence folder has `velodyne/`, `image_2/` and `labels/`. Then either place the data at `dataset/` in the repository root, or set environment variables:

```bash
export SEMKITTI_ROOT=/path/to/dataset/sequences/
export KITTI_CALIB_ROOT=/path/to/data_odometry_calib/dataset/sequences/
```

### Training

Both training notebooks use the same network, data split and settings (lr 0.001, batch size 8). The only difference is the RGB input:

- `notebooks/02_train_gcn3d_baseline_no_rgb.ipynb` is the initial LiDAR-only model.
- `notebooks/03_train_gcn3d_rgb_painting.ipynb` adds the camera RGB values (point painting).

The baseline keeps only the points that fall inside the camera image, just like the painted model, so the two are trained on the same points.

Open a notebook and run all cells. Models, checkpoints and logs are written to `outputs/`.

Logging to [Comet ML](https://www.comet.com) is optional. To turn it on, set `COMET_API_KEY` (and `COMET_WORKSPACE`).

### Using the trained weights

To start from the released model, set `"load": "../weights/gcn3d_rgb_painting_lr0.001_b8.pt"` in the `args` cell.

The file is a PyTorch `state_dict` saved from a `torch.nn.DataParallel` model, so its keys start with `module.`. The notebook's `Manager` wraps the model the same way before loading. To load it into a plain `GCN3D(class_num=19, support_num=1, neighbor_num=50)`, remove the `module.` prefix first.

## Acknowledgements

- The 3D-GCN layers (`Conv_surface`, `Conv_layer`, `Pool_layer` and the neighbour helpers) are taken from the official implementation: [zhihao-lin/3dgcn](https://github.com/zhihao-lin/3dgcn).
- `config/sem_kitti.yaml` and the calibration-loading helpers are adapted from the [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api) and [pykitti](https://github.com/utiasSTARS/pykitti).
- Datasets: SemanticKITTI (Behley et al., ICCV 2019) and KITTI (Geiger et al., CVPR 2012). Both are licensed under CC BY-NC-SA 3.0.

## References

- J. Behley et al. *SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences.* ICCV 2019.
- A. Geiger, P. Lenz, R. Urtasun. *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite.* CVPR 2012.
- Z.-H. Lin, S.-Y. Huang, Y.-C. F. Wang. *Convolution in the Cloud: Learning Deformable Kernels in 3D Graph Convolution Networks for Point Cloud Analysis.* CVPR 2020.
- C. R. Qi et al. *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.* CVPR 2017.
