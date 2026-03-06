## SocialDIPP (Social Navigation for Mobile Robots with Differentiable Integrated Motion Prediction and Planning)
This repo is a fork of the original repo based in the following paper:

**Differentiable Integrated Motion Prediction and Planning with Learnable Cost Function for Autonomous Driving**
<br> [Zhiyu Huang](https://mczhi.github.io/), [Haochen Liu](https://scholar.google.com/citations?user=iizqKUsAAAAJ&hl=en), [Jingda Wu](https://wujingda.github.io/), [Chen Lv](https://scholar.google.com/citations?user=UKVs2CEAAAAJ&hl=en) 
<br> [AutoMan Research Lab, Nanyang Technological University](https://lvchen.wixsite.com/automan)
<br> **[[Paper]](https://ieeexplore.ieee.org/document/10154577/)**&nbsp;**[[arXiv]](https://arxiv.org/abs/2207.10422)**&nbsp;**[[Project Website]](https://mczhi.github.io/DIPP/)**

## Dataset
Download dataset ETH-UCY (for funcionality testing)

## Installation
### Install dependency
```bash
sudo apt-get install libsuitesparse-dev
```

### Create conda env
```bash
conda env create -f environment.yml
conda activate DIPP
```

### Install Theseus
Install the [Theseus library](https://github.com/facebookresearch/theseus), follow the guidelines.

## Usage

Integrated planning and prediction with DIPP

### Data Processing
1.1 Process the data in joint scenes without considering the robot, you can configure the flags
```shell
python process_eth_ucy.py \
  --datasets_dir datasets \
  --out_dir DIPP_model/data \
  --leave_out ucy-zara02 \
  --split \
  --k_neighbors 10 \
  --neighbor_radius 3.0

```
Posteriormente si quieres visualizar tus datos ejecuta:
```shell
python vis.py --npz DIPP_model/data/train/data.npz --random --ego_frame --rotate
```
### 2.1 DIPP_model train (Whitout planning)
```shell
python DIPP_model/train.py \
  --name Exp1_noplan \
  --train_set DIPP_model/data/train/data.npz \
  --valid_set DIPP_model/data/val/data.npz \
  --device cuda \
  --batch_size 32 \
  --train_epochs 10
```

### 2.2 DIPP_model train (Integrated planning)
```shell
python DIPP_model/train.py \
  --name Exp1_plan \
  --train_set DIPP_model/data/train/data.npz \
  --valid_set DIPP_model/data/val/data.npz \
  --device cuda \
  --use_planning \
  --batch_size 32 \
  --train_epochs 10
```



### 2.3 Test and Visualization
If you are realizing the train without planning so you must use the flag --use_planning, in another case use this flag
```shell
python DIPP_model/test_eth_ucy.py \
  --model_path DIPP_model/training_log/Exp1_plan/model_10_0.2293.pth \
  --name Exp1_plan_vis \
  --test_set DIPP_model/data/test/data.npz \
  --device cuda \
  --use_planning \
  --min_neighbors 5
```
## Citation
If you find the repo or the paper useful, please use the following citation:
```
@article{huang2023differentiable,
  title={Differentiable integrated motion prediction and planning with learnable cost function for autonomous driving},
  author={Huang, Zhiyu and Liu, Haochen and Wu, Jingda and Lv, Chen},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```