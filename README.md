# SocialDIPP (Social Navigation for Mobile Robots with Differentiable Integrated Motion Prediction and Planning)
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
### Data Processing
Run ```process_eth_ucy.py``` to process the raw data for training and validation (If you need it). This will convert the original data format into a set of ```.npz``` files, each containing the data of a scene with the ego (first pedestrian of the tensor) and surrounding pedestrian. You need to specify the file path to the original data ```--load_path``` and the path to save the processed data ```--save_path``` . Example with Zara02: 
```shell
python process_eth_ucy.py 
--process_all 
--leave_out ucy-zara02 /
--output data/processed_leave_zara02_out /
--split
```

### Training
Run ```train.py``` to learn the predictor and planner (if set ```--use_planning```). You need to specify the file paths to training data ```--train_set``` and validation data ```--valid_set```. Leave other arguments vacant to use the default setting. Example with Zara1: 
```shell
python train.py \
--name leave_zara02_out \
--train_set data/processed_leave_zara02_out/train_combined \
--valid_set data/processed_leave_zara02_out/val_combined \
--pretrain_epochs 5 \
--train_epochs 20 \
--batch_size 32 \
--learning_rate 2e-4 \
--device cuda
```

### Open-loop testing
Run ```test_eth_ucy.py``` to test the trained planner in an open-loop manner. You need to specify the path to the original test dataset ```--test_set``` (path to the folder) and also the file path to the trained model ```--model_path```. Set ```--render``` to visualize the results and set ```--save``` to save the rendered images.
```shell
python test_eth_ucy.py \
--model_path training_log/leave_zara02_out_no_planning/model_11_0.1589.pth \
--test_set data/processed_leave_zara02_out/test \
--name leave_zara02_out_test \
--batch_size 32 \
--device cuda \
--visualize \
--num_vis_samples 20
```

### Closed-loop testing (Not at the moment)
Run ```closed_loop_test.py``` to do closed-loop testing. You need to specify the file path to the original test data ```--test_file``` (a single file) and also the file path to the trained model ```--model_path```. Set ```--render``` to visualize the results and set ```--save``` to save the videos.
```shell
python closed_loop_test.py \
--name closed_loop \
--test_file /path/to/original/test/data \
--model_path /path/to/saved/model \
--use_planning \
--render \
--save \
--device cpu
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
