## MMHAR-28: Human Action Recognition Across RGB, Depth, Thermal, and Event Modalities 

This repository provides the official implementation, dataset, and training scripts for MMHAR-28: Human Action Recognition Across RGB, Depth, Thermal, and Event Modalities paper. 

This repository contains two model pipelines for the same MMHAR-28 dataset:
- `videomamba/`: the VideoMamba-based implementation
- `tsm_timeSformer/`: an MMAction2-based TSM/TimeSformer training pipeline for MMHAR-28

This repository builds on top of [VideoMamba](https://github.com/OpenGVLab/VideoMamba/tree/main?tab=readme-ov-file) and includes the training code, evaluation scripts, and the local `mamba` and `causal-conv1d` dependencies required by the model implementation.

## Our MMHAR-28 Dataset

The project expects MMHA-28 data to be organized by split and modality. The CSV files in [`videomamba/video_sm/data`](videomamba/video_sm/data) show the expected path format.

Examples:

```text
data/train/session_1/sub_18/d_rgb/28/rgb_images,13
data/train/session_1/sub_7/d_rgb/26/depth_images,12
data/train/session_1/sub_33/thermal/9_1_0,8
data/train/session_1/sub_55/event-streams/15,7
```

Available split files include:

- [`videomamba/video_sm/data/train.csv`](videomamba/video_sm/data/train.csv)
- [`videomamba/video_sm/data/val.csv`](videomamba/video_sm/data/val.csv)
- [`videomamba/video_sm/data/test.csv`](videomamba/video_sm/data/test.csv)
- modality-specific test files under [`videomamba/video_sm/data`](videomamba/video_sm/data)

### 1.Download the MMHA-28 dataset from the official source
```
   https://huggingface.co/datasets/issai/MMHA_28
```
Alternatively, a mini-sample version is available, containing data from one subject in session_1 and session_2, across all human actions. This is option for testing and visualization:
[mini-mmha-28](https://huggingface.co/datasets/tomirisss/mini-mmha)
```
huggingface-cli upload tomirisss/mini-mmha . --repo-type=dataset
```

### 2. Visualization
To visualize data from the mini-sample, run the following script with appropriate parameters:
```
   python vis.py --path PATH_TO_DATA --session session_1 --exp_num EXP_NUMBER
```

## VideoMamba Pipeline
### Installation
### 1. Install the dependencies:

```bash
pip install -r requirements.txt
pip install -e ./mamba
pip install -e ./causal-conv1d
```

### 2. Optional: pull the Docker image

If you use the provided container workflow, pull the published image:

```bash
docker pull mmhm28/mmha-28:latest
```

### 3. Training

Training is launched from [`videomamba/video_sm/run.py`](videomamba/video_sm/run.py).

Before running training, review these values in that file:

- `--nproc_per_node` to match the number of GPUs on your machine
- `MODEL_PATH` if you want to start from a specific checkpoint
- `DATA_PATH` and `PREFIX` if your dataset is stored outside the default relative layout
- output directories such as `OUTPUT_DIR`

Start training with:

```bash
cd videomamba/video_sm
python run.py
```

### 4. Evaluation
To test a pretrained model, first download the final Multimodal VideoMamba checkpoint:
[MV-Mamba](https://huggingface.co/tomirisss/MV-Mamba)
or using this code:
```
   huggingface-cli upload tomirisss/MV-Mamba .
```
MV-Mamba is the final multimodal model. The filename also indicates the number of frames used during training (e.g., MV-Mamba_f16.pth was trained with --num_frames=16).
Then, run the script, updating the --num_frames parameter and specifying the appropriate paths for MODEL_PATH.
```
   python3 run_test.py
```

## TSM / TimeSformer Pipeline
### Installation

The `tsm` directory is a local MMAction2-based codebase. Install it from this repository, not from a separate external checkout.

Recommended setup:

```bash
cd tsm
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements/build.txt
pip install -r requirements/mminstall.txt
pip install -e .
```

Important dependency notes:

- local MMAction version: `1.2.0`
- required `mmcv`: `>=2.0.0rc4,<2.2.0`
- required `mmengine`: `>=0.7.1,<1.0.0`

If you already have PyTorch installed, keep it compatible with your CUDA setup before installing the MMAction stack.

Main entrypoints:

- [`tsm/README.md`](tsm/README.md)
- [`tsm/configs/recognition/tsm/tsm_multimodal_mmha28_32.py`](tsm/configs/recognition/tsm/tsm_multimodal_mmha28_32.py)
- [`tsm/tools/train.py`](tsm/tools/train.py)
- [`tsm/tools/test.py`](tsm/tools/test.py)

Run TSM training:

```bash
cd tsm
source .venv/bin/activate
python tools/train.py configs/recognition/tsm/tsm_multimodal_mmha28_32.py
```

Run TSM evaluation:

```bash
cd tsm
source .venv/bin/activate
python tools/test.py configs/recognition/tsm/tsm_multimodal_mmha28_32.py work_dirs/tsm_multimodal_mmha28_32/best_acc_top1_epoch_95.pth
```
MMAction2 reference:

- [OpenMMLab/mmaction2](https://github.com/open-mmlab/mmaction2)

## Citation

If you use the MMHA-28 dataset or code in your research, please cite our paper.

