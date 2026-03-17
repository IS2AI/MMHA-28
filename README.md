## MMHAR-28: Human Action Recognition Across RGB, Depth, Thermal, and Event Modalities 

This repository provides the official implementation, dataset, and training scripts for MMHAR-28: Human Action Recognition Across RGB, Depth, Thermal, and Event Modalities paper. 
This repository builds on top of [VideoMamba](https://github.com/OpenGVLab/VideoMamba/tree/main?tab=readme-ov-file) and includes the training code, evaluation scripts, and the local `mamba` and `causal-conv1d` dependencies required by the model implementation.

## Setup
### 1. Clone the repository

```bash
git clone https://github.com/IS2AI/MMHA-28.git
cd MMHA-28
```

### 2. Optional: pull the Docker image

If you use the provided container workflow, pull the published image:

```bash
docker pull mmhm28/mmha-28:latest
```

### 3. Install Python dependencies

Create and activate your environment first, then install the project requirements:

```bash
pip install -r requirements.txt
pip install -e ./mamba
pip install -e ./causal-conv1d
```

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
- 
### 1.Download the MMHA-28 dataset from the official source
```
   tbd
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

## Training

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

## Evaluation
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

