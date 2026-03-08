# RunPod Training Guide

RunPod is a cloud platform that allows you to rent high-performance GPUs (like NVIDIA A6000 or RTX 4090) by the hour for a very low cost (~$0.40/hr). This is where we will train our `deberta-v3-small` model.

## Step 1: Launch a Pod
1. Go to [RunPod.io](https://www.runpod.io) and create an account.
2. Go to **GPU Instances** -> **Deploy**.
3. Select an affordable GPU (e.g., **RTX 3090** or **RTX 4090**).
4. Choose the **PyTorch** template (this comes with Python and CUDA pre-installed).
5. Click **Deploy**.

## Step 2: Connect to the Pod
1. Once the Pod is "Running", click **Connect**.
2. Select **Connect via SSH** (or use the web-based **Jupyter Notebook** terminal).

## Step 3: Upload/Sync Your Code
The easiest way is to use Git or zip your folder:
```bash
# Inside the RunPod Terminal
git clone <your-repo-url>
cd challenge/
```
*Note: Make sure to copy the `datasets/` folder as well!*

## Step 4: Install Requirements
Run the following inside the Pod:
```bash
pip install transformers datasets torch scikit-learn sentencepiece accelerate
```

## Step 5: Start Training
Execute the training script:
```bash
python training/train.py
```

## Step 6: Download the Model
Once training is complete (usually ~15-30 minutes):
1. The model will be saved in `models/detector_v1/`.
2. Download this folder back to your local machine for deployment in the FastAPI server.
3. **CRITICAL**: Stop/Terminate the Pod on RunPod so you stop getting charged!
