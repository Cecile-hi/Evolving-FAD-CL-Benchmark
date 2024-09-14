# Towards Robust Audio Deepfake Detection: An Evolving Benchmark for Continual Learning

This repository hosts the official project of the paper ["Towards Robust Audio Deepfake Detection: An Evolving Benchmark for Continual Learning"](https://arxiv.org/abs/2405.08596)!

Test your deepfake audio detection model’s continual learning ability using our benchmark, which supports various state-of-the-art methods.

## 🚀 Quick Start

### Step 1: Set up the environment
```bash
conda create -n cl_fad --python=3.8
pip install -r requirement.txt
```
### Step 2: Configure your experiment Create a configuration file `config.yaml` or use the provided template located at `yaml/fad_feature.yaml`.

### Step 3: Train your model Run the following command to start training: 
```bash
bash train.sh
```
🗓️ Project Timeline --------------------
### 2024.05.08 
* 🎯 Added the **Equal Error Rate (EER)** metric to the benchmark. 
* 🚀 To start training, simply run: 
```bash 
bash train.sh
```
### 2024.07.21 
* ✅ Most methods can run normally. 
* 🔍 Currently debugging the **ELMA**
### 2024.08.13
* ✨ Major modifications to the original benchmark: It is now clearer and faster.
* 🔧 Currently running 8 experiments.
* 🔍 Continued debugging of ELMA.
### 2024.09.14
* 🎉 Successfully completed debugging of ELMA, which now shows great results!