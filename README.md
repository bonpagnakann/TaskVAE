# TaskVAE
The implementation of the paper: TaskVAE: A Task-Specific Variational Autoencoders for Exemplar Generation in Continual Learning for Human Activity Recognition 

## Table of Content
* [1. Installation](#1-installation)
  * [1.1 Dependencies](#11-dependencies)
  * [1.2 Data](#12-data)
* [2. Running our Experiments](#2-Running-our-Experiments)
  * [2.1 Important Parameters](#21-Important-Parameters)
  * [2.2 Replay-Based Methods](#22-Replay-Based-Methods)
  * [2.3 TaskVAE](#23-TaskVAE)
* [3. Detailed Results](#3-Detailed-Results)

## 1. Installation

### 1.1 Dependencies
This code was implemented with PyTorch 1.13.0. Our experiments were conducted on 4 NVIDIA GTX1080Ti GPUs (11GB each) and 2 Intel Xeon E5-2620 v4 processors operating at 2.10GHz. 

Please launch this command to install the necessary libraries to run our experiment.

```
pip install -r requirements.txt
```

### 1.2 Data

The dataset can be downloaded from this Google Drive link: https://drive.google.com/drive/folders/1zuVhJiePkv2y2q56Y-_8Fz3Jh_JNZhp7?usp=sharing . Once the download is completed, please store it in the 'datasets' folder.

## 2. Running our Experiments
### 2.1 Important Parameters

In the following commands of running experiment for each method, the necessary arguments are defined as follows:
- ```--dataset```: The dataset can be selected from ['motion', 'realworld', 'hhar', 'pamap', 'uci']

- ```--total_classes``` : 6 for MotionSense, HHAR and UCI dataset, 8 for RealWorld, and 10 for PAMAP2 Dataset.

- ```--base_classes``` : The number of classes in the first task.

- ```--new_classes``` : Order of number of classes from the second task. 

For example, ```--base_classes 5 --new_classes '32' ``` refers to the scenario where there are 5,3,2 classes in Task 1,2,3 respectively. 

### 2.2 Replay-Based Methods
After downloading the datasets and store them in 'datasets' folder, run the following commands for the experiments of each replay-based method. Below is a sample command to run experiments of Random, EWC-Replay, iCaRL, and LUCIR for scenario 2-3-1 on MotionSense dataset. 

- Random:
  ```
  python runner.py --dataset 'motion' --total_classes 6 --new_classes '31' --base_classes 2 --epochs 20 --method 'ce' --exemplar 'random' --person 0 --number 0
  ```
- EWC-Replay:
  ```
  python runner.py --dataset 'motion' --total_classes 6 --new_classes '31' --base_classes 2 --epochs 20 --method 'ce_ewc' --exemplar 'random' --person 0 --number 0
  ```
- iCarl:
  ```
  python runner.py --dataset 'motion' --total_classes 6 --new_classes '31' --base_classes 2 --epochs 20 --method 'kd_kldiv' --exemplar 'icarl' --person 0 --number 0
  ```
- LUCIR:
  ```
  python runner.py --dataset 'motion' --total_classes 6 --new_classes '31' --base_classes 2 --epochs 20 --method 'cn_lfc_mr' --exemplar 'icarl' --person 0 --number 0
  ```

### 2.3 TaskVAE
For TaskVAE approaches, *filtering process* can be applied with ```--latent_vec_filter``` and the value can be selected from from the following: 
```--latent_vec_filter 'probability'```: TaskVAE with filtering process
```--latent_vec_filter 'none'```: TaskVAE without filtering process

Below is the sample commands to run TaskVAE experiments for both with and without filtering process:

- TaskVAE (without filtering):
  ```
  python runner.py --dataset 'motion' --total_classes 6 --new_classes '31' --base_classes 2 --epochs 20 --method 'ce' --exemplar 'taskvae' --vae_lat_sampling 'boundary_box' --person 0
  ```
- TaskVAE (with filtering):
  ```
  python runner.py --dataset 'motion' --total_classes 6 --new_classes '31' --base_classes 2 --epochs 20 --method 'ce' --exemplar 'taskvae' --vae_lat_sampling 'boundary_box' --latent_vec_filter 'probability' --person 0
  ```

The output files are stored in 'output_reports/' folder which will automatically appears when a experiment command is launched. The output files contains necessary information including the data size, exemplar size, and the detailed results (Accuracy of all, old, and new classes) in each task. 
 
## 3. Detailed Results:
More details on the figures and tables of each dataset in this paper can be accessed through this link: https://bonpagnakann.github.io/TaskVAE_Vis/
