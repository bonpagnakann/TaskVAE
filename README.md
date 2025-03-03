# TaskVAE
The implementation of the paper: TaskVAE: A Task-Specific Variational Autoencoders for Exemplar Generation in Continual Learning for Human Activity Recognition 

The dataset can be downloaded from this GDrive link: https://drive.google.com/drive/folders/1zuVhJiePkv2y2q56Y-_8Fz3Jh_JNZhp7?usp=sharing . Once the download is completed, please store it in the 'datasets' folder.

Run the following commands for each CL method:
```--dataset``` can be selected from ['motion', 'realworld', 'hhar', 'pamap', 'uci']
```--total_classes``` : 6 for MotionSense, HHAR and UCI dataset, 8 for RealWorld, and 10 for PAMAP2 Dataset.
```--new_classes``` : Order of number of classes from the second task.

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
- VAE:
  ```
  python runner.py --dataset 'motion' --total_classes 6 --new_classes '31' --base_classes 2 --epochs 20 --method 'ce' --exemplar 'vae' --vae_lat_sampling 'boundary_box' --person 0
  python runner.py --dataset 'motion' --total_classes 6 --new_classes '31' --base_classes 2 --epochs 20 --method 'ce' --exemplar 'vae' --vae_lat_sampling 'boundary_box' --latent_vec_filter 'probability' --person 0
  ```
  

## Sample filtering method 
*Filtering process* can be applied with "--latent_vec_filter" and the value to be selected from from the following: ['probability', 'none']
 
## Detailed Results:
More details on the figures and tables of each dataset in this paper can be accessed through this link: https://bonpagnakann.github.io/TaskVAE_Vis/
