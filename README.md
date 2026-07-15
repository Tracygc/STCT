### STCT
STCT: Structure-Aware Token Contrastive Translation for Unpaired Infrared-to-Visible Traffic Scenes


### Availability of Datasets
```
InfraredCity and InfraredCity-Lite Dataset: The datasets and their more details are available in [InfiRay](http://openai.raytrontek.com/apply/Infrared_city.html/).
```
### Dataset Structure
```
dataset/
├── testA
├── testB
├── trainA
└── trainB
```

### Install Dependencies
```
Python 3.7 or higher
Pytorch 1.8.0, torchvison 0.9.0
Tensorboard, TensorboardX, Pyyaml, Pillow, dominate, visdom, timm, einops,  matplotlib, lpips, scikit-image, torch-fidelity
```

```
conda create translation python=3.13
conda activate translation
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu128 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install Tensorboard (TensorboardX, Pyyaml, Pillow, dominate, visdom, einops,  matplotlib, lpips, scikit-image, torch-fidelity ) -i https://pypi.tuna.tsinghua.edu.cn/simple

```

### Python
```
# Train for video mode
CUDA_VISIBLE_DEVICES=0  taskset -c 0-15 env PYTHONUNBUFFERED=1 TORCH_NUM_THREADS=8 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 nohup python -u train.py --dataroot datasets/Double/Traffic/City/clearday/ --name ICL_City_Double_STCT_balanced_135 --dataset_mode unaligned_double --model stct --local_nums 64 --no_flip --lambda_D_ViT 1.0 --lambda_GAN 1.0 --lambda_global 5.0 --lambda_NCE 0.2 --lambda_structure 1.0 --lambda_anti_hallucination 0.5 --lambda_temporal_token 1.0 --lambda_temporal_luma 0.5 --lambda_perception 0.0 --side_length 7 --atten_layers 1,3,5 --lr 0.00001 --n_epochs 50 --n_epochs_decay 50 --display_id -1 > logs/ICL_City_Double_STCT_balanced_135.log 2>&1 &

# Train for image mode
CUDA_VISIBLE_DEVICES=1 taskset -c 16-31 env PYTHONUNBUFFERED=1 TORCH_NUM_THREADS=8 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 nohup python -u train.py --dataroot datasets/Single/Traffic/City/clearday/ --name ICL_City_STCT_balanced_135 --dataset_mode unaligned --model stct --local_nums 64 --no_flip --lambda_D_ViT 1.0 --lambda_GAN 1.0 --lambda_global 5.0 --lambda_NCE 0.2 --lambda_structure 1.0 --lambda_anti_hallucination 0.5 --lambda_temporal_token 1.0 --lambda_temporal_luma 0.5 --lambda_perception 0.0 --side_length 7 --atten_layers 1,3,5 --lr 0.00001 --n_epochs 50 --n_epochs_decay 50 --display_id -1 > logs/ICL_City_STCT_balanced_135.log 2>&1 & 

##  Testing
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot /path/of/test_dataset --checkpoints_dir ./checkpoints --name ICL_City_STCT_balanced_135 --model stct --num_test 10000 --epoch latest
```

### Realism Evaluation
We use torch-fidelity (https://github.com/toshas/torch-fidelity) to evaluate the realism of the translated results.
```
FID
fidelity --gpu 0 --fid --input1  ./results/stct_name/test_latest/images/fake_B --input2 ./results/stct_name/test_latest/images/real_B
KID
fidelity --gpu 0 --kid --input1  ./results/stct_name/test_latest/images/fake_B --input2 ./results/stct_name/test_latest/images/real_B  --kid-subset-size 1000

CUDA_VISIBLE_DEVICES=1 fidelity --fid --kid --isc --kid-subset-size 1000 --input1  ./results/stct_name/test_latest/images/fake_B --input2  ./results/stct_name/test_latest/images/real_B > logs/fidelity/fidelity_IRVI_Monitor_stct_name_latest.log 2>&1 &
```

### Reference Links
```
https://github.com/BIT-DA/ROMA
https://github.com/silver-hzh/USTNet
```
