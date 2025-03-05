from numpy.distutils.system_info import openblas_clapack_info

### Requirements

```python
conda create -n hidisc_ssl
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install timm
pip install wandb
pip install tifffile
pip install dill
pip install hydra-core --upgrade
pip install matplotlib
pip install scikit-learn
pip install tqdm
pip install torchmetrics
pip install pandas
pip install opencv-python
pip install gdown
pip install ftfy regex tqdm
pip install transformers
## Adding mamba models ##
pip install packaging
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.1.4
pip install torchinfo timm numba
pip install monai==1.3.0

## Additional Packages for VMamba ##
pip install fvcore
cd VMamba/kernels/selective_scan && pip install .

```


### SSL Training

```python
# Single GPU

python main.py data.db_root=path/to/dataset data.dynamic_aug=False model.backbone=resnet50 training.attack=none
training.batch_size=10 root="./" out_dir="Results/single_gpu_ssl"

# Multi GPU

torchrun --nproc_per_node=$NUM_GPUS --master_port="$RANDOM" main.py data.db_root=path/to/dataset data.dynamic_aug=False model.backbone=resnet50 training.attack=none
training.batch_size=10 root="./" out_dir="Results/multi_gpu_ssl"
```

### SSL Training with Attack

```python
# Single GPU

python main.py data.db_root=path/to/dataset data.dynamic_aug=False model.backbone=resnet50 training.attack=pgd training.attack_eps=8.0
training.attack.alpha=2.0 training.attack.iters=5 training.batch_size=10 root="./" out_dir="Results/single_gpu_adv_ssl"

# Multi GPU

torchrun --nproc_per_node=$NUM_GPUS --master_port="$RANDOM" main.py data.db_root=path/to/dataset data.dynamic_aug=False model.backbone=resnet50 training.attack=pgd training.attack_eps=8.0
training.attack.alpha=2.0 training.attack.iters=5 training.batch_size=10 root="./" out_dir="Results/multi_gpu_adv_ssl"
```

All the logs and checkpoints will be saved in os.path.join(root, out_dir)

