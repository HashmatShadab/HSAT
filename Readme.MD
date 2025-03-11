# **HSAT:Hierarchical Self-Supervised Adversarial Training for Robust Vision Models in Histopathology**

[Hashmat Shadab Malik](https://github.com/HashmatShadab),
[Shahina Kunhimon](https://github.com/ShahinaKK),
[Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en),
[Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en),
and [Salman Khan](https://salman-h-khan.github.io)

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2406.08486)

[//]: # ([![Video]&#40;https://img.shields.io/badge/Video-Presentation-F9D371&#41;]&#40;https://drive.google.com/file/d/1ZdUV83RvyL4mqCyxlqqD468VbRRAGdDY/view?usp=sharing&#41;)

[//]: # ([![slides]&#40;https://img.shields.io/badge/Poster-PDF-87CEEB&#41;]&#40;https://drive.google.com/file/d/1fvR4KUFCAEFO7wZqr-f8isk5FYMQvsT9/view?usp=sharing&#41;)

[//]: # ([![slides]&#40;https://img.shields.io/badge/Presentation-Slides-B762C1&#41;]&#40;https://drive.google.com/file/d/1osaG-OsgUlODRfRqDPK6f79bOLmRcC42/view?usp=sharing&#41;)


Official PyTorch implementation

<hr />

# :fire: News

* **(March 10, 2025)**
    * Training and evaluation codes are released.

<hr />


> **<p align="justify"> Abstract:** *Adversarial attacks pose significant challenges for vision models in critical
fields like healthcare,
> where
> reliability is essential.
> Although adversarial training has been well studied in natural images, its application to biomedical and microscopy
> data remains limited.
> Existing self-supervised adversarial training methods overlook the hierarchical structure of histopathology images,
> where patient-slide-patch
> relationships provide valuable discriminative signals. To address this, we propose Hierarchical Self-Supervised
> Adversarial Training (HSAT),
> which exploits these properties to craft adversarial examples using multi-level contrastive learning and integrate it
> into adversarial training for enhanced robustness.
> We evaluate HSAT on multiclass histopathology cancer diagnosis
> dataset OpenSRH and the results show that HSAT outperforms existing methods from both biomedical and natural
> image domains. HSAT enhances robustness, achieving an average gain of 54.31% in the white-box setting and
> reducing performance drops to 3-4% in the black-box setting, compared to 25-30% for the baseline. These results set a
> new benchmark for adversarial training in this domain, paving the way for more robust models. Our code and pretrained
> models will be made publicly available.* </p>

<hr />

<div align="center">
    <img src="./assets/conceptfigure.png" alt="Robust-LLaVA Diagram" width="800">

<p align="justify">
We propose <tt>HSAT</tt>, a <b>min-max optimization framework</b> to learn <i>robust representations</i> via self-supervised contrastive learning. The inner maximization step, <b>𝓛<sub>max</sub></b>, generates adversarial perturbations by crafting a <i>hierarchy-wise attack</i>. The outer minimization step, <b>𝓛<sub>min</sub></b>, updates the encoder <b>f<sub>θ</sub></b> to minimize the hierarchical contrastive loss, encouraging <b>robust feature representations</b> across <i>patch</i>, <i>slide</i>, and <i>patient</i> levels.
</p>

</div>

<hr />

## Contents

1) [Installation](#Installation)
2) [Models and Data Preparation](#Models-and-Data-Preparation)
3) [Training](#Training)
4) [Robustness against White-Box Attacks](#Robustness-against-White-Box-Attacks)
5) [Robustness against Transfer-Based Black-Box Attacks](#Robustness-against-Transfer-Based-Black-Box-Attacks)
6) [BibTeX](#bibtex)
7) [Contact](#contact)
8) [References](#references)

<hr>
<hr>



<a name="Installation"/>

## 💿 Installation

```python

conda
create - n
hsat
conda
activate
hsat
conda
install
pytorch == 2.1
.2
torchvision == 0.16
.2
torchaudio == 2.1
.2
pytorch - cuda = 11.8 - c
pytorch - c
nvidia
pip
install
timm
pip
install
wandb
pip
install
tifffile
pip
install
dill
pip
install
hydra - core - -upgrade
pip
install
matplotlib
pip
install
scikit - learn
pip
install
tqdm
pip
install
torchmetrics
pip
install
pandas
pip
install
opencv - python
pip
install
gdown
pip
install
ftfy
regex
tqdm
pip
install
transformers

```

<a name="Models-and-Data-Preparation"/>

## 🛠️ Models and Data Preparation

### 📦 Available Models

| Model                                        | Identifier                 |
|----------------------------------------------|----------------------------|
| ResNet-50 (Scratch)                          | `resnet50`                 |
| ResNet-50 (ImageNet Pretrained)              | `resnet50_timm_pretrained` |
| ResNet-50 (Adversarial ImageNet Pretrained)  | `resnet50_at`              |
| WResNet-50 (ImageNet Pretrained)             | `wresnet50_normal`         |
| WResNet-50 (Adversarial ImageNet Pretrained) | `wresnet50_at`             |
| ResNet-101 (ImageNet Pretrained)             | `resnet101_normal`         |
| ResNet-101 (Adversarial ImageNet Pretrained) | `resnet101_at`             |

---

### 📊 OpenSRH Dataset

To access the **OpenSRH** dataset, please submit a data request [here](https://opensrh.mlins.org).

---



<a name="Training"/>

# 🚀 Training

## 📚 Baseline Non-Adversarial Training

To train different vision models on the OpenSRH dataset with the baseline non-adversarial training approach from HiDisc,
run the following command:

```python
torchrun - -nproc_per_node = < NUM_GPUS > main.py
data.db_root = < DATA_PATH > data.dynamic_aug = False
data.dynamic_aug_version = v0
model.backbone = < MODEL_NAME >
training.batch_size = < BATCH_SIZE > training.only_adv = False
training.attack.name = none
training.attack.eps = 8
training.attack.warmup_epochs = 0
training.attack.loss_type = p_s_pt
out_dir = Results / Baseline / < MODEL_NAME > _exp1
wandb.exp_name = Baseline_backbone_ < MODEL_NAME > _exp1
wandb.use = True
```

- **`<NUM_GPUS>`**: Number of GPUs used for distributed training.
- **`<DATA_PATH>`**: Path to the OpenSRH dataset.
- **`<MODEL_NAME>`**: Name of the vision model backbone (for example `resnet50`, `resnet50_at`)
- **`<BATCH_SIZE>`**: Number of patients per batch during training.
- **`data.dynamic_aug=False:`** Disables dynamic augmentation, keeping the dataset unchanged for baseline training.
- **`training.only_adv=False:`** Ensures that only clean samples (non-adversarial) are used for training.
- **`training.attack.name=none:`** Disables adversarial attacks for baseline training.
- **`out_dir=Results/Baseline/<MODEL_NAME>_exp1:`** Path to save the training results.
- **`wandb.exp_name=Baseline_backbone_<MODEL_NAME>_exp1:`** Sets the experiment name for tracking in Weights & Biases (
  W&B).
- **`wandb.use=True:`** Enables logging with Weights & Biases for visualizing training metrics.

## 📚 Hierarchical Self-Supervised Adversarial Training (HSAT)

To train different vision models on the OpenSRH dataset with the proposed Hierarchical Self-Supervised Adversarial
Training (HSAT) approach, run the following command:

```python
torchrun - -nproc_per_node = < NUM_GPUS > main.py
data.db_root = < DATA_PATH > data.dynamic_aug = True
data.dynamic_aug_version = v0
model.backbone = < MODEL_NAME >
training.batch_size = < BATCH_SIZE > training.only_adv = True
training.attack.name = pgd
training.attack.eps = 8
training.attack.warmup_epochs = 5000
training.attack.loss_type = p_s_pt
out_dir = Results / Adv / < MODEL_NAME > _dynamicaug_true_epsilon_warmup_5000_only_adv_exp2
wandb.exp_name = Adv_backbone_ < MODEL_NAME > _dynamicaug_true_epsilon_warmup_5000_only_adv_exp2
wandb.use = True
```

- **`training.attack.name=pgd:`** Adversarial attack used for HSAT training.

- **`training.attack.eps=8`:** Sets the maximum perturbation allowed for adversarial examples.

- **`training.attack.warmup_epochs=0:`** Number of warm-up epochs to linearly increase the adversarial attack strength.

- **`training.attack.loss_type=p_s_pt:`** Specifies the maximization loss type for crafting adversarial examples during
  training.
    - `p_s_pt` is a combination of the patch-level, slide-level, and patient-level contrastive losses.
    - `pt` is the patch-level contrastive loss.
    - `s_pt` is the combination of the slide-level and patch-level contrastive losses.

To train different vision models on the OpenSRH dataset with the proposed Hierarchical Self-Supervised Adversarial
Training (HSAT) approach but using
different levels of heirarchical contrastive losses during the training, run the following command:

```python
# HAT-Patch Testing the effect of patch loss only (L_max = L_min) on the adversarial training. Increase the batch size accordingly to keep the same effective batch size.
torchrun - -nproc_per_node = < NUM_GPUS > main.py
data.db_root = < DATA_PATH > data.dynamic_aug = True
data.dynamic_aug_version = v0
data.hidisc.num_slide_samples = 1
data.hidisc.num_patch_samples = 1
model.backbone = < MODEL_NAME >
training.batch_size = < BATCH_SIZE > training.only_adv = True
training.attack.name = pgd
training.attack.eps = 8
training.attack.warmup_epochs = 5000
training.attack.loss_type = p_s_pt
out_dir = Results / Adv / < MODEL_NAME > _dynamicaug_true_epsilon_warmup_5000_only_adv_hat_patch_exp13
wandb.exp_name = Adv_backbone_ < MODEL_NAME > _dynamicaug_true_epsilon_warmup_5000_only_adv_hat_patch_exp13
wandb.use = True
```

```python
#  HAT-Slide Testing the effect of patch-slide loss only (L_max = L_min) on the adversarial training. Increase the batch size accordingly to keep the same effective batch size.
torchrun - -nproc_per_node = < NUM_GPUS > main.py
data.db_root = < DATA_PATH > data.dynamic_aug = True
data.dynamic_aug_version = v0
data.hidisc.num_patch_samples = 1
model.backbone = < MODEL_NAME >
training.batch_size = < BATCH_SIZE > training.only_adv = True
training.attack.name = pgd
training.attack.eps = 8
training.attack.warmup_epochs = 5000
training.attack.loss_type = p_s_pt
out_dir = Results / Adv / < MODEL_NAME > _dynamicaug_true_epsilon_warmup_5000_only_adv_hat_slide_exp14
wandb.exp_name = Adv_backbone_ < MODEL_NAME > _dynamicaug_true_epsilon_warmup_5000_only_adv_hat_slide_exp14
wandb.use = True
```

We provide bash scripts to train different vision models on the OpenSRH dataset with the baseline and proposed
Hierarchical Self-Supervised Adversarial Training (HSAT) approach.

```python
bash
scripts / training.sh < NUM_GPUS > < BATCH_SIZE > < EXP_NUM > < MODEL_NAME > < DATA_PATH >

# For example, training ResNet-50(scratch) on OpenSRH dataset with Baseline training using 4 GPU, batch size 16
bash
scripts / training.sh
4
16
1
resnet50
path / to / dataset

# For example, training ResNet-50(scratch) on OpenSRH dataset with HSAT training using 4 GPU, batch size 16
bash
scripts / training.sh
4
16
2
resnet50
path / to / dataset
```

Please check the `scripts/training.sh` file for more details. The logs and trained models will be saved in the `Results`
folder.


<a name="Robustness-against-White-Box-Attacks"/>

## 🛡️ Robustness against White-Box Attacks

```python
# Pixel-based PGD attack on Volumetric Segmentation models
python
wb_attack.py - -model_name < MODEL_NAME > --in_channels
1 - -out_channel < NUM_CLASSES > --checkpoint_path < MODEL_CKPT_PATH > --dataset < DATASET_NAME >
--data_dir = < DATA_PATH > --json_list < DATA_JSON_FILE > --attack_name
pgd - -eps
8 - -steps
20

# Pixel-based CosPGD attack on Volumetric Segmentation models
python
wb_attack.py - -model_name < MODEL_NAME > --in_channels
1 - -out_channel < NUM_CLASSES > --checkpoint_path < MODEL_CKPT_PATH > --dataset < DATASET_NAME >
--data_dir = < DATA_PATH > --json_list < DATA_JSON_FILE > --attack_name
cospgd - -eps
8 - -steps
20

# Pixel-based FGSM attack on Volumetric Segmentation models
python
wb_attack.py - -model_name < MODEL_NAME > --in_channels
1 - -out_channel < NUM_CLASSES > --checkpoint_path < MODEL_CKPT_PATH > --dataset < DATASET_NAME >
--data_dir = < DATA_PATH > --json_list < DATA_JSON_FILE > --attack_name
fgsm - -eps
8

# Pixel-based GN attack on Volumetric Segmentation models
python
wb_attack.py - -model_name < MODEL_NAME > --in_channels
1 - -out_channel < NUM_CLASSES > --checkpoint_path < MODEL_CKPT_PATH > --dataset < DATASET_NAME >
--data_dir = < DATA_PATH > --json_list < DATA_JSON_FILE > --attack_name
fgsm - -std
8

# Frequency-based VAFA attack on Volumetric Segmentation models
python
wb_attack.py - -model_name < MODEL_NAME > --in_channels
1 - -out_channel < NUM_CLASSES > --checkpoint_path < MODEL_CKPT_PATH > --dataset < DATASET_NAME >
--data_dir = < DATA_PATH > --json_list < DATA_JSON_FILE > --attack_name
vafa - 3
d - --q_max
30 - -steps
20 - -block_size
32
32
32 - -use_ssim_loss
True
```

Available attacks: Fast Gradient Sign Method ([FGSM](https://arxiv.org/abs/1412.6572)), Projected Gradient
Descent ([PGD](https://arxiv.org/abs/1706.06083)), Cosine Projected Gradient
Descent ([CosPGD](https://arxiv.org/abs/2302.02213)), Gaussian Noise (GN), and Volumetric Adversarial Frequency
Attack ([VAFA](https://arxiv.org/abs/2307.07269))

`--eps`: Perturbation budget for Pixel-based adversarial attacks

`--std`: Perturbation budget for Gaussian Noise attack

`--q_max`: Maximum quantization level for VAFA attack

`--block_size`: Block size for VAFA attack

`--use_ssim_loss`: Use SSIM loss for VAFA attack

`--steps`: Number of attack steps for iterative attacks

To run the above attacks across all models and datasets, run the following scripts:

```python
# Pixel and Frequency-based attacks on Volumetric Segmentation models trained on BTCV dataset
bash
scripts / btcv / attacks.sh

# Pixel and Frequency-based attacks on Volumetric Segmentation models trained on Hecktor dataset
bash
scripts / hecktor / attacks.sh

# Pixel and Frequency-based attacks on Volumetric Segmentation models trained on ACDC dataset
bash
scripts / acdc / attacks.sh

# Pixel and Frequency-based attacks on Volumetric Segmentation models trained on Abdomen-CT dataset
bash
scripts / abdomen / attacks.sh
```

In the above scripts replace the following arguments:

`<DATA_DIR>`: Path to the dataset

`<model_names>`: name of the models and their corresponding checkpoints in `<ckpt_paths>`

The generated adversarial images and logs will be saved in the same folder as from where the model checkpoint was
loaded.


<a name="Robustness-against-Transfer-Based-Black-Box-Attacks"/>

## 🛡️ Robustness against Transfer-Based Black-Box Attacks

After generating adversarial examples using a surrogate model, the transferability of adversarial examples can be
reported by evaluating them on unseen target models trained on the same dataset.
To evaluate any target model on the adversarial examples, run the following script:

```python
# Transferability on BTCV adversarial examples
python
inference_on_adv_images.py - -model_name < MODEL_NAME > --in_channels
1 - -out_channel
14 - -checkpoint_path < BTCV_MODEL_CKPT_PATH > --dataset
btcv
--data_dir = < ORIG_BTCV_DATA_PATH > --json_list
dataset_synapse_18_12.json - -adv_imgs_dir < PATH_TO_BTCV_ADVERSARIAL_IMAGES >

# Transferability on Hecktor adversarial examples
python
inference_on_adv_images.py - -model_name < MODEL_NAME > --in_channels
1 - -out_channel
3 - -checkpoint_path < HECKTOR_MODEL_CKPT_PATH > --dataset
hecktor
--data_dir = < ORIG_HECKTOR_DATA_PATH > --json_list
dataset_hecktor.json - -adv_imgs_dir < PATH_TO_HECKTOR_ADVERSARIAL_IMAGES >

# Transferability on ACDC adversarial examples
python
inference_on_adv_images.py - -model_name < MODEL_NAME > --in_channels
1 - -out_channel
4 - -checkpoint_path < ACDC_MODEL_CKPT_PATH > --dataset
acdc
--data_dir = < ORIG_ACDC_DATA_PATH > --json_list
dataset_acdc_140_20_.json - -adv_imgs_dir < PATH_TO_ACDC_ADVERSARIAL_IMAGES >

# Transferability on Abdomen-CT adversarial examples
python
inference_on_adv_images.py - -model_name < MODEL_NAME > --in_channels
1 - -out_channel
14 - -checkpoint_path < ABDOMEN_MODEL_CKPT_PATH > --dataset
abdomen
--data_dir = < ORIG_ABDOMEN_DATA_PATH > --json_list
dataset_abdomen.json - -adv_imgs_dir < PATH_TO_ABDOMEN_ADVERSARIAL_IMAGES >
```

Furthermore, bash scripts are provided to evaluate transferability of adversarial examples across different models(given
the adversarial examples are generated first across all models and datasets):

```python
# Transferability of BTCV adversarial examples across all models
bash
scripts / btcv / transferability.sh

# Transferability of Hecktor adversarial examples across all models
bash
scripts / hecktor / transferability.sh

# Transferability of ACDC adversarial examples across all models
bash
scripts / acdc / transferability.sh

# Transferability of Abdomen-CT adversarial examples across all models
bash
scripts / abdomen / transferability.sh
```

The evaluation logs for target models will be saved in the same folder as from where the adversarial examples were
loaded.

## Quantitative Evaluation 📊

<div align="center">
    <img src="./assets/table1.png" alt="Robust-LLaVA Diagram" width="800">

<p align="justify">
We observe that vision models trained using our <tt>HSAT</tt> framework exhibit significantly higher adversarial robustness in white-box settings compared to baseline methods. Under the PGD attack at ϵ = <tt>8/255</tt> using a ResNet-50 backbone, <tt>HSAT</tt> achieves gains of <b>43.90%</b>, <b>60.70%</b>, and <b>58.33%</b> in patch, slide, and patient classification, respectively, compared to non-adversarial hierarchical training methods. Additionally, <tt>HSAT</tt> outperforms instance-level adversarial training (<tt>HSAT-Patch</tt>) by <b>6.68%</b>, <b>10.51%</b>, and <b>10%</b> across the same tasks. Despite a slight drop in clean accuracy compared to non-adversarial methods, <tt>HSAT</tt> maintains a superior clean performance with improvements of <b>15.68%</b>, <b>17.12%</b>, and <b>20%</b> over <tt>HSAT-Patch</tt>.
</p>



<div align="center">
    <img src="./assets/table2.png" alt="Robust-LLaVA Diagram" width="800">

<p align="justify">
We observe that target vision models trained using our <tt>HSAT</tt> framework are 
significantly more robust against transferability of adversarial examples crafted across different 
surrogate models. On average, <tt>HSAT</tt> trained target models show a performance 
drop of around <b>3-4%</b>, while target models trained using 
<a href="https://arxiv.org/abs/your-citation-link" target="_blank" style="color: #007bff; text-decoration: underline;">Hidisc</a> 
show a performance drop of around <b>25-30%</b>.
</p>

</div>

<div align="center">
    <img src="./assets/table3_4.png" alt="Robust-LLaVA Diagram" width="800">

<p align="justify">
Ablation studies for <i>Hierarchical Adversarial Training</i> (<tt>HSAT</tt>) in Tables <b>3</b> and <b>4</b> show that increasing hierarchical discrimination—progressing from patch-level (<tt>HSAT-Patch</tt>) to slide-level (<tt>HSAT-Slide</tt>) and patient-level (<tt>HSAT-Patient</tt>)—consistently improves adversarial robustness. Table <b>2</b> further reveals that combining loss terms, from patch-level <b>𝓛<sub>con</sub><sup>patch</sup></b> to slide-level <b>𝓛<sub>con</sub><sup>slide</sup></b> and patient-level <b>𝓛<sub>con</sub><sup>patient</sup></b>, enhances model resilience. These results highlight the effectiveness of multi-level adversarial training in aligning robust representations across hierarchical levels.
</p>

</div>



---


<a name="bibtex"/>

## 📚 BibTeX

```bibtex

```

<hr />

<a name="contact"/>

## 📧 Contact

Should you have any question, please create an issue on this repository or contact at hashmat.malik@mbzuai.ac.ae

<hr />

<a name="references"/>

## 📚 References

Our code is based
on [hidisc](https://github.com/MLNeurosurg/hidisc). We thank them for open-sourcing their codebase.


