# VDTR: Video Deblurring with Transformer

[Mingdeng Cao](https://github.com/ljzycmd), [Yanbo Fan](https://sites.google.com/site/yanbofan0124/), [Yong Zhang](https://yzhang2016.github.io/yongnorriszhang.github.io/), [Jue Wang](https://juew.org/) and [Yujiu Yang](https://scholar.google.com/citations?user=4gH3sxsAAAAJ).

---

[[arXiv](https://arxiv.org/abs/2204.08023)]

We propose Video Deblurring Transformer (VDTR), a simple yet effective model that takes advantage of the long-range and relation modeling characteristics of the Transformer for video deblurring. VDTR utilizes pure Transformer for both spatial and temporal modeling and obtains highly competitive performance on the popular video deblurring benchmarks.

<div align=center> 
<img src=./assets/deblur_demo.gif>
</div>

<div align=center> 
<img src=./assets/deblur_demo2.gif>
</div>

VDTR surpasses CNN-based state-of-the-art methods more than 1.5dB PSNR with moderate computational costs:
<div align=center>
<img src=assets/comparison_teaser.png width=50% />
</div>

---
> Spatio-temporal learning is significant for video deblurring, which is dominated by convolution-based methods. This paper presents VDTR, an effective Transformer-based model that makes the first attempt to adapt the Transformer for video deblurring. VDTR exploits the superior long-range and relation modeling capabilities of Transformer for both spatial and temporal modeling. However, it is challenging to design an appropriate Transformer-based model for video deblurring due to the high computational costs for high-resolution spatial modeling and the misalignment across frames for temporal modeling. To address these problems, VDTR advocates performing attention within non-overlapping windows and exploiting the hierarchical structure for long-range dependencies modeling. For frame-level spatial modeling, we propose an encoder-decoder Transformer that utilizes multi-scale features for deblurring. For multi-frame temporal modeling, we adapt the Transformer to fuse multiple spatial features efficiently. Compared with CNN-based methods, the proposed method achieves highly competitive results on both synthetic and real-world video deblurring benchmarks, including DVD, GOPRO, REDS and BSD. We hope such a pure Transformer-based architecture can serve as a powerful alternative baseline for video deblurring and other video restoration tasks.

<div align=center> 
<img src=./assets/model_arch.png>
Model Architecture
</div>

### Environment
* Python 3.8
* PyTorch >= 1.5
```bash
git clone  git clone https://github.com/ljzycmd/SimDeblur.git

# install the SimDeblur
cd SimDeblur
bash Install.sh
```

### Quick Start

1. Clone the codes of VDTR
```bash
git clone https://github.com/ljzycmd/VDTR.git
```

2. Download and unzip the datasets

* [DVD](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/)

* [GOPRO](https://seungjunnah.github.io/Datasets/gopro)

* [BSD](https://github.com/zzh-tech/ESTRNN)

Then create the soft links of the datasets to the `./datasets` folder.

3. Run the training script

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10086 train.py ./configs/vdtr/vdtr_dvd.yaml --gpus=4
```
the training logs are saved in ./workdir/*

4. Run the testing script (single GPU, minimum requirement RTX 2080Ti)

```bash
python test.py ./configs/vdtr/vdtr_dvd.yaml $Checkpoint_path
```
the testing logs and frames are saved in `./workdir/*`. 

**Pretrained checkpoints are listed:**

| Model  | Dataset | Download |
|:------:|:------:|:--------:|
|  VDTR  | DVD | [Google drive](https://drive.google.com/file/d/1wNukRvhBcfG2KlU8WUzcCxhSKLvjpIEw/view?usp=sharing) |
|  VDTR  | GOPRO | [Google drive](https://drive.google.com/file/d/1zmAmjOa0sdXl0rNWxB6tF7QlDYuQpqDg/view?usp=sharing)|
|  VDTR  | BSD-1ms | [Goodle drive](https://drive.google.com/file/d/1W5z2b6k3iEA0qS3Ytvt_4893_tbklPBD/view?usp=share_link) |
|  VDTR  | BSD-2ms | [Goodle drive](https://drive.google.com/file/d/1w1bUjDV8OoT8s89BE0_rdbwAQjcre8_F/view?usp=share_link) |
|  VDTR  | BSD-3ms | [Goodle drive](https://drive.google.com/file/d/1yt2_28LlWHwB32_QrouIOtWLk5xVJkOH/view?usp=share_link) |
### Experimental Results

VDTR achieves competitive PSNR and SSIM on both synthetic and real-world deblurring datasets.

**Quantitative results on popular video deblurring datasets: DVD, GOPRO, REDS**
![qualitative_comparison](./assets/results/results_on_synthetic_datasets.png)

**Qualitative comparison to state-of-the-art video deblurring methods on GOPRO**
![qualitative_comparison](./assets/results/vis_comparison_gopro.png)

**Quantitative results on real-world video deblurring datasets: BSD**
<div align=center>
<img src=./assets/results/results_on_realworld.png>
</div>

**Qualitative comparison to state-of-the-art video deblurring methods on BSD**
<div align=center>
<img src=./assets/results/vis_comparison_bsd.png>
</div>

### Citation

If the proposed model is useful for your research, please consider citing

```bibtex
@article{cao2022vdtr,
  title   = {VDTR: Video Deblurring with Transformer},
  author  = {Mingdeng Cao and Yanbo Fan and Yong Zhang and Jue Wang and Yujiu Yang},
  journal = {arXiv:2204.08023},
  year    = {2022}
}
```
