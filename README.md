<div align="center">

<h2> Audio-driven Facial Reenactment using Text-to-speech (TTS) <br/> <span style="font-size:12px">Audio-driven Facial Reenactment using Text-to-speech (TTS)</span> </h2> 

  <a href='https://arxiv.org/abs/2211.14758'><img src='https://img.shields.io/badge/ArXiv-2211.14758-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://vinthony.github.io/video-retalking/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

<div>
    <a target='_blank'>Rémi Decelle <sup>*,1,2</sup> </a>&emsp;
    <a href='https://vinthony.github.io/' target='_blank'>Serge Miguet <sup>*,2</a>&emsp;
</div>
<br>
<div>
    <sup>1</sup> Université de Lyon 2 &emsp; <sup>2</sup> Mon Petit Placement
</div>
<br>
<i><strong><a href='https://sa2022.siggraph.org/' target='_blank'>Conference/Journal Name</a></strong></i>
<br>
<br>
Link to img

<div align="justify"> We present something</div>
<br>

Link to img

</div>

## Results in the Wild （contains audio）

Link to video


## Requirements

- Python 3.8.16
- torch 1.10.1+cu113
- torchvision 0.11.2+cu113
- ffmpeg

We recommend to install [pytorch](https://pytorch.org/) firstly，and then install related toolkit by running
```
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```
You also can use environment.yml to install relevant environment by running
```
conda env create -f environment.yml
```

## Quick Inference

We trained a pretrained model on LRS2 dataset. You can quickly try it by running:
```
python inference.py --checkpoint_path_BASE=checkpoints/model_lrs2.pth 
```
The result is saved (by default) in `results/result_video.mp4`. To inference on other videos, please specify the `--face` and `--audio` option and see more details in code.
You can also precise `--text`, if not the text will be automatically extracted from the audio.


## Pre-processing datas for Training
```
conda -n mfa-env
conda install -c conda-forge montreal-forced-aligner
mfa model download acoustic french_mfa
mfa model download dictionary french_mfa
```

Use another acoustic and dictionnary if needed : [acoustic models](https://mfa-models.readthedocs.io/en/latest/acoustic/index.html) and  [dictionaries](https://mfa-models.readthedocs.io/en/latest/dictionary/index.html)

#### Converting video to audio (wav format)
```
python3 preprocessing/video2audio.py [CORPUS_DIRECTORY]
```

#### Extract encodec from audio
```
python3 preprocessing/audio2codes.py [CORPUS_DIRECTORY]
```

#### Align text to audio
```
conda activate mfa-env
mfa align CORPUS_DIRECTORY DICTIONARY_NAME ACOUSTIC_MODEL_NAME OUTPUT_DIRECTORY
conda deactivate
```

#### Convert phonemes to one-hot vector
(Really needed ?)
```
python3 preprocessing/phoneme2vector.py
```

#### Pretrained Models
Please download our [pre-trained models](https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0?usp=share_link) and put them in `./checkpoints`.

<!-- We also provide some [example videos and audio](https://drive.google.com/drive/folders/14OwbNGDCAMPPdY-l_xO1axpUjkPxI9Dv?usp=share_link). Please put them in `./examples`. -->

#### Inference

```
python3 inference.py \
  --face examples/face/1.mp4 \
  --audio examples/audio/1.wav \
  --outfile results/1_1.mp4
```
This script includes data preprocessing steps. You can test any talking face videos without manual alignment. But it is worth noting that DNet cannot handle extreme poses.

You can also control the expression by adding the following parameters:

```--exp_img```: Pre-defined expression template. The default is "neutral". You can choose "smile" or an image path.

```--up_face```: You can choose "surprise" or "angry" to modify the expression of upper face with [GANimation](https://github.com/donydchen/ganimation_replicate).


## Acknowledgement
Thanks to
[Wav2Lip](https://github.com/Rudrabha/Wav2Lip),
[PIRenderer](https://github.com/RenYurui/PIRender), 
[GFP-GAN](https://github.com/TencentARC/GFPGAN), 
[GPEN](https://github.com/yangxy/GPEN),
[ganimation_replicate](https://github.com/donydchen/ganimation_replicate),
[STIT](https://github.com/rotemtzaban/STIT),
[IP_LAP](https://github.com/Weizhi-Zhong/IP_LAP),
[VideoRetalking](https://github.com/OpenTalker/video-retalking),
for sharing their code.

## Related Work
- [StyleHEAT: One-Shot High-Resolution Editable Talking Face Generation via Pre-trained StyleGAN (ECCV 2022)](https://github.com/FeiiYin/StyleHEAT)
- [CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior (CVPR 2023)](https://github.com/Doubiiu/CodeTalker)
- [SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023)](https://github.com/Winfredy/SadTalker)
- [DPE: Disentanglement of Pose and Expression for General Video Portrait Editing (CVPR 2023)](https://github.com/Carlyx/DPE)
- [3D GAN Inversion with Facial Symmetry Prior (CVPR 2023)](https://github.com/FeiiYin/SPI/)
- [T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations (CVPR 2023)](https://github.com/Mael-zys/T2M-GPT)

##  Disclaimer

This is not an official product of Tencent. This repository can only be used for personal/research/non-commercial purposes.

