# AudioQR

- Our work has been accpted by IJCAI-2023.
- Our paper has been highlighted by IJCAI-2023 officially. [🔬The research #AI4SG1368 on "AudioQR: Deep Neural Audio Watermarks for QR Code" by Xinghua Qu, et al. presents an innovative application scenario for visually impaired individuals.](https://www.linkedin.com/feed/update/urn:li:activity:7085199256826830849/)

<br>
<p align="center">
  <img src="images/Image-Audio-QR.png" align="center" width="55%">
  <br>
</p>
<br>

[Youtube Video for Audio QR Code](https://www.youtube.com/watch?v=FmcZgRgMwEM)
[<img width="80%" src="https://github.com/xinghua-qu/AudioQR/assets/36146785/d736cc14-2a20-46aa-bad2-c881e38a4dd8">](https://www.youtube.com/watch?v=FmcZgRgMwEM "Audio QR Code")

## Abstract
Image-based quick response (QR) code is frequently used, but creates barriers for the visual impaired people. With the goal of AI for good, this paper proposes the AudioQR, a barrier-free QR coding mechanism for the visually impaired population via deep neural audio watermarks. Previous audio watermarking approaches are mainly based on handcrafted pipelines, which is less secure and difficult to apply in large-scale scenarios. In contrast, AudioQR is the first comprehensive end-to-end pipeline that hides watermarks in audio imperceptibly and robustly. To achieve this, we jointly train an encoder and decoder, where the encoder is structured as a concatenation of transposed convolutions and multi-receptive field fusion modules. Moreover, we customize the decoder training with a stochastic data augmentation chain to make the watermarked audio robust towards different audio distortions, such as environment background, room impulse response when playing through the air, music surrounding, and Gaussian noise. Experiment results indicate that AudioQR can efficiently hide arbitrary information into audio without introducing significant perceptible difference.
<br>
<p align="center">
  <img src="images/train_pipeline.png" align="center" width="90%">
  <br>
</p>
<br>

## Demos
TO-BE appear


## Prepared the environment

```pip3 install -r requirements.txt ```

## Download the required datasets

```bash dataset_download.sh```

## Run Training

```python3 train.py -c configs/ljs_base.json -m AUGsimple_10-20_Dual_decoder_mixed_mel_10_20k_20k_agmt_1_2k_200 --ptb_type mixed --mel_w 10 --mel_start 20000 --mel_len 20000 --agmt_w 1 --agmt_start 2000 --agmt_len 200 --batch_size 64 --msg_dim 50 --max_step 1000000```
