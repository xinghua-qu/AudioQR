import os
import torch
from torch import nn, optim
from torch.nn import functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import commons
import utils
from data_utils import (
  TextAudioLoader,
  TextAudioCollate,
  DistributedBucketSampler
)
from models import NFTAudio
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from torchmetrics import SignalNoiseRatio
import math
from torch_audiomentations import Compose, Gain, PolarityInversion, AddBackgroundNoise, PitchShift, ApplyImpulseResponse, AddColoredNoise,HighPassFilter,LowPassFilter, Shift
from pathlib import Path
import torchaudio
torchaudio.set_audio_backend("sox_io")

def audio_ptb(input_audio, ptb_type):
    input_audio = input_audio.unsqueeze(1)
    if ptb_type=='none':
        new_audio = input_audio
        new_audio = torch.clamp(new_audio, -1, 1)
    if ptb_type=='gain':
        agmt = Gain(min_gain_in_db=9,max_gain_in_db=10,p=1)
        new_audio = agmt(input_audio, sample_rate=22050)
        new_audio = torch.clamp(new_audio, -1, 1)
    if ptb_type=='noise':
        new_audio = input_audio + 0.01*torch.randn_like(input_audio)
        new_audio = torch.clamp(new_audio, -1, 1)
    if ptb_type=='inversion':
        agmt = PolarityInversion(p=1)
        new_audio = agmt(input_audio, sample_rate = 22050)
        new_audio = torch.clamp(new_audio, -1, 1)
    if ptb_type=='shift':
        agmt = Shift(min_shift=0.1,max_shift=0.5,shift_unit="fraction",rollover=True,p=1)
        new_audio = agmt(input_audio, sample_rate = 22050)
        new_audio = torch.clamp(new_audio, -1, 1)
    if ptb_type=='env_background':
        env_wav_dir = './background_noise/environment/'
        agmt = AddBackgroundNoise(env_wav_dir, 1, p=1.0)
        new_audio = agmt(input_audio, sample_rate = 22050)
        new_audio = torch.clamp(new_audio, -1, 1)
    if ptb_type=='music_background':
        env_wav_dir = './background_noise/music/'
        agmt = AddBackgroundNoise(env_wav_dir, 1, p=1.0)
        new_audio = agmt(input_audio, sample_rate = 22050)
        new_audio = torch.clamp(new_audio, -1, 1)
    if ptb_type=='rir':
        rir_dir = './background_noise/rir_audios/'
        agmt = ApplyImpulseResponse(p=1,ir_paths = rir_dir, sample_rate = 22050)
        new_audio = agmt(input_audio, sample_rate = 22050)
        new_audio = torch.clamp(new_audio, -1, 1)
    return new_audio

def SNR(watermarked, original):
    # audio data type: tensor; dimension batch*8192.
    snr = SignalNoiseRatio().cuda()
    rate = snr(watermarked, original)
    return rate 

def evaluate(run_name, model, eval_loader, hps, ptb_type):
    model.eval()
    wave_length = 8192
    msg_length = 100
    l1_loss  = torch.nn.L1Loss()
    wav_index = np.random.randint(0, 32)
    amplitude = np.iinfo(np.int16).max
    ori_errors = []
    ptb_errors = []
    SNR_list = []
    ptb_SNR_list = []
    with torch.no_grad():
        for batch_idx, (message,spec, spec_lengths, wav, wav_lengths) in enumerate(eval_loader):
            spec    = spec.cuda(0)
            wav, wav_lengths = wav.cuda(0), wav_lengths.cuda(0)
            message = message.cuda(0)
            wav = wav.squeeze()

            watermarked_wav = torch.zeros_like(wav)# don't use empty like function
            watermarked_wav.copy_(wav)
            ptb_watermarked_wav = torch.zeros_like(wav)
            ptb_watermarked_wav.copy_(wav)

            error_rate = 0
            ptb_error_rate = 0
            watermark_times = wav_lengths[wav_index].item()//8192
            for i in range(watermark_times):
                wav_slice = torch.zeros_like(wav[wav_index,i*8192:(i+1)*8192])
                wav_slice.copy_(wav[wav_index,i*8192:(i+1)*8192])
                wav_slice = wav_slice.repeat(wav.size()[0],1)

                message_new = torch.zeros_like(message[wav_index,:])
                message_new.copy_(message[wav_index,:])
                message_new = message_new.repeat(wav.size()[0],1)

                recon_wav = torch.zeros_like(wav_slice)
                recon_msg = torch.zeros_like(message_new)
                recon_wav, recon_msg = nft_model(wav_slice, message_new)
                watermarked_wav[wav_index, i*8192:(i+1)*8192] = recon_wav[wav_index,:]

                err = torch.abs(message[wav_index].squeeze().round() -recon_msg[wav_index].squeeze().round()).sum()
                ori_errors.append(err.item())

            entire_ori_mel = mel_spectrogram_torch(
                  wav[wav_index:wav_index+1,:].float(), 
                  hps.data.filter_length, 
                  hps.data.n_mel_channels, 
                  hps.data.sampling_rate, 
                  hps.data.hop_length, 
                  hps.data.win_length, 
                  hps.data.mel_fmin, 
                  hps.data.mel_fmax
                )

            recon_mel = mel_spectrogram_torch(
                  watermarked_wav[wav_index:wav_index+1,:].float(), 
                  hps.data.filter_length, 
                  hps.data.n_mel_channels, 
                  hps.data.sampling_rate, 
                  hps.data.hop_length, 
                  hps.data.win_length, 
                  hps.data.mel_fmin, 
                  hps.data.mel_fmax
                )

            delta_mel = mel_spectrogram_torch(
                  watermarked_wav[wav_index:wav_index+1,:].float()-wav[wav_index:wav_index+1,:].float(), 
                  hps.data.filter_length, 
                  hps.data.n_mel_channels, 
                  hps.data.sampling_rate, 
                  hps.data.hop_length, 
                  hps.data.win_length, 
                  hps.data.mel_fmin, 
                  hps.data.mel_fmax
                )

            ptb_watermarked_wav = audio_ptb(watermarked_wav, ptb_type)
            for i in range(watermark_times):
                ptb_wav_slice = ptb_watermarked_wav[wav_index:wav_index+1,:,i*8192:(i+1)*8192]
                ptb_recon_msg = nft_model.decoder(ptb_wav_slice)
                ptb_err = torch.abs(message[wav_index].squeeze().round() -ptb_recon_msg[wav_index].squeeze().round()).sum()
                ptb_errors.append(ptb_err.item())
            ptb_watermarked_wav = ptb_watermarked_wav.squeeze()
            snr = SNR(watermarked_wav[wav_index,:], wav[wav_index,:])
            ptb_snr = SNR(ptb_watermarked_wav[wav_index,:], wav[wav_index,:])
            SNR_list.append(snr.item())
            ptb_SNR_list.append(ptb_snr.item())
            save_ori_wav = np.asarray(wav[wav_index:wav_index+1,:].cpu().detach().numpy()*amplitude)[0]
            save_wtm_wav = np.asarray(watermarked_wav[wav_index:wav_index+1,:].cpu().detach().numpy()*amplitude)[0]
            save_ptb_wtm_wav = np.asarray(ptb_watermarked_wav[wav_index:wav_index+1,:].cpu().detach().numpy()*amplitude)[0]
            delta = save_wtm_wav - save_ori_wav

            model_dir = os.path.join("./results", run_name)
            log_path = '{}/evaluations/{}_ptb/'.format(model_dir, ptb_type)
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            write("{}/original_{}.wav".format(log_path, batch_idx), 22050, save_ori_wav.astype(np.int16))
            write("{}/watermarked_{}.wav".format(log_path, batch_idx), 22050, save_wtm_wav.astype(np.int16))
            write("{}/{}_ptb_watermarked_{}.wav".format(log_path, ptb_type,batch_idx), 22050, save_ptb_wtm_wav.astype(np.int16))

            if batch_idx>=20:
                break
    x = np.arange(0, delta.shape[0])
    plt.figure()
    plt.plot(x, save_ori_wav, label = 'Original')
    plt.plot(x, save_wtm_wav, label = 'Watermarked')
    plt.plot(x, delta, label = 'delta')
    plt.legend()
    plt.savefig('{}/wav_comparison_{}.pdf'.format(log_path, batch_idx),bbox_inches='tight', dpi=600)
    plt.close() 

    x = np.arange(0, delta.shape[0])
    plt.figure()
    plt.plot(x, save_ori_wav, label = 'Original')
    plt.plot(x, save_wtm_wav, label = 'Watermarked')
    plt.plot(x, save_ptb_wtm_wav, label = '{}_Watermarked'.format(ptb_type))
    plt.legend()
    plt.savefig('{}/ptb_vs_nonptb_{}.pdf'.format(log_path, batch_idx),bbox_inches='tight', dpi=600)
    plt.close()

    np.savetxt("{}/ber.csv".format(log_path), ori_errors, delimiter =", ", fmt ='%1.9f')
    np.savetxt("{}/{}_ptb_ber.csv".format(log_path, ptb_type), ptb_errors, delimiter =", ", fmt ='%1.9f')
    np.savetxt("{}/snr.csv".format(log_path), SNR_list, delimiter =", ", fmt ='%1.9f')

