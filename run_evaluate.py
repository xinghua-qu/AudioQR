import os
import argparse
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
from augmentation_utils import _get_sample
from torch_audiomentations import Compose, Gain, PolarityInversion, AddBackgroundNoise, PitchShift, ApplyImpulseResponse, AddColoredNoise,HighPassFilter,LowPassFilter, Shift
from pathlib import Path
import torchaudio
torchaudio.set_audio_backend("sox_io")

def audio_ptb(input_audio, ptb_type):
    input_audio = input_audio.unsqueeze(1)
    agmenter = audio_augmenter(ptb_type)
    new_audio = agmenter(input_audio, sample_rate = 22050)
    new_audio = torch.clamp(new_audio, -1, 1)
    return new_audio

def SNR(watermarked, original):
    # audio data type: tensor; dimension batch*8192.
    snr = SignalNoiseRatio().cuda()
    rate = snr(watermarked, original)
    return rate  

def audio_augmenter(ptb_type):
    if ptb_type=='noise':
        agmt = AddColoredNoise(p=1.0, min_snr_in_db=15, max_snr_in_db=20, min_f_decay=0, max_f_decay=0.0000001)
    if ptb_type=='env_background':
        env_wav_dir = './background_noise_simple/environment/'
        agmt = AddBackgroundNoise(env_wav_dir, p=1.0, min_snr_in_db=10, max_snr_in_db=20)
    if ptb_type=='music_background':
        music_wav_dir = './background_noise_simple/music/'
        agmt = AddBackgroundNoise(music_wav_dir,  p=1.0, min_snr_in_db=10, max_snr_in_db=20)
    if ptb_type=='rir':
        rir_dir = './background_noise_simple/rir_audios/'
        agmt = ApplyImpulseResponse(p=1,ir_paths = rir_dir, sample_rate = 22050, compensate_for_propagation_delay=False)
    if ptb_type=='mixed':
        env_wav_dir   = './background_noise_simple/environment/'
        music_wav_dir = './background_noise_simple/music/'
        rir_wav_dir   = './background_noise_simple/rir_audios/'
        ptb_prob      = 0.75
        agmt = Compose(transforms=[
            AddColoredNoise(p=ptb_prob, min_snr_in_db=15, max_snr_in_db=20, min_f_decay=0, max_f_decay=0.0000001),
            AddBackgroundNoise(env_wav_dir, p=ptb_prob, min_snr_in_db=10, max_snr_in_db=20),
            AddBackgroundNoise(music_wav_dir, p=ptb_prob, min_snr_in_db=10, max_snr_in_db=20),
            ApplyImpulseResponse(p=ptb_prob,ir_paths = rir_wav_dir, sample_rate = 22050, compensate_for_propagation_delay=False)])
    return agmt

def main(args):
    run_name = args.model
    step = args.step
    model_path = './results/{}/AS_model.pth'.format(run_name)
    wave_length = 8192
    msg_dim    = args.msg_dim

    config_save_path = './configs/ljs_base.json'
    with open(config_save_path, "r") as f:
        data = f.read()
    import json
    from utils import HParams
    config = json.loads(data)
    hps = HParams(**config)
    hps.train.segment_size = 8192
    hps.train.eval_interval = 10000
    hps.train.log_interval = 500
    hps.train.batch_size = 1
    collate_fn = TextAudioCollate(msg_dim)
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data, msg_dim)
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=True,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    
    ptb_list = ['env_background','music_background',  'noise', 'rir', 'mixed']
    for ptb_type in ptb_list:
        net_nft = NFTAudio(wave_length, msg_dim, ptb_type).cuda(0)
        check_point = torch.load(model_path)
        net_nft.load_state_dict(check_point['model'])
        net_nft.eval()
        l1_loss  = torch.nn.L1Loss()
        wav_index = 0
        amplitude = np.iinfo(np.int16).max
        ori_errors = []
        ptb_errors = []
        SNR_list = []
        ptb_SNR_list = []
        with torch.no_grad():
            for batch_idx, (message,wav, wav_lengths) in enumerate(eval_loader):
                wav, wav_lengths = wav.cuda(0), wav_lengths.cuda(0)
                message = message.cuda(0)
                wav = wav.squeeze(dim=1)

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
                    recon_wav, recon_msg, recon_features, agmt_recon_wav, agmt_recon_msgs, agmt_features =  net_nft(wav_slice, message_new)
                    recon_msg = recon_msg.unsqueeze(0)
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
                watermarked_wav = watermarked_wav.squeeze(dim=1)
                for i in range(watermark_times):
                    ptb_wav_slice = ptb_watermarked_wav[wav_index:wav_index+1,:,i*8192:(i+1)*8192]
                    ptb_recon_msg,_ = net_nft.robust_decoder(ptb_wav_slice)                
                    ptb_err = torch.abs(message[wav_index].squeeze().round() -ptb_recon_msg.squeeze().round()).sum()
                    ptb_errors.append(ptb_err.item())
                ptb_watermarked_wav = ptb_watermarked_wav.squeeze(dim=1)
                snr = SNR(watermarked_wav[wav_index,:].squeeze(), wav[wav_index,:].squeeze())
                ptb_snr = SNR(ptb_watermarked_wav[wav_index,:].squeeze(), wav[wav_index,:].squeeze())
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
                write("{}/watermarked_step_{}.wav".format(log_path, batch_idx), 22050, save_wtm_wav.astype(np.int16))
                write("{}/ptb_watermarked_step_{}.wav".format(log_path, batch_idx), 22050, save_ptb_wtm_wav.astype(np.int16))
                if batch_idx>=99:
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

    
if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                      help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True,
                      help='Model name')
    parser.add_argument('-p', '--ptb_type', type=str, required=True,
                      help='The type of perturbation during training')
    parser.add_argument('-s', '--step', type=int, required=True,
                      help='The step of model checkpoint')
    parser.add_argument('--msg_dim', type=int, default=50)
    args = parser.parse_args()
    main(args)