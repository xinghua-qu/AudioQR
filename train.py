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
from augmentation_utils import _get_sample
import random
import math
from torch_audiomentations import Compose, Gain, PolarityInversion, AddBackgroundNoise, PitchShift, ApplyImpulseResponse, AddColoredNoise
import torchaudio
from torchmetrics import SignalNoiseRatio
from pqmf import PQMF
from losses import feature_loss, hidden_feature_loss

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    assert torch.cuda.is_available(), "CPU training is not allowed."
    n_gpus = torch.cuda.device_count()
    
    hps = utils.get_hparams()
    hps.train.segment_size = 8192
    hps.train.eval_interval = 1000
    hps.train.log_interval = 500
    hps.train.batch_size = hps.batch_size
    
    global global_step
    rank = 0
    n_gpus = 1
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)

    log_path = os.path.join(hps.model_dir, 'tensorboard')
    writer = SummaryWriter(log_path)
    log_path = os.path.join(hps.model_dir, 'tensorboard/eval')
    writer_eval = SummaryWriter(log_path)
   

    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.manual_seed_all(hps.train.seed)
    msg_length  = hps.msg_dim

    train_dataset = TextAudioLoader(hps.data.training_files, hps.data, msg_length)
    train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True) # this line of code ensures the length of audio data has similar input lengths in a batch

    collate_fn = TextAudioCollate(msg_length)
    train_loader = DataLoader(train_dataset, num_workers=16, shuffle=False, pin_memory=True, prefetch_factor=4,
      collate_fn=collate_fn, batch_sampler=train_sampler)

    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data, msg_length)
    eval_batch_size = hps.train.batch_size
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False, prefetch_factor=2,
        batch_size=1, pin_memory=True, collate_fn=collate_fn)

    wave_length = hps.train.segment_size
    lr  = hps.lr ## learning rate setting
    net_nft = NFTAudio(wave_length, msg_length, hps.ptb_type).cuda(rank)
    optim_nft = torch.optim.AdamW(
      net_nft.parameters(), 
      lr, 
      betas=hps.train.betas, 
      eps=hps.train.eps)

    epoch_str = 1
    global_step = 0

    scheduler_nft = torch.optim.lr_scheduler.ExponentialLR(optim_nft, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(rank, epoch, hps, net_nft, optim_nft, scheduler_nft, scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
        scheduler_nft.step()

def train_and_evaluate(rank, epoch, hps, net_nft, optims, schedulers, scaler, loaders, logger, writers):
    optim_nft= optims
    scheduler_nft= schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_nft.train()
    l1_loss  = torch.nn.L1Loss()
    celoss   = torch.nn.BCELoss()
    
    pqmf_analyser = PQMF(subbands=4, taps=62, cutoff_ratio=0.15, beta=9.0, device='cuda')
    pqmf_weight = torch.tensor([10, 1, 0.1, 0.01])
    pqmf_weights= pqmf_weight.repeat(hps.train.batch_size, 1).cuda()

    for batch_idx, (message, wav, wav_lengths) in enumerate(train_loader):
        wav, wav_lengths = wav.cuda(rank, non_blocking=True), wav_lengths.cuda(rank, non_blocking=True)# wav is in [-1,1]
        message  = message.cuda(rank, non_blocking=True)
        for _ in range(hps.iter):
            wav_slice, _ = commons.rand_slice_segments(wav, wav_lengths, hps.train.segment_size)
            wav_slice = wav_slice.squeeze()
            message = message.squeeze()
            recon_wav, recon_msg, recon_features, agmt_recon_wav, agmt_recon_msgs, agmt_features = net_nft(wav_slice, message)

            ori_mel = mel_spectrogram_torch(
              wav_slice.squeeze(1).float(), 
              hps.data.filter_length, 
              hps.data.n_mel_channels, 
              hps.data.sampling_rate, 
              hps.data.hop_length, 
              hps.data.win_length, 
              hps.data.mel_fmin, 
              hps.data.mel_fmax
            )
            recon_mel = mel_spectrogram_torch(
              recon_wav.squeeze(1).float(), 
              hps.data.filter_length, 
              hps.data.n_mel_channels, 
              hps.data.sampling_rate, 
              hps.data.hop_length, 
              hps.data.win_length, 
              hps.data.mel_fmin, 
              hps.data.mel_fmax
            )

            message   = message.squeeze()
            recon_msg = recon_msg.squeeze()
            recon_wav = recon_wav.squeeze()
                    
            loss_message      = celoss(recon_msg, message)
            loss_agmt_message = celoss(agmt_recon_msgs, message)

            ## PQMF calculates the similarity loss
            ori_subbands   = pqmf_analyser.analysis( wav_slice.unsqueeze(1).float())
            recon_subbands = pqmf_analyser.analysis( recon_wav.unsqueeze(1).float())
            delta_subbands = torch.abs(recon_subbands-ori_subbands)
            delta_subbands = delta_subbands.mean(2)
            delta_subbands = (pqmf_weights*delta_subbands).mean(1)
            loss_pqmf      = delta_subbands.mean()

            loss_mel     = l1_loss(ori_mel, recon_mel)     
            loss_norm    = l1_loss(torch.zeros_like(wav_slice), recon_wav-wav_slice)
            loss_feature = hidden_feature_loss(recon_features, agmt_features, hps.hidden_index)
            
            wtm_weight   = max(0, min(hps.mel_w*(global_step-hps.mel_start)/hps.mel_len, hps.mel_w))
            aug_weight   = max(0, min(hps.agmt_w*(global_step-hps.agmt_start)/hps.agmt_len, hps.agmt_w))
            loss_all     = wtm_weight*(loss_mel + loss_norm + 0.1*loss_pqmf)  + loss_message + aug_weight*(0.5*loss_agmt_message+0.1*loss_feature)

            optim_nft.zero_grad()
            loss_all.backward()
            optim_nft.step()

            if global_step % hps.train.log_interval == 0:
                lr = optim_nft.param_groups[0]['lr']
                losses = [loss_all, loss_mel, loss_norm, loss_pqmf, loss_message, loss_agmt_message]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                  epoch, 100. * batch_idx / len(train_loader)))
                logger.info([x.item() for x in losses] + [global_step, lr])

            if global_step % 50 == 0:
                writer.add_scalar('loss/all', loss_all, global_step)
                writer.add_scalar('loss/recon_msg', loss_message, global_step)
                writer.add_scalar('loss/recon_msg_agmt', loss_agmt_message, global_step)
                writer.add_scalar('loss/recon_feature', loss_feature, global_step)
                writer.add_scalar('loss/recon_pqmf', loss_pqmf, global_step)
                writer.add_scalar('loss/recon_mel', loss_mel, global_step)
                writer.add_scalar('loss/loss_norm', loss_norm, global_step)
                new_ori_mel   = ori_mel[0,:,:]
                new_recon_mel = recon_mel[0,:,:]
                writer.add_image('input/ori_mel',  utils.plot_spectrogram_to_numpy(new_ori_mel.data.cpu().numpy()), global_step, dataformats='HWC')
                writer.add_image('input/recon_mel',  utils.plot_spectrogram_to_numpy(new_recon_mel.data.cpu().numpy()), global_step, dataformats='HWC')
                writer.add_audio('audio/original',    wav_slice[0:1,:], global_step, sample_rate=22050)
                writer.add_audio('audio/watermarked', recon_wav[0:1,:], global_step, sample_rate=22050)
            ## mel spectrogram is with two dimension, e.g., 80*32 or 80*534
            hps.train.eval_interval = 1000
            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_nft, eval_loader, writer_eval)
            if global_step % 5e3 == 0:
                utils.save_checkpoint(net_nft, optim_nft, hps.train.learning_rate, epoch, os.path.join(hps.save_dir, "AS_model.pth"))
            if global_step>=hps.max_step:
                utils.save_checkpoint(net_nft, optim_nft, hps.train.learning_rate, epoch, os.path.join(hps.save_dir, "AS_model.pth"))
                os.system('python3 run_evaluate.py -c configs/ljs_base.json -m {} -s {} -p mixed --msg_dim {}'.format(hps.run_name, hps.max_step, hps.msg_dim))
                os.system('tar cvf {}.tar {}'.format(hps.run_name, hps.save_dir))
                os._exit(0)
            global_step += 1
    
def SNR(watermarked, original):
    # audio data type: tensor; dimension batch*8192.
    snr = SignalNoiseRatio().cuda()
    rate = snr(watermarked, original)
    return rate  
 
def evaluate(hps, net_nft, eval_loader, writer_eval):
    net_nft.eval()
    l1_loss  = torch.nn.L1Loss()
    wav_index = 0
    amplitude = np.iinfo(np.int16).max
    
    ori_errors = []
    ptb_errors = []
    SNR_list = []
    ptb_SNR_list = []
    save_num = 0
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
                recon_wav, recon_msg, _, _, _, _ =  net_nft(wav_slice, message_new)
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
            ptb_watermarked_wav = audio_ptb(watermarked_wav, hps.ptb_type)
            watermarked_wav = watermarked_wav.squeeze(dim=1)
            for i in range(watermark_times):
                ptb_wav_slice = ptb_watermarked_wav[wav_index:wav_index+1,:,i*8192:(i+1)*8192]
                ptb_recon_msg, _ = net_nft.robust_decoder(ptb_wav_slice)                
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
            
            writer_eval.add_scalar('scalars/ber_mean', torch.tensor(np.mean(ori_errors)), global_step)
            writer_eval.add_scalar('scalars/ber_std', torch.tensor(np.std(ori_errors)), global_step)
            writer_eval.add_scalar('scalars/agmt_ber', torch.tensor(np.mean(ptb_errors)), global_step)
            writer_eval.add_scalar('scalars/snr', torch.tensor(np.mean(SNR_list)), global_step)
            writer_eval.add_scalar('scalars/agmt_snr', torch.tensor(np.mean(ptb_SNR_list)), global_step)
            writer_eval.add_audio('audio/watermarked', watermarked_wav[wav_index:wav_index+1, :], global_step, sample_rate=22050)
            writer_eval.add_audio('audio/agmt_recon_wav', ptb_watermarked_wav[wav_index:wav_index+1, :], global_step, sample_rate=22050)
            
            log_path = '{}/log_files/'.format(hps.save_dir)
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            write("{}/original_{}.wav".format(log_path, save_num), 22050, save_ori_wav.astype(np.int16))
            write("{}/watermarked_step_{}.wav".format(log_path, save_num), 22050, save_wtm_wav.astype(np.int16))
            write("{}/ptb_watermarked_step_{}.wav".format(log_path, save_num), 22050, save_ptb_wtm_wav.astype(np.int16))

            if batch_idx>=5:
                break
            save_num += 1
    net_nft.train()
    
def audio_ptb(input_audio, ptb_type):
    input_audio = input_audio.unsqueeze(1)
    agmenter = audio_augmenter(ptb_type)
    new_audio = agmenter(input_audio, sample_rate = 22050)
    new_audio = torch.clamp(new_audio, -1, 1)
    return new_audio

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
                          
if __name__ == "__main__":
    main()
