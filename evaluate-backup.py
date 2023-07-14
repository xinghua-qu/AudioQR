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

torch.backends.cudnn.benchmark = True
global_step = 0

def SNR(watermarked, original):
    # audio data type: tensor; dimension batch*8192.
    snr = SignalNoiseRatio().cuda()
    rate = snr(watermarked, original)
    return rate


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '80002'

  hps = utils.get_hparams()
  hps.train.segment_size = 8192
  hps.train.eval_interval = 10000
  hps.train.log_interval = 500
  run(rank=0, n_gpus=1, hps=hps)

def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
  
    ## Prepare the tensorboard: whether on local machine or Arnold cloud (hdfs savings)
    tensorboard = 0
    if tensorboard==0:
        log_path = os.path.join(hps.save_dir, 'tensorboard/final_evaluate')
        writer_eval = SummaryWriter(log_path)
    else:
        HDFS_log_dir = os.environ.get("ARNOLD_OUTPUT")
        if HDFS_log_dir:  # if Arnold supports remote Tensorboard
      
          log_path = f'{HDFS_log_dir}/logs/{hps.run_name}/final_evaluate'
          cmd = f'hdfs dfs -mkdir -p {log_path}'
          print(cmd)
          os.system(cmd)
          writer_eval = SummaryWriter(log_path)  # this line alone will create the folder

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.manual_seed_all(hps.train.seed)
  hps.train.batch_size = 2
  msg_dim = 100

  train_dataset = TextAudioLoader(hps.data.training_files, hps.data, msg_dim)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True) # this line of code ensures the length of audio data has similar input lengths in a batch

  collate_fn = TextAudioCollate(msg_dim)
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data, msg_dim)
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=True,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)

  wave_length = hps.train.segment_size
  msg_length  = 100
  ## Model defination
  nft_model = NFTAudio(wave_length, msg_length).cuda(rank)
  optim_nft = torch.optim.AdamW(
      nft_model.parameters(), 
      0.00005, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_nft = DDP(nft_model, device_ids=[rank])

  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "NFTAudio_*.pth"), net_nft, optim_nft)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_nft = torch.optim.lr_scheduler.ExponentialLR(optim_nft, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scaler = GradScaler(enabled=hps.train.fp16_run)
  epoch = 0
  train_and_evaluate(rank, epoch, hps, net_nft, optim_nft, scheduler_nft, scaler, [train_loader, eval_loader], logger, writer_eval)


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writer_eval):
  net_nft = nets
  optim_nft= optims
  scheduler_nft= schedulers
  train_loader, eval_loader = loaders
  
#   evaluate(hps, net_nft, eval_loader, writer_eval)
#   evaluate_non_watermarked(hps, net_nft, eval_loader, writer_eval)
#   evaluate_resample(hps, net_nft, eval_loader, writer_eval)
  evaluate_noise(hps, net_nft, eval_loader, writer_eval)
#   evaluate_crop(hps, net_nft, eval_loader, writer_eval)
#   evaluate_rir(hps, net_nft, eval_loader, writer_eval)

def evaluate(hps, net_nft, eval_loader, writer_eval):
    net_nft.eval()
    l1_loss  = torch.nn.L1Loss()
    wav_index = 0
    amplitude = np.iinfo(np.int16).max
    Total_errors = []
    SNR_list = []
    with torch.no_grad():
      for batch_idx, (message,spec, spec_lengths, wav, wav_lengths) in enumerate(eval_loader):
        spec    = spec.cuda(0)
        wav, wav_lengths = wav.cuda(0), wav_lengths.cuda(0)
        message = message.cuda(0)
        watermark_times = wav_lengths[wav_index].item()//8192
        wav = wav.squeeze()
        
        watermarked_wav = torch.zeros_like(wav)
        watermarked_wav.copy_(wav)
        # don't use empty like function
        error_rate = 0
        for i in range(watermark_times):
            wav_slice = torch.zeros_like(wav[wav_index,i*8192:(i+1)*8192])
            wav_slice.copy_(wav[wav_index,i*8192:(i+1)*8192])
            wav_slice = wav_slice.repeat(wav.size()[0],1)
            
            message_new = torch.zeros_like(message[wav_index,:])
            message_new.copy_(message[wav_index,:])
            message_new = message_new.repeat(wav.size()[0],1)
            
            recon_wav = torch.zeros_like(wav_slice)
            recon_msg = torch.zeros_like(message_new)
            recon_wav, recon_msg = net_nft(wav_slice, message_new)
            watermarked_wav[wav_index, i*8192:(i+1)*8192] = recon_wav[wav_index,:]

            err = torch.abs(message[wav_index].squeeze().round() -recon_msg[wav_index].squeeze().round()).sum()
            error_rate += err.item()
            Total_errors.append(err.item())
        
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
        snr = SNR(watermarked_wav[wav_index,:], wav[wav_index,:])
        SNR_list.append(snr.item())
        print(snr.item())
        save_ori_wav = np.asarray(wav[wav_index:wav_index+1,:].cpu().detach().numpy()*amplitude)[0]
        save_wtm_wav = np.asarray(watermarked_wav[wav_index:wav_index+1,:].cpu().detach().numpy()*amplitude)[0]
        delta = save_wtm_wav - save_ori_wav
        log_path = os.path.join(hps.save_dir, 'tensorboard/clean_evaluate')
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        write("{}/original_{}.wav".format(log_path, batch_idx), 22050, save_ori_wav.astype(np.int16))
        write("{}/watermarked_{}.wav".format(log_path, batch_idx), 22050, save_wtm_wav.astype(np.int16))
        write("{}/delta_{}.wav".format(log_path, batch_idx), 22050, delta.astype(np.int16))
        
        x = np.arange(0, delta.shape[0])
        plt.figure()
        plt.plot(x, save_ori_wav, label = 'Original')
        plt.plot(x, save_wtm_wav, label = 'Watermarked')
        plt.plot(x, delta, label = 'delta')
        plt.legend()
        plt.savefig('{}/wav_comparison_{}.pdf'.format(log_path, batch_idx),bbox_inches='tight', dpi=600)
        plt.close() 
        
        print('{} error: {}'.format(batch_idx, error_rate/watermark_times))
        writer_eval.add_scalar('error_rate',      error_rate/watermark_times, batch_idx)
        writer_eval.add_image('input/ori_mel',     utils.plot_spectrogram_to_numpy(entire_ori_mel[0].data.cpu().numpy()), batch_idx, dataformats='HWC')
        writer_eval.add_image('input/recon_mel',   utils.plot_spectrogram_to_numpy(recon_mel[0].data.cpu().numpy()), batch_idx, dataformats='HWC')
        writer_eval.add_image('input/recon_mel',   utils.plot_spectrogram_to_numpy(delta_mel[0].data.cpu().numpy()), batch_idx, dataformats='HWC')
        writer_eval.add_audio('audio/original',    wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        writer_eval.add_audio('audio/watermarked', watermarked_wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        writer_eval.add_audio('audio/delta', watermarked_wav[wav_index:wav_index+1,:]-wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        print('Mean: {}  STD: {}'.format(np.mean(Total_errors), np.std(Total_errors)))
      np.savetxt("{}/accuracy.csv".format(log_path), Total_errors, delimiter =", ", fmt ='%1.9f')
      np.savetxt("{}/snr.csv".format(log_path), SNR_list, delimiter =", ", fmt ='%1.9f')

def evaluate_non_watermarked(hps, net_nft, eval_loader, writer_eval):
    net_nft.eval()
    l1_loss  = torch.nn.L1Loss()
    wav_index = 0
    amplitude = np.iinfo(np.int16).max
    Total_errors = []
    with torch.no_grad():
      for batch_idx, (message,spec, spec_lengths, wav, wav_lengths) in enumerate(eval_loader):
        spec    = spec.cuda(0)
        wav, wav_lengths = wav.cuda(0), wav_lengths.cuda(0)
        message = message.cuda(0)
        watermark_times = wav_lengths[wav_index].item()//8192
        wav = wav.squeeze()
        
        watermarked_wav = torch.zeros_like(wav)
        watermarked_wav.copy_(wav)
        # don't use empty like function
        error_rate = 0
        for i in range(watermark_times):
            wav_slice = torch.zeros_like(wav[wav_index,i*8192:(i+1)*8192])
            wav_slice.copy_(wav[wav_index,i*8192:(i+1)*8192])
            wav_slice = wav_slice.repeat(wav.size()[0],1)
            
            message_new = torch.zeros_like(message[wav_index,:])
            message_new.copy_(message[wav_index,:])
            message_new = message_new.repeat(wav.size()[0],1)
            
            recon_msg = net_nft.module.reconstructor(wav_slice)
            
            err = torch.abs(message[wav_index].squeeze().round() -recon_msg[wav_index].squeeze().round()).sum()
            error_rate += err.item()
            print('non watermarked: {}'.format(err.item()))
            Total_errors.append(err.item())
        
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
    
       
        save_ori_wav = np.asarray(wav[wav_index:wav_index+1,:].cpu().detach().numpy()*amplitude)[0]
        log_path = os.path.join(hps.save_dir, 'tensorboard/nonwatermark_evaluate')
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        
        print('{} error: {}'.format(batch_idx, error_rate/watermark_times))
        writer_eval.add_scalar('error_rate',      error_rate/watermark_times, batch_idx)
        writer_eval.add_audio('audio/original',    wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        print('Mean: {}  STD: {}'.format(np.mean(Total_errors), np.std(Total_errors)))
      np.savetxt("{}/accuracy.csv".format(log_path), Total_errors, delimiter =", ", fmt ='%1.9f')        

def evaluate_rir(hps, net_nft, eval_loader, writer_eval):
    from augmentation_utils import _get_sample
    sample_rate = 22050
    RIR_DIR = './rir_audio/rir.wav'
    rir_raw, sample_rate = _get_sample(RIR_DIR, resample=None)
    rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
    rir = rir / torch.norm(rir, p=2)
    RIR = torch.flip(rir, [1])
    RIR = RIR.cuda()
    
    net_nft.eval()
    l1_loss  = torch.nn.L1Loss()
    wav_index = 0
    amplitude = np.iinfo(np.int16).max
    Total_errors = []
    with torch.no_grad():
      for batch_idx, (message,spec, spec_lengths, wav, wav_lengths) in enumerate(eval_loader):
        spec    = spec.cuda(0)
        wav, wav_lengths = wav.cuda(0), wav_lengths.cuda(0)
        message = message.cuda(0)
        watermark_times = wav_lengths[wav_index].item()//8192
        wav = wav.squeeze()
        
        watermarked_wav = torch.zeros_like(wav)
        watermarked_wav.copy_(wav)
        # don't use empty like function
        error_rate = 0
        for i in range(watermark_times):
            wav_slice = torch.zeros_like(wav[wav_index,i*8192:(i+1)*8192])
            wav_slice.copy_(wav[wav_index,i*8192:(i+1)*8192])
            wav_slice = wav_slice.repeat(wav.size()[0],1)
            
            message_new = torch.zeros_like(message[wav_index,:])
            message_new.copy_(message[wav_index,:])
            message_new = message_new.repeat(wav.size()[0],1)
            
            recon_wav = torch.zeros_like(wav_slice)
            recon_msg = torch.zeros_like(message_new)
            recon_wav, recon_msg = net_nft(wav_slice, message_new)
            watermarked_wav[wav_index, i*8192:(i+1)*8192] = recon_wav[wav_index,:]

        cur_watermarked_wav = torch.zeros_like(watermarked_wav[wav_index:wav_index+1,:])
        cur_watermarked_wav.copy_(watermarked_wav[wav_index:wav_index+1,:])
        sample_rate = 22050
        print('aaa:', cur_watermarked_wav.size(), RIR.size())
        speech_ = torch.nn.functional.pad(cur_watermarked_wav, (RIR.shape[1]-1, 0))
        new_watermarked_wav = torch.nn.functional.conv1d(speech_[None, ...], RIR[None, ...])[0]
        watermark_times = new_watermarked_wav.size()[-1]//8192
        for i in range(watermark_times):
            wav_slice = torch.zeros_like(new_watermarked_wav[wav_index,i*8192:(i+1)*8192])
            wav_slice.copy_(new_watermarked_wav[wav_index, i*8192:(i+1)*8192])
            wav_slice = wav_slice.unsqueeze(0)
            wav_slice = wav_slice.unsqueeze(0)
            wav_slice = wav_slice.repeat(wav.size()[0],1,1)
            
            message_new = torch.zeros_like(message[wav_index,:])
            message_new.copy_(message[wav_index,:])
            message_new = message_new.repeat(wav.size()[0],1)
            
            recon_msg = net_nft.module.reconstructor(wav_slice)
            
            err = torch.abs(message[wav_index].squeeze().round() -recon_msg[wav_index].squeeze().round()).sum()
            error_rate += err.item()
            print('rired: {}'.format(err.item()))
            Total_errors.append(err.item())
        
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
        resampled_mel = mel_spectrogram_torch(
              new_watermarked_wav[wav_index:wav_index+1,:].float(), 
              hps.data.filter_length, 
              hps.data.n_mel_channels, 
              hps.data.sampling_rate, 
              hps.data.hop_length, 
              hps.data.win_length, 
              hps.data.mel_fmin, 
              hps.data.mel_fmax
            )
        
        save_ori_wav = np.asarray(wav[wav_index:wav_index+1,:].cpu().detach().numpy()*amplitude)[0]
        save_wtm_wav = np.asarray(watermarked_wav[wav_index:wav_index+1,:].cpu().detach().numpy()*amplitude)[0]
        start = int(new_watermarked_wav.size()[0]*0)
        end   = int(new_watermarked_wav.size()[0]*1)
        save_resample_wav = np.asarray(new_watermarked_wav[start:end].cpu().detach().numpy()*amplitude)[0]
        delta = save_wtm_wav - save_ori_wav
        log_path = os.path.join(hps.save_dir, 'tensorboard/rir_evaluate')
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        write("{}/watermarked_{}.wav".format(log_path, batch_idx), 22050, save_wtm_wav.astype(np.int16))
        write("{}/rir_{}.wav".format(log_path, batch_idx), 22050, save_resample_wav.astype(np.int16))
        write("{}/delta_{}.wav".format(log_path, batch_idx), 22050, delta.astype(np.int16))
        
        x = np.arange(0, delta.shape[0])
        plt.figure()
        plt.plot(x, delta)
        plt.savefig('{}/delta_{}.pdf'.format(log_path, batch_idx),bbox_inches='tight', dpi=600)
        plt.close()        
        
        print('{} error: {}'.format(batch_idx, error_rate/watermark_times))
        writer_eval.add_scalar('error_rate',      error_rate/watermark_times, batch_idx)
        writer_eval.add_image('input/ori_mel',     utils.plot_spectrogram_to_numpy(entire_ori_mel[0].data.cpu().numpy()), batch_idx, dataformats='HWC')
        writer_eval.add_image('input/recon_mel',   utils.plot_spectrogram_to_numpy(recon_mel[0].data.cpu().numpy()), batch_idx, dataformats='HWC')
        writer_eval.add_image('input/delta_mel',   utils.plot_spectrogram_to_numpy(delta_mel[0].data.cpu().numpy()), batch_idx, dataformats='HWC')
        writer_eval.add_audio('audio/original',    wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        writer_eval.add_audio('audio/watermarked', watermarked_wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        writer_eval.add_audio('audio/delta', watermarked_wav[wav_index:wav_index+1,:]-wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        print('Mean: {}  STD: {}'.format(np.mean(Total_errors), np.std(Total_errors)))
      np.savetxt("{}/accuracy.csv".format(log_path), Total_errors, delimiter =", ", fmt ='%1.9f')
    
def evaluate_resample(hps, net_nft, eval_loader, writer_eval):
    net_nft.eval()
    l1_loss  = torch.nn.L1Loss()
    wav_index = 0
    amplitude = np.iinfo(np.int16).max
    Total_errors = []
    with torch.no_grad():
      for batch_idx, (message,spec, spec_lengths, wav, wav_lengths) in enumerate(eval_loader):
        spec    = spec.cuda(0)
        wav, wav_lengths = wav.cuda(0), wav_lengths.cuda(0)
        message = message.cuda(0)
        watermark_times = wav_lengths[wav_index].item()//8192
        wav = wav.squeeze()
        
        watermarked_wav = torch.zeros_like(wav)
        watermarked_wav.copy_(wav)
        # don't use empty like function
        error_rate = 0
        for i in range(watermark_times):
            wav_slice = torch.zeros_like(wav[wav_index,i*8192:(i+1)*8192])
            wav_slice.copy_(wav[wav_index,i*8192:(i+1)*8192])
            wav_slice = wav_slice.repeat(wav.size()[0],1)
            
            message_new = torch.zeros_like(message[wav_index,:])
            message_new.copy_(message[wav_index,:])
            message_new = message_new.repeat(wav.size()[0],1)
            
            recon_wav = torch.zeros_like(wav_slice)
            recon_msg = torch.zeros_like(message_new)
            recon_wav, recon_msg = net_nft(wav_slice, message_new)
            watermarked_wav[wav_index, i*8192:(i+1)*8192] = recon_wav[wav_index,:]

        cur_watermarked_wav = torch.zeros_like(watermarked_wav[wav_index:wav_index+1,:])
        cur_watermarked_wav.copy_(watermarked_wav[wav_index:wav_index+1,:])
        sample_rate = 22050
        resample_rate = 16000
        resampler = T.Resample(sample_rate, resample_rate)
        new_watermarked_wav = resampler(cur_watermarked_wav)
        new_watermarked_wav = new_watermarked_wav.squeeze()
        
        watermark_times = new_watermarked_wav.size()[0]//8192
        for i in range(watermark_times):
            wav_slice = torch.zeros_like(wav[wav_index,i*8192:(i+1)*8192])
            wav_slice.copy_(new_watermarked_wav[i*8192:(i+1)*8192])
            wav_slice = wav_slice.repeat(wav.size()[0],1)
            
            message_new = torch.zeros_like(message[wav_index,:])
            message_new.copy_(message[wav_index,:])
            message_new = message_new.repeat(wav.size()[0],1)
            
            recon_msg = net_nft.module.reconstructor(wav_slice)
            
            err = torch.abs(message[wav_index].squeeze().round() -recon_msg[wav_index].squeeze().round()).sum()
            error_rate += err.item()
            print('resampled: {}'.format(err.item()))
            Total_errors.append(err.item())
        
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
        resampled_mel = mel_spectrogram_torch(
              new_watermarked_wav.unsqueeze(0).float(), 
              hps.data.filter_length, 
              hps.data.n_mel_channels, 
              hps.data.sampling_rate, 
              hps.data.hop_length, 
              hps.data.win_length, 
              hps.data.mel_fmin, 
              hps.data.mel_fmax
            )
    
        save_ori_wav = np.asarray(wav[wav_index:wav_index+1,:].cpu().detach().numpy()*amplitude)[0]
        save_wtm_wav = np.asarray(watermarked_wav[wav_index:wav_index+1,:].cpu().detach().numpy()*amplitude)[0]
        start = int(new_watermarked_wav.size()[0]*0)
        end   = int(new_watermarked_wav.size()[0]*1)
        save_resample_wav = np.asarray(new_watermarked_wav[start:end].unsqueeze(0).cpu().detach().numpy()*amplitude)[0]
        delta = save_wtm_wav - save_ori_wav
        log_path = os.path.join(hps.save_dir, 'tensorboard/resample_evaluate')
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        write("{}/watermarked_{}.wav".format(log_path, batch_idx), 22050, save_wtm_wav.astype(np.int16))
        write("{}/resampled_{}.wav".format(log_path, batch_idx), 16000, save_resample_wav.astype(np.int16))
        write("{}/delta_{}.wav".format(log_path, batch_idx), 22050, delta.astype(np.int16))
        
        x = np.arange(0, delta.shape[0])
        plt.figure()
        plt.plot(x, delta)
        plt.savefig('{}/delta_{}.pdf'.format(log_path, batch_idx),bbox_inches='tight', dpi=600)
        plt.close()        
        
        print('{} error: {}'.format(batch_idx, error_rate/watermark_times))
        writer_eval.add_scalar('error_rate',      error_rate/watermark_times, batch_idx)
        writer_eval.add_image('input/ori_mel',     utils.plot_spectrogram_to_numpy(entire_ori_mel[0].data.cpu().numpy()), batch_idx, dataformats='HWC')
        writer_eval.add_image('input/recon_mel',   utils.plot_spectrogram_to_numpy(recon_mel[0].data.cpu().numpy()), batch_idx, dataformats='HWC')
        writer_eval.add_image('input/delta_mel',   utils.plot_spectrogram_to_numpy(delta_mel[0].data.cpu().numpy()), batch_idx, dataformats='HWC')
        writer_eval.add_audio('audio/original',    wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        writer_eval.add_audio('audio/watermarked', watermarked_wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        writer_eval.add_audio('audio/delta', watermarked_wav[wav_index:wav_index+1,:]-wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        print('Mean: {}  STD: {}'.format(np.mean(Total_errors), np.std(Total_errors)))
      np.savetxt("{}/accuracy.csv".format(log_path), Total_errors, delimiter =", ", fmt ='%1.9f')
    
def evaluate_noise(hps, net_nft, eval_loader, writer_eval):
    net_nft.eval()
    l1_loss  = torch.nn.L1Loss()
    wav_index = 0
    amplitude = np.iinfo(np.int16).max
    Total_errors = []
    with torch.no_grad():
      for batch_idx, (message,spec, spec_lengths, wav, wav_lengths) in enumerate(eval_loader):
        spec    = spec.cuda(0)
        wav, wav_lengths = wav.cuda(0), wav_lengths.cuda(0)
        message = message.cuda(0)
        watermark_times = wav_lengths[wav_index].item()//8192
        wav = wav.squeeze()
        
        watermarked_wav = torch.zeros_like(wav)
        watermarked_wav.copy_(wav)
        # don't use empty like function
        error_rate = 0
        for i in range(watermark_times):
            wav_slice = torch.zeros_like(wav[wav_index,i*8192:(i+1)*8192])
            wav_slice.copy_(wav[wav_index,i*8192:(i+1)*8192])
            wav_slice = wav_slice.repeat(wav.size()[0],1)
            
            message_new = torch.zeros_like(message[wav_index,:])
            message_new.copy_(message[wav_index,:])
            message_new = message_new.repeat(wav.size()[0],1)
            
            recon_wav = torch.zeros_like(wav_slice)
            recon_msg = torch.zeros_like(message_new)
            recon_wav, _ = net_nft(wav_slice, message_new)
            
            ## adding noise
#             speech_power = wav_slice.norm(p=2)
#             noise = torch.randn_like(wav_slice)
#             noise_power = noise.norm(p=2)
#             db_list = [40, 20, 10]
#             snr_db = db_list[2]
#             snr = math.exp(snr_db / 10)
#             scale = snr * noise_power / speech_power
#             noise_wav = (scale * wav_slice + noise) / 2
            noise_wav = wav_slice + 0.001*torch.randn_like(wav_slice)
            noise_wav = noise_wav.unsqueeze(1)
            noise_wav = torch.clamp(noise_wav, -1, 1)
            
            recon_msg = net_nft.module.reconstructor(noise_wav)
            watermarked_wav[wav_index, i*8192:(i+1)*8192] = noise_wav[wav_index,0,:]

            err = torch.abs(message[wav_index].squeeze().round() -recon_msg[wav_index].squeeze().round()).sum()
            error_rate += err.item()
            print('noise: {}'.format(err.item()))
            Total_errors.append(err.item())
        
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
    
        save_ori_wav = np.asarray(wav[wav_index:wav_index+1,:].cpu().detach().numpy()*amplitude)[0]
        save_wtm_wav = np.asarray(watermarked_wav[wav_index:wav_index+1,:].cpu().detach().numpy()*amplitude)[0]
        delta = save_wtm_wav - save_ori_wav
        log_path = os.path.join(hps.save_dir, 'tensorboard/noise_evaluate')
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        write("{}/original_{}.wav".format(log_path, batch_idx), 22050, save_ori_wav.astype(np.int16))
        write("{}/watermarked_{}.wav".format(log_path, batch_idx), 22050, save_wtm_wav.astype(np.int16))
        write("{}/delta_{}.wav".format(log_path, batch_idx), 22050, delta.astype(np.int16))
        
        x = np.arange(0, delta.shape[0])
        plt.figure()
        plt.plot(x, delta)
        plt.savefig('{}/delta_{}.pdf'.format(log_path, batch_idx),bbox_inches='tight', dpi=600)
        plt.close()        
        
        print('{} error: {}'.format(batch_idx, error_rate/watermark_times))
        writer_eval.add_scalar('error_rate',      error_rate/watermark_times, batch_idx)
        writer_eval.add_image('input/ori_mel',     utils.plot_spectrogram_to_numpy(entire_ori_mel[0].data.cpu().numpy()), batch_idx, dataformats='HWC')
        writer_eval.add_image('input/recon_mel',   utils.plot_spectrogram_to_numpy(recon_mel[0].data.cpu().numpy()), batch_idx, dataformats='HWC')
        writer_eval.add_image('input/recon_mel',   utils.plot_spectrogram_to_numpy(delta_mel[0].data.cpu().numpy()), batch_idx, dataformats='HWC')
        writer_eval.add_audio('audio/original',    wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        writer_eval.add_audio('audio/watermarked', watermarked_wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        writer_eval.add_audio('audio/delta', watermarked_wav[wav_index:wav_index+1,:]-wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        print('Mean: {}  STD: {}'.format(np.mean(Total_errors), np.std(Total_errors)))
      np.savetxt("{}/accuracy.csv".format(log_path), Total_errors, delimiter =", ", fmt ='%1.9f')

def evaluate_crop(hps, net_nft, eval_loader, writer_eval):
    net_nft.eval()
    l1_loss  = torch.nn.L1Loss()
    wav_index = 0
    amplitude = np.iinfo(np.int16).max
    Total_errors = []
    with torch.no_grad():
      for batch_idx, (message,spec, spec_lengths, wav, wav_lengths) in enumerate(eval_loader):
        spec    = spec.cuda(0)
        wav, wav_lengths = wav.cuda(0), wav_lengths.cuda(0)
        message = message.cuda(0)
        watermark_times = wav_lengths[wav_index].item()//8192
        wav = wav.squeeze()
        
        watermarked_wav = torch.zeros_like(wav)
        watermarked_wav.copy_(wav)
        # don't use empty like function
        error_rate = 0
        for i in range(watermark_times):
            wav_slice = torch.zeros_like(wav[wav_index,i*8192:(i+1)*8192])
            wav_slice.copy_(wav[wav_index,i*8192:(i+1)*8192])
            wav_slice = wav_slice.repeat(wav.size()[0],1)
            
            message_new = torch.zeros_like(message[wav_index,:])
            message_new.copy_(message[wav_index,:])
            message_new = message_new.repeat(wav.size()[0],1)
            
            recon_wav = torch.zeros_like(wav_slice)
            recon_msg = torch.zeros_like(message_new)
            recon_wav, recon_msg = net_nft(wav_slice, message_new)
            watermarked_wav[wav_index, i*8192:(i+1)*8192] = recon_wav[wav_index,:]

        crop_start = np.random.randint(8192)
        new_watermarked_wav = torch.zeros_like(watermarked_wav[wav_index,crop_start:])
        new_watermarked_wav.copy_(watermarked_wav[wav_index,crop_start:])
        watermark_times = (wav_lengths[wav_index].item()-crop_start)//8192
        print(wav.size(), watermarked_wav.size(), new_watermarked_wav.size())
        for i in range(watermark_times):
            wav_slice = torch.zeros_like(wav[wav_index,i*8192:(i+1)*8192])
            wav_slice.copy_(new_watermarked_wav[i*8192:(i+1)*8192])
            wav_slice = wav_slice.repeat(wav.size()[0],1)
            
            message_new = torch.zeros_like(message[wav_index,:])
            message_new.copy_(message[wav_index,:])
            message_new = message_new.repeat(wav.size()[0],1)
            
            recon_msg = net_nft.module.reconstructor(wav_slice)
            
            err = torch.abs(message[wav_index].squeeze().round() -recon_msg[wav_index].squeeze().round()).sum()
            error_rate += err.item()
            print('cropped: {}'.format(err.item()))
            Total_errors.append(err.item())
        
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
        croped_mel = mel_spectrogram_torch(
              new_watermarked_wav.unsqueeze(0).float(), 
              hps.data.filter_length, 
              hps.data.n_mel_channels, 
              hps.data.sampling_rate, 
              hps.data.hop_length, 
              hps.data.win_length, 
              hps.data.mel_fmin, 
              hps.data.mel_fmax
            )
    
        save_ori_wav = np.asarray(wav[wav_index:wav_index+1,:].cpu().detach().numpy()*amplitude)[0]
        save_wtm_wav = np.asarray(watermarked_wav[wav_index:wav_index+1,:].cpu().detach().numpy()*amplitude)[0]
        start = int(new_watermarked_wav.size()[0]*0.3)
        end   = int(new_watermarked_wav.size()[0]*0.85)
        save_crp_wav = np.asarray(new_watermarked_wav[start:end].unsqueeze(0).cpu().detach().numpy()*amplitude)[0]
        delta = save_wtm_wav - save_ori_wav
        log_path = os.path.join(hps.save_dir, 'tensorboard/crop_evaluate')
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        write("{}/watermarked_{}.wav".format(log_path, batch_idx), 22050, save_wtm_wav.astype(np.int16))
        write("{}/cropped_{}.wav".format(log_path, batch_idx), 22050, save_crp_wav.astype(np.int16))
        write("{}/delta_{}.wav".format(log_path, batch_idx), 22050, delta.astype(np.int16))
        
        x = np.arange(0, delta.shape[0])
        plt.figure()
        plt.plot(x, delta)
        plt.savefig('{}/delta_{}.pdf'.format(log_path, batch_idx),bbox_inches='tight', dpi=600)
        plt.close()        
        
        print('{} error: {}'.format(batch_idx, error_rate/watermark_times))
        writer_eval.add_scalar('error_rate',      error_rate/watermark_times, batch_idx)
        writer_eval.add_image('input/ori_mel',     utils.plot_spectrogram_to_numpy(entire_ori_mel[0].data.cpu().numpy()), batch_idx, dataformats='HWC')
        writer_eval.add_image('input/recon_mel',   utils.plot_spectrogram_to_numpy(recon_mel[0].data.cpu().numpy()), batch_idx, dataformats='HWC')
        writer_eval.add_image('input/recon_mel',   utils.plot_spectrogram_to_numpy(delta_mel[0].data.cpu().numpy()), batch_idx, dataformats='HWC')
        writer_eval.add_audio('audio/original',    wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        writer_eval.add_audio('audio/watermarked', watermarked_wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        writer_eval.add_audio('audio/delta', watermarked_wav[wav_index:wav_index+1,:]-wav[wav_index:wav_index+1,:], batch_idx, sample_rate=22050)
        print('Mean: {}  STD: {}'.format(np.mean(Total_errors), np.std(Total_errors)))
      np.savetxt("{}/accuracy.csv".format(log_path), Total_errors, delimiter =", ", fmt ='%1.9f')

                          
if __name__ == "__main__":
  main()
