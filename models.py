import math
import torch
from torch import nn
import modules
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from torch_audiomentations import Compose, Gain, PolarityInversion, AddBackgroundNoise, PitchShift, ApplyImpulseResponse, AddColoredNoise, Shift

#inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes,gin_channels
#192 1 [3, 7, 11] [[1, 3, 5], [1, 3, 5], [1, 3, 5]] [8, 8, 2, 2] 512 [16, 16, 4, 4] 0

class DecoderP(torch.nn.Module):
    def __init__(self, period, fc_d, msg_dim, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DecoderP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.fc = nn.Linear(fc_d, msg_dim)

    def forward(self, x):
        fmap = []
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.2)
            fmap.append(x)
        x = self.conv_post(x)
        x = F.leaky_relu(x, 0.2)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        x = self.fc(x)
        return x, fmap
    
class MP_Decoder(torch.nn.Module):
    def __init__(self, msg_dim, device = 'cpu'):
        super(MP_Decoder, self).__init__()
        self.decoders = nn.ModuleList([
            DecoderP(3, 102,  msg_dim ),
            DecoderP(7, 105,  msg_dim ),
            DecoderP(11, 110, msg_dim )])
        self.to(device)
        self.device = device

    def forward(self, y_hat):
        msg_recons = []
        feature_maps = []
        for i, d in enumerate(self.decoders):
            msg_recon, f_map = d(y_hat)
            msg_recons.append(msg_recon)
            feature_maps.append(f_map)
        recon_msg = torch.stack(msg_recons, dim=1)
        recon_msg = recon_msg.mean(dim=1)
        recon_msg = recon_msg.squeeze()
        return  torch.sigmoid(recon_msg), feature_maps


class Message_Encoder(torch.nn.Module):
    def __init__(self,initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Message_Encoder, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        
        self.second_dim = 32
        self.initial_channel = initial_channel
        self.fc1 = nn.Linear(1000, self.initial_channel*self.second_dim)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x):
        batch_sz = x.size()[0]
        x = self.fc1(x)
        x = x.view(batch_sz, self.initial_channel, self.second_dim )
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = 0.05*F.tanh(x)
        return x

class Wav_Hidden(torch.nn.Module):
    def __init__(self,hidden_dim):
        super(Wav_Hidden, self).__init__()
        norm_f = weight_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.fc = nn.Linear(128, hidden_dim)

    def forward(self, x, g=None):
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
        x = self.conv_post(x)
        x = F.leaky_relu(x, modules.LRELU_SLOPE)
        x = torch.flatten(x, 1, -1)
        x = self.fc(x)
        return x


class NFTAudio(nn.Module):   
    def __init__(self,wav_len,msg_dim, ptb_type)-> None:
        super(NFTAudio, self).__init__()
        self.initial_channel = 192
        self.resblock = 1
        self.resblock_kernel_sizes = [3, 7, 11]
        self.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.upsample_rates= [8, 8, 2, 2] 
        self.upsample_initial_channel = 512
        self.upsample_kernel_sizes= [16, 16, 4, 4]
        self.gin_channels =0
        self.watermarker = Message_Encoder(self.initial_channel, self.resblock, self.resblock_kernel_sizes, self.resblock_dilation_sizes, self.upsample_rates, self.upsample_initial_channel, self.upsample_kernel_sizes, gin_channels=0)
        hidden_dim = 500
        self.hidden_wav = Wav_Hidden(hidden_dim)
        self.hidden_message = torch.nn.Sequential(
          nn.Linear(msg_dim, 500),
          nn.ReLU(),
          nn.Linear(500, hidden_dim))
        self.decoder = MP_Decoder(msg_dim)
        self.robust_decoder = MP_Decoder(msg_dim)
        self.augmenter = self.audio_augmenter(ptb_type)

    def forward(self, audio, message): #audio.size()=bs*8192
        audio = torch.unsqueeze(audio, 1)
        wav_hidden = self.hidden_wav(audio)
        msg_hidden = self.hidden_message(message)
        hidden = torch.cat((wav_hidden,msg_hidden), -1)
        delta_audio = self.watermarker(hidden)
        delta_audio = delta_audio.squeeze(dim=1)
        recon_audio = delta_audio.squeeze(dim=1) + audio.squeeze(dim=1)
        recon_audio = torch.clamp(recon_audio, -1, 1)
        recon_wav = torch.unsqueeze(recon_audio, 1)
        
        agmt_recon_wav  = self.augmenter(recon_wav, sample_rate=22050)
        agmt_recon_wav  = torch.clamp(agmt_recon_wav, -1, 1)
        agmt_recon_msgs, agmt_features  = self.robust_decoder(agmt_recon_wav) 
        recon_msgs, recon_features      = self.decoder(recon_wav)
        return recon_wav, recon_msgs, recon_features, agmt_recon_wav, agmt_recon_msgs, agmt_features
    
    def audio_augmenter(self, ptb_type):
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
