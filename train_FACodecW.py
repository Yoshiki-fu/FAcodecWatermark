import warnings
import argparse
import torch
import os
import yaml
#from tqdm import tqdm

warnings.simplefilter('ignore')

from modules.commons import *
from losses import *
import time

import torchaudio
import librosa
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import Librilight, collate_fn
from optimizers import build_optimizer
import watermark_hparams as hp

from audiotools import AudioSignal
from torch.utils.tensorboard import SummaryWriter

# set seeds
seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_watermark_model(args):
    ckpt_path = args.ckpt_path
    config_path = args.config_path
    config = yaml.safe_load(open(config_path))
    model_params = recursive_munch(config['model_params'])
    new_model = build_model(model_params, 'watermarking')

    # 古いモデルチェックポイントの読み込み
    ckpt_params = torch.load(ckpt_path)
    ckpt_params = ckpt_params['net'] if 'net' in ckpt_params else ckpt_params  # adapt to format of self-trained checkpoints
    ckpt_params = {key: value for key, value in ckpt_params.items() if key in ['encoder','quantizer','decoder','discriminator']}        # fa_predictorを除く

    new_layers = []
    for block_i, block_j in zip(new_model, ckpt_params):
        new_state_dict = new_model[block_i].state_dict()
        old_state_dict = ckpt_params[block_j]
        # 共通するキーの重みだけコピー
        for layer in new_state_dict:
            if layer in old_state_dict and new_state_dict[layer].shape == old_state_dict[layer].shape:
                new_state_dict[layer] = old_state_dict[layer]
            elif layer not in old_state_dict:
                print(layer)
                
        new_model[block_i].load_state_dict(new_state_dict)
        # 事前学習した重みを凍結
        if block_i in ['encoder', 'quantizer']:
            for param in new_model[block_i].parameters():
                param.requires_grad = False
    
    # 学習したい重み
    for param in new_model['quantizer'].watermark_emb.parameters():
        param.requires_grad = True  # watermark_embのパラメータは学習可能にする

    for param in new_model['quantizer'].msg_linear.parameters():
        param.requires_grad = True
    
    _ = [new_model[key].train() for key in new_model]
    _ = [new_model[key].to(device) for key in new_model]

    return new_model


def make_watermark_extracter(args):
    ckpt_path = args.ckpt_path
    config_path = args.config_path
    config = yaml.safe_load(open(config_path))
    model_params = recursive_munch(config['model_params'])
    new_model = build_model(model_params, 'extracter')
    
    # 古いモデルのチェックポイントの読み込み
    ckpt_params = torch.load(ckpt_path)
    ckpt_params = ckpt_params['net'] if 'net' in ckpt_params else ckpt_params  # adapt to format of self-trained checkpoints
    ckpt_params = {key: value for key, value in ckpt_params.items() if key in ['encoder','quantizer','decoder','discriminator']}        # fa_predictorを除く

    new_layers = []
    for block_i, block_j in zip(new_model, ckpt_params):
        print(block_i, block_j)
        new_state_dict = new_model[block_i].state_dict()
        old_state_dict = ckpt_params[block_j]
        # 共通するキーの重みだけコピー
        for layer in new_state_dict:
            if layer in old_state_dict and new_state_dict[layer].shape == old_state_dict[layer].shape:
                new_state_dict[layer] = old_state_dict[layer]
            elif layer not in old_state_dict:
                print(layer)

        new_model[block_i].load_state_dict(new_state_dict)

    _ = [new_model[key].train() for key in new_model]
    _ = [new_model[key].to(device) for key in new_model]
    
    return new_model

    
def main(args):
    writer = SummaryWriter("./tensorboard_logs")

    # モデルの作成
    watermark_model = make_watermark_model(args)
    extracter = make_watermark_extracter(args)

    # データセットの準備
    train_dataset = Librilight()
    train_loader = DataLoader(train_dataset,
                              batch_size=hp.batch_size, 
                              num_workers=4, 
                              drop_last=True, 
                              shuffle=True, 
                              collate_fn=collate_fn, 
                              pin_memory=True)

    # otimizerの準備
    scheduler_params = {
        "warmup_steps": 200,
        "base_lr": 0.0001
    }
    watermark_optimizer = build_optimizer({key: watermark_model[key] for key in ['quantizer', 'decoder', 'discriminator']},
                                           scheduler_params_dict={key: scheduler_params.copy() for key in ['quantizer', 'decoder', 'discriminator']},
                                           lr=float(scheduler_params['base_lr']))
    extracter_optimizer = build_optimizer({key: extracter[key] for key in extracter},
                                           scheduler_params_dict={key: scheduler_params.copy() for key in extracter},
                                           lr=float(scheduler_params['base_lr']))


    # 損失関数の準備
    content_criterion = FocalLoss(gamma=2).to(device)
    stft_criterion = MultiScaleSTFTLoss().to(device)
    mel_criterion = MelSpectrogramLoss(
        n_mels=[5, 10, 20, 40, 80, 160, 320],
        window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
        mel_fmin=[0, 0, 0, 0, 0, 0, 0],
        mel_fmax=[None, None, None, None, None, None, None],
        pow=1.0,
        mag_weight=0.0,
        clamp_eps=1e-5,
    ).to(device)
    l1_criterion = L1Loss().to(device)
    msg_criterion = nn.MSE().to(device)


    # 学習
    start_epoch = 0
    iters = 0

    for epoch in range(start_epoch, hp.epoch):
        start_time = time.time()
        _ = [watermark_model[key].train() for key in watermark_model]
        _ = [extracter[key].train() for key in extracter]
        last_time = time.time()
        for i, batch in enumerate(train_loader):
            watermark_optimizer.zero_grad()
            extracter_optimizer.zero_grad()

            train_start_time = time.time()

            batch = [b.to(device, non_blocking=True) for b in batch]
            waves, mels, wav_lengths, mel_input_length = batch      # waves shape is (batch_size, sample_num), mel shape is (batch_size, freq_bin, frame)

            # get clips
            mel_seg_len = min([int(mel_input_length.min().item()), hp.max_frame_len])

            gt_mel_seg = []
            wav_seg = []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item())

                random_start = np.random.randint(0, mel_length - mel_seg_len) if mel_length != mel_seg_len else 0
                gt_mel_seg.append(mels[bib, :, random_start:random_start + mel_seg_len])

                y = waves[bib][random_start * 300:(random_start + mel_seg_len) * 300]

                wav_seg.append(y.to(device))

            gt_mel_seg = torch.stack(gt_mel_seg).detach()

            wav_seg = torch.stack(wav_seg).float().detach().unsqueeze(1)

            wav_seg_input = wav_seg
            wav_seg_target = wav_seg

            z = watermark_model.encoder(wav_seg_input)

            # prepare message
            msg = np.random.choice([0,1], [hp.batch_size, 1, hp.msg_len])

            z, quantized, commitment_loss, codebook_loss, timbre, z_c_emb = watermark_model.quantizer(z, 
                                                                                            wav_seg_input, 
                                                                                            msg, 
                                                                                            n_c=2, 
                                                                                            full_waves=waves, 
                                                                                            wave_lens=wave_lengths)
            pred_wave = watermark_model.decoder(z)

            pred_msg = extracter.encoder(pred_wave)

            msg_loss = msg_criterion(pred_msg, msg)

            len_diff = wav_seg_target.size(-1) - pred_wave.size(-1)
            if len_diff > 0:
                wav_seg_target = wav_seg_target[..., len_diff // 2:-len_diff // 2]

            # dicriminator loss
            d_fake = watermark_model.discriminator(pred_wave.detach())
            d_real = watermark_model.discriminator(wav_seg_target)
            loss_d = 0
            for x_fake, x_real in zip(d_fake, d_real):
                loss_d += torch.mean(x_fake[-1] ** 2)
                loss_d += torch.mean((1 - x_real[-1]) ** 2)
                
            loss_d.backward()
            grad_norm_d = torch.nn.utils.clip_grad_norm_(watermark_model.discriminator.parameters(), 10.0)
            watermark_optimizer.step('discriminator')
            watermark_optimizer.scheduler(key='discriminator')

            # generator loss
            signal = AudioSignal(wav_seg_target, sample_rate=24000)
            recons = AudioSignal(pred_wave, sample_rate=24000)
            stft_loss = stft_criterion(recons, signal)
            mel_loss = mel_criterion(recons, signal)
            waveform_loss = l1_criterion(recons, signal)

            d_fake = watermark_model.discriminator(pred_wave)
            d_real = watermark_model.discriminator(wav_seg_target)

            loss_g = 0
            for x_fake in d_fake:
                loss_g += torch.mean((1 - x_fake[-1]) ** 2)

            loss_feature = 0

            for i in range(len(d_fake)):
                for j in range(len(d_fake[i]) - 1):
                    loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
                
            loss_gen_all = (waveform_loss + mel_loss + stft_loss) * 1.0 + loss_feature * 1.0 + loss_g * 1.0 + msg_loss * 1.0
            loss_gen_all.backward()

            grad_norm_g2 = torch.nn.utils.clip_grad_norm_(watermark_model.decoder.parameters(), 1000.0)
            grad_norm_g3 = torch.nn.utils.clip_grad_norm_(watermark_model.quantizer.parameters(), 1000.0)
            grad_norm_g4 = torch.nn.utils.clip_grad_norm_(extracter.encoder.parameters(), 10.0)

            watermark_optimizer.step('quantizer')
            watermark_optimizer.step('decoder')
            extracter_optimizer.step('encoder')

            watermark_optimizer.scheduler(key='quantizer')
            watermark_optimizer.scheduler(key='decoder')
            extracter_optimizer.scheduler(key='encoder')

            train_time_per_step = time.time() - train_start_time

            if iters % hp.log_step == 0:
                with torch.no_grad():
                    print("Epoch %d, Iteration %d, Total Loss: %.4f, Disc Loss: %.4f, mel Loss: %.4f, msg Loss: %.4f, Time: %.4f" % (epoch, iters, loss_gen_all.item(), loss_d.item(), mel_loss.item(), msg_loss.item(), train_time_per_step))
                

            

            iters = iters + 1


            

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and print model parameters")
    parser.add_argument('--ckpt_path', type=str, default='/home/facodec-checkpoints/pytorch_model.bin')
    parser.add_argument('--config_path', type=str, default='/home/FAcodecWatermark/configs/config.yml')
    
    args = parser.parse_args()
    
    #make_watermark_model(args)
    make_watermark_extracter(args)