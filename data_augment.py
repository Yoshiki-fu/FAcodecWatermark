import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mask_random_wave_frames(wav: torch.Tensor, vad_labels: torch.Tensor, frame_size: int=300, mask_prob: float=0.1):
    """
    Randomly mask frames in the waveform.
    Args:
        wav: Tensor of shape (batch_size, samples) 1D waveform
        frame_size: sample nums of 1 frame, example 300 for 12.5ms at 24kHz
        mask_prob: Probability of masking a frame
        Returns:
        masked_wav: Tensor of shape (batch_size, samples) 1D waveform with masked frames
        vad_labels: Tensor of shape (batch_size, num_frames) VAD labels
    returns:
        masked_wav: Tensor of shape (batch_size, samples) 1D waveform with masked frames
        vad_labels: Tensor of shape (batch_size, num_frames) VAD labels edited to reflect the changes
    """
    B, S = wav.size()
    total_samples = S
    num_frames = total_samples // frame_size
    tmp_wav = wav[:, :num_frames * frame_size]  # フレーム分割可能な長さに切り詰め

    frames = tmp_wav.view(B, num_frames, frame_size)
    mask = (torch.rand(B, num_frames) > mask_prob).float().to(device)
    mask = mask.unsqueeze(-1)  # (B, num_frames, 1)に変形
    masked_frames = frames * mask  # 無音化
    masked_wav = masked_frames.view(B, -1)
    # 波形に無音処理を適用
    #wav[:, :masked_wav.size(1)] = masked_wav
    
    # ラベルを貼り直す処理
    mask = mask.squeeze(-1)
    vad_labels = vad_labels * mask

    return masked_wav, vad_labels


def partial_watermarking_filtering(original_wav: torch.Tensor, watermarked_wav: torch.Tensor, vad_labels: torch.Tensor, frame_size: int=1200, prob=0.9):
    """
    Replace 50ms segments of the watermarked waveform with original waveform at 50% probability.
    Args:
        original_wav: Tensor of shape (batch_size, samples) 1D waveform
        watermarked_wav: Tensor of shape (batch_size, samples) 1D waveform with watermark
        frame_size: sample nums of 1 frame, example 1200 for 50ms at 24kHz
        segment_ms: Length of each segment in milliseconds
        prob: Probability of replacing a segment with the original signal
    Returns:
        watermarked_wav: Tensor of shape (batch_size, samples) 1D waveform with replaced segments
        vad_labels: Tensor of shape (batch_size, num_frames) VAD labels edited to reflect the changes
    """
    assert original_wav.size() == watermarked_wav.size(), "Original and watermarked waveforms must have the same shape."

    B, S = original_wav.size()
    total_samples = S
    num_frames = total_samples // frame_size
    tmp_original_wav = original_wav[:, :num_frames * frame_size]  # フレーム分割可能な長さに切り詰め
    original_frames = tmp_original_wav.view(B, num_frames, frame_size)
    mask = (torch.rand(B, num_frames) > prob).to(device)
    mask = mask.unsqueeze(-1).float()  # (B, num_frames, 1)に変形
    original_frames = original_frames * mask        # 元の音声のフレームを選択
    
    replace_mask = (~mask).float()       # 置き換えないフレームは1.0, 置き換えるフレームは0.0
    tmp_watermarked_wav = watermarked_wav[:, :num_frames * frame_size]  # フレーム分割可能な長さに切り詰め
    watermarked_frames = tmp_watermarked_wav.view(B, num_frames, frame_size)
    watermarked_frames = watermarked_frames * replace_mask  # 元の音声で置き換えるフレームをゼロにする

    combined_frames = original_frames + watermarked_frames  # フレームを結合
    combined_wav = combined_frames.view(B, -1)  # 1D waveform

    watermarked_wav[:, :combined_wav.size(1)] = combined_wav

    # ラベルを貼り直す処理
    replace_mask = replace_mask.squeeze(1).repeat_interleave(4, dim=1)      # 4=0.05/0.0125 1のフレームを4倍に拡張
    vad_labels[:, :replace_mask.size(1)] = vad_labels[:, :replace_mask.size(1)] * replace_mask

    return watermarked_wav, vad_labels


import torch
import torch.nn as nn
import torchaudio
import numpy as np

class PseudoVCInjector(nn.Module):
    def __init__(self, sample_rate=24000, n_mels=80, n_fft=1024, hop_length=256, attack_prob=0.5):
        super().__init__()
        self.sample_rate = sample_rate
        self.attack_prob = attack_prob
        
        # 1. Mel-Resynthesis用 (微分可能なVocoder劣化の近似)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, 
            n_mels=n_mels, normalized=True
        )
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, hop_length=hop_length, n_iter=16 # 軽めに設定
        )
        
        # 2. リサンプリング用
        self.resample_down = torchaudio.transforms.Resample(sample_rate, 16000)
        self.resample_up = torchaudio.transforms.Resample(16000, sample_rate)

    def forward(self, waveform):
        """
        waveform: (B, 1, T) or (B, T)
        """
        # 確率で攻撃をスキップ
        if np.random.rand() > self.attack_prob:
            return waveform

        # 攻撃の種類をランダム選択
        attack_type = np.random.choice(['noise', 'resample', 'mel_resynth'], p=[0.4, 0.3, 0.3])
        
        attacked_wav = waveform.clone()

        if attack_type == 'noise':
            # 量子化ノイズのシミュレーション (SNR 20dB-40dB程度)
            noise_level = torch.rand(1).item() * 0.02 + 0.001
            noise = torch.randn_like(attacked_wav) * noise_level
            attacked_wav = attacked_wav + noise

        elif attack_type == 'resample':
            # 帯域制限攻撃
            attacked_wav = self.resample_up(self.resample_down(attacked_wav))
            # 長さが微妙に変わる場合があるので合わせる
            if attacked_wav.shape[-1] != waveform.shape[-1]:
                min_len = min(attacked_wav.shape[-1], waveform.shape[-1])
                attacked_wav = attacked_wav[..., :min_len]
                waveform = waveform[..., :min_len] # 元波形も合わせておく(Loss計算用)

        elif attack_type == 'mel_resynth':
            # メルスペクトログラム経由での再合成 (Vocoder劣化)
            # ※ GriffinLimへの勾配伝播は不安定な場合があるので、ここだけ no_grad にするか、
            # 構造的劣化として割り切るのが一般的です。今回は学習させるため勾配を通してみます。
            mels = self.mel_transform(attacked_wav)
            attacked_wav = self.griffin_lim(mels)
            
            # 長さ合わせ
            if attacked_wav.shape[-1] != waveform.shape[-1]:
                min_len = min(attacked_wav.shape[-1], waveform.shape[-1])
                attacked_wav = attacked_wav[..., :min_len]
                # waveformは呼び出し元で調整が必要になるため、ここではattacked_wavをpadding/cropして戻すのが安全
                if attacked_wav.shape[-1] < waveform.shape[-1]:
                    pad = waveform.shape[-1] - attacked_wav.shape[-1]
                    attacked_wav = torch.nn.functional.pad(attacked_wav, (0, pad))
                else:
                    attacked_wav = attacked_wav[..., :waveform.shape[-1]]

        return attacked_wav