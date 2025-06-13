import torch
import torch.nn as nn
from torch.utils.data import Dataset
import watermark_hparams as hp

class Librilight(Dataset):
    def __init__(self, sr=24000, range=(1, 30)):
        self.data_list = self.get_all_data_path(dir_path=hp.train_data_path, extensions=['.wav'])
        self.sr = sr
        self.duration_range = range

     def get_all_data_path(self, dir_path, extensions=None):
        file_paths = []
        if extensions is not None:
            extensions = tuple(extensions)
        for root, _, files in os.walk(dir_path):
            for file in files:
                if extensions is None or file.endswith(extensions):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
        return file_paths 
    
    def __len__(self):
        return len(self.data_list)

        def __getitem__(self, idx):
        # replace this with your own data loading
        wave, sr = librosa.load(self.data_list[idx], sr=self.sr)
        # wave = np.random.randn(self.sr * random.randint(*self.duration_range))
        # wave = wave / np.max(np.abs(wave))
        mel = preprocess(wave).squeeze(0)
        wave = torch.from_numpy(wave).float()
        return wave, mel 
    
def collate(batch):
     # batch[0] = wave, mel, text, f0, speakerid
    batch_size = len(batch)

    # sort by mel length        mel shape (mel_freq_bin, frame)
    lengths = [b[1].shape[1] for b in batch]
    batch_indexes = np.argsort(lengths)[::-1]
    batch = [batch[bid] for bid in batch_indexes]

    nmels = batch[0][1].size(0)
    max_mel_length = max([b[1].shape[1] for b in batch])
    max_wave_length = max([b[0].size(0) for b in batch])

    mels = torch.zeros((batch_size, nmels, max_mel_length)).float() - 10
    waves = torch.zeros((batch_size, max_wave_length)).float()

    mel_lengths = torch.zeros(batch_size).long()
    wave_lengths = torch.zeros(batch_size).long()

    # メルスペクトログラムと波形のそれぞれのpadding処理
    for bid, (wave, mel) in enumerate(batch):
        mel_size = mel.size(1)
        mels[bid, :, :mel_size] = mel
        waves[bid, : wave.size(0)] = wave
        mel_lengths[bid] = mel_size
        wave_lengths[bid] = wave.size(0)

    return waves, mels, wave_lengths, mel_lengths