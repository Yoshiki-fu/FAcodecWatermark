from meldataset import PseudoDataset, collate, preprocess
from torch.utils.data import DataLoader
import librosa
import torch
import os


if __name__ == "__main__":
    dataset = PseudoDataset()
    dataset_size = dataset.__len__()
    wave, mel = dataset.__getitem__(0)
    print(f"dataset size is {dataset_size}")
    print(f"wave shape is {wave.size()}")
    print(f"mel shape is {mel.size()}")

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate)

    for epoch in range(1):
        for i, batch in enumerate(dataloader):
            wave, mel, wav_legnth, mel_length = batch
            print(f"wave shape in training: {wave.unsqueeze(1).size()}")
            print(f"mel shape in training: {mel.shape}")
            print(len(wav_legnth))
            print(len(mel_length))

            if i == 0:
                break

    path = os.path.join("/workspace/DS_10283_3443/VCTK_corpus_uk/trim_wav48/p225", "p225_149_mic2.wav")
    wave, sr = librosa.load(path, sr=24000)
    mel = preprocess(wave).squeeze(0)
    wave = torch.from_numpy(wave).float()
    print("mel shape is {0}".format(mel.shape))