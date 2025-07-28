import os
import argparse
import numpy as np
import librosa
import parselmouth

"""wave = librosa.load("/home/FAcodecWatermark/reconstructed/p299_001_mic1.wav", sr=24000)[0]
print(f"Wave shape: {wave.shape}")
snd = parselmouth.Sound("/home/FAcodecWatermark/reconstructed/p299_001_mic1.wav")
print(snd)
pitch = snd.to_pitch(time_step=0.01)

vad_labels = []
sample_sec = 0.00004167
for i in range(wave.shape[0]):
    f0 = pitch.get_value_at_time(sample_sec * i)
    vad_labels.append(1 if f0 and f0 > 0 else 0)

assert len(vad_labels) == wave.shape[0], "VAD labels length does not match wave length"
print(len(vad_labels))
vad_labels = np.array(vad_labels)
print(vad_labels.shape)"""

"""name = os.path.basename("/home/FAcodecWatermark/reconstructed/p299_001_mic1.wav").split(".")[0]
print(name)
save_path = f"./reconstructed/{name}_vad_labels.npy"
np.save(save_path, vad_labels)"""

def create_vad_labels(args):
    dir_path = args.data_dir_path
    sample_sec = 1 / args.sr
    extensions = ['.wav']
    missing_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(tuple(extensions)):
                file_path = os.path.join(root, file)
                wave = librosa.load(file_path, sr=24000)[0]
                snd = parselmouth.Sound(file_path)
                pitch = snd.to_pitch(time_step=0.01)
                vad_labels = []
                for i in range(wave.shape[0]):
                    f0 = pitch.get_value_at_time(sample_sec* i)
                    vad_labels.append(1 if f0 and f0 > 0 else 0)
                assert len(vad_labels) == wave.shape[0], "VAD labels length does not match wave length"
                if len(vad_labels) != wave.shape[0]:
                    print(f"Length mismatch for {file_path}: {len(vad_labels)} != {wave.shape[0]}")
                    missing_files.append(file_path)
                    continue
                else:
                    vad_labels = np.array(vad_labels)
                    save_name = os.path.basename(file_path).split(".")[0]
                    save_path = os.path.join(root, f"{save_name}_vad_labels.npy")
                    np.save(save_path, vad_labels)

    print(f"Missing files: {missing_files}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir_path', type=str)
    parser.add_argument('--sr', type=int, default=24000)
    args = parser.parse_args()
    create_vad_labels(args)