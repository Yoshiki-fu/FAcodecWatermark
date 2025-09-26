import shutil
import warnings
import argparse
import torch
import os
import yaml

warnings.simplefilter('ignore')

from modules.commons import *
from losses import *
import time

import torchaudio
import librosa

import watermark_hparams as hp
from modules.watermarking import hex_to_bin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(args):
    emb_ckpt_path = args.emb_ckpt_path      # 埋め込みモデルのチェックポイントパス
    extract_ckpt_path = args.extract_ckpt_path  # 抽出モデルのチェックポイントパス
    config_path = args.config_path
    config = yaml.safe_load(open(config_path))
    model_params = recursive_munch(config['model_params'])
    watemark_model = build_model(model_params, 'watermarking')
    extracter = build_model(model_params, 'extracter_v2')

    emb_ckpt_params = torch.load(emb_ckpt_path)
    emb_ckpt_params = emb_ckpt_params['net'] if 'net' in emb_ckpt_params else emb_ckpt_params  # adapt to format of self-trained checkpoints

    for key in emb_ckpt_params:
        watemark_model[key].load_state_dict(emb_ckpt_params[key])

    _ = [watemark_model[key].eval() for key in watemark_model]
    _ = [watemark_model[key].to(device) for key in watemark_model]

    extract_ckpt_params = torch.load(extract_ckpt_path)
    extract_ckpt_params = extract_ckpt_params['net'] if 'net' in extract_ckpt_params else extract_ckpt_params  # adapt to format of self-trained checkpoints

    for key in extract_ckpt_params:
        extracter[key].load_state_dict(extract_ckpt_params[key])
    
    _ = [extracter[key].eval() for key in extracter]
    _ = [extracter[key].to(device) for key in extracter]

    return watemark_model, extracter

@torch.no_grad()
def main(args):

    watermark_model, extracter = load_model(args)
    source = args.source
    source_audio = librosa.load(source, sr=24000)[0]
    # crop only the first 30 seconds
    source_audio = source_audio[:24000 * 30]
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)

    # prepare message
    msg = np.random.choice([0,1], [1, hp.msg_len])
    msg = torch.tensor(msg)
    msg = msg.to(device)

    # without timbre norm
    z = watermark_model.encoder(source_audio[None, ...].to(device).float())
    z, quantized, commitment_loss, codebook_loss, timbre, z_c_emb, content_vec2, wm_content_vec2, chunk_label = watermark_model.quantizer(z,
                                                                                                                                          source_audio[None, ...].to(device).float(),
                                                                                                                                          msg,
                                                                                                                                          n_c=2)

    pred_wave = watermark_model.decoder(z)

    h = extracter.encoder(pred_wave)
    logit_vad, logit_chunk = extracter.quantizer(h, pred_wave, n_c=2)
    binary_message = hex_to_bin(logit_chunk)

    os.makedirs("reconstructed", exist_ok=True)
    source_name = source.split("/")[-1].split(".")[0]
    torchaudio.save(f"reconstructed/{source_name}.wav", pred_wave[0].cpu(), 24000)

    decoder_acc = [((binary_message).eq(msg).sum().float() / msg.numel()).item()]
    print(f"Decoder accuracy: {decoder_acc[0]*100}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_ckpt_path', type=str, default='/workspace/checkpoints/modelv2_1/watermarv2_model_epoch_11_iter_487817.pth')
    parser.add_argument('--extract_ckpt_path', type=str, default='/workspace/checkpoints/modelv2_1/extracterv2_model_epoch_11_iter_487817.pth')
    parser.add_argument('--config_path', type=str, default='/home/FAcodecWatermark/configs/config.yml')
    parser.add_argument('--source', type=str, default='/workspace/DS_10283_3443/wav48_data/p299/p299_001_mic1.wav')
    args = parser.parse_args()
    main(args)