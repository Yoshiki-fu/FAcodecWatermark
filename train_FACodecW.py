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
    # モデルの定義
    watermark_model = make_watermark_model(args)
    extracter = make_watermark_extracter(args)

    # データセットの準備

    # 学習

    return

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and print model parameters")
    parser.add_argument('--ckpt_path', type=str, default='/home/facodec-checkpoints/pytorch_model.bin')
    parser.add_argument('--config_path', type=str, default='/home/FAcodecWatermark/configs/config.yml')
    
    args = parser.parse_args()
    
    #make_watermark_model(args)
    make_watermark_extracter(args)