import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from pathlib import Path
import time
import os

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence


def synthesize(model, file_name, step, configs, batch, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values
    device = model_config['device']
    
    import os
    duration = torch.from_numpy(np.load(os.path.join('preprocessed_data/Persian/duration', 'single_speaker-duration-' + file_name + '.npy'))).long().to(device).unsqueeze(0)

    batch = to_device(batch, device)
    with torch.no_grad():
        # Forward
        output = model(
            *(batch[1:]),

            d_targets=duration,

            p_control=pitch_control,
            e_control=energy_control,
            d_control=duration_control
        )
        specs = output[1].transpose(1, 2)
    return specs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    
    parser.add_argument(
        "--train_text_data",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    device = model_config['device']
    model = get_model(args, configs, device, train=False)

    data_path = args.train_text_data
    f = open(data_path, 'r')
    data = []
    for line in f.readlines():
        line = line.strip()
        line_parts = line.split('|')
        data.append((line_parts[0], line_parts[2]))
        
    i = 0
    for file_name, text in data: 
        i += 1
        print(str(i) + ") text file: " + str(file_name))
        t = str(time.time()).split('.')[0]
        ids = ['speaker_id_' + str(args.speaker_id) + '_' + t]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "fa":
            texts = np.array([text_to_sequence(text, preprocess_config['preprocessing']['text']['text_cleaners'])])
        text_lens = np.array([len(texts[0])])
        batch = (ids, speakers, texts, text_lens, max(text_lens))

        control_values = args.pitch_control, args.energy_control, args.duration_control

        specs = synthesize(model, file_name, args.restore_step, configs, batch, control_values)
        
        np.save(
            os.path.join(args.out_dir, file_name),
            specs.cpu(),
        )
        
    
    ## python create_specs.py --restore_step 300000 --train_text_data dataset/train.txt --out_dir output/synthesized_specs/ --preprocess_config config/Persian/preprocess.yaml  --model_config config/Persian/model.yaml --train_config config/Persian/train.yaml

