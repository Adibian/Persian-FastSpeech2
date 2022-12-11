import re
import argparse
from string import punctuation
import time

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values
    device = model_config['device']
    
    # path = '/mnt/hdd1/adibian/FastSpeech2/Single-Speaker-FastSpeech2-new-spec/output/check_vectors/012989-1/features'
    #path = '/mnt/hdd1/adibian/FastSpeech2/Single-Speaker-FastSpeech2-new-spec/output/check_vectors/001571-1'
    #path = '/mnt/hdd1/adibian/FastSpeech2/Single-Speaker-FastSpeech2-new-spec/output/check_vectors/006880-1'
    #path = '/mnt/hdd1/adibian/FastSpeech2/Single-Speaker-FastSpeech2-new-spec/output/check_vectors/012989-1'
    # import os
    # duration = torch.from_numpy(np.load(os.path.join(path, 'duration_target.npy'))).long().to(device).unsqueeze(0)
    # pitch = torch.from_numpy(np.load(os.path.join(path, 'pitch_target.npy'))).float().to(device).unsqueeze(0)
    # energy = torch.from_numpy(np.load(os.path.join(path, 'energy_target.npy'))).to(device).unsqueeze(0)
    # print(duration.shape)
    # print(pitch.shape)
    # print(energy.shape)
    # energy = None
    
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[1:]),
                
                # p_targets=pitch,
                # e_targets=energy,
                # d_targets=duration,
                
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
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

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    device = model_config['device']

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        t = str(time.time()).split('.')[0]
        ids = ['speaker_id_' + str(args.speaker_id) + '_' + t]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "fa":
            texts = np.array([text_to_sequence(args.text, preprocess_config['preprocessing']['text']['text_cleaners'])])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)

