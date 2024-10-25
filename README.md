# Persian FastSpeech2

**Persian FastSpeech2** is an adaptation of Microsoft's FastSpeech2, optimized for generating high-quality speech from Persian text. This project is based on [official implementation](https://github.com/ming024/FastSpeech2) 
and includes language-specific modifications to handle Persian phonemes and text structure.

## Overview
FastSpeech2 is a non-autoregressive TTS model that offers faster and more robust speech synthesis than autoregressive models like Tacotron.

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YourUsername/Persian-FastSpeech2.git
   cd Persian-FastSpeech2

1. **Install Requirements**  
   ```bash
   pip3 install -r requirements.txt

## Quickstart
### Inference
  1. *Prepare Pretrained Models*
     
     Place pretrained model files in output/ckpt/ or any specified folder.

  3. *Run Inference*
     
     Generate audio by specifying text and model checkpoints:
     Synthesized audio will appear in output/result/.
     
      ```bash
      python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step YOUR_CHECKPOINT_STEP --mode
    
  ### Controllability
  Adjust pitch, volume, and speaking rate by specifying control ratios. For example:
  
  ```
     python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step YOUR_CHECKPOINT_STEP --mode single -p config/Preprocess.yaml -m config/Model.yaml -t config/Train.yaml --duration_control 0.8 --energy_control 0.8
  ```

### Training

1. *Prepare Data*
  * Organize Persian audio data and phoneme sequences.
  * Follow preprocessing steps to align text and audio.

2. *Run Preprocessing*

  ```
    python3 preprocess.py config/Preprocess.yaml
  ```

3. *Train Model*
  Start training with:

  ```
    python3 train.py -p config/Preprocess.yaml -m config/Model.yaml -t config/Train.yaml
  ```


## Acknowledgments
This project builds on:

* [FastSpeech2 paper](https://arxiv.org/abs/2006.04558)
* [FastSpeech2 implementation](https://github.com/ming024/FastSpeech2)
