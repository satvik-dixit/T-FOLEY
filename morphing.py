import argparse
import json
import os

import numpy as np
import torch
import torchaudio as T
import soundfile as sf
from model import UNet
from sampler import SDESampling_batch
from scipy.io.wavfile import write
from sde import VpSdeCos
from utils import (adjust_audio_length, get_event_cond, high_pass_filter,
                   normalize, pooling, resample_audio)

LABELS = ['DogBark', 'Footstep', 'GunShot', 'Keyboard', 'MovingMotorVehicle', 'Rain', 'Sneeze_Cough']


def load_ema_weights(model, model_path):
    checkpoint = torch.load(model_path)
    dic_ema = {}
    for (key, tensor) in zip(checkpoint['model'].keys(), checkpoint['ema_weights']):
        dic_ema[key] = tensor
    model.load_state_dict(dic_ema)
    return model

def generate_samples(target_event, class_idx1, class_idx2, sampler, cond_scale, device, N, audio_length):
    print(f"Generate {N} samples of class '{LABELS[class_idx1]}' and '{LABELS[class_idx2]}' using target audio...")

    # Generate gen_audio_1 using class_idx1 + target_event and gen_audio_2 using class_idx2 + target_event
    noise_1 = torch.randn(N, audio_length, device=device)
    classes_1 = torch.tensor([class_idx1] * N, device=device)
    sampler.batch_size = N
    gen_audio_1 = sampler.predict(noise_1, 100, classes_1, target_event, cond_scale=cond_scale)
    
    noise_2 = torch.randn(N, audio_length, device=device)
    classes_2 = torch.tensor([class_idx2] * N, device=device)
    gen_audio_2 = sampler.predict(noise_2, 100, classes_2, target_event, cond_scale=cond_scale)
    
    return gen_audio_1, gen_audio_2

def save_samples(gen_audio_1, gen_audio_2, target_audio, output_dir, sr, class_name_1, class_name_2, beta):
    # Save the target audio
    sf.write(f"{output_dir}/target_audio.wav", target_audio.cpu().numpy(), sr)
    
    # Direct summation with user-defined beta
    combined_samples = beta * gen_audio_1 + (1 - beta) * gen_audio_2
    
    for j in range(combined_samples.shape[0]):
        combined_sample = combined_samples[j].cpu()
        combined_sample = high_pass_filter(combined_sample)
        write(f"{output_dir}/{class_name_1}_{class_name_2}_combined_{beta}_{str(j+1).zfill(3)}.wav", sr, combined_sample)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Set model and sampler
    T.set_audio_backend('sox_io')
    device = torch.device('cuda')
    
    with open(args.param_path) as f:
        params = json.load(f)
    sample_rate = params['sample_rate']
    audio_length = sample_rate * 4
    model = UNet(len(LABELS), params).to(device)
    model = load_ema_weights(model, args.model_path)

    sde = VpSdeCos()
    sampler = SDESampling_batch(model, sde, batch_size=args.N, device=device)
    
    # Load and prepare target audio
    target_audio, sr = T.load(args.target_audio_path)
    if sr != sample_rate:
        target_audio = resample_audio(target_audio, sr, sample_rate)
    
    # Ensure audio is the correct length
    target_audio = adjust_audio_length(target_audio, audio_length)
    
    target_event = get_event_cond(target_audio, params['event_type']).repeat(args.N, 1).to(device)
    
    # Generate samples based on target audio
    class_idx1 = LABELS.index(args.class_name_1)
    class_idx2 = LABELS.index(args.class_name_2)
    gen_audio_1, gen_audio_2 = generate_samples(target_event, class_idx1, class_idx2, sampler, args.cond_scale, device, args.N, audio_length)
    
    # Save target and generated samples
    save_samples(gen_audio_1, gen_audio_2, target_audio, args.output_dir, sample_rate, args.class_name_1, args.class_name_2, args.beta)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./pretrained/block-49_epoch-500.pt')
    parser.add_argument('--param_path', type=str, default='./pretrained/params.json')
    parser.add_argument('--target_audio_path', type=str, required=True, help='Path to the target audio file.')
    parser.add_argument('--class_name_1', type=str, required=True, help='First class name for generating samples.', choices=LABELS)
    parser.add_argument('--class_name_2', type=str, required=True, help='Second class name for generating samples.', choices=LABELS)
    parser.add_argument('--output_dir', type=str, default="./results_idea_2")
    parser.add_argument('--cond_scale', type=int, default=3)
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--beta', type=float, default=0.5, help='Weighting factor for combining gen_audio_1 and gen_audio_2.')
    args = parser.parse_args()
    
    main(args)
