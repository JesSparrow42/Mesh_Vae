import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
from data_loader import create_data_loader
from ptseries.models import PTGenerator
from ptseries.algorithms.gans.utils import infiniteloop
from utils import *

def main():
    ### Hyperparameters and Boson Sampler Settings
    boson_sampler_params = {
        "input_state": [1, 0, 1, 0, 1, 0, 1, 0],  # Quantum state |1,0,1,0,1,0,1,0>
        "tbi_params": {
            "input_loss": 0.0,
            "detector_efficiency": 1,
            "bs_loss": 0,
            "bs_noise": 0,
            "distinguishable": False,
            "n_signal_detectors": 0,
            "g2": 0,
            "tbi_type": "multi-loop",
            "n_loops": 3,
            "loop_lengths": [2, 3, 4],
            "postselected": True
        },
        "n_tiling": 1
    }
    
    NUM_ITER = 1  # Number of training iterations (for demonstration)
    latent_size = 256  # Discrete latent space size (number of tokens)

    ### Data Loading
    # Specify folders for CT and PET scans
    ct_folder = 'NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/2.000000-CT IMAGES-25805'
    pet_folder = 'NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/1.000000-PET NAC-24000'
    
    num_workers = 4  # Adjust based on available hardware resources
    dicom_files = create_data_loader(ct_folder=ct_folder, pet_folder=pet_folder, num_workers=num_workers, augment=False)
    
    ### Initialize the Latent Space Generator
    # This generator uses a boson sampler-inspired approach to produce a latent distribution.
    latent_space = PTGenerator(**boson_sampler_params)
    
    ### Set Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    ### Obtain Boson Sampler Distribution
    # Here we assume that PTGenerator has a method 'sample_latent_distribution'
    # that returns a tensor representing the boson sampler distribution over latent tokens.
    boson_distribution_tensor = latent_space.sample_latent_distribution(latent_size)  # <-- Replace with actual method if different
    boson_distribution = boson_distribution_tensor.detach().cpu().numpy()  # Convert to NumPy for further processing
    
    ### Simulate Transformer Prediction Distribution
    # In a transformer model, predictions over discrete tokens (latent space) can be generated.
    transformer_probs = np.random.dirichlet(np.ones(latent_size))
    
    ### Combine Distributions for Token Sampling
    alpha = 0.6  # Weight: 60% transformer, 40% boson sampler
    combined_probs = alpha * transformer_probs + (1 - alpha) * boson_distribution
    combined_probs /= combined_probs.sum()  # Normalize to ensure the sum is 1
    
    # Sample a latent token from the combined distribution.
    next_token = np.random.choice(latent_size, p=combined_probs)
    print("Selected latent token index:", next_token)
    
    ### Load a Batch of Data
    looper = infiniteloop(dicom_files)
    pet_images, ct_images = next(looper)
    pet_images = pet_images.to(device)
    ct_images = ct_images.to(device)
    batch_size = pet_images.shape[0]
    print("Batch size:", batch_size)
    
    ### TODO: Integrate with the Transformer-Based Image Generator
    # The sampled 'next_token' can be used as part of the input sequence (or conditioning variable)
    # for a transformer model that generates CT images from PET scans.
    # This is where you would add the forward pass, loss computation, backpropagation, etc.
    
if __name__ == '__main__':
    main()
