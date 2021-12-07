import torch

def convert_model_z_to_latent(z):
    return z[0].reshape(-1).tolist()


def convert_latent_to_model_z(latent):
    return torch.tensor(latent)
