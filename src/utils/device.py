"""module contenant une fonction renvoyant l'accelerateur gpu disponible adapté a la machine"""
import torch
def get_device():
    """
    Renvoie le device PyTorch le plus adapté disponible.
    Priorité : CUDA > MPS > CPU
    """
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"[Device] Using {device}")
    return device