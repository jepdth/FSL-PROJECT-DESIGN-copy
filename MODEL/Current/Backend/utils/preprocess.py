import numpy as np
import torch

def process_input(data):
    """
    Convert incoming JSON data into a PyTorch tensor
    Expected: data["features"] = list of 188 floats
    Returns tensor with shape (1, 1, 188)
    """
    arr = np.array(data["features"], dtype=np.float32)

    # Validate length
    if arr.ndim != 1:
        arr = arr.flatten()
    if len(arr) < 188:
        arr = np.pad(arr, (0, 188 - len(arr)), 'constant')
    elif len(arr) > 188:
        arr = arr[:188]

    tensor = torch.tensor(arr).view(1, 1, 188)
    return tensor
