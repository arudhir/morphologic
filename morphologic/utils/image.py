from skimage.exposure import rescale_intensity

RGB_MAP = {
    1: {
        'rgb': np.array([19, 0, 249]),
        'range': [0, 51]
    },
    2: {
        'rgb': np.array([42, 255, 31]),
        'range': [0, 107]
    },
    3: {
        'rgb': np.array([255, 0, 25]),
        'range': [0, 64]
    },
    4: {
        'rgb': np.array([45, 255, 252]),
        'range': [0, 191]
    },
    5: {
        'rgb': np.array([250, 0, 253]),
        'range': [0, 89]
    },
    6: {
        'rgb': np.array([254, 255, 40]),
        'range': [0, 191]
    }
}

def normalize_channel(channel: np.ndarray) -> np.ndarray:
    """Normalize image channel."""
    return rescale_intensity(channel, in_range='image', out_range=(0, 1))

def convert_to_rgb(self, tensor: np.ndarray, vmax: int = 255) -> np.ndarray:
    """
    Converts the 6-channel image tensor to RGB using Recursion's color mapping.
    
    Parameters:
    -----------
    tensor : np.ndarray
        Image tensor of shape (height, width, 6)
    vmax : int
        Maximum value for scaling
        
    Returns:
    --------
    np.ndarray : RGB image
    """
    colored_channels = []
    for i in range(6):
        channel_num = i + 1
        x = (tensor[:, :, i] / vmax) / \
            ((RGB_MAP[channel_num]['range'][1] - RGB_MAP[channel_num]['range'][0]) / 255) + \
            RGB_MAP[channel_num]['range'][0] / 255
        x = np.clip(x, 0, 1)
        x_rgb = np.array(
            np.outer(x, RGB_MAP[channel_num]['rgb']).reshape(tensor.shape[0], tensor.shape[1], 3),
            dtype=int)
        colored_channels.append(x_rgb)
    
    # Combine all channels
    combined = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    combined = np.clip(combined, 0, 255)
    return combined
