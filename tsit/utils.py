import numpy as np
from PIL import Image


def is_image_file(filename):
    """Function for checking the parameter value has image extension.

    Args:
        filename (String): Filename of checking target.

    Returns:
        Bool: Check it is image file or not.
    """
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    """Function for loading image by path

    Args:
        filepath (String): File path

    Returns:
        PIL Image: PIL Image of following path
    """
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    """Simple image saving function.

    Args:
        image_tensor (Torch Tensor)
        filename (String)
    """
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))
