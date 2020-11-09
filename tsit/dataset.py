from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from tsit.utils import is_image_file


class DatasetFromFolder(data.Dataset):
    """Dataset Class inherited by PyTorch data.Dataset.
    This is designed for image-to-image translation task which has a, b pairs separated in different folders.
    DatasetFromFolder consists of __init__, __getitem__, __len__.
    """
    def __init__(self, image_dir, direction):
        """Function which initializes class DatasetFromFolder.

        Args:
            image_dir (String): Root folder of target dataset
            direction (String): Select a2b or b2a
        """
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        """Get items based on item list made in initialization function by index.

        Args:
            index (int)

        Returns:
            Torch Tensor, Torch Tensor
        """
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        a = a.resize((740, 740), Image.BICUBIC)
        b = b.resize((740, 740), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        w_offset = random.randint(0, max(0, 740 - 720 - 1))
        h_offset = random.randint(0, max(0, 740 - 720 - 1))
    
        a = a[:, h_offset:h_offset + 720, w_offset:w_offset + 720]
        b = b[:, h_offset:h_offset + 720, w_offset:w_offset + 720]
    
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        if random.random() < 0.5:
            idx = [i for i in range(a.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            a = a.index_select(2, idx)
            b = b.index_select(2, idx)

        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.image_filenames)
