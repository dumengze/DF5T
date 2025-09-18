import torch.utils.data as data 
import torchvision.transforms as transforms
from PIL import Image



def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageDataset(data.Dataset):
    def __init__(self,
                 root_dir,
                 meta_file,
                 transform=None,
                 image_size=128,
                 normalize=True):
        """
        Args:
            root_dir (str): Path to the root directory with images.
            meta_file (str): Path to the txt file containing the list of image paths.
            transform (callable, optional): A function/transform to apply to the image.
            image_size (int, optional): Resize images to this size. Default is 128.
            normalize (bool, optional): Whether to normalize the images. Default is True.
        """
        self.root_dir = root_dir
        self.meta_file = meta_file
        
        # Set the transform based on whether normalization is enabled
        if transform is not None:
            self.transform = transform
        else:
            norm_mean = [0.5, 0.5, 0.5]
            norm_std = [0.5, 0.5, 0.5]
            if normalize:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor()
                ])
        
        # Read the .txt file that contains image paths and labels
        with open(meta_file, 'r') as f:
            lines = f.readlines()
        
        self.num = len(lines)
        self.metas = []
        
        # Assuming the txt file format is:
        # <image_path> <label>
        suffix = ".png"
        for line in lines:
            line_split = line.rstrip().split()
            if len(line_split) == 2:
                self.metas.append((line_split[0] + suffix, int(line_split[1])))
            else:
                self.metas.append((line_split[0] + suffix, -1))

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = self.root_dir + '/' + self.metas[idx][0]
        cls = self.metas[idx][1]
        img = default_loader(filename)

        # Apply transformation
        if self.transform is not None:
            img = self.transform(img)

        return img, cls  # Optionally you can also return the image path if needed
