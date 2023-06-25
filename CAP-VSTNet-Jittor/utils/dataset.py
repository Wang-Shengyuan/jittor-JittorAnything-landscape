import os.path
import random
from PIL import Image
import jittor as jt
from PIL import ImageFile
from utils.MattingLaplacian import compute_laplacian

import numpy as np

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated



IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class ImageFolder(jt.dataset.Dataset):

    def __init__(self, root, transform=None, use_lap=False):
        imgs = []
        if isinstance(root, list):
            for r in root:
                imgs = imgs + sorted(make_dataset(r))
        elif isinstance(root, str):
            imgs = sorted(make_dataset(root))

        self.imgs = imgs
        self._length = len(self.imgs)
        if self._length == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\nSupported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.transform = transform
        self.to_tensor = jt.transform.Compose([jt.transform.ToTensor()])
        self.use_lap = use_lap

    def __getitem__(self, index):
        try:
            path = self.imgs[index]
            img = Image.open(path).convert('RGB')
        except OSError as err:
            # print(err)
            return self.__getitem__(random.randint(0, self._length - 1))
        if self.transform is not None:
            img = self.transform(img)

        if self.use_lap:
            laplacian_m = compute_laplacian(img)
        else:
            laplacian_m = None

        img = self.to_tensor(img)
        return {'img': img, 'laplacian_m': laplacian_m}

    def __len__(self):
        return self._length


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(jt.dataset.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def collate_fn(batch):
    img = [b['img'] for b in batch]
    img = jt.stack(img, dim=0)

    laplacian_m = [b['laplacian_m'] for b in batch]

    return {'img': img, 'laplacian_m': laplacian_m}


def get_data_loader_folder(input_folder, batch_size, new_size=288, height=256, width=256, use_lap=False, num_workers=None):
    transform_list = []
    transform_list = [jt.transform.RandomCrop((height, width))] + transform_list
    transform_list = [jt.transform.Resize(new_size)] + transform_list
    transform = jt.transform.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform, use_lap=use_lap)
    
    if num_workers is None:
        num_workers = 2*batch_size
    loader = jt.dataset.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True, num_workers=num_workers, sampler=InfiniteSamplerWrapper(dataset), collate_fn=collate_fn)
    return loader


