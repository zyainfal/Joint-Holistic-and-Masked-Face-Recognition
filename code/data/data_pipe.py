from torch.utils.data import DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import numpy as np
import bcolz
import torch
from torch.utils.data import Dataset

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class MaskedImageFolder(ImageFolder):
    def __init__(self, root, transform=None, mask_ratio=0.0):
        super(MaskedImageFolder, self).__init__(root, transform)
        self.mask_ratio = mask_ratio

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        ismask = 0
        if np.random.random_sample() < self.mask_ratio:
            path = path.replace("imgs", "masked")
            try: # as some imgs are not masked
                img = self.loader(path)
                ismask = 1
            except:
                ismask = 0

        if self.transform is not None:
            img = self.transform(img)

        return img, ismask, target

def get_train_dataset(conf):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = MaskedImageFolder(conf.data_path+"/imgs", train_transform, conf.masked_faces)
    if "faces_emore" in conf.data_path:
        class_num = ds[-1][-1] + 1
    elif "glint" in conf.data_path:
        class_num = 360232
    elif "WebFace" in conf.data_path:
        class_num = 2058932 #2059906
    else:
        raise ValueError
    
    return ds, class_num

def get_train_loader(conf):
    emore_ds, emore_class_num = get_train_dataset(conf)

    loader = DataLoader(emore_ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory,
                        num_workers=conf.num_workers)
    return loader, emore_class_num 


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = os.path.join(path, name), mode='r')
    issame = np.load(os.path.join(path,'{}_list.npy'.format(name)))
    return carray, issame

def get_val_data(conf):
    data_path = conf.data_path
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame
