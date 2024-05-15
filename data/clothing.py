# borrow some code from https://github.com/askmuhsin/clothing1M_experiments/blob/main/data/clothing1m.py
# path on dbcloud: /remote-home/share/datasets/Clothing1M/Clothing1M/data

from torch.utils.data import DataLoader, Dataset
# import torch
import os
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

class Clothing1M_dataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        data_file,
        data_label_file,
        transform,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.data_file = os.path.join(root_dir, data_file)
        self.data_label_file = os.path.join(root_dir, data_label_file)
        self.transform = transform
        self.get_datas()
    
    def get_datas(self):
        self.imgs = []
        img2index = {}
        
        with open(self.data_file, 'r') as f:
            lines = f.read().splitlines()
        print('getting data paths')
        for l in tqdm(lines):
            img_path = os.path.join(self.root_dir, l)
            self.imgs.append(img_path)
            img2index[img_path] = len(self.imgs) - 1
        print(f'Found {len(self.imgs)} images')
        self.labels = [None] * len(self.imgs)
        with open(self.data_label_file, 'r') as f:
            lines = f.read().splitlines()
        print('getting labels')
        for l in tqdm(lines):
            entry = l.split()
            img_path = os.path.join(self.root_dir, entry[0])
            label = int(entry[1])
            index = img2index.get(img_path, None)
            if index is not None:
                self.labels[index] = label
        # index where label is None
        print('checking labels')
        for i in tqdm(range(len(self.labels))):
            if self.labels[i] is None:
                print(f'index {i} has no label')
        assert all([l is not None for l in self.labels])
        del img2index, lines

            # if img_path in self.imgs:
            # self.labels[img_path] = label
        # assert len(self.imgs) == len(self.labels)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return {
            'input': img,
            'target': label,
            'index': index
        }

def Clothing1M(config, logger):
    img_resize = (256, 256) if 'img_resize' not in config['dataset'] else config['dataset']['img_resize']
    img_size = (224, 224) if 'img_size' not in config['dataset'] else config['dataset']['img_size']
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    num_classes = 14

    transform = transforms.Compose([
                transforms.Resize(img_resize),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    test_transform = transforms.Compose([
                transforms.Resize(img_resize),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    train_dset = Clothing1M_dataset(
        root_dir=config['dataset']['root'],
        data_file=config['dataset']['train_data_file'],
        data_label_file=config['dataset']['train_data_label_file'],
        transform=transform,
    )
    test_dset = Clothing1M_dataset(
        root_dir=config['dataset']['root'],
        data_file=config['dataset']['test_data_file'],
        data_label_file=config['dataset']['test_data_label_file'],
        transform=test_transform,
    )
    config['training_opt']['test_batch_size'] = config['training_opt']['batch_size'] if 'test_batch_size' not in config['training_opt'] else config['training_opt']['test_batch_size']

    test_loader = DataLoader(
        test_dset, batch_size = config['training_opt']['test_batch_size'], shuffle=False, num_workers=config['training_opt']['num_workers'], pin_memory=True, drop_last=False
    )
    return {
        'num_classes': num_classes,
        'train_dset': train_dset,
        'test_loader': test_loader,
        'num_train_samples': len(train_dset)
    }
    



if __name__ == '__main__':
    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) # meanstd transformation
    dset = Clothing1M_dataset(
        root_dir='/remote-home/share/datasets/Clothing1M/Clothing1M/data',
        data_file='noisy_train_key_list.txt',
        data_label_file='noisy_label_kv.txt',
        transform=transform,
    )
    print(len(dset))
    print(dset[0]['input'].shape)
    print(dset[0]['target'])
    print(dset[0]['index'])
    print(dset[1]['input'].shape)
    print(dset[1]['target'])
    print(dset[1]['index'])

    labels = dset.labels
    # get number of unique labels
    print(len(set(labels)))
        