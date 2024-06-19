from torchvision import datasets, transforms
import torch
import os
# import requests
# import zipfile
# from tqdm import tqdm

class wrapped_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # self.targets = dataset.targets
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return {
            'input': self.dataset[index][0],
            'target': self.dataset[index][1],
            'index': index
        }

def TinyImageNet(config, logger):
    if not os.path.exists(os.path.join(config['dataset']['root'], "tiny-imagenet-200")):
        # url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"  # 248MB
        # logger.info("Downloading Tiny-ImageNet")
        # r = requests.get(url, stream=True)
        # file_size = int(r.headers.get('Content-Length', 0))
        # progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)

        # with open(os.path.join(config['dataset']['root'], "tiny-imagenet-200.zip"), 'wb') as file:
        #     for chunk in r.iter_content(chunk_size=1024):
        #         file.write(chunk)
        #         progress_bar.update(len(chunk))
        # # with open(os.path.join(config['dataset']['root'], "tiny-imagenet-200.zip"), "wb") as f:
        # #     for chunk in r.iter_content(chunk_size=1024):
        # #         if chunk:
        # #             f.write(chunk)

        # logger.info("Unzipping Tiny-ImageNet")
        # with zipfile.ZipFile(os.path.join(config['dataset']['root'], "tiny-imagenet-200.zip"), 'r') as zip_ref:
        #     zip_ref.extractall(config['dataset']['root'])
        # logger.info("getting Tiny-ImageNet done")
        raise ValueError("Dataset not found")

    im_size = (32, 32) if 'downsize' in config['dataset'] and config['dataset']['downsize'] else (64, 64)
    num_classes = 200
    mean = [0.4802, 0.4481, 0.3975]
    std = (0.2770, 0.2691, 0.2821)

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)
        ])
    test_transform =  transforms.Compose([transforms.RandomCrop(im_size),transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    dst_train = datasets.ImageFolder(root=os.path.join(config['dataset']['root'], 'tiny-imagenet-200/train'), transform=transform)
    dst_test = datasets.ImageFolder(root=os.path.join(config['dataset']['root'], 'tiny-imagenet-200/val'), transform=test_transform)
    # print(dst_test.targets)
    config['training_opt']['test_batch_size'] = config['training_opt']['batch_size'] if 'test_batch_size' not in config['training_opt'] else config['training_opt']['test_batch_size']
    test_loader = torch.utils.data.DataLoader(
        wrapped_dataset(dst_test), batch_size = config['training_opt']['test_batch_size'],
        shuffle=False, num_workers = config['training_opt']['num_data_workers'], pin_memory=True, drop_last=False
    )

    return {
        'num_classes': num_classes,
        'train_dset': wrapped_dataset(dst_train),
        'test_loader': test_loader,
        'num_train_samples': len(dst_train)
    }


    