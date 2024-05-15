from torchvision import datasets, transforms
import torch
import os

class wrapped_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, noise_file = None):
        self.dataset = dataset
        # self.targets = dataset.targets
        self.noisy_labels = torch.load(noise_file)['noisy_label'] if noise_file is not None else None
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return {
            'input': self.dataset[index][0],
            'clean_label': self.dataset[index][1],
            'target': self.noisy_labels[index],
            'index': index
        } if self.noisy_labels is not None else {
            'input': self.dataset[index][0],
            'target': self.dataset[index][1],
            'index': index
        }

def CIFAR10N(config, logger):
    im_size = (32, 32) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    num_classes = 10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 32 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )

    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 32 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )
    noise_file = os.path.join(config['dataset']['root'], 'CIFAR_N/CIFAR-10_human.pt')
    dst_train = datasets.CIFAR10(
        config['dataset']['root'], train=True, download=True, transform= transform
    )
    dst_test = datasets.CIFAR10(config['dataset']['root'], train=False, download=True, transform = test_transform)
    config['training_opt']['test_batch_size'] = config['training_opt']['batch_size'] if 'test_batch_size' not in config['training_opt'] else config['training_opt']['test_batch_size']
    test_loader = torch.utils.data.DataLoader(
        wrapped_dataset(dst_test), batch_size = config['training_opt']['test_batch_size'],
        shuffle=False, num_workers = config['training_opt']['num_data_workers'], pin_memory=True, drop_last=False
    )

    return {
        'num_classes': num_classes,
        'train_dset': wrapped_dataset(dst_train, noise_file),
        'test_loader': test_loader,
        'num_train_samples': len(dst_train)
    }

def CIFAR100N(config, logger):
    im_size = (32, 32) if 'im_size' not in config['dataset'] else config['dataset']['im_size']
    print(f'Image size: {im_size}')
    num_classes = 100
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    transform = transforms.Compose(
        [transforms.RandomCrop(im_size, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        ) if im_size[0] == 32 else transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )
    
    test_transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]) if im_size[0] == 32 else transforms.Compose(
        [transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
        )
    noise_file = os.path.join(config['dataset']['root'], 'CIFAR_N/CIFAR-100_human.pt')
    dst_train = datasets.CIFAR100(
        config['dataset']['root'], train=True, download=True, transform= transform
    )
    dst_test = datasets.CIFAR100(config['dataset']['root'], train=False, download=True, transform = test_transform)
    # class_names = dst_train.classes
    dst_train.targets = torch.tensor(dst_train.targets, dtype=torch.long)
    dst_test.targets = torch.tensor(dst_test.targets, dtype=torch.long)
    config['training_opt']['test_batch_size'] = config['training_opt']['batch_size'] if 'test_batch_size' not in config['training_opt'] else config['training_opt']['test_batch_size']
    test_loader = torch.utils.data.DataLoader(
        wrapped_dataset(dst_test), batch_size = config['training_opt']['test_batch_size'],
        shuffle=False, num_workers = config['training_opt']['num_data_workers'], pin_memory=True, drop_last=False
    )

    return {
        'num_classes': num_classes,
        'train_dset': wrapped_dataset(dst_train, noise_file),
        'test_loader': test_loader,
        'num_train_samples': len(dst_train)
    }
