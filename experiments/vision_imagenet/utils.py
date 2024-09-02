import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from datetime import datetime

from autoaugment import CIFAR10Policy, SVHNPolicy, ImageNetPolicy
from da import RandomCropPaste


def get_model(args):
    if args.model_name == 'vit':
        import sys; sys.path.append('../..')
        from vision_models import VisionTransformer
        net = VisionTransformer(
            image_shape=args.image_shape,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            d_model=args.d_model,
            dff=args.dff,
            n_layers=args.n_layers,
            n_heads=args.sa,
            activation=args.activation,
            norm_first=args.norm_first,
            bias=args.bias,
            dropout_rate=args.dropout_rate,
            pool=args.pool
            )

    elif args.model_name == 'vidat':
        import sys; sys.path.append('../..')
        from vision_models import VisionDualAttnTransformer
        net = VisionDualAttnTransformer(
            image_shape=args.image_shape,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            d_model=args.d_model,
            dff=args.dff,
            n_layers=args.n_layers,
            n_heads_sa=args.sa,
            n_heads_ra=args.ra,
            symbol_retrieval=args.symbol_type,
            symbol_retrieval_kwargs=args.symbol_retrieval_kwargs,
            ra_type=args.ra_type,
            ra_kwargs=args.ra_kwargs,
            sa_kwargs=args.sa_kwargs,
            activation=args.activation,
            norm_first=args.norm_first,
            bias=args.bias,
            dropout_rate=args.dropout_rate,
            pool=args.pool
            )
    else:
        raise NotImplementedError(f"{args.model_name} is not implemented yet...")

    return net

def get_transform(args):
    train_transform = []
    test_transform = []
    if args.dataset == 'imagenet':
        train_transform += [
            # need to resize because the images are not the same size
            transforms.RandomResizedCrop(size=args.size),
        ]
    else:
        train_transform += [
            transforms.RandomCrop(size=args.size, padding=args.padding)
        ]

    if args.dataset != 'svhn':
        train_transform += [transforms.RandomHorizontalFlip()]

    if args.autoaugment:
        if args.dataset == 'imagenet':
            train_transform.append(ImageNetPolicy())
        elif args.dataset == 'cifar10' or args.dataset=='cifar100':
            train_transform.append(CIFAR10Policy())
        elif args.dataset == 'svhn':
            train_transform.append(SVHNPolicy())
        else:
            print(f"No AutoAugment for {args.dataset}")

    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]
    if args.rcpaste:
        train_transform += [RandomCropPaste(size=args.size)]

    if args.dataset == 'imagenet':
        test_transform += [
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ]

    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform


def get_dataset(args):
    root = "data"
    if args.dataset == "imagenet":
        args.in_c = 3
        args.num_classes=1000
        args.size = 224
        args.padding = None # 28 NOTE

        from torch.utils.data import DataLoader
        from imagenet_data_utils import ImageNetKaggle

        args.mean = [0.485, 0.456, 0.406]
        args.std = [0.229, 0.224, 0.225]

        train_transform, val_transform = get_transform(args)

        root = '/home/ma2393/scratch/datasets/imagenet' # location of imagenet dataset
        train_ds = ImageNetKaggle(root, "train", train_transform)
        test_ds = ImageNetKaggle(root, "val", val_transform)


    elif args.dataset == "cifar10":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "cifar100":
        args.in_c = 3
        args.num_classes=100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "svhn":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.SVHN(root, split="train",transform=train_transform, download=True)
        test_ds = torchvision.datasets.SVHN(root, split="test", transform=test_transform, download=True)

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")

    return train_ds, test_ds

def get_experiment_name(args):
    datetimestr = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    experiment_name = f"{args.model_name}_{args.dataset}"
    if args.model_name == 'vidat':
        experiment_name += f"-sa={args.sa}-ra={args.ra}-nr={args.n_relations}-symrel={args.symmetric_rels}-symb={args.symbol_type}"
    if args.n_kv_heads is not None:
        experiment_name += f"-n_kv_heads={args.n_kv_heads}"
    if args.autoaugment:
        experiment_name+="_aa"
    if args.label_smoothing:
        experiment_name+="_ls"
    if args.rcpaste:
        experiment_name+="_rc"
    if args.cutmix:
        experiment_name+="_cm"
    if args.mixup:
        experiment_name+="_mu"
    # if args.pool == 'mean':
        # experiment_name+="_gap"
    print(f"Experiment:{experiment_name}")

    experiment_name += f'-pool={args.pool}'

    run_name = f'{experiment_name} ({datetimestr})'
    return experiment_name, run_name