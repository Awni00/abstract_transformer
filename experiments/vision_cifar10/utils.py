import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from datetime import datetime

from autoaugment import CIFAR10Policy, SVHNPolicy
from da import RandomCropPaste

# FIXME
# def get_model(args):
#     if args.model_name == 'vit':
#         from vit import ViT
#         net = ViT(
#             args.in_c,
#             args.num_classes,
#             img_size=args.size,
#             patch=args.patch,
#             dropout=args.dropout,
#             mlp_hidden=args.mlp_hidden,
#             num_layers=args.num_layers,
#             hidden=args.hidden,
#             head=args.head,
#             is_cls_token=args.is_cls_token
#             )

#     elif args.model_name == 'my-vit':
#         import sys; sys.path.append('../..')
#         from vision_models import VisionTransformer
#         img_shape = (args.in_c, args.size, args.size)
#         patch_size = args.size // args.patch
#         patch_size = (patch_size, patch_size)
#         pool = 'cls' if args.is_cls_token else 'mean'
#         net = VisionTransformer(
#             image_shape=img_shape,
#             patch_size=patch_size,
#             num_classes=args.num_classes,
#             d_model=args.hidden,
#             n_layers=args.num_layers,
#             n_heads=args.head,
#             dff=args.mlp_hidden,
#             activation='gelu',
#             norm_first=True,
#             bias=True,
#             dropout_rate=args.dropout,
#             pool=pool
#             )
#     else:
#         raise NotImplementedError(f"{args.model_name} is not implemented yet...")

#     return net

def get_model(args):
    if args.model_name == 'vit':
        from vit import ViT

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
    train_transform += [
        transforms.RandomCrop(size=args.size, padding=args.padding)
    ]
    if args.dataset != 'svhn':
        train_transform += [transforms.RandomHorizontalFlip()]

    if args.autoaugment:
        if args.dataset == 'cifar10' or args.dataset=='cifar100':
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

    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std)
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform


def get_dataset(args):
    root = "data"
    if args.dataset == "cifar10":
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
    experiment_name = f"{args.model_name}_{args.dataset}-{datetimestr}"
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
    if args.pool == 'mean':
        experiment_name+="_gap"
    print(f"Experiment:{experiment_name}")
    return experiment_name