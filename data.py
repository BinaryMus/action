import torch
import random
import numpy as np

from collections import defaultdict
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def split_data(num_client, train_set, bs=256, num_workers=10, alpha=5):
    labels = np.array(train_set.targets if hasattr(train_set, 'targets') else [s[1] for s in train_set.samples])
    num_classes = np.max(labels) + 1

    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = defaultdict(list)

    for c in range(num_classes):
        idx = class_indices[c]
        proportions = np.random.dirichlet(alpha=np.ones(num_client) * alpha)
        proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        split_idx = np.split(idx, proportions)
        for i in range(num_client):
            client_indices[i].extend(split_idx[i])

    g = torch.Generator()
    g.manual_seed(42)
    client_loaders = []
    for i in range(num_client):
        indices = client_indices[i]
        if len(indices) == 0:
            indices = np.random.choice(len(train_set), size=10, replace=False)
        subset = Subset(train_set, indices)
        loader = DataLoader(subset, batch_size=bs, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
        client_loaders.append(loader)
    return client_loaders

def split_nlp_data(num_client, dataset, bs=32, num_workers=4, alpha=5):
    labels = np.array(dataset['label'])
    num_classes = np.max(labels) + 1

    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = defaultdict(list)

    for c in range(num_classes):
        idx = class_indices[c]
        proportions = np.random.dirichlet(np.ones(num_client) * alpha)
        proportions = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        split_idx = np.split(idx, proportions)
        for i in range(num_client):
            client_indices[i].extend(split_idx[i])

    g = torch.Generator()
    g.manual_seed(42)
    client_loaders = []
    for i in range(num_client):
        indices = client_indices[i]
        if len(indices) == 0:
            indices = np.random.choice(len(dataset), size=10, replace=False)
        subset = dataset.select(indices).with_format("torch")
        loader = DataLoader(subset, batch_size=bs, shuffle=True, collate_fn=default_data_collator,
                            num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
        client_loaders.append(loader)

    return client_loaders

def cifar10(num_client, root='/dataset/cifar10', bs=256, alpha=5):
    transform_train = Compose([ 
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = CIFAR10(root=root, train=True, transform=transform_train, download=True)
    validate_set = CIFAR10(root=root, train=False, transform=transform_test)
    train_loaders = split_data(num_client, train_set, bs, alpha=alpha)
    validate_loader = DataLoader(validate_set, bs, num_workers=20, shuffle=False)
    return train_loaders, validate_loader

def cifar100(num_client, root='/dataset/cifar100', bs=256, alpha=5):
    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = Compose([
        ToTensor(),
        Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
     ])
    train_set = CIFAR100(root=root, train=True, transform=transform_train, download=True)
    validate_set = CIFAR100(root=root, train=False, transform=transform_test)
    train_loaders = split_data(num_client, train_set, bs, alpha=alpha)
    validate_loader = DataLoader(validate_set, bs, shuffle=False)
    return train_loaders, validate_loader

def tiny_imagenet(num_client, root='/dataset/tiny-imagenet-200/', bs=256, alpha=5):
    transform_train = Compose([ 
        RandomCrop(64, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_val = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_set = ImageFolder(root+'train', transform=transform_train)
    validate_set = ImageFolder(root+'val', transform=transform_val)
    train_loaders = split_data(num_client, train_set, bs, num_workers=10, alpha=alpha)
    validate_loader = DataLoader(validate_set, bs, shuffle=False, num_workers=10)
    return train_loaders, validate_loader


def yahoo(num_client, bs=256, alpha=5):
    dataset = load_dataset("yahoo_answers_topics")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(example):
        return tokenizer(example["question_title"], example["question_content"],
                         truncation=True, padding="max_length", max_length=128)
    dataset = dataset.rename_column("topic", "label")

    train_set = dataset["train"].map(tokenize, batched=True, num_proc=8)
    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_set = dataset["test"].map(tokenize, batched=True, num_proc=8)
    test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loaders = split_nlp_data(num_client, train_set, bs=bs, alpha=alpha)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=10)
    print('load data finish!')
    return train_loaders, test_loader

def agnews(num_client, bs=256, alpha=5):
    dataset = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    train_set = dataset["train"].map(tokenize, batched=True, num_proc=8)
    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    test_set = dataset["test"].map(tokenize, batched=True, num_proc=8)
    test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loaders = split_nlp_data(num_client, train_set, bs=bs, alpha=alpha)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=10, collate_fn=default_data_collator)
    print('load data finish!')
    return train_loaders, test_loader

def emotion(num_client, bs=128, alpha=5):
    dataset = load_dataset("emotion")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    train_set = dataset["train"].map(tokenize, batched=True, num_proc=8)
    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    test_set = dataset["test"].map(tokenize, batched=True, num_proc=8)
    test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loaders = split_nlp_data(num_client, train_set, bs=bs, alpha=alpha)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=10, collate_fn=default_data_collator)
    print('load data finish!')
    return train_loaders, test_loader

def trec(num_client, bs=128, alpha=5):
    dataset = load_dataset("trec", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    dataset = dataset.rename_column("coarse_label", "label")

    train_set = dataset["train"].map(tokenize, batched=True, num_proc=8)
    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    test_set = dataset["test"].map(tokenize, batched=True, num_proc=8)
    test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loaders = split_nlp_data(num_client, train_set, bs=bs, alpha=alpha)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=10, collate_fn=default_data_collator)
    print('load data finish!')
    return train_loaders, test_loader