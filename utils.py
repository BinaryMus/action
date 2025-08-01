import os
import random
import torch
import numpy as np
from data import cifar10, cifar100, tiny_imagenet, yahoo, agnews, emotion, trec
from transformers import get_scheduler
from models import vgg16, resnet18, vit, bert, vgg11, tibert
from copy import deepcopy

def seed_it(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def vgg11_cifar10(num_client, num_classes, epoch, device, bs, alpha):
    train_loaders, validate_loader = cifar10(num_client=num_client, bs=bs, alpha=alpha)
    global_model = vgg11(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) for i in client_models]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers]
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers


def vgg16_cifar10(num_client, num_classes, epoch, device, bs, alpha):
    train_loaders, validate_loader = cifar10(num_client=num_client, bs=bs, alpha=alpha)
    global_model = vgg16(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) for i in client_models]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers]
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers

def resnet18_cifar100(num_client, num_classes, epoch, device, bs, alpha):
    train_loaders, validate_loader = cifar100(num_client=num_client, bs=bs, alpha=alpha)
    global_model = resnet18(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) for i in client_models]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers]
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers

def vit_tinyin(num_client, num_classes, epoch, device, bs, alpha):
    train_loaders, validate_loader = tiny_imagenet(num_client=num_client, bs=bs, alpha=alpha)
    global_model = vit(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) for i in client_models]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers]
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers

def bert_yahoo(num_client, num_classes, epoch, device, bs, alpha):
    train_loaders, validate_loader = yahoo(num_client=num_client, bs=bs, alpha=alpha)
    global_model = bert(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4) for i in client_models]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers]
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers

def bert_agnews(num_client, num_classes, epoch, device, bs, alpha):
    train_loaders, validate_loader = agnews(num_client=num_client, bs=bs, alpha=alpha)
    global_model = bert(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.SGD(i.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4) for i in client_models]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(i, T_max=epoch) for i in optimizers]
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers

def bert_emotion(num_client, num_classes, epoch, device, bs, alpha):
    # def lr_lambda(current_step: int):
    #     return max(0.0, 1.0 - current_step / epoch)
    train_loaders, validate_loader = emotion(num_client=num_client, bs=bs, alpha=alpha)
    global_model = bert(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.AdamW(i.parameters(), lr=2e-5, weight_decay=0.01) for i in client_models]
    # schedulers = [torch.optim.lr_scheduler.LambdaLR(i, lr_lambda) for i in optimizers]
    schedulers = [get_scheduler("linear", optimizer=i, num_warmup_steps=0, num_training_steps=epoch) for i in optimizers]
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers

def tibert_emotion(num_client, num_classes, epoch, device, bs, alpha):
    # def lr_lambda(current_step: int):
    #     return max(0.0, 1.0 - current_step / epoch)
    train_loaders, validate_loader = emotion(num_client=num_client, bs=bs, alpha=alpha)
    global_model = tibert(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.AdamW(i.parameters(), lr=2e-5, weight_decay=0.01) for i in client_models]
    # schedulers = [torch.optim.lr_scheduler.LambdaLR(i, lr_lambda) for i in optimizers]
    schedulers = [get_scheduler("linear", optimizer=i, num_warmup_steps=0, num_training_steps=epoch) for i in optimizers]
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers

def tibert_trec(num_client, num_classes, epoch, device, bs, alpha):
    # def lr_lambda(current_step: int):
    #     return max(0.0, 1.0 - current_step / epoch)
    train_loaders, validate_loader = trec(num_client=num_client, bs=bs, alpha=alpha)
    global_model = tibert(num_classes=num_classes).to(device)
    client_models = [deepcopy(global_model).to(device) for _ in range(num_client)]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(num_client)]
    optimizers = [torch.optim.AdamW(i.parameters(), lr=1e-3, weight_decay=0.01) for i in client_models]
    # schedulers = [torch.optim.lr_scheduler.LambdaLR(i, lr_lambda) for i in optimizers]
    schedulers = [get_scheduler("linear", optimizer=i, num_warmup_steps=0, num_training_steps=epoch) for i in optimizers]
    return train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers
