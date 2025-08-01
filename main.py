import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import numpy as np

from core import FederatedLearning
from compression import Baseline
from compression import VanillaQuantization, VanillaSparsification, VanillaLowRank
from compression import ConstraintQuantization, ConstraintSparsification, ConstraintLowRank

from compression import QuantileQuantization, EnhancedQuantileQuantization
from compression import RandTopkSparsification
from compression import THCQuantization, EnhancedTHCQuantization
from compression import FedTC, EnhancedFedTC

from utils import seed_it, vgg11_cifar10, vgg16_cifar10, resnet18_cifar100, vit_tinyin, bert_yahoo, bert_agnews, bert_emotion, tibert_emotion, tibert_trec

seed_it(42)

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, choices=['tibert_trec', 'tibert_emotion', 'vgg11_cifar10', 'vgg16_cifar10', 'resnet18_cifar100', 'vit_tinyin', 'bert_yahoo', 'bert_agnews', 'bert_emotion'], default='vgg16_cifar10')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--bs', type=int, default=256)
parser.add_argument('--nlp', type=int, default=0)
parser.add_argument('--comp', type=str, choices=['baseline', 'vs', 'cs', 'vq', 'cq', 'vlr', 'clr', 'vqq', 'eqq', 'rt','vtq','etq','vtc','etc'])

parser.add_argument('--num_client', type=int, default=10)
parser.add_argument('--alpha', type=float, default=5)
parser.add_argument('--global_epoch', type=int, default=100)
parser.add_argument('--local_epoch', type=int, default=1)
parser.add_argument('--estimate_type', type=str, choices=['last', 'avg'], default='last')
parser.add_argument('--beta', type=float, default=0.1)

parser.add_argument('--device', default='cuda:0')

parser.add_argument('--r', type=float, default=0.1)
parser.add_argument('--positive', type=int, default=1)
parser.add_argument('--bit', type=int, default=2)
parser.add_argument('--tol', type=float, default=0.45)
parser.add_argument('--sample', type=int, default=1)
parser.add_argument('--sample_size', type=int, default=10000)
parser.add_argument('--lower_percentile', type=float, default=1)
parser.add_argument('--upper_percentile', type=float, default=99)
parser.add_argument('--offset', default=0.99)


parser.add_argument('--output', type=str)

arg = parser.parse_args()

train_loaders, validate_loader, global_model, client_models, criterions, optimizers, schedulers = {
    'vgg16_cifar10': vgg16_cifar10,
    'vgg11_cifar10': vgg11_cifar10,
    'resnet18_cifar100': resnet18_cifar100,
    'vit_tinyin': vit_tinyin,
    'tibert_emotion': tibert_emotion,
    'bert_yahoo': bert_yahoo,
    'bert_agnews': bert_agnews,
    'bert_emotion': bert_emotion,
    'tibert_trec': tibert_trec
}[arg.benchmark](num_client=arg.num_client, num_classes=arg.num_classes, epoch=arg.global_epoch * arg.local_epoch, device=arg.device, bs=arg.bs, alpha=arg.alpha)

comp = {
    'baseline': Baseline,
    'vs': VanillaSparsification,
    'cs': ConstraintSparsification,
    'vq': VanillaQuantization,
    'cq': ConstraintQuantization,
    'vlr': VanillaLowRank,
    'clr': ConstraintLowRank,
    'vqq': QuantileQuantization,
    'eqq': EnhancedQuantileQuantization,
    'rt': RandTopkSparsification,
    'vtq': THCQuantization,
    'etq': EnhancedTHCQuantization,
    'vtc': FedTC,
    'etc': EnhancedFedTC
}[arg.comp](ratio=arg.r, positive=arg.positive, bit=arg.bit, lower_percentile=arg.lower_percentile, upper_percentile=arg.upper_percentile, tol=arg.tol, offset=arg.offset, sample=arg.sample, sample_size=arg.sample_size)

fl = FederatedLearning(
    global_model=global_model,
    client_models=client_models,
    criterions=criterions,
    optimizers=optimizers,
    schedulers=schedulers,
    dataloaders=train_loaders,
    valloader=validate_loader,
    device=arg.device,
    comp=comp,
    global_epoch=arg.global_epoch,
    local_epoch=arg.local_epoch,
    beta=arg.beta,
    estimate_type=arg.estimate_type,
    nlp=arg.nlp
)
loss_lst, acc1_lst, acc5_lst = fl.run()

np.save(f'results/{arg.output}_loss.npy', np.array(loss_lst))
np.save(f'results/{arg.output}_acc1.npy', np.array(acc1_lst))
np.save(f'results/{arg.output}_acc5.npy', np.array(acc5_lst))
