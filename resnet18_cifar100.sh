python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp baseline --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_baseline

python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp vs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_vs_0.001       --r 0.001
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp vq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_vq_1bit       --bit 1
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp vlr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_vlr_0.01       --r 0.01

# EMA

python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_cs_0.001_true_avg  --r 0.001 --positive 1 --estimate_type avg --beta 0.1
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_cs_0.001_false_avg --r 0.001 --positive 0 --estimate_type avg --beta 0.1

python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_cq_1bit_true_avg  --bit 1 --positive 1 --estimate_type avg --beta 0.1
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_cq_1bit_false_avg --bit 1 --positive 0 --estimate_type avg --beta 0.1

python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_clr_0.001_true_avg  --r 0.001 --positive 1 --estimate_type avg --beta 0.1
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_clr_0.001_false_avg --r 0.001 --positive 0 --estimate_type avg --beta 0.1

# SSM

python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_cs_0.001_true  --r 0.001 --positive 1
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_cs_0.001_false --r 0.001 --positive 0

python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_cq_1bit_true  --bit 1 --positive 1
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_cq_1bit_false --bit 1 --positive 0

python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_clr_0.001_true  --r 0.001 --positive 1
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_clr_0.001_false --r 0.001 --positive 0
