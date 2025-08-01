python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp baseline --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_baseline

python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp vs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_vs_0.001       --r 0.001
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp vq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output vgg_vq_1bit        --bit 1 --lower_percentile 3 --upper_percentile 97
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp vlr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_vlr_0.001       --r 0.001

# EMA

python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_cs_0.001_true_avg  --r 0.001 --positive 1 --estimate_type avg --beta 0.1
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_cs_0.001_false_avg --r 0.001 --positive 0 --estimate_type avg --beta 0.1

python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_cq_1bit_true_avg   --bit 1 --lower_percentile 3 --upper_percentile 97 --positive 1 --estimate_type avg --beta 0.1
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_cq_1bit_false_avg  --bit 1 --lower_percentile 3 --upper_percentile 97 --positive 0 --estimate_type avg --beta 0.1

python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_clr_0.001_true_avg  --r 0.001 --positive 1 --estimate_type avg --beta 0.1
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_clr_0.001_false_avg --r 0.001 --positive 0 --estimate_type avg --beta 0.1

# SSM

python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_cs_0.001_true  --r 0.001 --positive 1
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_cs_0.001_false --r 0.001 --positive 0

python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_cq_1bit_true   --bit 1 --lower_percentile 3 --upper_percentile 97 --positive 1
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:2 --output vgg_cq_1bit_false  --bit 1 --lower_percentile 3 --upper_percentile 97 --positive 0

python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_clr_0.01_true  --r 0.01 --positive 1
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_clr_0.01_false --r 0.01 --positive 0

