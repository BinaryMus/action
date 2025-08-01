python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_cs_0.001_true_avg_beta_0.2  --r 0.001 --positive 1 --estimate_type avg --beta 0.2
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_cs_0.001_true_avg_beta_0.3  --r 0.001 --positive 1 --estimate_type avg --beta 0.3
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_cs_0.001_true_avg_beta_0.4  --r 0.001 --positive 1 --estimate_type avg --beta 0.4

python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_cq_1bit_true_avg_beta_0.2   --bit 1 --lower_percentile 3 --upper_percentile 97 --positive 1 --estimate_type avg --beta 0.2
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_cq_1bit_true_avg_beta_0.3   --bit 1 --lower_percentile 3 --upper_percentile 97 --positive 1 --estimate_type avg --beta 0.3
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_cq_1bit_true_avg_beta_0.4   --bit 1 --lower_percentile 3 --upper_percentile 97 --positive 1 --estimate_type avg --beta 0.4

python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_clr_0.001_true_avg_beta_0.2  --r 0.001 --positive 1 --estimate_type avg --beta 0.2
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_clr_0.001_true_avg_beta_0.3  --r 0.001 --positive 1 --estimate_type avg --beta 0.3
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_clr_0.001_true_avg_beta_0.4  --r 0.001 --positive 1 --estimate_type avg --beta 0.4

python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_cs_0.001_true_avg_beta_0.2  --r 0.001 --positive 1 --estimate_type avg --beta 0.2
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_cs_0.001_true_avg_beta_0.3  --r 0.001 --positive 1 --estimate_type avg --beta 0.3
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_cs_0.001_true_avg_beta_0.4  --r 0.001 --positive 1 --estimate_type avg --beta 0.4

python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_cq_1bit_true_avg_beta_0.2  --bit 1 --positive 1 --estimate_type avg --beta 0.2
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_cq_1bit_true_avg_beta_0.3  --bit 1 --positive 1 --estimate_type avg --beta 0.3
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_cq_1bit_true_avg_beta_0.4  --bit 1 --positive 1 --estimate_type avg --beta 0.4

python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_clr_0.001_true_avg_beta_0.2  --r 0.001 --positive 1 --estimate_type avg --beta 0.2
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_clr_0.001_true_avg_beta_0.3  --r 0.001 --positive 1 --estimate_type avg --beta 0.3
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_clr_0.001_true_avg_beta_0.4  --r 0.001 --positive 1 --estimate_type avg --beta 0.4

python main.py --benchmark vit_tinyin --num_classes 200 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vit_cs_0.01_true_avg_beta_0.2  --r 0.01 --positive 1 --estimate_type avg --beta 0.2
python main.py --benchmark vit_tinyin --num_classes 200 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vit_cs_0.01_true_avg_beta_0.3  --r 0.01 --positive 1 --estimate_type avg --beta 0.3
python main.py --benchmark vit_tinyin --num_classes 200 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vit_cs_0.01_true_avg_beta_0.4  --r 0.01 --positive 1 --estimate_type avg --beta 0.4

python main.py --benchmark vit_tinyin --num_classes 200 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output vit_cq_2bit_true_avg_beta_0.2  --bit 2 --positive 1 --estimate_type avg --beta 0.2
python main.py --benchmark vit_tinyin --num_classes 200 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output vit_cq_2bit_true_avg_beta_0.3  --bit 2 --positive 1 --estimate_type avg --beta 0.3
python main.py --benchmark vit_tinyin --num_classes 200 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output vit_cq_2bit_true_avg_beta_0.4  --bit 2 --positive 1 --estimate_type avg --beta 0.4

python main.py --benchmark vit_tinyin --num_classes 200 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vit_clr_0.1_true_avg_beta_0.2  --r 0.1 --positive 1 --estimate_type avg --beta 0.2
python main.py --benchmark vit_tinyin --num_classes 200 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vit_clr_0.1_true_avg_beta_0.3  --r 0.1 --positive 1 --estimate_type avg --beta 0.3
python main.py --benchmark vit_tinyin --num_classes 200 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vit_clr_0.1_true_avg_beta_0.4  --r 0.1 --positive 1 --estimate_type avg --beta 0.4


