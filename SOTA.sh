python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp vqq      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output vgg_vqq_2bit  --bit 2
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp eqq      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_eqq_2bit  --bit 2

python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp vqq      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_vqq_2bit  --bit 2
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp eqq      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output resnet_eqq_2bit  --bit 2

python main.py --benchmark bert_emotion --num_classes 6 --comp vqq      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output bert_vqq_2bit  --bit 2 --nlp 1 --bs 128
python main.py --benchmark bert_emotion --num_classes 6 --comp eqq      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output bert_eqq_2bit  --bit 2 --nlp 1 --bs 128

##

python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp vtq      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output vgg_vtq_2bit  --bit 2
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp etq      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_etq_2bit  --bit 2

python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp vtq      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_vtq_2bit  --bit 2
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp etq      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output resnet_etq_2bit  --bit 2

python main.py --benchmark bert_emotion --num_classes 6 --comp vtq      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output bert_vtq_2bit  --bit 2 --nlp 1 --bs 128
python main.py --benchmark bert_emotion --num_classes 6 --comp etq      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output bert_etq_2bit  --bit 2 --nlp 1 --bs 128

##

python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp vtc      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output vgg_vtc_2bit_0.1  --bit 2 --r 0.1
python main.py --benchmark vgg16_cifar10 --num_classes 10 --comp etc      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vgg_etc_2bit_0.1  --bit 2 --r 0.1

python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp vtc      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output resnet_vtc_2bit_0.1  --bit 2 --r 0.1
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp etc      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output resnet_etc_2bit_0.1  --bit 2 --r 0.1

python main.py --benchmark bert_emotion --num_classes 6 --comp vtc      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output bert_vtc_2bit_0.1  --bit 2 --r 0.1 --nlp 1 --bs 128
python main.py --benchmark bert_emotion --num_classes 6 --comp etc      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output bert_etc_2bit_0.1  --bit 2 --r 0.1 --nlp 1 --bs 128
