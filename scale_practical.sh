# IID ResNet
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp vqq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:0 --output 100client_resnet_vqq_4bit_IID       --bit 4
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp eqq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:0 --output 100client_resnet_eqq_4bit_IID       --bit 4

python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp vtq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:0 --output 100client_resnet_vtq_4bit_IID       --bit 4
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp etq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:0 --output 100client_resnet_etq_4bit_IID       --bit 4
# NonIID ResNet
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp vqq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:0 --output 100client_resnet_vqq_4bit_NONIID       --bit 4 --alpha 0.1
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp eqq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:0 --output 100client_resnet_eqq_4bit_NONIID       --bit 4 --alpha 0.1

python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp vtq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:0 --output 100client_resnet_vtq_4bit_NONIID       --bit 4 --alpha 0.1
python main.py --benchmark resnet18_cifar100 --num_classes 100 --comp etq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:0 --output 100client_resnet_etq_4bit_NONIID       --bit 4 --alpha 0.1

##############

# IID tibert
python main.py --benchmark tibert_trec --num_classes 6 --comp vqq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:0 --output 100client_tibert_vqq_4bit_IID       --bit 4 --nlp 1
python main.py --benchmark tibert_trec --num_classes 6 --comp eqq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:1 --output 100client_tibert_eqq_4bit_IID       --bit 4 --nlp 1

python main.py --benchmark tibert_trec --num_classes 6 --comp vtq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:0 --output 100client_tibert_vtq_4bit_IID       --bit 4 --nlp 1
python main.py --benchmark tibert_trec --num_classes 6 --comp etq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:0 --output 100client_tibert_etq_4bit_IID       --bit 4 --nlp 1
# NonIID tibert
python main.py --benchmark tibert_trec --num_classes 6 --comp vqq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:0 --output 100client_tibert_vqq_4bit_NONIID       --bit 4 --alpha 0.1 --nlp 1
python main.py --benchmark tibert_trec --num_classes 6 --comp eqq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:1 --output 100client_tibert_eqq_4bit_NONIID       --bit 4 --alpha 0.1 --nlp 1

python main.py --benchmark tibert_trec --num_classes 6 --comp vtq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:0 --output 100client_tibert_vtq_4bit_NONIID       --bit 4 --alpha 0.1 --nlp 1
python main.py --benchmark tibert_trec --num_classes 6 --comp etq       --num_client 100 --global_epoch 100 --local_epoch 1 --device cuda:1 --output 100client_tibert_etq_4bit_NONIID       --bit 4 --alpha 0.1 --nlp 1