python main.py --benchmark bert_emotion --num_classes 6 --comp baseline --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output bert_baseline --nlp 1

python main.py --benchmark bert_emotion --num_classes 6 --comp vs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output bert_vs_0.001       --r 0.001  --nlp 1
python main.py --benchmark bert_emotion --num_classes 6 --comp vq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output bert_vq_1bit        --bit 1    --nlp 1
python main.py --benchmark bert_emotion --num_classes 6 --comp vlr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output bert_vlr_0.001      --r 0.001 --nlp 1

