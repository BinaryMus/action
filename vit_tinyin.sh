python main.py --benchmark vit_tinyin --num_classes 200 --comp baseline --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vit_baseline

python main.py --benchmark vit_tinyin --num_classes 200 --comp vs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vit_vs_0.01       --r 0.01
python main.py --benchmark vit_tinyin --num_classes 200 --comp vq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vit_vq_2bit       --bit 2
python main.py --benchmark vit_tinyin --num_classes 200 --comp vlr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output vit_vlr_0.1       --r 0.1

# EMA

python main.py --benchmark vit_tinyin --num_classes 200 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vit_cs_0.01_true_avg  --r 0.01 --positive 1 --estimate_type avg --beta 0.1
python main.py --benchmark vit_tinyin --num_classes 200 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vit_cs_0.01_false_avg --r 0.01 --positive 0 --estimate_type avg --beta 0.1

python main.py --benchmark vit_tinyin --num_classes 200 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output vit_cq_2bit_true_avg  --bit 2 --positive 1 --estimate_type avg --beta 0.1
python main.py --benchmark vit_tinyin --num_classes 200 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output vit_cq_2bit_false_avg --bit 2 --positive 0 --estimate_type avg --beta 0.1

python main.py --benchmark vit_tinyin --num_classes 200 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vit_clr_0.1_true_avg  --r 0.1 --positive 1 --estimate_type avg --beta 0.1
python main.py --benchmark vit_tinyin --num_classes 200 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output vit_clr_0.1_false_avg --r 0.1 --positive 0 --estimate_type avg --beta 0.1

# SSM

python main.py --benchmark vit_tinyin --num_classes 200 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vit_cs_0.01_true  --r 0.01 --positive 1
python main.py --benchmark vit_tinyin --num_classes 200 --comp cs       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vit_cs_0.01_false --r 0.01 --positive 0

python main.py --benchmark vit_tinyin --num_classes 200 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output vit_cq_2bit_true  --bit 2 --positive 1
python main.py --benchmark vit_tinyin --num_classes 200 --comp cq       --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output vit_cq_2bit_false --bit 2 --positive 0

python main.py --benchmark vit_tinyin --num_classes 200 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:1 --output vit_clr_0.1_true  --r 0.1 --positive 1
python main.py --benchmark vit_tinyin --num_classes 200 --comp clr      --num_client 10 --global_epoch 100 --local_epoch 1 --device cuda:0 --output vit_clr_0.1_false --r 0.1 --positive 0
