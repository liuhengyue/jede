#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/baseline/test_0.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/R101-FPN_test_0_gn.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_attn_gn.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_gn_randcrop.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_copypastemix.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_pretrain.yaml
#python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_0_baseline.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_randcrop.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_bn.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_ltrb.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_ltrb_gn_pretrain_copypastemix.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_kptsfeatureonly.yaml --resume
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_attn.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_attn_28x28_cat.yaml --resume
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_person_kpts_28x28_cat.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/digit_twochannels/test_0.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/digit_twochannels/test_0_gn.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/digit_twochannels/test_0_gn_pe.yaml
python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/digit_twochannels/test_0_gn_attn_parallel.yaml
python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml