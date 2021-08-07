#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/baseline/test_0.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_gn.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_attn_gn.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_gn_randcrop.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_copypastemix.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_pretrain.yaml
#python train_net.py --num-gpus 2 --config-file configs/faster_rcnn/test_0_baseline.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_randcrop.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_bn.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_ltrb.yaml
#python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_ltrb_gn_pretrain_copypastemix.yaml
python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_kptsfeatureonly.yaml --resume
python train_net.py --num-gpus 2 --config-file configs/pg_rcnn/ablations/test_0_attn.yaml