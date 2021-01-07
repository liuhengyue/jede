cd ..
# train, cross validation on four videos

python train_net.py \
        --config-file configs/pg_rcnn/pg_rcnn_R_50_FPN_1x_ablation_num_proposals_10_test_0.yaml \
        --num-gpus 2 \
        --resume

python train_net.py \
        --config-file configs/pg_rcnn/pg_rcnn_R_50_FPN_1x_ablation_num_proposals_100_test_0.yaml \
        --num-gpus 2

