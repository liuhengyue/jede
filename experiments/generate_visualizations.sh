# JEDE baseline
python visualize_json_results.py --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml DATASETS.TRAIN_VIDEO_IDS [1,2,3] DATASETS.TEST_VIDEO_IDS [0] OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_0"

python visualize_json_results.py --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml DATASETS.TRAIN_VIDEO_IDS [0,2,3] DATASETS.TEST_VIDEO_IDS [1] OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_1"

python visualize_json_results.py --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,3] DATASETS.TEST_VIDEO_IDS [2] OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_2"

python visualize_json_results.py --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,2] DATASETS.TEST_VIDEO_IDS [3] OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_3"

python visualize_json_results.py --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] DATASETS.TEST_VIDEO_IDS [4] OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_4"

#python visualize_json_results.py --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn.yaml DATASETS.TRAIN_VIDEO_IDS [4] DATASETS.TEST_VIDEO_IDS [0,1,2,3] INPUT.AUG.COPY_PASTE_MIX 0 OUTPUT_DIR "./output/jede_R_50_FPN_baseline/test_5" INPUT.MAX_SIZE_TEST 400 INPUT.MIN_SIZE_TEST 240

# JEDE augmented

python visualize_json_results.py --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml

python visualize_json_results.py --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml DATASETS.TRAIN_VIDEO_IDS [0,2,3] DATASETS.TEST_VIDEO_IDS [1] OUTPUT_DIR "./output/jede_R_50_FPN/test_1"

python visualize_json_results.py --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,3] DATASETS.TEST_VIDEO_IDS [2] OUTPUT_DIR "./output/jede_R_50_FPN/test_2"

python visualize_json_results.py --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,2] DATASETS.TEST_VIDEO_IDS [3] OUTPUT_DIR "./output/jede_R_50_FPN/test_3"

python visualize_json_results.py --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] DATASETS.TEST_VIDEO_IDS [4] OUTPUT_DIR "./output/jede_R_50_FPN/test_4"

#python visualize_json_results.py --config-file configs/pg_rcnn/digit_twochannels/test_0_parallel_gn_pe_pretrain_copypastemix_swapdigit_less_anchors_unfreeze_b8.yaml DATASETS.TRAIN_VIDEO_IDS [4] DATASETS.TEST_VIDEO_IDS [0,1,2,3] INPUT.AUG.COPY_PASTE_MIX 0 INPUT.MAX_SIZE_TEST 400 INPUT.MIN_SIZE_TEST 240 OUTPUT_DIR "./output/jede_R_50_FPN/test_5"

# faster rcnn
#python visualize_json_results.py --config-file configs/faster_rcnn/test_0.yaml
#python visualize_json_results.py --config-file configs/faster_rcnn/test_1.yaml
#python visualize_json_results.py --config-file configs/faster_rcnn/test_2.yaml
#python visualize_json_results.py --config-file configs/faster_rcnn/test_3.yaml
#python visualize_json_results.py --config-file configs/faster_rcnn/test_4.yaml
#python visualize_json_results.py --config-file configs/faster_rcnn/test_5.yaml INPUT.MAX_SIZE_TEST 400 INPUT.MIN_SIZE_TEST 240


# cascade rcnn

#python visualize_json_results.py --config-file configs/cascade_rcnn/cascade_rcnn_base.yaml DATASETS.TRAIN_VIDEO_IDS [1,2,3] DATASETS.TEST_VIDEO_IDS [0] OUTPUT_DIR "./output/cascade_rcnn/test_0"
#
#python visualize_json_results.py --config-file configs/cascade_rcnn/cascade_rcnn_base.yaml DATASETS.TRAIN_VIDEO_IDS [0,2,3] DATASETS.TEST_VIDEO_IDS [1] OUTPUT_DIR "./output/cascade_rcnn/test_1"
#
#python visualize_json_results.py --config-file configs/cascade_rcnn/cascade_rcnn_base.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,3] DATASETS.TEST_VIDEO_IDS [2] OUTPUT_DIR "./output/cascade_rcnn/test_2"
#
#python visualize_json_results.py --config-file configs/cascade_rcnn/cascade_rcnn_base.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,2] DATASETS.TEST_VIDEO_IDS [3] OUTPUT_DIR "./output/cascade_rcnn/test_3"
#
#python visualize_json_results.py --config-file configs/cascade_rcnn/cascade_rcnn_base.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] DATASETS.TEST_VIDEO_IDS [4] OUTPUT_DIR "./output/cascade_rcnn/test_4"

#python visualize_json_results.py --config-file configs/cascade_rcnn/cascade_rcnn_base.yaml DATASETS.TRAIN_VIDEO_IDS [4] DATASETS.TEST_VIDEO_IDS [0,1,2,3] OUTPUT_DIR "./output/cascade_rcnn/test_5" INPUT.MAX_SIZE_TEST 400 INPUT.MIN_SIZE_TEST 240

# tridentnet
#
#python visualize_json_results.py --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml DATASETS.TRAIN_VIDEO_IDS [1,2,3] DATASETS.TEST_VIDEO_IDS [0] OUTPUT_DIR "./output/trident_net/test_0"
#
#python visualize_json_results.py --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml DATASETS.TRAIN_VIDEO_IDS [0,2,3] DATASETS.TEST_VIDEO_IDS [1] OUTPUT_DIR "./output/trident_net/test_1"
#
#python visualize_json_results.py --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,3] DATASETS.TEST_VIDEO_IDS [2] OUTPUT_DIR "./output/trident_net/test_2"
#
#python visualize_json_results.py --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,2] DATASETS.TEST_VIDEO_IDS [3] OUTPUT_DIR "./output/trident_net/test_3"
#
#python visualize_json_results.py --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml DATASETS.TRAIN_VIDEO_IDS [0,1,2,3] DATASETS.TEST_VIDEO_IDS [4] OUTPUT_DIR "./output/trident_net/test_4"

#python visualize_json_results.py --config-file configs/tridentnet/tridentnet_fast_R_50_C4_1x_test_0.yaml DATASETS.TRAIN_VIDEO_IDS [4] DATASETS.TEST_VIDEO_IDS [0,1,2,3] OUTPUT_DIR "./output/trident_net/test_5" INPUT.MAX_SIZE_TEST 400 INPUT.MIN_SIZE_TEST 240