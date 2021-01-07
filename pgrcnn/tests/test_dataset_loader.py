import cv2
from detectron2.engine import default_argument_parser
from pgrcnn.data.custom_mapper import CustomDatasetMapper
from pgrcnn.data.build import build_detection_train_loader
from pgrcnn.utils.custom_visualizer import JerseyNumberVisualizer
from detectron2.data import MetadataCatalog
from pgrcnn.utils.launch_utils import setup

def visualize_training(batched_inputs, cfg):
    """
    A function used to visualize images and gts after any data augmentation
    used, the inputs here are the actual data fed into the model, so most of
    the fields are tensors.

    Modified from func visualize_training().

    Args:
        batched_inputs (list): a list that contains input to the model.

    """

    jnw_metadata = MetadataCatalog.get("jerseynumbers_train")
    for input in batched_inputs:
        img = input["image"].cpu().numpy()
        assert img.shape[0] == 3, "Images should have 3 channels."
        if cfg.INPUT.FORMAT == "RGB":
            img = img[::-1, :, :]
        img = img.transpose(1, 2, 0)
        v_gt = JerseyNumberVisualizer(img, metadata=jnw_metadata)
        # v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
        v_gt = v_gt.draw_dataloader_instances(input)
        vis_img = v_gt.get_image()
        # vis_img = vis_img.transpose(2, 0, 1)
        vis_name = " 1. GT bounding boxes"
        cv2.imshow(vis_name, vis_img)
        cv2.waitKey()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # lazy add config file
    args.config_file = "../../configs/pg_rcnn/pg_rcnn_R_50_FPN_1x_extend_aug_test_0.yaml"
    # args.config_file = "../../configs/faster_rcnn_R_50_FPN_3x.yaml"
    cfg = setup(args)
    dataloader = build_detection_train_loader(cfg, mapper=CustomDatasetMapper(cfg, True))
    # data = next(iter(dataloader))
    # print(data)
    # visualize_training(data, cfg)

    for data in dataloader:
        # print(data)
        visualize_training(data, cfg)