import logging
import cv2
from detectron2.engine import default_argument_parser
from pgrcnn.data.mapper import JerseyNumberDatasetMapper
from pgrcnn.data.build import build_detection_train_loader
from pgrcnn.utils.custom_visualizer import JerseyNumberVisualizer
from detectron2.data import MetadataCatalog
from pgrcnn.utils.launch_utils import setup
logger = logging.getLogger("pgrcnn")


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
    assert len(batched_inputs) == 1, "visualize_training() needs batch size of 1"
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
        return input['file_name'], vis_img

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # lazy add config file
    args.config_file = "configs/pg_rcnn/pg_rcnn_test.yaml"
    # args.config_file = "../../configs/faster_rcnn_R_50_FPN_3x.yaml"
    cfg = setup(args)
    dataloader = build_detection_train_loader(cfg, mapper=JerseyNumberDatasetMapper(cfg, True))

    show_data = False
    for data in dataloader:
        try:
            logger.info(f"{data[0]['file_name']}")
            if show_data:
                file_name, vis_img = visualize_training(data, cfg)
                cv2.imshow(file_name, vis_img)
                k = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if k == 27:
                    break
        except:
            raise Exception(f"Error when processing {data[0]['file_name']}")