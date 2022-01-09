import logging
import cv2
from detectron2.engine import default_argument_parser
from pgrcnn.data.dataset_mapper import JerseyNumberDatasetMapper
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
    batched_inputs = [batched_inputs[0]] if len(batched_inputs) > 1 else batched_inputs
    # assert len(batched_inputs) == 1, "visualize_training() needs batch size of 1"
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

def test_base_dataloader(cfg, show_data=False):
    dataloader = build_detection_train_loader(cfg, mapper=JerseyNumberDatasetMapper(cfg, True))

    for data in dataloader:
        logger.info(f"{data[0]['file_name']}")
        logger.info(f"{data[0]}")
        if show_data:
            file_name, vis_img = visualize_training(data, cfg)
            cv2.imshow(file_name, vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    args = default_argument_parser().parse_args()
    # lazy add config file if you want
    if not args.config_file:
        args.config_file = "configs/pg_rcnn/tests/baseline.yaml"
    cfg = setup(args)
    test_base_dataloader(cfg, show_data=True)
