import os
import random, cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_argument_parser
from pgrcnn.utils.launch_utils import setup
from pgrcnn.utils.custom_visualizer import JerseyNumberVisualizer
# dataset test
VIS_DATASET = False
NUM_IMAGE_SHOW = -1

if __name__ == "__main__":
    # register dataset
    args = default_argument_parser().parse_args()
    print(args)
    # lazy add config file
    # args.config_file = "../../configs/pg_rcnn_R_50_FPN_1x_test_2.yaml"
    args.config_file = "configs/pg_rcnn/tests/baseline.yaml"
    args.output = "output/vis_results"
    cfg = setup(args)
    dataset_root = os.path.join('../../../../', 'datasets/jnw') # working dir is the current file
    dataset_dir = os.path.join(dataset_root, 'total/')
    annotation_dir = os.path.join(dataset_root, 'annotations/processed_annotations.json')
    dataset_dicts = DatasetCatalog.get("jerseynumbers_train")
    jnw_metadata = MetadataCatalog.get("jerseynumbers_train")
    dataset_dicts = random.sample(dataset_dicts, NUM_IMAGE_SHOW) if NUM_IMAGE_SHOW > 0 else dataset_dicts
    for d in dataset_dicts:
        print("file name: ", os.path.abspath(d['file_name']))
        # print(d)
        basename = os.path.basename(d["file_name"])
        basename_wo_extension = os.path.splitext(basename)[0]
        img = cv2.imread(d["file_name"])
        visualizer = JerseyNumberVisualizer(img[:, :, ::-1], metadata=jnw_metadata, scale=2)
        vis = visualizer.draw_dataset_dict(d)
        if VIS_DATASET:
            winname = "example"
            cv2.namedWindow(winname)  # Create a named window
            cv2.moveWindow(winname, -1000, 500)  # Move it the main monitor if you have two monitors
            cv2.imshow(winname, vis.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        out_path = os.path.join(args.output, basename_wo_extension)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        vis.save(os.path.join(out_path, "gt.pdf"))