import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_argument_parser
from pgrcnn.utils.launch_utils import setup

# dataset test
VIS_DATASET = True
NUM_IMAGE_SHOW = 3

if __name__ == "__main__":
    # register dataset
    args = default_argument_parser().parse_args()
    print(args)
    # lazy add config file
    # args.config_file = "../../configs/pg_rcnn_R_50_FPN_1x_test_2.yaml"
    args.config_file = "../../configs/faster_rcnn_R_50_FPN_3x.yaml"
    cfg = setup(args)
    from pgrcnn.utils.custom_visualizer import JerseyNumberVisualizer
    dataset_root = os.path.join('../../../../', 'datasets/jnw') # working dir is the current file
    dataset_dir = os.path.join(dataset_root, 'total/')
    annotation_dir = os.path.join(dataset_root, 'annotations/processed_annotations.json')
    # register_jerseynumbers()
    # dataset_dicts = get_dicts("jerseynumbers", annotation_dir, split=[0,1,2,3])
    # register_jerseynumbers()
    dataset_dicts = DatasetCatalog.get("jerseynumbers_train")
    jnw_metadata = MetadataCatalog.get("jerseynumbers_train")
    import random, cv2
    if VIS_DATASET:
        for d in random.sample(dataset_dicts, NUM_IMAGE_SHOW):
            print("file name: ", os.path.abspath(d['file_name']))
            print(d)
            img = cv2.imread(d["file_name"])
            visualizer = JerseyNumberVisualizer(img[:, :, ::-1], metadata=jnw_metadata, scale=2)
            vis = visualizer.draw_dataset_dict(d)
            winname = "example"
            cv2.namedWindow(winname)  # Create a named window
            cv2.moveWindow(winname, -1000, 500)  # Move it the main monitor if you have two monitors
            cv2.imshow(winname, vis.get_image()[:, :, ::-1])
            cv2.waitKey(0)
            cv2.destroyAllWindows()