import os
import random
import logging
import argparse
from collections import OrderedDict
import torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.evaluation.sem_seg_evaluation import SemSegEvaluator


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator = COCOEvaluator(dataset_name, output_dir=output_folder)
        return evaluator

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def train():
    register_coco_instances(
        "ycb_coco_train",
        {},
        "./datasets/ycb_coco/annotations/ycb_coco_train.json",
        "./datasets/ycb_coco/train",
    )
    register_coco_instances(
        "ycb_coco_val",
        {},
        "./datasets/ycb_coco/annotations/ycb_coco_val.json",
        "./datasets/ycb_coco/val",
    )

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("ycb_coco_train",)
    cfg.DATASETS.TEST = ("ycb_coco_val",) 
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.OUTPUT_DIR = "./output"
    cfg.DATALOADER.NUM_WORKERS = 6
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.BASE_LR = 0.001
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 256]]
    cfg.INPUT.MIN_SIZE_TRAIN = (720,)
    cfg.INPUT.MIN_SIZE_TEST = 720
    cfg.SOLVER.MAX_ITER = 40000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15  # 15 classes
    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def test(weights_file):
    register_coco_instances(
        "ycb_coco_train",
        {},
        "./datasets/ycb_coco/annotations/ycb_coco_train.json",
        "./datasets/ycb_coco/train",
    )
    register_coco_instances(
        "ycb_coco_val",
        {},
        "./datasets/ycb_coco/annotations/ycb_coco_val.json",
        "./datasets/ycb_coco/val",
    )

    ycb_metadata = MetadataCatalog.get("ycb_coco_val")
    dataset_dicts = DatasetCatalog.get("ycb_coco_val")
    print(ycb_metadata)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("ycb_coco_train",)
    cfg.DATASETS.TEST = ("ycb_coco_val",)
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATALOADER.NUM_WORKERS = 6
    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 256]]
    cfg.INPUT.MIN_SIZE_TRAIN = (720,)
    cfg.INPUT.MIN_SIZE_TEST = 720
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15  # 15 classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.freeze()

    predictor = DefaultPredictor(cfg)
    for d in random.sample(dataset_dicts, 10):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=ycb_metadata,
            scale=1,
            instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels
        )
        outputs = outputs["instances"].to("cpu")
        v = v.draw_instance_predictions(outputs)
        cv2.imshow("test", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test demo")
    parser.add_argument(
        "--weight-file",
        default="output/model_final.pth",
        metavar="FILE",
        help="path to weight file",
    )
    parser.add_argument("--test", action="store_true", help="Test.")
    args = parser.parse_args()

    if args.test:
        test(args.weight_file)
    else:
        train()