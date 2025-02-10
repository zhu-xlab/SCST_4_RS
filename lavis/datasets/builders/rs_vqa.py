from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.common.registry import registry
from lavis.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset

@registry.register_builder("rs_vqa")  # replace "your_dataset_name" with the actual name of your dataset
class RS_VQA(BaseDatasetBuilder):  # replace "YourDatasetBuilder" with a name of your choice
    train_dataset_cls = COCOVQADataset
    eval_dataset_cls = COCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rsvqa/defaults_cap_rsvqa.yaml"  # replace with the actual path to your dataset config
    }
