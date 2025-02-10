from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.common.registry import registry
from lavis.datasets.datasets.rs import CaptionDataset, CaptionEvalDataset

@registry.register_builder("rs_caption")  # replace "your_dataset_name" with the actual name of your dataset
class RSCaptioning(BaseDatasetBuilder):  # replace "YourDatasetBuilder" with a name of your choice
    train_dataset_cls = CaptionDataset
    eval_dataset_cls = CaptionEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/rs/defaults_cap.yaml"  # replace with the actual path to your dataset config
    }
