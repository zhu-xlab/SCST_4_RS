import os
from PIL import Image
from PIL import ImageFile

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset

import json

ImageFile.LOAD_TRUNCATED_IMAGES = True

def extract_name(s):
    name_part = s.split('_')[0]  # Extracts the part before the first underscore
    name_with_spaces = name_part.replace('_', ' ')  # Replaces remaining underscores with spaces
    return name_with_spaces

def get_pseudo_caption(filename, json_filepath):
    # Load the bounding box data from the JSON file
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    if filename not in data:
        print(f"No bounding box data found for {filename}.")
        return None

    result = data[filename]
    
    return result

class CaptionDataset(CaptionDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images 
        ann_paths (list): List of paths to the annotation files
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {
            "image": image,
            "text_input": 'a photo of ',# + str(get_pseudo_caption(ann["image"], 'pseudo_labels_rsicd.json')),
            "text_output": caption,
            "image_id": ann["image_id"],
        }

class CaptionEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images 
        ann_paths (list): List of paths to the annotation files
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {
            "image": image,
            "image_id": ann["image_id"],
            "prompt": 'a photo of ',# + str(get_pseudo_caption(ann["image"], 'pseudo_labels_rsicd.json')),
            #"instance_id": ann["instance_id"],
        }
