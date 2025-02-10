#RS backbone
"""from features_extractor import load_model
from MAEPretrain_SceneClassification.models_mae_vitae import MaskedAutoencoderViTAE  # replace this with the import line for your actual model
from MAEPretrain_SceneClassification.util.pos_embed import interpolate_pos_embed
import torch.nn as nn
import torch

class FeatureExtractor(MaskedAutoencoderViTAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features=1408
        self.linear = nn.Linear(768, self.num_features)
        
    def forward(self, x, mask_ratio=0.75):
        features, _, _ = self.forward_encoder(x, mask_ratio)
        features = self.linear(features)
        return features

def convert_weights_to_fp16(model: nn.Module):
    #Convert applicable model parameters to fp16

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

def create_vit_rs(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FeatureExtractor() 

    model = model.to(device)

    state_dict = torch.load(model_path, map_location="cpu")    
    interpolate_pos_embed(model,state_dict)

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    convert_weights_to_fp16(model)
        
    return model"""

#ViT-G-LAION

import torch
import torch.nn as nn
import open_clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomCLIPModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs, **kwargs):
        vision_outputs = self.model.visual(*inputs)
        patch_features = vision_outputs

        # Then re-insert the pixel values into kwargs and continue with the original forward method
        outputs = self.model.encode_image(*inputs)
        
        return outputs, patch_features

class FeatureExtractor(CustomCLIPModel):
    def __init__(self, model):
        super().__init__(model)
        self.num_features = 1408
        self.embedding_transform = nn.Linear(1024, self.num_features)
        self.dummy_input_ids = torch.zeros(1,1, dtype=torch.long).to(device) # Add this line

    def forward(self, *inputs, **kwargs):
        outputs, patch_features = super().forward(*inputs, **kwargs)
        transformed_patch_features = self.embedding_transform(patch_features)
        return transformed_patch_features

def convert_weights_to_fp16(model: nn.Module):
    #Convert applicable model parameters to fp16

    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.MultiheadAttention)):
            m.half()

def create_vit_rs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k')
    tokenizer = open_clip.get_tokenizer('hf-hub:timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k')

    model = FeatureExtractor(model)
    model = model.to(device)

    convert_weights_to_fp16(model)
        
    return model

"""from features_extractor import load_model
from open_clip import create_model_and_transforms, get_tokenizer  # replace this with the import line for your actual model
import torch.nn as nn
import torch

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='ViT-L-14', num_features=1408, ckpt_path="RemoteCLIP/RemoteCLIP-ViT-L-14.pt"):
        super().__init__()
        self.model, _, _ = create_model_and_transforms(model_name)
        self.tokenizer = get_tokenizer(model_name)
        self.num_features = num_features
        self.linear = nn.Linear(768, self.num_features)  # 768 might need to be changed depending on the output size of the model's encoder
        
    def forward(self, x):
        with torch.no_grad(), torch.cuda.amp.autocast():
            features = self.model.encode_image(x)
            features /= features.norm(dim=-1, keepdim=True)
        features = self.linear(features)
        return features

def convert_weights_to_fp16(model: nn.Module):
    #Convert applicable model parameters to fp16

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

def create_vit_rs(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FeatureExtractor() 

    model = model.to(device)

    state_dict = torch.load(model_path, map_location="cpu")    

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    convert_weights_to_fp16(model)
        
    return model"""