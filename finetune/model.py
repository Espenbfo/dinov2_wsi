import torch
from torch import nn
from dinov2.models.vision_transformer import vit_base
class Model(nn.Module):
    def __init__(self, backbone, emb_dim, num_classes):
        super(Model, self).__init__()
        self.transformer = backbone
        self.hidden_dim=1024
        self.embed_dim = emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim*2, num_classes))

    def forward(self, x):
        output = self.transformer(x, is_training=True)
        cls = output["x_norm_clstoken"]
        average_patch = output["x_norm_patchtokens"].mean(axis=1)
        concat = torch.cat((cls, average_patch), 1)
        x = self.classifier(concat)
        return x


def extract_teacher_weights(ordered_dict):
    new_dict = {}
    for key in ordered_dict.keys():
        if "teacher.backbone." in key:
            new_key = key.replace("teacher.backbone.", "")
            new_dict[new_key] = ordered_dict[key]
    return new_dict


def init_model(classes, pretrained_path=None):
    vit_kwargs = dict(
        img_size=224,
        patch_size=16,
        init_values=1.0e-05,
        ffn_layer="swiglufused",
        block_chunks=4,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        num_register_tokens=0,
        interpolate_offset=0.1,
        interpolate_antialias=False,
    )
    backbone = vit_base(**vit_kwargs)

    emb_dim = backbone.embed_dim

    if pretrained_path == "dino":
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    elif pretrained_path is not None:
        data = torch.load(pretrained_path)
        state_dict = extract_teacher_weights(data["model"])
        backbone.load_state_dict(state_dict)

    print(f"Embedding dimension: {emb_dim}]")
    model = Model(backbone, emb_dim, classes)
    return model

def load_model(classes, filename):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')

    model = Model(model, classes)
    m_state_dict = torch.load(filename)
    model.load_state_dict(m_state_dict)
    return model