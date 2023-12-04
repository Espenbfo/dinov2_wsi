import torch
from torch import nn
from dinov2.models import build_model
class Model(nn.Module):
    def __init__(self, backbone, emb_dim, num_classes):
        super(Model, self).__init__()
        self.transformer = backbone
        self.hidden_dim=1024
        self.embed_dim = emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim*2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, num_classes))

    def forward(self, x):
        output = self.transformer(x, is_training=True)
        cls = output["x_norm_clstoken"]
        average_patch = output["x_norm_patchtokens"].mean(axis=1)
        concat = torch.cat((cls, average_patch), 1)
        x = self.classifier(concat)
        return x


def init_model(classes, pretrained_path=None):
    vit_kwargs = dict(
        img_size=224,
        arch="vit_base",
        patch_size=args.patch_size,
        init_values=args.layerscale,
        ffn_layer=args.ffn_layer,
        block_chunks=args.block_chunks,
        qkv_bias=args.qkv_bias,
        proj_bias=args.proj_bias,
        ffn_bias=args.ffn_bias,
        num_register_tokens=args.num_register_tokens,
        interpolate_offset=args.interpolate_offset,
        interpolate_antialias=args.interpolate_antialias,
    )

    backbone, emb_dim = build_model(**vit_kwargs, only_teacher=True)

    if pretrained_path is not None:
        data = torch.load(pretrained_path)
        state_dict = data["model"].transformer
        backbone.load_state_dict(state_dict)

    model = Model(backbone, emb_dim, classes)
    return model

def load_model(classes, filename):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')

    model = Model(model, classes)
    m_state_dict = torch.load(filename)
    model.load_state_dict(m_state_dict)
    return model