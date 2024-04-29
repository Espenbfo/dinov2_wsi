import torch
from torch import nn
from dinov2.models.vision_transformer import vit_base, vim_tiny, vim_base, vim_small, vim_base_orig, vim_tiny_orig, vmamba_base, vmamba_small, vmamba_tiny
from dinov2.fsdp import FSDPCheckpointer
class Model(nn.Module):
    def __init__(self, backbone, emb_dim, num_classes, is_phikon=False):
        super(Model, self).__init__()
        self.transformer = backbone
        self.embed_dim = emb_dim
        self.is_phikon = is_phikon
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim*(2 if not self.is_phikon else 1), num_classes))

    def forward(self, x):
        if self.is_phikon:
            concat = self.transformer(x)
        else:
            output = self.transformer(x, is_training=True)
            cls = output["x_norm_clstoken"]
            average_patch = output["x_norm_patchtokens"].mean(axis=1)
            concat = torch.cat((cls, average_patch), 1)
        concat = torch.nn.functional.layer_norm(concat, (concat.shape[1],))
        x = self.classifier(concat)
        return x


def extract_teacher_weights(ordered_dict):
    new_dict = {}
    for key in ordered_dict.keys():
        if "teacher.backbone." in key:
            new_key = key.replace("teacher.backbone.", "")
            new_dict[new_key] = ordered_dict[key]
        if "backbone." in key:
            new_key = key.replace("backbone.", "")
            new_dict[new_key] = ordered_dict[key]
    return new_dict


def init_model(classes, pretrained_path=None, teacher_checkpoint=True, mode="normal"):
    is_phikon = False
    if mode == "dino":
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        emb_dim = backbone.embed_dim
    elif mode == "phikon":
        from HistoSSLscaling.rl_benchmarks.models.feature_extractors.ibot_vit import iBOTViT
        backbone = iBOTViT(weights_path="weights/ibot_vit_base_pancan.pth", encoder="student")
        emb_dim = backbone.feature_extractor.num_features
        is_phikon = True
    elif mode == "normal":
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
        #torch.distributed.init_process_group(rank=0, world_size=1, store=torch.distributed.Store())
        backbone = vmamba_tiny(**vit_kwargs)

        emb_dim = backbone.embed_dim
        if pretrained_path:
            if (teacher_checkpoint):
                data = torch.load(pretrained_path)
                state_dict = extract_teacher_weights(data["teacher"])
                backbone.load_state_dict(state_dict)
            else:
                data = torch.load(pretrained_path)

                state_dict = extract_teacher_weights(data["model"])
                backbone.load_state_dict(state_dict)

    print(f"Embedding dimension: {emb_dim}]")
    model = Model(backbone, emb_dim, classes, is_phikon)
    return model

def load_model(classes, filename):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')

    model = Model(model, classes)
    m_state_dict = torch.load(filename)
    model.load_state_dict(m_state_dict)
    return model