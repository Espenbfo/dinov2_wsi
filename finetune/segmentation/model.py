import torch
from torch import nn
from dinov2.models.vision_transformer import vit_base
from dinov2.fsdp import FSDPCheckpointer
class Model(nn.Module):
    def __init__(self, emb_dim, num_classes, models):
        super(Model, self).__init__()
        self.hidden_dim=32
        self.embed_dim = emb_dim
        self.backbones = nn.ModuleList(model[0] for model in models)
        self.modes = [model[1] for model in models]

        total_emb_size = 0
        for mode in self.modes:
            if mode:
                total_emb_size += self.embed_dim
            else:
                total_emb_size += self.embed_dim*2
        self.classifier = nn.Sequential(
            nn.Linear(total_emb_size, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, num_classes))

    def forward(self, *xs):
        embs = []
        for i in range(len(self.backbones)):
            x = xs[i]
            if self.modes[i]:
                cls_tokens = self.backbones[i](x)
                embs.append(cls_tokens)
            else:
                output = self.backbones[i](x, is_training=True)
                cls_tokens = output["x_norm_clstoken"]
                patch_tokens = output["x_norm_patchtokens"].mean(axis=1)
                embs.append(cls_tokens)
                embs.append(patch_tokens)
        #average_patch = output["x_norm_patchtokens"].mean(axis=1)
        concat = torch.cat(embs, 1)
        x = self.classifier(concat)
        return x


def extract_teacher_weights(ordered_dict):
    new_dict = {}
    for key in ordered_dict.keys():
        if "teacher.backbone." in key:
            new_key = key.replace("teacher.backbone.", "")
            new_dict[new_key] = ordered_dict[key]
        elif "backbone." in key:
            new_key = key.replace("backbone.", "")
            new_dict[new_key] = ordered_dict[key]
    return new_dict


def init_model(classes, model_configs):
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
    models = []
    for config in model_configs:
        size, mode, path = config

        is_phikon = False
        if mode == "dino":
            backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
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
            backbone = vit_base(**vit_kwargs)

            emb_dim = backbone.embed_dim
            if path:
                data = torch.load(path)

                state_dict = extract_teacher_weights(data["teacher"])
                backbone.load_state_dict(state_dict)
        models.append((backbone, is_phikon))

    model = Model(emb_dim, classes, models)
    return model

def load_model(classes, filename):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')

    model = Model(model, classes)
    m_state_dict = torch.load(filename)
    model.load_state_dict(m_state_dict)
    return model