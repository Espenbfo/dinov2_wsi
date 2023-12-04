import torch
from torch import nn
class Model(nn.Module):
    def __init__(self, dino, num_classes):
        super(Model, self).__init__()
        self.transformer = dino
        self.hidden_dim=1024
        self.embed_dim = self.transformer.embed_dim
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


def init_model(classes):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')

    model = Model(model, classes)
    return model

def load_model(classes, filename):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')

    model = Model(model, classes)
    m_state_dict = torch.load(filename)
    model.load_state_dict(m_state_dict)
    return model