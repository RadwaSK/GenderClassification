import torch.nn as nn
import timm


class ViT(nn.Module):
    def __init__(self, CUDA, pretrained_ViT=True, freeze_ViT=False):
        super(ViT, self).__init__()

        self.CUDA = CUDA

        self.model = timm.create_model("vit_base_patch32_224", pretrained=pretrained_ViT)
        self.linear_fc_layer1 = nn.Linear(1000, 100)
        self.linear_fc_layer2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

        # if freeze_resnet is True, requires grad is False
        for param in self.model.parameters():
            param.requires_grad = not freeze_ViT

    def forward(self, data):
        output = self.model(data)
        output = self.linear_fc_layer1(output)
        output = self.linear_fc_layer2(output)
        return self.sigmoid(output)
