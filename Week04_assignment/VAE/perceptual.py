from torchvision.models import vgg16
from torch import nn
from torch.nn.functional import mse_loss
from torchvision.transforms import Resize
from torch.nn import functional as F


from torchvision.transforms import Resize

class PerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15], reduction='sum'):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.layers = layers
        self.vgg = vgg
        self.reduction = reduction
        self.resize = Resize((224, 224))  # 이미지 크기 조정

    def forward(self, x, y):
        x = self.expand_to_rgb(x)
        y = self.expand_to_rgb(y)
        x = self.resize(x)  # 크기 조정
        y = self.resize(y)  # 크기 조정
        x_features = self.extract_features(x)
        y_features = self.extract_features(y)
        loss = 0.0
        for xf, yf in zip(x_features, y_features):
            loss += F.mse_loss(xf, yf, reduction=self.reduction)
        return loss

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features

    def expand_to_rgb(self, x):
        return x.repeat(1, 3, 1, 1)  # Grayscale → RGB 변환
