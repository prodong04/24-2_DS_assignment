import torch
import torch.nn as nn

"""
GAN의 Generator를 자유롭게 개선해주세요!!
단순 논문 구현한 Generator에 이것 저것을 추가해도 좋고, 변경해도 좋습니다!

Hint:
1. Batch Normalization
2. Dropout
3. Deep Layer
4. etc...

Layer가 깊을수록 성능이 좋아질까요??
 
"""

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=1):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # (z_dim, 1, 1) -> (128, 7, 7)
            nn.ConvTranspose2d(z_dim, 128, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # (128, 7, 7) -> (64, 14, 14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # (64, 14, 14) -> (img_channels, 28, 28)
            nn.ConvTranspose2d(64, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output: [-1, 1]로 스케일링
        )

    def forward(self, z):
        return self.gen(z)