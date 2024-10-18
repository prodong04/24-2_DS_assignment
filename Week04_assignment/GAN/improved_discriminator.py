import torch
import torch.nn as nn

"""
GAN의 Discriminator를 자유롭게 개선해주세요!!
단순 논문 구현한 Discriminator에 이것 저것을 추가해도 좋고, 변경해도 좋습니다!

Hint:
1. Batch Normalization
2. Dropout
3. Deep Layer
4. etc...

Layer가 깊을수록 성능이 좋아질까요?? 

"""
class Discriminator(nn.Module):
    def __init__(self, img_channels=1):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),  # (1, 28, 28) -> (64, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (64, 14, 14) -> (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),  # (128, 7, 7) -> (128 * 7 * 7)
            nn.Linear(128 * 7 * 7, 1),  # Output: 1 (진짜/가짜 확률)
            nn.Sigmoid()  # 확률 값으로 변환
        )

    def forward(self, x):
        return self.main(x)