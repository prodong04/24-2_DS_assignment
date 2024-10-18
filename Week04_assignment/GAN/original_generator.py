import torch
import torch.nn as nn

"""
정말 간단하게 논문 구현만 해봅시다
논문에서 제시한대로 구현하면 됩니다!!
데이터셋은 Fashion MNIST를 사용하기 때문에, 이미지의 크기(=img_dim)는 28 * 28 = 784입니다.
z_dim은 latent vector z의 차원이고, default로 64로 설정했습니다.

Hint: nn.Sequential을 사용하면 간단하게 구현할 수 있습니다.

"""


class Original_Generator(nn.Module):

    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape  

        self.gen = nn.Sequential(
            nn.Linear(latent_dim, 128),    
            nn.ReLU(),                     
            nn.Linear(128, 256),           
            nn.BatchNorm1d(256, 0.8),     
            nn.ReLU(),                   
            nn.Linear(256, 512),           
            nn.BatchNorm1d(512, 0.8),      
            nn.ReLU(),                     
            nn.Linear(512, int(torch.prod(torch.tensor(img_shape)))),  
            nn.Tanh()                      
        )

    def forward(self, z):
        img_flat = self.gen(z)  # (batch_size, flattened_img_size)
        return img_flat