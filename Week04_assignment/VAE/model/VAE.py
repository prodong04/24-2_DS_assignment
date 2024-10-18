import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(CNNVAE, self).__init__()

        # 인코더 정의 (Conv layers)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, 7, 7)
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout(0.1),

            nn.Flatten(),  # (B, 64*7*7)
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)  # 평균
        self.fc_log_var = nn.Linear(64 * 7 * 7, latent_dim)  # 로그 분산

        # 잠재 공간 -> 디코더 입력
        self.fc_decoder_input = nn.Linear(latent_dim, 64 * 7 * 7)

        # 디코더 정의 (Transposed Conv layers)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.1),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1, 28, 28)
            nn.Sigmoid(),  # [0, 1]로 스케일링
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)

        # 인코더를 통해 잠재 변수 추출
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        z = self.reparameterize(mu, log_var)

        # 디코더 입력 준비
        decoder_input = self.fc_decoder_input(z).view(batch_size, 64, 7, 7)

        # 디코더를 통해 재구성
        reconstructed = self.decoder(decoder_input)

        return reconstructed, mu, log_var
