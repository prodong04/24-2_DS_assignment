from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model.VAE import CNNVAE  # 수정된 VAE 클래스 가져오기
#from CRVAE.models import CNNVAE
# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 데이터셋 준비 (Fashion MNIST)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)



# 모델 초기화 및 GPU 이동
#model = CNNVAE(config).to(device)
model = CNNVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# VAE 손실 함수 정의 (KL 가중치 포함)
def vae_loss(reconstructed, original, mu, log_var, beta=4.0):
    reconstruction_loss = F.mse_loss(reconstructed, original, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + beta * kld_loss

# 데이터 로드 및 전처리
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 모델 및 옵티마이저 설정
model = CNNVAE(latent_dim=16)
optimizer = Adam(model.parameters(), lr=1e-3)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# 학습 루프
num_epochs = 50
for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0

    for batch, (x, _) in enumerate(train_loader):
        x = x.to(torch.float32)

        optimizer.zero_grad()
        reconstructed, mu, log_var = model(x)
        loss = vae_loss(reconstructed, x, mu, log_var, beta=4)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    #scheduler.step(train_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader.dataset):.4f}")
