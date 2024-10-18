from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim import Adam
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#from model.VQVAE import VQVAE
from model.RE_VQVAE import VQVAE
import os
from perceptual import PerceptualLoss
# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# VQ-VAE 손실 함수 정의
def vq_vae_loss(reconstructed, original, vq_loss):
    reconstruction_loss = F.mse_loss(reconstructed, original, reduction='sum')
    total_loss = reconstruction_loss + vq_loss
    return total_loss

# 데이터셋 준비 (Fashion MNIST)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
perceptual_loss_fn = PerceptualLoss().to(device)

# VQ-VAE 모델 설정
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25

model = VQVAE(1, embedding_dim, num_embeddings, commitment_cost).to(device)
model.load_state_dict(torch.load('/home/dongryeol/24-2_DS_assignment/Week04_assignment/VAE/saved_models/vqvae_epoch121_loss74.0698.pt'))
optimizer = Adam(model.parameters(), lr=0.001)

# 저장 폴더 생성
os.makedirs('./saved_models', exist_ok=True)

# 학습 루프 설정
num_epochs = 10000
best_loss = float('inf')  # 최소 손실을 추적하기 위한 변수

for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0

    for batch, (x, _) in enumerate(train_loader):
        x = x.to(device).float()

        # Forward pass
        optimizer.zero_grad()
        reconstructed, vq_loss = model(x)

        # Perceptual Loss 계산
        p_loss = perceptual_loss_fn(reconstructed, x)
        total_loss = p_loss + vq_loss  # VQ Loss와 결합

        # Backward pass 및 매개변수 업데이트
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()

    avg_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 최소 손실일 경우 모델 저장
    if avg_loss < best_loss:
        best_loss = avg_loss
        model_path = f'./saved_models/revqvae_epoch{epoch + 1}_loss{avg_loss:.4f}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"최소 손실 갱신, 모델 저장: {model_path}")