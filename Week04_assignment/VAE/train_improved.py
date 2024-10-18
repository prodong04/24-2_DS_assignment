import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model.my_VAE import my_VAE, loss_function  # VAE 모델과 손실 함수 가져오기
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # 학습 진행 표시
import os

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 데이터셋 준비 (Fashion MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 모델 초기화 및 GPU로 이동
model = my_VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TensorBoard 설정
writer = SummaryWriter("runs/VAE_FashionMNIST")

# 모델 저장 함수
def save_checkpoint(model, optimizer, epoch, loss, filename="vae_best.pth"):
    print(f"=> Saving model with loss {loss:.4f} at epoch {epoch}")
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)

# 학습 설정
num_epochs = 50
best_loss = float('inf')  # 최저 손실 초기화

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    loop = tqdm(train_loader, leave=True)  # 학습 진행 표시
    for batch_idx, (data, _) in enumerate(loop):
        data = data.to(device)  # 데이터를 GPU로 이동

        # Forward 및 손실 계산
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)

        # Backward 및 최적화
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        # tqdm 진행 표시 업데이트
        loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())

    # 에포크 별 평균 손실 계산 및 출력
    avg_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    writer.add_scalar("Loss/train", avg_loss, epoch)

    # 최저 손실 갱신 시 모델 저장
    if avg_loss < best_loss:
        best_loss = avg_loss  # 최저 손실 갱신
        save_checkpoint(model, optimizer, epoch + 1, best_loss, filename="vae_best.pth")

# TensorBoard 종료
writer.close()
print("학습 완료!")
