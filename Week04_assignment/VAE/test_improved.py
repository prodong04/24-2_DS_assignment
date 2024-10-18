import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model.my_VAE import my_VAE
import os

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 데이터셋 준비 (Fashion MNIST) - 정규화 추가
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # [-1, 1] 범위로 변환
])
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)  # 배치 크기 조정

# 디렉토리 생성
origin_dir = "original_images"
os.makedirs(origin_dir, exist_ok=True)

improved_save_dir = "improved_generated_images"
os.makedirs(improved_save_dir, exist_ok=True)

# 개선된 모델 불러오기
checkpoint = torch.load('/home/dongryeol/Week04_assignment/VAE/vae_best.pth', map_location=device)
model_improved = my_VAE().to(device)

# 체크포인트에서 state_dict만 추출해서 로드
model_improved.load_state_dict(checkpoint['state_dict'])
model_improved.eval()
print("개선 모델 불러오기 완료!")

# 이미지 저장 함수
def save_images(original, improved_reconstructed, num_images=100):
    """원본과 재구성된 이미지를 저장합니다."""
    original = original[:num_images]
    improved_reconstructed = improved_reconstructed[:num_images]

    for i in range(num_images):
        # 원본 이미지 저장
        plt.imsave(
            os.path.join(origin_dir, f"original_image_{i+1}.png"),
            (original[i].squeeze().cpu().numpy() + 1) / 2,  # [-1, 1] -> [0, 1]
            cmap='gray'
        )
        # 개선된 모델의 재구성 이미지 저장
        plt.imsave(
            os.path.join(improved_save_dir, f"improved_reconstructed_image_{i+1}.png"),
            (improved_reconstructed[i].squeeze().cpu().numpy() + 1) / 2,  # [-1, 1] -> [0, 1]
            cmap='gray'
        )

# 테스트 데이터에서 재구성 이미지 생성 및 저장
with torch.no_grad():
    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.to(device)

        # 개선된 모델의 재구성 이미지 생성
        improved_recon_batch, _, _ = model_improved(data)

        # 배치에 대해 이미지 저장
        save_images(data, improved_recon_batch, num_images=100)  # 배치 크기 조정
