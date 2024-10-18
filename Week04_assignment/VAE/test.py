import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model.VAE import CNNVAE
from model.my_VAE import my_VAE
import os

from model.VQVAE import VQVAE



path = '/home/dongryeol/24-2_DS_assignment/Week04_assignment'
origin_image_dir = path + '/VAE/original_generated_images' #생성된 이미지가 저장된 디렉토리 경로 넣어주세요!
improved_image_dir = path + '/VAE/improved_generated_images'

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 데이터셋 준비 (Fashion MNIST)
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
save_dir = "original_generated_images"
os.makedirs(save_dir, exist_ok=True)
# 모델 불러오기 및 GPU로 이동
origin_dir = "original_images"
os.makedirs(origin_dir, exist_ok=True)

# VAE 설정
class Config:
    height = 28
    width = 28
    color_channel = 1  # 흑백 이미지
    hidden_dim = 400
    latent_dim = 20
    dropout = 0.2  # Dropout 비율

config = Config()
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25

# 모델 및 옵티마이저 설정
model = VQVAE(1, embedding_dim, num_embeddings, commitment_cost).to(device)
model.load_state_dict(torch.load('/home/dongryeol/24-2_DS_assignment/Week04_assignment/VAE/saved_models/vqvae_epoch244_loss182.0044.pt'))

model.eval()

print("개선 모델 불러오기 완료!")
improved_save_dir = "improved_generated_images"
os.makedirs(improved_save_dir, exist_ok=True)
def save_images(original, reconstructed, num_images=1000):
    """원본과 재구성된 이미지를 저장합니다."""
    original = original[:num_images]
    reconstructed = reconstructed[:num_images]

    for i in range(num_images):
        # 784 크기의 벡터를 28x28 이미지로 변환
        img = reconstructed[i].view(28, 28).detach().cpu().numpy()

        # 이미지 저장
        #plt.imsave(os.path.join(save_dir, f"reconstructed_image_{i+1}.png"), img, cmap='gray')
        plt.imsave(os.path.join(improved_save_dir, f"reconstructed_image_{i+1}.png"), img, cmap='gray')

with torch.no_grad():
    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.to(device)  # 데이터를 GPU로 이동
        recon_batch, _ = model(data)
        save_images(data, recon_batch, num_images=1000)
        break
# 첫 배치만 사용하여 원본과 재구성 이미지 저장
  # 첫 배치만 사용하여 원본과 재구성 이미지 저장cd ../




