import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from improved_discriminator import Discriminator
from improved_generator import Generator
from dataset import get_fashion_mnist_loader
import os
import argparse

def save_checkpoint(model, optimizer, directory_name="improved_model", filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, os.path.join(directory_name, filename))

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def main(model_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    z_dim = 100  # Generator의 입력 잠재 공간 차원
    img_channels = 1  # Fashion MNIST는 흑백 이미지
    least_epoch = 4  # 최소 에포크 설정
    best_lossG = float('inf')  # 최저 Generator 손실 초기화
    directory_name = "improved_model"
    save_model = True

    # 모델 초기화
    disc = Discriminator(img_channels).to(device)
    gen = Generator(z_dim, img_channels).to(device)

    fixed_noise = torch.randn((64, z_dim, 1, 1)).to(device)  # 고정된 입력 벡터
    loader = get_fashion_mnist_loader(batch_size=64)

    # 옵티마이저 및 손실 함수 설정
    opt_disc = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_gen = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(50):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)

            # Discriminator 학습
            noise = torch.randn(real.size(0), z_dim, 1, 1).to(device)
            fake = gen(noise)

            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2

            disc.zero_grad()
            lossD.backward()
            opt_disc.step()

            # Generator 학습
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))

            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/50], Batch {batch_idx}, Loss D: {lossD.item()}, Loss G: {lossG.item()}")

        # 일정 에포크 이상부터 최저 Generator 손실을 갱신할 때만 체크포인트 저장
        if epoch > least_epoch:
            if save_model and lossG < best_lossG:
                best_lossG = lossG  # 최저 손실 갱신
                save_checkpoint(gen, opt_gen, directory_name=directory_name, filename=f"improved_generator_best_v2.pth.tar")
                print(f"New best model saved at epoch {epoch} with lossG: {lossG}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="improved", help="Type of model: 'original' or 'improved'")
    args = parser.parse_args()
    main(args.model_type)
