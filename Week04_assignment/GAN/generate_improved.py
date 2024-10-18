import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from original_generator import Original_Generator
from improved_generator import Generator 
import os
import argparse

"""
훈련한 모델을 불러와서 이미지를 생성하는 코드입니다!
명령줄 인자를 사용하여 simple GAN과 improved GAN을 선택할 수 있습니다.

단순 구현한 GAN 모델 불러올 시 (터미널에 입력)
python generate.py --model_type original 

개선한 GAN 모델 불러올 시 (터미널에 입력)
python generate.py --model_type improved

checkpoint_file은 커맨드로 입력하면 너무 길어지기 때문에 직접 코드에서 수정해주세요!

여기서 생성한 이미지의 디렉토리를 IS Score를 계산하는 코드에 입력해주세요!
"""

def show_generated_image(generated_image, save_path=None):
    # 이미지 값을 [0, 1] 범위로 변환합니다.
    generated_image = (generated_image.squeeze(0) + 1) / 2  
    generated_image = generated_image.detach().cpu().numpy()

    plt.imshow(generated_image, cmap="gray")
    plt.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.imsave(save_path, generated_image, cmap="gray")
        print(f"Image saved to {save_path}")

    plt.close()

def generate_random_image(model_type, checkpoint_file, z_dim, num_images):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델의 종류에 따라 Generator 선택
    if model_type == "original":
        gen = Original_Generator(z_dim, 1).to(device)  # 흑백 이미지 생성
        save_dir = "original_generated_image"
    elif model_type == "improved":
        gen = Generator(z_dim, 1).to(device)  # Conv 기반 Improved Generator
        save_dir = "improved_generated_image"
    else:
        raise ValueError("model_type should be either 'original' or 'improved'")

    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_file, map_location=device)
    gen.load_state_dict(checkpoint['state_dict'], strict=False)
    gen.eval()

    # 잠재 벡터 생성 (batch_size, z_dim, 1, 1)
    noise = torch.randn(num_images, z_dim, 1, 1).to(device)

    # 이미지 생성 및 저장
    for idx in range(num_images):
        n = noise[idx].unsqueeze(0)  # 개별 이미지 생성 시 (1, z_dim, 1, 1)

        with torch.no_grad():
            generated_image = gen(n)  # (1, 1, 28, 28) 형태로 출력
            generated_image = generated_image.squeeze(0)  # (1, 28, 28)로 변환

        # 생성된 이미지 저장
        show_generated_image(generated_image, save_path=f"{save_dir}/generated_image_{idx}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, help="Type of model: 'original' or 'improved'")
    parser.add_argument("--z_dim", type=int, default=100, help="Dimensionality of latent vector z")
    parser.add_argument("--num_images", type=int, default=1000, help="Number of images to generate")
    args = parser.parse_args()

    # Generator의 체크포인트 경로 설정
    generator_checkpoint = "/home/dongryeol/Week04_assignment/GAN/improved_model/improved_generator_best_v2.pth.tar"

    # 이미지 생성 함수 호출
    generate_random_image(args.model_type, generator_checkpoint, args.z_dim, args.num_images)
