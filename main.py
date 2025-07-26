import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader


from datasets.psgan_dataset import get_psgan_data
from datasets.seed_patch_dataset import get_seed_patch_data
from models.target import YOLOv8TargetModel

from psgan import PSGAN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YOLO_MODEL_DIR = './pretrained_models/yolov8x-best.pt'

BATCH_SIZE = 16
NUM_WORKERS = 8

IMG_SIZE = 256
PATCH_SIZE = 32
PATCH_RATIO = 0.05

DISTORTION_THRESHOLD = 50.0

SIGMA_GAN = 0.01
LAMBDA_PATCH = 0.002

GAMMA_ADV = 1.0
DISCRIMINATOR_LR = 0.0002
GENERATOR_LR = 0.0002
DG_RATIO = 2
EPOCHS = 100


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
  download_params = {
      'api_key': "Yp0GigDoIjDjIoTrGiyA",
      'workspace': "mrsyaban",
      'project': "traffic-signs-id", 
      'version': 2
  }
  
  psgan_dataloader, psgan_dataset = get_psgan_data(
      dataset_path=None,
      img_size=IMG_SIZE,
      split='train',
      batch_size=BATCH_SIZE,
      num_workers=NUM_WORKERS,
      download_params=download_params,
      attentive_dir="./constant/attentive_coordinate.json",
      keypoints_dir="./constant/traffic_signs_keypoints.json",
      relight_dir="./constant/relighting_params.json"
  )

  seed_patch_dataloader, seed_patch_dataset = get_seed_patch_data(
    patch_size=PATCH_SIZE,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
  )

  yolo_model = YOLOv8TargetModel(
    model_path=YOLO_MODEL_DIR,
    confidence_threshold=0.1,
    device=DEVICE
  )

  # Initialize PSGAN
  psgan = PSGAN(
    patch_size=PATCH_SIZE, 
    image_size=IMG_SIZE,
    lambda_patch=LAMBDA_PATCH,
    gamma_adv=GAMMA_ADV,
    sigma_gan=SIGMA_GAN,
    patch_area_ratio=PATCH_RATIO,
    distortion_threshold=DISTORTION_THRESHOLD
  )

  # Training with multiple epochs
  epoch_metrics = psgan.train_step(
      seed_patch_dataloader=seed_patch_dataloader,
      psgan_dataloader=psgan_dataloader, 
      epochs=EPOCHS,
      target_model=yolo_model,
      k_steps=DG_RATIO,
      d_lr=DISCRIMINATOR_LR,
      g_lr=GENERATOR_LR,
      save_interval=1,
      save_path='./checkpoints_16/',
      lambda_decay=False,
      verbose=True
  )

  # Print training history
  for metrics in epoch_metrics:
      print(f"Epoch {metrics['epoch']}: D_loss={metrics['d_loss']:.4f}, G_loss={metrics['g_loss']:.4f}")

if __name__ == '__main__':
  main()
