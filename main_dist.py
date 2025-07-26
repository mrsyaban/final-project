import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import datetime  # Add this import
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

# Set multiprocessing start method
torch.multiprocessing.set_start_method('spawn', force=True)

from datasets.psgan_dataset import get_psgan_data
from datasets.seed_patch_dataset import get_seed_patch_data
from models.target import YOLOv8TargetModel
from psgan import PSGAN

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Add NCCL environment variables for better stability
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    os.environ['NCCL_SOCKET_TIMEOUT'] = '1800'
    os.environ['NCCL_DEBUG'] = 'INFO'
    
    # Use timedelta for timeout, not dist.store
    timeout = datetime.timedelta(seconds=180)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)
    
    # Set device explicitly
    torch.cuda.set_device(rank)
    
def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Configuration
    IMG_SIZE = 256
    PATCH_SIZE = 32
    PATCH_RATIO = 0.1
    DISTORTION_THRESHOLD = 10.0
    LAMBDA_PATCH = 0.00387
    SIGMA_GAN = 1.0
    GAMMA_ADV = 2.0
    DISCRIMINATOR_LR = 0.0001
    GENERATOR_LR = 0.0002
    DG_RATIO = 9
    EPOCHS = 250
    BATCH_SIZE = 16  # Per GPU batch size
    NUM_WORKERS = 2
    YOLO_MODEL_DIR = './pretrained_models/yolov8x-best.pt'
    DISTRIBUTED = True
    
    # Set up device for this process
    torch.cuda.set_device(rank)
    
    # Create datasets with DistributedSampler
    download_params = {
        'api_key': "Yp0GigDoIjDjIoTrGiyA",
        'workspace': "mrsyaban",
        'project': "traffic-signs-id", 
        'version': 2
    }
    
    try:
        psgan_dataloader, psgan_dataset = get_psgan_data(
            dataset_path=None,
            img_size=IMG_SIZE,
            split='train',
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            download_params=download_params,
            attentive_dir="./constant/attentive_coordinate.json",
            keypoints_dir="./constant/traffic_signs_keypoints.json",
            relight_dir="./constant/relighting_params.json",
            distributed=DISTRIBUTED,
            rank=rank,
            world_size=world_size,
        )

        seed_patch_dataloader, seed_patch_dataset = get_seed_patch_data(
            patch_size=PATCH_SIZE,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            distributed=DISTRIBUTED,
            rank=rank,
            world_size=world_size,
        )
        
        # Initialize target model
        yolo_model = YOLOv8TargetModel(
            model_path=YOLO_MODEL_DIR,
            confidence_threshold=0.1,
            device=torch.device(f"cuda:{rank}")
        )
        
        # Initialize PSGAN for distributed training
        psgan = PSGAN(
            patch_size=PATCH_SIZE, 
            image_size=IMG_SIZE,
            lambda_patch=LAMBDA_PATCH,
            gamma_adv=GAMMA_ADV,
            sigma_gan=SIGMA_GAN,
            patch_area_ratio=PATCH_RATIO,
            distortion_threshold=DISTORTION_THRESHOLD,
            distributed=DISTRIBUTED,
            local_rank=rank
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
            save_path=f'./checkpoints/finalr9_g-1.0_l-0.00387/',
            log_path='./logging/finalr9_g-1.0_l-0.00387.txt',
            # save_path=f'./checkpoints/temp/',
            # log_path='./logging/temp.txt',
            lambda_decay=False,
            verbose=(rank == 0)  # Only print from main process
        )
        
    except Exception as e:
        print(f"Error on rank {rank}: {str(e)}")
        # Properly shutdown even if there's an error
        cleanup()
        raise e
    
    cleanup()

if __name__ == "__main__":
    # Configure environment variables for better multiprocessing
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")
    
    try:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Training failed with error: {str(e)}")