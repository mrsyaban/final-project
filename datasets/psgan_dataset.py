import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
import yaml
from roboflow import Roboflow
import json
from torch.utils.data.distributed import DistributedSampler

class PSGANDataset(Dataset):
    def __init__(self, dataset_path, split='train', transform=None, target_transform=None, attentive_dir=None, keypoints_dir=None, relight_dir=None):
        """
        Initialize dataset for YOLOv8 Roboflow format
        
        Args:
            dataset_path: Path to downloaded roboflow dataset
            split: 'train', 'valid', or 'test'
            transform: Image transformations
            target_transform: Label transformations
        """
        self.dataset_path = dataset_path
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load hottest points if provided
        self.hottest_points = {}
        if attentive_dir and os.path.exists(attentive_dir):
            with open(attentive_dir, 'r') as f:
                self.hottest_points = json.load(f)
                # print(f"Loaded hottest points for {len(self.hottest_points)} images")
        
        # Load keypoints data if provided
        self.keypoints_data = {}

        if keypoints_dir and os.path.exists(keypoints_dir):
            print(f"Loading keypoints from {keypoints_dir}...")
            with open(keypoints_dir, 'r') as f:
                self.keypoints_data = json.load(f)
            print(f"Loaded keypoints for {len(self.keypoints_data)} images")
        else:
            print("gadaaa")
        
        # Load relighting parameters if provided
        self.relighting_params = {}
        if relight_dir and os.path.exists(relight_dir):
            with open(relight_dir, 'r') as f:
                self.relighting_params = json.load(f)
                # print(f"Loaded relighting parameters for {len(self.relighting_params)} images")

        # Load dataset configuration
        self.config_path = os.path.join(dataset_path, 'data.yaml')
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get class information
        self.classes = self.config['names']
        self.num_classes = self.config['nc']
        self.idx_to_class = {i: name for i, name in enumerate(self.classes)}
        
        # Get split directory
        split_dir = os.path.join(dataset_path, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory '{split}' not found in {dataset_path}")
        
        # Get image and label directories
        self.images_dir = os.path.join(split_dir, 'images')
        self.labels_dir = os.path.join(split_dir, 'labels')
        
        # Get all image files
        self.image_files = []
        self.label_files = []
        
        if os.path.exists(self.images_dir):
            for img_file in os.listdir(self.images_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(self.images_dir, img_file)
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    label_path = os.path.join(self.labels_dir, label_file)
                    
                    # Only include images that have corresponding label files
                    if os.path.exists(label_path):
                        self.image_files.append(img_path)
                        self.label_files.append(label_path)
                    else:
                        # Include images without labels (empty labels)
                        self.image_files.append(img_path)
                        self.label_files.append(None)
        
        print(f"Loaded {len(self.image_files)} images from {split} split")
        # print(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.label_files[idx]

        # Open the image
        image = Image.open(img_path).convert('RGB')

        # Read the labels
        labels = []
        if label_path and os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line:  # Skip empty lines
                        # Parse class and bounding box information
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id, x_center, y_center, width, height = map(float, parts[:5])
                            labels.append([class_id, x_center, y_center, width, height])

        # Convert to tensor (handle empty labels)
        if labels:
            labels = torch.tensor(labels, dtype=torch.float32)
        else:
            # Empty tensor for images without annotations
            labels = torch.zeros((0, 5), dtype=torch.float32)


        # Get keypoints if available
        keypoints, shapes = self._get_keypoints(img_path)

        # Get hottest points for this image
        hotttest_points = self._get_hottestpoints(img_path)

         # Get relighting coefficients for this image
        relighting_coeffs = self._get_relighting_coeffs(img_path)


        if self.transform:
            image = self.transform(image)

        filename = os.path.basename(img_path)

        return image, labels, filename, keypoints, shapes[0], hotttest_points, relighting_coeffs

    def get_original_image(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        return image, img_path
    
    def _get_relighting_coeffs(self, img_path):
        """Get relighting coefficients (alpha and beta) for specific image"""
        relighting_coeffs = []
        filename = os.path.basename(img_path)
        key = f"{self.split}/{filename}"
        
        if key in self.relighting_params and self.relighting_params[key]:
            # Extract alpha and beta values from each entry
            for params in self.relighting_params[key]:
                if "alpha" in params and "beta" in params:
                    relighting_coeffs.append([params["alpha"], params["beta"]])
        
        # Convert to tensor
        if relighting_coeffs:
            relighting_coeffs = torch.tensor(relighting_coeffs, dtype=torch.float32)
        else:
            # Return empty tensor if no coefficients found
            relighting_coeffs = torch.zeros((0, 2), dtype=torch.float32)
        
        return relighting_coeffs
        
    def _get_hottestpoints(self, img_path):
        hottest_points = []
        filename = os.path.basename(img_path)
        key = f"{self.split}/{filename}"
        
        if key in self.hottest_points and self.hottest_points[key]:
            # Extract just the hottest_point coordinates from each point
            points_data = self.hottest_points[key]
            for point_data in points_data:
                if "hottest_point" in point_data:
                    hottest_points.append(point_data["hottest_point"])
            
        # Convert hottest points to tensor
        if hottest_points:
            hottest_points = torch.tensor(hottest_points, dtype=torch.float32)
        else:
            hottest_points = torch.zeros((0, 2), dtype=torch.float32)
        return hottest_points
    
    def _get_keypoints(self, img_path):
        """Get keypoints for specific image"""
        # Create key for keypoints lookup
        img_filename = os.path.basename(img_path)
        keypoints_key = f"{self.split}/{img_filename}"
        
        if keypoints_key in self.keypoints_data:
            keypoints_list = self.keypoints_data[keypoints_key]
            
            # Convert to tensor format
            all_keypoints = []
            shapes = []
            for kp_data in keypoints_list:
                if 'keypoints' in kp_data and kp_data['keypoints']:
                    # Flatten keypoints and convert to tensor
                    kp_tensor = torch.tensor(kp_data['keypoints'], dtype=torch.float32)
                    all_keypoints.append(kp_tensor)

                    shape = kp_data['shape']
                    shapes.append(shape)
            
            if all_keypoints:
                # Stack all keypoints for this image
                return torch.stack(all_keypoints, dim=0), shapes
        
        # Return empty tensor if no keypoints found
        return torch.zeros((0, 2), dtype=torch.float32), []  # Adjust shape based on your keypoints format


def check_dataset_exists(project_name, version, base_path="./"):
    """
    Check if dataset already exists in Colab environment
    
    Args:
        project_name: Name of the project
        version: Dataset version
        base_path: Base path to check (default: /content for Colab)
    
    Returns:
        Path to dataset if exists, None otherwise
    """
    dataset_name = f"{project_name}-{version}"
    dataset_path = os.path.join(base_path, dataset_name)
    
    # Check if directory exists and has required files
    if os.path.exists(dataset_path):
        data_yaml_path = os.path.join(dataset_path, 'data.yaml')
        if os.path.exists(data_yaml_path):
            print(f"Dataset found at: {dataset_path}")
            return dataset_path
        else:
            print(f"Dataset directory exists but missing data.yaml: {dataset_path}")
    
    return None

def download_roboflow_dataset(api_key, workspace, project, version, format_type="yolov8"):
    """
    Download dataset from Roboflow
    
    Args:
        api_key: Your Roboflow API key
        workspace: Workspace name
        project: Project name  
        version: Dataset version
        format_type: Dataset format (default: "yolov8")
    
    Returns:
        Path to downloaded dataset
    """
    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project).version(version).download(format_type)
    return dataset.location

def get_or_download_dataset(download_params, base_path="./"):
    """
    Get dataset path, checking if it exists first before downloading
    
    Args:
        download_params: Dict with keys: api_key, workspace, project, version
        base_path: Base path to check for existing dataset
    
    Returns:
        Path to dataset
    """
    project = download_params['project']
    version = download_params['version']
    
    # First check if dataset already exists
    existing_path = check_dataset_exists(project, version, base_path)
    if existing_path:
        return existing_path
    
    # If not found, download it
    print("Dataset not found locally. Downloading from Roboflow...")
    dataset_path = download_roboflow_dataset(**download_params)
    print(f"Dataset downloaded to: {dataset_path}")
    
    return dataset_path




def yolo_collate_fn(batch):
    images = []
    targets = []
    filenames = []
    keypoints_list = []
    shapes_list = []
    hottest_points_list = []
    relighting_coeffs_list = []

    for i, (img, boxes, filename, keypoints, shape, hottest_points, relighting_coeffs) in enumerate(batch):
        images.append(img)
        filenames.append(filename)
        shapes_list.append(shape)
        
        
        # Handle bounding boxes
        if len(boxes) > 0:
            # Add image index for tracking which image the boxes belong to
            img_idx = torch.full((boxes.shape[0], 1), i, dtype=torch.float32)
            target = torch.cat([img_idx, boxes], dim=1)  # [num_boxes, 6]
            targets.append(target)
            
            # Handle keypoints
            if keypoints.shape[0] > 0:
                # Keypoints should be [num_boxes, num_keypoints_per_box, 2]
                # We need to keep the hierarchical structure
                img_idx_kp = i  # Just use the integer index
                # Store the image index separately with the keypoints
                keypoints_list.append((img_idx_kp, keypoints))
            
            # Handle hottest points
            if hottest_points.shape[0] > 0:
                # Hottest points should be [num_boxes, 2]
                # Instead of concatenating, associate with image index
                img_idx_hp = i
                hottest_points_list.append((img_idx_hp, hottest_points))
            
            # Handle relighting coefficients
            if relighting_coeffs.shape[0] > 0:
                # Relighting coeffs should be [num_boxes, 2]
                # Instead of concatenating, associate with image index
                img_idx_rc = i
                relighting_coeffs_list.append((img_idx_rc, relighting_coeffs))

    images = torch.stack(images, dim=0)  # [B, 3, H, W]
    
    # Process targets
    if targets:
        targets = torch.cat(targets, dim=0)  # [N_total_boxes, 6]
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)
    
            # Process keypoints
    if keypoints_list:
        # Create structured batch with proper dimensions
        keypoints_batch = []
        for img_idx, keypoints in keypoints_list:
            keypoints_batch.append({
                'image_idx': img_idx,
                'keypoints': keypoints  # Shape: [num_keypoints_per_box, 2]
            })
    else:
        keypoints_batch = []

    # Process hottest points
    if hottest_points_list:
        hottest_points_batch = []
        for img_idx, points in hottest_points_list:
            hottest_points_batch.append({
                'image_idx': img_idx,
                'points': points  # Shape: [num_points, 2]
            })
    else:
        hottest_points_batch = []

    # Process relighting coefficients
    if relighting_coeffs_list:
        relighting_coeffs_batch = []
        for img_idx, coeffs in relighting_coeffs_list:
            relighting_coeffs_batch.append({
                'image_idx': img_idx,
                'coeffs': coeffs  # Shape: [num_coeffs, 2]
            })
    else:
        relighting_coeffs_batch = []      

    
    return (images, targets, filenames, keypoints_batch, shapes_list, 
            hottest_points_batch, relighting_coeffs_batch)


def get_psgan_data(
    dataset_path="./traffic-signs-id-2",
    img_size=640, 
    split='train',
    batch_size=32,
    num_workers=8,
    download_params=None,
    attentive_dir=None,
    keypoints_dir=None,
    relight_dir=None,
    distributed=False, 
    rank=0, 
    world_size=1
):
    """
    Get PSGAN dataloader for YOLOv8 Roboflow dataset
    
    Args:
        dataset_path: Path to dataset or None if downloading/auto-detecting
        img_size: Image size for resizing
        split: Dataset split ('train', 'valid', 'test')
        batch_size: Batch size
        num_workers: Number of worker processes
        download_params: Dict with keys: api_key, workspace, project, version
        attentive_dir: Path to JSON file with hottest points
        keypoints_dir: Path to JSON file with keypoints
        relight_dir: Path to JSON file with relighting parameters
    
    Returns:
        dataloader, dataset
    """
    
    # Get dataset path (check existing or download)
    if dataset_path is None and download_params:
        dataset_path = get_or_download_dataset(download_params)
    elif dataset_path is None:
        raise ValueError("Either dataset_path or download_params must be provided")
    
    # Setup Dataset
    psgan_transform = T.Compose([
        T.ToTensor(),
        T.Resize((img_size, img_size)),
        T.Normalize(mean=[0.5, 0.5, 0.5],
               std=[0.5, 0.5, 0.5])
    ])
    psgan_dataset = PSGANDataset(
        dataset_path=dataset_path, 
        split=split,
        transform=psgan_transform,
        attentive_dir=attentive_dir,
        keypoints_dir=keypoints_dir,
        relight_dir=relight_dir
    )
    
    if distributed:
        sampler = DistributedSampler(
            psgan_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        psgan_dataloader = DataLoader(
            psgan_dataset, 
            batch_size=batch_size, 
            sampler=sampler,  # Use sampler instead of shuffle
            shuffle=False,    # Don't use shuffle when using sampler
            collate_fn=yolo_collate_fn, 
            num_workers=num_workers,
            pin_memory=False
        )
    else:
        psgan_dataloader = DataLoader(
            psgan_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=yolo_collate_fn, 
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

    return psgan_dataloader, psgan_dataset