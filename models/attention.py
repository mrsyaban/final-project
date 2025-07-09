import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from typing import List, Tuple, Dict, Union


def create_shape_mask(keypoints: List[Tuple[float, float]], 
                     shape: str, 
                     image_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    if len(keypoints) < 4:
        return mask
    
    if shape == "circle":
        return create_circle_mask(keypoints, image_shape)
    elif shape == "octagon":
        return create_octagon_mask(keypoints, image_shape)
    elif shape in ["rect", "diamond"]:
        return create_polygon_mask(keypoints, image_shape)
    else:
        # Default to polygon for unknown shapes
        return create_polygon_mask(keypoints, image_shape)

def create_circle_mask(keypoints: List[Tuple[float, float]], 
                      image_shape: Tuple[int, int]) -> np.ndarray:
    """Create a circular/elliptical mask fitted to the rectangle defined by keypoints."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    if len(keypoints) < 4:
        return mask
    
    # Convert keypoints to numpy array
    points = np.array(keypoints)
    
    # Calculate center from keypoints
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    center = (center_x, center_y)
    
    # Calculate actual length and width of the rectangle from keypoints
    # Assuming keypoints are ordered as [top-left, top-right, bottom-right, bottom-left]
    # Calculate the actual sides of the rectangle
    
    # Side 1: top edge (top-left to top-right)
    side1_length = np.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
    
    # Side 2: right edge (top-right to bottom-right)  
    side2_length = np.sqrt((points[2][0] - points[1][0])**2 + (points[2][1] - points[1][1])**2)
    
    # Determine which is length (major axis) and which is width (minor axis)
    if side1_length >= side2_length:
        # side1 is length, side2 is width
        length = side1_length
        width = side2_length
        # Calculate angle from the longer side (top edge)
        dx = points[1][0] - points[0][0]  # top-right - top-left
        dy = points[1][1] - points[0][1]
        angle = np.degrees(np.arctan2(dy, dx))
    else:
        # side2 is length, side1 is width
        length = side2_length
        width = side1_length
        # Calculate angle from the longer side (right edge)
        dx = points[2][0] - points[1][0]  # bottom-right - top-right
        dy = points[2][1] - points[1][1]
        angle = np.degrees(np.arctan2(dy, dx))
    
    # Semi-axes for the ellipse
    a = length / 2   # Semi-major axis (half of rectangle length)
    b = width / 2    # Semi-minor axis (half of rectangle width)
    
    # Create ellipse parameters
    axes = (int(a * 2), int(b * 2))  # cv2.ellipse expects full axes lengths
    ellipse = (center, axes, angle)
    
    # Add bounds checking
    if axes[0] <= 0 or axes[1] <= 0:
        return mask
    
    # Draw the ellipse
    cv2.ellipse(mask, ellipse, 1, -1)
    
    return mask

def create_octagon_mask(keypoints: List[Tuple[float, float]], 
                       image_shape: Tuple[int, int]) -> np.ndarray:
    """Create an octagonal mask by connecting all keypoints with convex hull."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    if len(keypoints) < 8:
        return mask
    
    # Convert keypoints to numpy array
    points = np.array(keypoints, dtype=np.float32)
    
    # Create convex hull from all keypoints
    hull = cv2.convexHull(points)
    
    # Convert to integer coordinates for drawing
    pts = hull.astype(np.int32)
    
    # Fill the convex hull polygon
    cv2.fillPoly(mask, [pts], 1)
    
    return mask

def create_polygon_mask(keypoints: List[Tuple[float, float]], 
                       image_shape: Tuple[int, int]) -> np.ndarray:
    """Create a polygon mask for rectangle or diamond shapes."""
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(keypoints, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask


class AttentionModel:
    def __init__(self, image_size=640):
        self.image_size = image_size
        self.device = None

    def generate_attention_map(self, images, labels, random_seed=None):
        """
        Generate attention maps with maximum values at the center of bounding boxes

        Args:
            images: Batch of images [batch_size, channels, height, width]
            labels: Tensor of bounding boxes [N_total_boxes, 6] where each row is
                    [img_idx, class_id, x_center, y_center, width, height]
            random_seed: Optional seed for reproducibility

        Returns:
            attention_maps: Batch of attention maps [batch_size, 1, height, width]
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        batch_size = images.shape[0]
        self.device = images.device

        # Initialize attention maps with low values
        attention_maps = torch.ones(batch_size, 1, self.image_size, self.image_size, device=self.device) * 0.1

        # Group boxes by image index
        for i in range(batch_size):
            # Get boxes for this image
            img_boxes = labels[labels[:, 0] == i]

            if len(img_boxes) == 0:
                continue

            # Create attention peaks at the center of each bounding box
            for box in img_boxes:
                _, class_id, x_center, y_center, width, height = box

                # Convert normalized bbox coordinates to pixel coordinates
                center_x = int(x_center * self.image_size)
                center_y = int(y_center * self.image_size)
                box_width = int(width * self.image_size)
                box_height = int(height * self.image_size)

                # Create a Gaussian-like attention peak at the center
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(self.image_size, device=self.device),
                    torch.arange(self.image_size, device=self.device),
                    indexing='ij'
                )
                
                # Calculate distance from center
                dist_sq = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
                
                # Create Gaussian peak with standard deviation based on box size
                sigma = min(box_width, box_height) / 4
                gaussian_peak = torch.exp(-dist_sq / (2 * sigma ** 2))
                
                # Only apply within the bounding box area
                x1 = max(0, int(center_x - box_width / 2))
                y1 = max(0, int(center_y - box_height / 2))
                x2 = min(self.image_size, int(center_x + box_width / 2))
                y2 = min(self.image_size, int(center_y + box_height / 2))
                
                # Create mask for bounding box area
                bbox_mask = torch.zeros_like(gaussian_peak)
                bbox_mask[y1:y2, x1:x2] = 1.0
                
                # Apply masked Gaussian to attention map
                attention_maps[i, 0] = torch.maximum(
                    attention_maps[i, 0], 
                    gaussian_peak * bbox_mask
                )

        return attention_maps
    
    def generate_patch_mask_v2(self, labels, hottest_points_batch, keypoints_batch, shapes, patch_area_ratio=0.05):
        """
        Generate patch masks based on hottest points and shape masks
        
        Args:
            labels: Tensor of bounding boxes [N_total_boxes, 6] where each row is
                    [img_idx, class_id, x_center, y_center, width, height]
            hottest_points: Tensor of hottest points [N_total_points, 3] where each row is
                        [img_idx, x, y]
            keypoints: Tensor of keypoints [N_total_keypoints, 3] where each row is 
                    [img_idx, x, y]
            shapes: List of shape names for each image (e.g. "circle", "rect", "diamond", "octagon")
            patch_area_ratio: Ratio of the patch area to the bounding box area (default: 0.05)
        
        Returns:
            patch_masks: Batch of patch masks [batch_size, 1, height, width]
        """
        # Get batch size from unique image indices in labels
        # Get batch size from unique image indices in labels
        img_indices = labels[:, 0].unique()
        batch_size = len(img_indices)
        device = labels.device
        
        # Initialize patch masks as zeros
        patch_masks = torch.zeros(batch_size, 1, self.image_size, self.image_size, device=device)
        
        # Create lookup dictionaries for faster access
        hottest_points_dict = {item['image_idx']: item['points'] for item in hottest_points_batch}
        keypoints_dict = {item['image_idx']: item['keypoints'] for item in keypoints_batch}
        
        for i in range(batch_size):
            # Get boxes for this image
            img_boxes = labels[labels[:, 0] == i]
                
            # Get hottest points for this image
            img_points = hottest_points_dict.get(i, torch.zeros((0, 2), device=device))
            
            # Get keypoints for this image - should be [num_boxes, num_keypoints_per_box, 2]
            img_keypoints = keypoints_dict.get(i, torch.zeros((0, 0, 2), device=device))
            
            # Create shape name for this image
            shape_name = shapes[i] if i < len(shapes) else "rect"
            
            current_mask = torch.zeros(self.image_size, self.image_size, device=device)
            
            # Process each bounding box and corresponding hottest point
            for j, box in enumerate(img_boxes):
                # Skip if we don't have enough hottest points
                if j >= len(img_points):
                    continue
                    
                _, class_id, x_center, y_center, width, height = box
                
                
                # Convert to pixel coordinates
                box_width = int(width * self.image_size)
                box_height = int(height * self.image_size)
                
                # Calculate patch size based on bounding box area
                box_area = box_width * box_height
                patch_area = box_area * patch_area_ratio
                patch_size = int(math.sqrt(patch_area))
                patch_size = max(patch_size, 1)  # Minimum size of 1
                
                # Get keypoints for this specific bounding box if available
                box_keypoints = []
                if j < img_keypoints.shape[0]:  # Check if we have keypoints for this box
                    # Convert keypoints to list of tuples for create_shape_mask
                    for kp in img_keypoints[j]:
                        # Convert from normalized to pixel coordinates
                        x = int(kp[0])
                        y = int(kp[1])
                        box_keypoints.append((x, y))
                    
                    # Create shape mask for this specific bounding box
                    if len(box_keypoints) > 0:
                        shape_mask_np = create_shape_mask(
                            box_keypoints,
                            shape_name, 
                            (self.image_size, self.image_size)
                        )
                        shape_mask = torch.from_numpy(shape_mask_np).to(device).float()
                    else:
                        # Default empty mask if no keypoints
                        shape_mask = torch.zeros(self.image_size, self.image_size, device=device)
                else:
                    # Default empty mask if no keypoints for this box
                    shape_mask = torch.zeros(self.image_size, self.image_size, device=device)
                
                # Get hottest point coordinates
                point = img_points[j]
                hot_x = int(point[0])
                hot_y = int(point[1])
                
                # Create square patch centered at hottest point
                patch_x1 = max(0, hot_x - patch_size // 2)
                patch_y1 = max(0, hot_y - patch_size // 2)
                patch_x2 = min(self.image_size, patch_x1 + patch_size)
                patch_y2 = min(self.image_size, patch_y1 + patch_size)
                
                # Check if patch overlaps with shape mask
                patch_mask = torch.zeros_like(current_mask)
                patch_mask[patch_y1:patch_y2, patch_x1:patch_x2] = 1.0

                # Check if patch is not fully contained within shape mask
                if not torch.all((patch_mask * shape_mask) == patch_mask):
                    # import pdb; pdb.set_trace()
                    # Find center and bounds of shape mask
                    shape_points = torch.where(shape_mask > 0)
                    if len(shape_points[0]) > 0 and len(shape_points[1]) > 0:
                        shape_center_y = shape_points[0].float().mean().int()
                        shape_center_x = shape_points[1].float().mean().int()
                        
                        # Find bounds of the shape
                        min_y, max_y = shape_points[0].min().item(), shape_points[0].max().item()
                        min_x, max_x = shape_points[1].min().item(), shape_points[1].max().item()
                        
                        # Try to find best position with minimal movement
                        best_overlap = 0
                        best_pos = (patch_x1, patch_y1)
                        
                        # Grid search for best position
                        for offset_y in range(-patch_size, patch_size+1, max(1, patch_size//8)):
                            for offset_x in range(-patch_size, patch_size+1, max(1, patch_size//8)):
                                # New potential position
                                new_x1 = max(0, hot_x - patch_size//2 + offset_x)
                                new_y1 = max(0, hot_y - patch_size//2 + offset_y)
                                new_x2 = min(self.image_size, new_x1 + patch_size)
                                new_y2 = min(self.image_size, new_y1 + patch_size)
                                
                                # Create test mask
                                test_mask = torch.zeros_like(current_mask)
                                test_mask[new_y1:new_y2, new_x1:new_x2] = 1.0
                                
                                # Calculate overlap with shape
                                overlap = (test_mask * shape_mask).sum()
                                
                                # If fully contained, or better than previous best
                                if torch.all((test_mask * shape_mask) == test_mask):
                                    # Calculate distance from original position
                                    dist = ((new_x1 - patch_x1)**2 + (new_y1 - patch_y1)**2)**0.5
                                    
                                    # If this is the first fully contained position or closer than previous
                                    if best_overlap < test_mask.sum() or (best_overlap == test_mask.sum() and 
                                                                        dist < ((best_pos[0] - patch_x1)**2 + 
                                                                                (best_pos[1] - patch_y1)**2)**0.5):
                                        best_pos = (new_x1, new_y1)
                                        best_overlap = test_mask.sum()
                        
                        # If we found a better position, use it
                        if best_overlap > 0:
                            patch_x1, patch_y1 = best_pos
                            patch_x2 = min(self.image_size, patch_x1 + patch_size)
                            patch_y2 = min(self.image_size, patch_y1 + patch_size)
                            
                            # Update patch mask
                            patch_mask = torch.zeros_like(current_mask)
                            patch_mask[patch_y1:patch_y2, patch_x1:patch_x2] = 1.0
                
                # import pdb; pdb.set_trace()
                # Add to current mask (handle overlapping patches by taking maximum)
                current_mask = torch.maximum(current_mask, patch_mask)
            
            patch_masks[i, 0] = current_mask

        return patch_masks

    def apply_patch(self, images, patch_masks, adv_patches):
        """
        PyTorch-only version using flood fill to find connected components
        """
        batch_size = images.shape[0]
        channels = images.shape[1]
        device = images.device
        
        patched_images = images.clone()
        
        for i in range(batch_size):
            mask = (patch_masks[i, 0] > 0.5).float()
            
            if mask.sum() == 0:
                continue
            
            # Simple connected component finding using erosion/dilation
            # This is a simplified approach - for complex shapes, scipy version is better
            processed_mask = torch.zeros_like(mask)
            component_id = 0
            
            while (mask - processed_mask).sum() > 0:
                component_id += 1
                
                # Find first unprocessed pixel
                remaining = mask - processed_mask
                first_pixel = torch.nonzero(remaining, as_tuple=False)[0]
                
                # Flood fill from this pixel (simplified version)
                current_component = torch.zeros_like(mask)
                current_component[first_pixel[0], first_pixel[1]] = 1
                
                # Expand the component (simple dilation approach)
                for _ in range(max(mask.shape)):  # Max iterations
                    old_sum = current_component.sum()
                    
                    # Dilate and intersect with mask
                    kernel = torch.ones(3, 3, device=device)
                    dilated = F.conv2d(
                        current_component.unsqueeze(0).unsqueeze(0),
                        kernel.unsqueeze(0).unsqueeze(0),
                        padding=1
                    ) > 0
                    current_component = (dilated.squeeze() * mask).float()
                    
                    if current_component.sum() == old_sum:
                        break  # No more expansion
                
                # Process this component
                if current_component.sum() > 0:
                    self._apply_patch_to_component(
                        patched_images[i], current_component, 
                        adv_patches[i], channels, component_id
                    )
                
                processed_mask += current_component

        return patched_images

    def _apply_patch_to_component(self, image, component_mask, patch, channels, component_id):
        """Helper function to apply patch to a single component"""
        hole_indices = torch.nonzero(component_mask, as_tuple=False)
        if hole_indices.numel() == 0:
            return
        
        y_min, x_min = hole_indices.min(dim=0)[0]
        y_max, x_max = hole_indices.max(dim=0)[0]
        
        hole_height = y_max - y_min + 1
        hole_width = x_max - x_min + 1
        
        # Resize patch to fit this hole
        resized_patch = F.interpolate(
            patch.unsqueeze(0),
            size=(hole_height, hole_width),
            mode='bilinear',
            align_corners=False
        )[0]
        
        # Extract hole region mask
        hole_region_mask = component_mask[y_min:y_max+1, x_min:x_max+1]
        
        # Apply patch
        for c in range(channels):
            original_region = image[c, y_min:y_max+1, x_min:x_max+1]
            patch_region = resized_patch[c]
            
            blended_region = (original_region * (1.0 - hole_region_mask) + 
                            patch_region * hole_region_mask)
            image[c, y_min:y_max+1, x_min:x_max+1] = blended_region

    def visualize_attention_and_masks(self, images, attention_maps, patch_masks, patch_sizes, targets, num_samples=4):
        """
        Visualize attention maps, patch masks, and final results
        
        Args:
            images: Batch of images [batch_size, channels, height, width]
            attention_maps: Batch of attention maps [batch_size, 1, height, width]
            patch_masks: Batch of patch masks [batch_size, 1, height, width]
            patch_sizes: List of patch sizes
            targets: Tensor of bounding boxes [N_total_boxes, 6]
            num_samples: Number of samples to visualize
        """
        plt.figure(figsize=(20, 5 * min(num_samples, len(images))))

        for i in range(min(num_samples, len(images))):
            # Original image with bounding boxes
            plt.subplot(num_samples, 4, i * 4 + 1)
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            plt.imshow(img)

            # Draw bounding boxes
            img_boxes = targets[targets[:, 0] == i]
            for box in img_boxes:
                _, class_id, x_center, y_center, width, height = box

                # Convert to pixel coordinates
                x_center = x_center * self.image_size
                y_center = y_center * self.image_size
                box_width = width * self.image_size
                box_height = height * self.image_size

                # Calculate box corners
                x1 = x_center - box_width / 2
                y1 = y_center - box_height / 2

                # Draw rectangle
                rect = plt.Rectangle((x1, y1), box_width, box_height,
                                    fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)

            plt.title(f"Image {i+1} with Boxes")
            plt.axis('off')

            # Attention map
            plt.subplot(num_samples, 4, i * 4 + 2)
            att_map = attention_maps[i, 0].cpu().numpy()
            plt.imshow(att_map, cmap='hot')
            plt.title(f"Attention Map {i+1}")
            plt.axis('off')
            plt.colorbar()

            # Patch mask
            plt.subplot(num_samples, 4, i * 4 + 3)
            mask = patch_masks[i, 0].cpu().numpy()
            plt.imshow(mask, cmap='gray')
            plt.title(f"Patch Mask {i+1}\nMax Size: {patch_sizes[i]}")
            plt.axis('off')

            # Image with mask overlay
            plt.subplot(num_samples, 4, i * 4 + 4)
            img_with_mask = img.copy()
            mask_overlay = mask[..., np.newaxis]  # Add channel dimension
            img_with_mask = img_with_mask * (1 - mask_overlay) + np.array([1.0, 0.0, 0.0]) * mask_overlay
            
            plt.imshow(img_with_mask)
            plt.title(f"Image with Mask Overlay {i+1}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def apply_patch_to_images(self, images, attention_maps, adv_patches, patch_area_ratio=0.1, labels=None):
        """
        Complete pipeline: generate masks and apply patches
        
        Args:
            images: Batch of images [batch_size, channels, height, width]
            attention_maps: Batch of attention maps [batch_size, 1, height, width]
            adv_patches: Batch of adversarial patches
            patch_area_ratio: Ratio of patch area to bounding box area
            labels: Bounding box labels (required for mask generation)
        
        Returns:
            patched_images: Images with patches applied
            patch_masks: Generated patch masks
            patch_sizes: List of patch sizes
        """
        if labels is None:
            raise ValueError("Labels are required for patch mask generation")
        
        # Generate patch masks
        patch_masks, patch_sizes = self.generate_patch_mask(attention_maps, labels, patch_area_ratio)
        
        # Apply patches
        patched_images = self.apply_patch(images, patch_masks, adv_patches)
        
        return patched_images, patch_masks, patch_sizes
    

