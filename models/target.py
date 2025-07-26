import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import torchvision
import os
import matplotlib.patches as patches


# def save_debug_images(images, all_boxes, all_confs, save_dir="debug_outputs_fianl_9dg_0.00388_patch15", prefix="batch"):
def save_debug_images(images, all_boxes, all_confs, save_dir="debug/finalr9_g-1.0_l-0.00387", prefix="batch"):
    """
    Save a grid of images with the max-confidence predicted bounding box for each label.

    Args:
        images: Tensor [B, C, H, W] in [0,1] range
        all_boxes: List of lists of [4] arrays (xyxy format per label per image, or None)
        all_confs: List of lists of floats (confidence per label per image, or None)
        save_dir: Directory to save images
        prefix: Prefix for saved file name
    """
    os.makedirs(save_dir, exist_ok=True)
    images = images.detach().cpu()
    B = images.shape[0]
    cols = 4
    rows = (B + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    axes = axes.flatten()

    for i in range(B):
        img = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        # Draw all max-confidence boxes for this image (one per label)
        for box, conf in zip(all_boxes[i], all_confs[i]):
            if box is not None:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
                axes[i].add_patch(rect)
                axes[i].text(x1, y1, f"{conf:.2f}", color='yellow', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
    # Hide unused axes
    for j in range(B, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{prefix}_debug.png")
    plt.savefig(save_path)
    plt.close(fig)

class YOLOv8TargetModel:
    """
    YOLOv8 model used as a target for adversarial patch training
    """
    def __init__(self, model_path=None, confidence_threshold=0.25, device=None):
        """
        Initialize the YOLOv8 model

        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Detection confidence threshold
            device: Device to run inference on
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold

        if model_path is None:
            raise ValueError("model_path is required and cannot be None")
        
        self.model = YOLO(model_path)
        self.model.model.to(self.device) 


    def predict(self, images, return_results=False):
        """
        Run YOLOv8 inference on images

        Args:
            images: Batch of images [batch_size, channels, height, width]
            return_results: Whether to return full results or just detection counts

        Returns:
            If return_results=True: Full detection results
            If return_results=False: Number of detections per image
        """
        # Convert tensor to list of PIL Images
        if isinstance(images, torch.Tensor):
            from PIL import Image
            images_list = []
            for i in range(images.shape[0]):
                # Get single image [C, H, W]
                img = images[i]
                # Convert to [H, W, C] and scale to 0-255
                img_np = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                # Convert to PIL Image
                pil_img = Image.fromarray(img_np)
                images_list.append(pil_img)
        else:
            images_list = images

        # Run inference - pass each image individually
        results = []
        for img in images_list:
            result = self.model(img, conf=self.confidence_threshold, verbose=False)
            results.extend(result)  # result is a list, so extend instead of append

        if return_results:
            return results
        else:
            # Return the number of detections per image as a simple metric
            detection_counts = [len(result.boxes) for result in results]
            return torch.tensor(detection_counts, device=self.device)

    def _get_confidences_with_gradients(self, image_tensor, label):
        """
        Extract confidence scores while maintaining gradients.
        This is the tricky part - we need to access YOLO's internal forward pass.
        
        Args:
            image_tensor: Single image tensor [1, C, H, W]
            
        Returns:
            Tensor of confidence scores with gradients
        """
        class_id = int(label[1].item())  

        # Convert tensor to format expected by YOLO model
        # Scale from [0,1] to [0,255] if needed
        if image_tensor.min() <= 0.0:
            image_input = (image_tensor + 1) / 2.0
        else:
            image_input = image_tensor
        
        # Ensure the tensor is on the right device and has the right dtype
        image_input = image_input.to(self.device).float()
        
        with torch.set_grad_enabled(True):
            raw_preds = self.model.model(image_input)
            
            # Extract first element from tuple: shape [1, 25, 8400]
            predictions = raw_preds[0]
            
            # Extract class confidences (skip first 4 bbox coordinates)
            # Shape: [1, 21, 8400] - 21 classes
            class_confidences = predictions[:, 4:, :]
            
            # Get confidence for the true label class
            # label should be the class index (0-20 for 21 classes)
            true_class_confidences = class_confidences[:, class_id, :]  # Shape: [1, 8400]
            
            # Apply sigmoid to convert to probabilities
            true_class_confidences = torch.sigmoid(true_class_confidences)
            
            # Flatten and return max confidence
            confidences = true_class_confidences.flatten()
            max_confidence = torch.max(confidences)
            
            return max_confidence

    def compute_adv_loss(self, images, labels, iou_thresh=0.5, conf_thresh=0.1, debug=True, debug_prefix="batch"):
        """
        Computes adversarial loss based on max confidence of detections
        overlapping ground truth objects.
        """
        images = (images + 1.0) / 2.0
        batch_losses = []

        # debug
        all_boxes = []
        all_confs = []

        for i in range(images.shape[0]):
            single_image = images[i:i+1]
            image_labels = labels[labels[:,0] == i]
            gt_class = int(image_labels[0][1].item())
            
            raw_preds = self.model.model(single_image) # shape: tuple((1, 22, 1344), [(1, 82, 32, 32), (1, 82, 16, 16), (1, 82, 8, 8)])
            results = raw_preds[0][0].permute(1, 0) # shape: (1344, 22)

            # filter grid cells that have probability on target class > threshold
            class_results = results[results[:, 4+gt_class] >= conf_thresh] # class 0 will be in 5th

            fallback = results[:, 4+gt_class].max()
            pred_confs = class_results[:, 4+gt_class]
            # pred_confs = class_results[:, 4+gt_class]

            confidences = []

            # debug
            boxes_per_image = []
            confs_per_image = []

            for gt in image_labels:
                gt_box = gt[2:].unsqueeze(0)  # [1, 4]

                class_boxes = class_results[:, :4]

                if len(class_boxes) == 0:
                    confidences.append(fallback)
                    boxes_per_image.append(None)
                    confs_per_image.append(None)
                    continue

                
                gt_box_xyxy = torchvision.ops.box_convert(gt_box, in_fmt='cxcywh', out_fmt='xyxy')
                gt_box_xyxy = gt_box_xyxy * single_image.shape[2]

                class_boxes = torchvision.ops.box_convert(class_boxes, in_fmt='cxcywh', out_fmt='xyxy')
                
                # Compute IoU between predicted boxes and gt_box
                ious = torchvision.ops.box_iou(gt_box_xyxy, class_boxes)
                matched = ious > iou_thresh
                
                if matched.any():
                    matched_scores = pred_confs[matched[0]]
                    confidence = matched_scores.max()

                    # debug
                    matched_boxes = class_boxes[matched[0]]
                    max_idx = torch.argmax(matched_scores)
                    max_box = matched_boxes[max_idx].detach().cpu().numpy()
                    boxes_per_image.append(max_box)
                    confs_per_image.append(confidence.item())
                else:
                    confidence = fallback

                    # debug
                    boxes_per_image.append(None)
                    confs_per_image.append(None)

                confidences.append(confidence)

            if confidences:
                image_loss = torch.stack(confidences).mean()
            else:
                image_loss = fallback

            batch_losses.append(image_loss)

            # debug
            all_boxes.append(boxes_per_image)
            all_confs.append(confs_per_image)
        
        if debug:
            save_debug_images(images, all_boxes, all_confs, prefix=debug_prefix)

        return torch.stack(batch_losses).mean()
        

    # def compute_adv_loss(self, images, labels, iou_thresh=0.5):
    #     """
    #     Computes adversarial loss based on max confidence of detections
    #     overlapping ground truth objects.
    #     """
    #     images = (images + 1.0) / 2.0
    #     batch_losses = []

    #     for i in range(images.shape[0]):
    #         single_image = images[i:i+1]

    #         results = self.model(single_image, verbose=False)
    #         boxes = results[0].boxes

    #         # Filter detections for each ground truth object
    #         image_labels = labels[labels[:,0] == i]

    #         confidences = []

    #         for gt in image_labels:
    #             gt_class = int(gt[1].item())
    #             gt_box = gt[2:].unsqueeze(0)  # [1, 4]

    #             class_mask = boxes.cls == gt_class
    #             class_boxes = boxes[class_mask]

    #             if len(class_boxes) == 0:
    #                 confidences.append(fallback)
    #                 continue

    #             gt_box_xyxy = torchvision.ops.box_convert(gt_box, in_fmt='cxcywh', out_fmt='xyxy')
                
    #             # Compute IoU between predicted boxes and gt_box
    #             ious = torchvision.ops.box_iou(gt_box_xyxy, class_boxes.xyxyn)
    #             matched = ious > iou_thresh

    #             if matched.any():
    #                 matched_scores = class_boxes.conf[matched[0]]
    #                 confidence = matched_scores.max()
    #             else:
    #                 confidence = fallback

    #             confidences.append(confidence)

    #         if confidences:
    #             image_loss = torch.stack(confidences).mean()
    #         else:
    #             image_loss = fallback

    #         batch_losses.append(image_loss)

    #     if batch_losses:
    #         return torch.stack(batch_losses).mean()
    #     else:
    #         return fallback


    def compute_loss(self, images, labels, confidence_threshold=1e-6):
        """
        Enhanced loss computation with multiple options.
        
        Args:
            images: Batch of images tensor [B, C, H, W]
            strategy: 'max' (use max confidence), 'all' (use all detections), 'top_k' (use top-k)
            aggregation: 'mean' or 'sum' (how to aggregate multiple confidences when strategy != 'max')
            top_k: Number of top confidences to use (only used when strategy='top_k')
            min_confidence: Minimum confidence threshold to consider a detection
            reduction: 'mean', 'sum', or 'none'
            
        Returns:
            Adversarial loss tensor
        """
        images = (images + 1.0) / 2.0

        batch_losses = []

        if labels.shape[0] == 0:
        # No objects in entire batch, return zero loss
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for i in range(images.shape[0]):
            single_image = images[i:i+1]

            image_labels = labels[labels[:,0] == i]

            if image_labels.shape[0] == 0:
                # No objects in this image, contribute zero to loss
                batch_losses.append(torch.tensor(0.0, device=self.device, requires_grad=True))
                continue
            
            image_confidences = []
            for j in range(image_labels.shape[0]):
                single_label = image_labels[j]  # [6] - [img_idx, class_id, x, y, w, h]
                confidence = self._get_confidences_with_gradients(single_image, single_label)
                
                # Apply minimum confidence filter
                confidence = torch.where(
                    confidence > confidence_threshold,
                    confidence,
                    torch.zeros_like(confidence)
                )
                
                image_confidences.append(confidence)
            
            if image_confidences:
                # Take mean of all object confidences in this image
                image_loss = torch.stack(image_confidences).mean()
            else:
                image_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            batch_losses.append(image_loss)
            
        
        if batch_losses:
            losses = torch.stack(batch_losses)
            return losses.mean()
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)