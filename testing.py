import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from psgan import PSGAN
from datasets.psgan_dataset import get_psgan_data
from datasets.seed_patch_dataset import get_seed_patch_data
import json
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class PSGANInference:
    def __init__(self, checkpoint_dir, patch_size=56, image_size=640, device=None):
        """
        Initialize PSGAN for inference with multiple epoch evaluation

        Args:
            checkpoint_dir: Directory containing checkpoint files
            patch_size: Size of the patches
            image_size: Size of the images
            device: Device to run inference on
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.image_size = image_size
        self.checkpoint_dir = checkpoint_dir

        # Initialize PSGAN (will load weights later)
        self.psgan = PSGAN(
            patch_size=patch_size,
            image_size=image_size,
            device=self.device
        )

        # Set models to evaluation mode
        self.psgan.generator.eval()
        self.psgan.discriminator.eval()

        # Store latest epoch data for visualization
        self.latest_epoch_data = None

    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.psgan.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.psgan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        return epoch

    def get_available_epochs(self):
        """Get list of available epoch checkpoints"""
        epochs = []
        if not os.path.exists(self.checkpoint_dir):
            print(f"Checkpoint directory {self.checkpoint_dir} does not exist")
            return epochs

        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith('psgan_epoch_') and filename.endswith('.pth'):
                try:
                    epoch_num = int(filename.split('_')[-1].split('.')[0])
                    epochs.append(epoch_num)
                except ValueError:
                    continue

        return sorted(epochs)

    def apply_seed_patches(self, images, labels, seed_patches, patch_area_ratio=0.05):
        """Apply seed patches directly to images without using the generator"""
        with torch.no_grad():
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device) if labels is not None else torch.empty(0, 6).to(self.device)
            seed_patches = seed_patches.to(self.device)

            # Ensure same batch size
            batch_size = min(images.size(0), seed_patches.size(0))
            images = images[:batch_size]
            seed_patches = seed_patches[:batch_size]

            # Reindex labels for current batch
            if len(labels) > 0:
                mask = labels[:, 0] < batch_size
                labels = labels[mask]

            # Preprocess seed patches if needed
            if seed_patches.min() >= 0 and seed_patches.max() <= 1:
                seed_patches = seed_patches * 2 - 1

            # Generate attention maps using new interface
            attention_maps = self.psgan.attention_model.generate_attention_map(images, labels)

            # Generate patch masks based on attention maps and labels
            patch_masks, patch_sizes = self.psgan.attention_model.generate_patch_mask(
                attention_maps, labels, patch_area_ratio=patch_area_ratio
            )

            # Apply seed patches directly to images (without generator)
            # Convert seed patches back to [0,1] range
            seed_patches_vis = seed_patches.clone()
            if seed_patches_vis.min() < 0:
                seed_patches_vis = (seed_patches_vis + 1) / 2

            # Apply patches to images using attention model
            patched_images = self.psgan.attention_model.apply_patch(images, patch_masks, seed_patches)

            return {
                'original_images': images,
                'seed_patches': seed_patches,
                'applied_patches': seed_patches_vis,  # For visualization
                'attention_maps': attention_maps,
                'patch_masks': patch_masks,
                'patch_sizes': patch_sizes,
                'patched_images': patched_images,
                'labels': labels
            }

    def generate_adversarial_images_detailed(self, images, labels, seed_patches, patch_area_ratio=0.05):
        """Generate adversarial images and return detailed information for visualization"""
        with torch.no_grad():
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device) if labels is not None else torch.empty(0, 6).to(self.device)
            seed_patches = seed_patches.to(self.device)

            # Ensure same batch size
            batch_size = min(images.size(0), seed_patches.size(0))
            images = images[:batch_size]
            seed_patches = seed_patches[:batch_size]

            # Reindex labels for current batch
            if len(labels) > 0:
                mask = labels[:, 0] < batch_size
                labels = labels[mask]

            # Preprocess seed patches if needed
            if seed_patches.min() >= 0 and seed_patches.max() <= 1:
                seed_patches = seed_patches * 2 - 1

            # Generate adversarial patches
            adv_patches = self.psgan.generator(seed_patches)

            # Generate attention maps using new interface
            attention_maps = self.psgan.attention_model.generate_attention_map(images, labels)

            # Generate patch masks based on attention maps and labels
            patch_masks, patch_sizes = self.psgan.attention_model.generate_patch_mask(
                attention_maps, labels, patch_area_ratio=patch_area_ratio
            )

            # Apply patches to images using new interface
            patched_images = self.psgan.attention_model.apply_patch(images, patch_masks, adv_patches)

            # Convert adv_patches back to [0,1] range for visualization
            adv_patches_vis = adv_patches.clone()
            if adv_patches_vis.min() < 0:
                adv_patches_vis = (adv_patches_vis + 1) / 2

            return {
                'original_images': images,
                'seed_patches': seed_patches,
                'adv_patches': adv_patches_vis,  # For visualization
                'attention_maps': attention_maps,
                'patch_masks': patch_masks,
                'patch_sizes': patch_sizes,
                'patched_images': patched_images,
                'labels': labels
            }

    def generate_adversarial_images(self, images, labels, seed_patches, patch_area_ratio=0.05):
        """Generate adversarial images using the current loaded generator"""
        result = self.generate_adversarial_images_detailed(images, labels, seed_patches, patch_area_ratio)
        return result['patched_images']

    def visualize_latest_adversarial_samples(self, yolo_model, num_samples=6, save_path=None):
        """
        Visualize sample adversarial images from the latest epoch

        Args:
            yolo_model: YOLO model for getting detection results
            num_samples: Number of samples to visualize
            save_path: Path to save the visualization
        """
        if self.latest_epoch_data is None:
            print("No latest epoch data available for visualization!")
            return

        data = self.latest_epoch_data
        num_samples = min(num_samples, data['original_images'].shape[0])

        input_yolo_original = (data['original_images'] + 1.0) / 2.0
        input_yolo_adv = (data['patched_images'] + 1.0) / 2.0
        # Get YOLO detections for comparison
        with torch.no_grad():
            original_results = yolo_model.predict(input_yolo_original[:num_samples], return_results=True)
            adversarial_results = yolo_model.predict(input_yolo_adv[:num_samples], return_results=True)

        # Create visualization
        fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Helper function to safely convert tensors
            def to_numpy_safe(tensor):
                if hasattr(tensor, 'cpu'):
                    return tensor.cpu().numpy()
                return tensor

            # Original image with detections
            ax = axes[i, 0]
            orig_img = to_numpy_safe(data['original_images'][i].permute(1, 2, 0))
            
            orig_img = (orig_img + 1.0) / 2.0
            ax.imshow(orig_img)

            # Draw original detections
            orig_result = original_results[i]
            if orig_result.boxes is not None and len(orig_result.boxes) > 0:
                boxes = orig_result.boxes.xyxy.cpu().numpy()
                scores = orig_result.boxes.conf.cpu().numpy()
                classes = orig_result.boxes.cls.cpu().numpy()

                for box, score, cls in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                       fill=False, edgecolor='green', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, f'{int(cls)}: {score:.2f}',
                           color='green', fontsize=8, fontweight='bold')

            ax.set_title(f'Original Image {i+1}\nDetections: {len(orig_result.boxes) if orig_result.boxes else 0}')
            ax.axis('off')

            # Seed patch
            ax = axes[i, 1]
            seed_patch = to_numpy_safe(data['seed_patches'][i].permute(1, 2, 0))
            # Convert from [-1,1] to [0,1] if needed
            if seed_patch.min() < 0:
                seed_patch = (seed_patch + 1) / 2
            seed_patch = np.clip(seed_patch, 0, 1)

            ax.imshow(seed_patch)
            ax.set_title(f'Seed Patch {i+1}')
            ax.axis('off')

            # Adversarial patch
            ax = axes[i, 2]
            adv_patch = to_numpy_safe(data['adv_patches'][i].permute(1, 2, 0))
            adv_patch = np.clip(adv_patch, 0, 1)

            ax.imshow(adv_patch)
            ax.set_title(f'Adversarial Patch {i+1}')
            ax.axis('off')

            # Attention map with patch mask overlay
            ax = axes[i, 3]
            att_map = to_numpy_safe(data['attention_maps'][i, 0])
            patch_mask = to_numpy_safe(data['patch_masks'][i, 0])

            # Show original image with attention overlay
            ax.imshow(orig_img)
            im = ax.imshow(att_map, alpha=0.4, cmap='hot')

            # Overlay patch mask in blue
            patch_mask_colored = np.zeros((*patch_mask.shape, 4))
            patch_mask_colored[:, :, 2] = patch_mask  # Blue channel
            patch_mask_colored[:, :, 3] = patch_mask * 0.6  # Alpha channel
            ax.imshow(patch_mask_colored)

            patch_size = data['patch_sizes'][i]
            ax.set_title(f'Attention + Patch Mask {i+1}\nMax Patch Size: {patch_size}')
            ax.axis('off')

            # Adversarial image with detections
            ax = axes[i, 4]
            adv_img = to_numpy_safe(data['patched_images'][i].permute(1, 2, 0))
            adv_img = (adv_img + 1.0) / 2.0
            ax.imshow(adv_img)

            # Draw adversarial detections
            adv_result = adversarial_results[i]
            if adv_result.boxes is not None and len(adv_result.boxes) > 0:
                boxes = adv_result.boxes.xyxy.cpu().numpy()
                scores = adv_result.boxes.conf.cpu().numpy()
                classes = adv_result.boxes.cls.cpu().numpy()

                for box, score, cls in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                       fill=False, edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, f'{int(cls)}: {score:.2f}',
                           color='red', fontsize=8, fontweight='bold')

            # Calculate detection reduction
            orig_count = len(orig_result.boxes) if orig_result.boxes else 0
            adv_count = len(adv_result.boxes) if adv_result.boxes else 0
            reduction = orig_count - adv_count

            ax.set_title(f'Adversarial Image {i+1}\nDetections: {adv_count} (↓{reduction})')
            ax.axis('off')

        # Add overall title and legend
        fig.suptitle(f'Adversarial Sample Results - Latest Epoch\n'
                    f'Green boxes: Original detections, Red boxes: Adversarial detections, Blue overlay: Patch masks',
                    fontsize=16, y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Adversarial samples visualization saved to {save_path}")

        plt.show()

        # Print detection summary
        print("\n=== Detection Summary ===")
        orig_total = sum(len(result.boxes) if result.boxes else 0 for result in original_results)
        adv_total = sum(len(result.boxes) if result.boxes else 0 for result in adversarial_results)

        print(f"Total detections on original images: {orig_total}")
        print(f"Total detections on adversarial images: {adv_total}")
        print(f"Total detection reduction: {orig_total - adv_total}")
        print(f"Detection reduction rate: {((orig_total - adv_total) / max(orig_total, 1)) * 100:.1f}%")

        # Per-image breakdown
        for i in range(num_samples):
            orig_count = len(original_results[i].boxes) if original_results[i].boxes else 0
            adv_count = len(adversarial_results[i].boxes) if adversarial_results[i].boxes else 0
            print(f"Image {i+1}: {orig_count} → {adv_count} detections")

    def evaluate_yolo_model(self, yolo_model, images, labels, dataset_config):
        """
        Evaluate YOLO model on given images and return mAP using torchmetrics
        """
        if images.min() < 0.0:
            images = (images + 1.0) / 2.0

        # Get YOLO predictions
        results = yolo_model.predict(images, return_results=True)

        # Prepare predictions and targets for torchmetrics
        preds = []
        targets = []
        for i, result in enumerate(results):
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu()
                scores = result.boxes.conf.cpu()
                classes = result.boxes.cls.cpu().int()
                preds.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': classes
                })
            else:
                preds.append({
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty((0,)),
                    'labels': torch.empty((0,), dtype=torch.int64)
                })

            # Prepare ground truth for this image
            if len(labels) > 0:
                img_labels = labels[labels[:, 0] == i]
                if len(img_labels) > 0:
                    gt_boxes = []
                    gt_classes = []
                    for label in img_labels:
                        _, cls, x_center, y_center, width, height = label
                        x_center = x_center * self.image_size
                        y_center = y_center * self.image_size
                        width = width * self.image_size
                        height = height * self.image_size
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_classes.append(int(cls))
                    targets.append({
                        'boxes': torch.tensor(gt_boxes),
                        'labels': torch.tensor(gt_classes, dtype=torch.int64)
                    })
                else:
                    targets.append({
                        'boxes': torch.empty((0, 4)),
                        'labels': torch.empty((0,), dtype=torch.int64)
                    })
            else:
                targets.append({
                    'boxes': torch.empty((0, 4)),
                    'labels': torch.empty((0,), dtype=torch.int64)
                })

        # Compute metrics using torchmetrics
        metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])
        metric.update(preds, targets)
        results = metric.compute()

        map_score = float(results['map'].item())
        return {
            'map': map_score
        }

    def test_against_yolo_epochs(self, dataset_path, dataset_config, yolo_model,
                                max_epochs=None, max_batches=20, save_results=True):
        """
        Test adversarial effectiveness across multiple epochs using Roboflow dataset
        """
        available_epochs = self.get_available_epochs()
        if not available_epochs:
            print("No checkpoint files found!")
            return {}

        if max_epochs is not None:
            available_epochs = available_epochs[:max_epochs]

        print(f"Evaluating {len(available_epochs)} epochs: {available_epochs}")

        # Load validation dataset
        print("Loading testing dataset...")
        test_dataloader, test_dataset = get_psgan_data(
            dataset_path=dataset_path,
            img_size=self.image_size,
            split='test',  # Use validation split for evaluation
            batch_size=4,  # Smaller batch size for evaluation
            num_workers=2,
            download_params=dataset_config
        )

        # Load seed patches
        seed_patch_dataloader, seed_patch_dataset = get_seed_patch_data(
            patch_size=self.patch_size,
            batch_size=4,
            num_workers=2
        )

        # Collect evaluation data
        print("Collecting evaluation data...")
        all_images = []
        all_labels = []
        all_seed_patches = []

        val_iter = iter(test_dataloader)
        seed_iter = iter(seed_patch_dataloader)

        batch_count = 0
        try:
            while batch_count < max_batches:
                try:
                    images, labels = next(val_iter)
                    seed_patches = next(seed_iter)

                    batch_size = min(images.size(0), seed_patches.size(0))
                    images = images[:batch_size]
                    seed_patches = seed_patches[:batch_size]

                    # Reindex labels for global indexing
                    if len(labels) > 0:
                        mask = labels[:, 0] < batch_size
                        labels = labels[mask]
                        labels[:, 0] = labels[:, 0] + batch_count * batch_size

                    all_images.append(images)
                    all_labels.append(labels)
                    all_seed_patches.append(seed_patches)

                    batch_count += 1

                except StopIteration:
                    break

        except Exception as e:
            print(f"Error collecting data: {e}")

        if not all_images:
            print("No evaluation data collected!")
            return {}
        

        # Concatenate all data
        eval_images = torch.cat(all_images, dim=0)
        eval_labels = torch.cat(all_labels, dim=0) if all_labels and len(all_labels[0]) > 0 else torch.empty(0, 6)
        eval_seed_patches = torch.cat(all_seed_patches, dim=0)

        print(f"Evaluation data: {eval_images.shape[0]} images, {len(eval_labels)} labels")

        # Calculate baseline mAP (original images)
        print("Calculating baseline mAP...")
        baseline_metrics = self.evaluate_yolo_model(yolo_model, eval_images, eval_labels, test_dataset.config)
        baseline_map = baseline_metrics['map']
        print(f"Baseline mAP: {baseline_map:.4f}")

        # Calculate seed patch mAP (before using generator)
        print("Calculating seed patch mAP (without generator)...")
        seed_patch_result = self.apply_seed_patches(eval_images, eval_labels, eval_seed_patches)
        seed_patch_metrics = self.evaluate_yolo_model(yolo_model, seed_patch_result['patched_images'], eval_labels, test_dataset.config)
        seed_patch_map = seed_patch_metrics['map']
        print(f"Seed patch mAP: {seed_patch_map:.4f}")
        print(f"Seed patch reduction: {baseline_map - seed_patch_map:.4f} ({((baseline_map - seed_patch_map) / max(baseline_map, 1e-6)) * 100:.1f}%)")

        # Evaluate each epoch
        epoch_results = {}
        map_scores = []
        epochs_evaluated = []

        for epoch in available_epochs:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'psgan_epoch_{epoch}.pth')

            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint for epoch {epoch} not found, skipping...")
                continue

            print(f"Evaluating epoch {epoch}...")

            try:
                # Load model for this epoch
                self.load_checkpoint(checkpoint_path)

                # Generate adversarial images with detailed info for latest epoch
                if epoch == available_epochs[-1]:  # Latest epoch
                    print("Generating detailed data for latest epoch visualization...")
                    # Use a smaller subset for visualization
                    vis_batch_size = min(8, eval_images.shape[0])
                    self.latest_epoch_data = self.generate_adversarial_images_detailed(
                        eval_images[:vis_batch_size],
                        eval_labels[eval_labels[:, 0] < vis_batch_size] if len(eval_labels) > 0 else torch.empty(0, 6),
                        eval_seed_patches[:vis_batch_size]
                    )

                    # Generate full adversarial dataset for evaluation
                    full_adversarial_images = self.generate_adversarial_images(
                        eval_images, eval_labels, eval_seed_patches
                    )
                else:
                    full_adversarial_images = self.generate_adversarial_images(
                        eval_images, eval_labels, eval_seed_patches
                    )

                # Evaluate on adversarial images
                adv_metrics = self.evaluate_yolo_model(yolo_model, full_adversarial_images, eval_labels, test_dataset.config)
                adv_map = adv_metrics['map']

                # Store results
                epoch_results[epoch] = {
                    'map': adv_map,
                    'map_reduction': baseline_map - adv_map,
                    'map_reduction_percentage': ((baseline_map - adv_map) / max(baseline_map, 1e-6)) * 100
                }

                map_scores.append(adv_map)
                epochs_evaluated.append(epoch)

                print(f"Epoch {epoch}: mAP = {adv_map:.4f} (reduction: {epoch_results[epoch]['map_reduction']:.4f}, "
                      f"{epoch_results[epoch]['map_reduction_percentage']:.1f}%)")

            except Exception as e:
                print(f"Error evaluating epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Create results summary
        results = {
            'baseline_map': baseline_map,
            'baseline_metrics': baseline_metrics,
            'seed_patch_map': seed_patch_map,
            'seed_patch_metrics': seed_patch_metrics,
            'epochs': epochs_evaluated,
            'map_scores': map_scores,
            'epoch_results': epoch_results,
            'dataset_info': {
                'num_images': len(eval_images),
                'num_labels': len(eval_labels),
                'classes': test_dataset.classes,
                'num_classes': test_dataset.num_classes
            }
        }

        # Save results if requested
        if save_results:
            results_file = os.path.join(self.checkpoint_dir, 'map_evaluation_results.json')
            # Convert tensors to lists for JSON serialization
            json_results = {
                'baseline_map': float(baseline_map),
                'seed_patch_map': float(seed_patch_map),
                'epochs': epochs_evaluated,
                'map_scores': [float(x) for x in map_scores],
                'dataset_info': results['dataset_info']
            }
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"Results saved to {results_file}")

        return results

    def visualize_map_decline(self, results, save_path=None, title="mAP Decline Across Training Epochs"):
        """
        Visualize mAP decline across training epochs
        """
        if not results or 'epochs' not in results:
            print("No results to visualize!")
            return

        epochs = results['epochs']
        map_scores = results['map_scores']
        baseline_map = results['baseline_map']
        seed_patch_map = results.get('seed_patch_map', None)

        plt.figure(figsize=(12, 8))

        # Plot mAP scores
        plt.plot(epochs, map_scores, 'b-', linewidth=2, label='PS-GAN Adversarial mAP', marker='o', markersize=4)

        # Plot baseline
        plt.axhline(y=baseline_map, color='r', linestyle='--', linewidth=2, label='Baseline mAP (Original Images)')

        # Plot seed patch mAP if available
        if seed_patch_map is not None:
            plt.axhline(y=seed_patch_map, color='orange', linestyle='--', linewidth=2, label='Seed Patch mAP (No Generator)')

        # Formatting
        plt.xlabel('Training Epochs', fontsize=12)
        plt.ylabel('mAP (Detection Accuracy)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

        # Set y-axis limits
        y_values = map_scores + [baseline_map]
        if seed_patch_map is not None:
            y_values.append(seed_patch_map)
        plt.ylim(0, max(1.0, max(y_values) * 1.1))

        # Add statistics text
        if map_scores:
            min_map = min(map_scores)
            final_map = map_scores[-1]
            max_reduction = baseline_map - min_map

            stats_text = f'Dataset: {results.get("dataset_info", {}).get("num_classes", "Unknown")} classes\n'
            stats_text += f'Baseline mAP: {baseline_map:.3f}\n'
            if seed_patch_map is not None:
                stats_text += f'Seed Patch mAP: {seed_patch_map:.3f}\n'
            stats_text += f'Final mAP: {final_map:.3f}\n'
            stats_text += f'Max Reduction: {max_reduction:.3f}\n'
            stats_text += f'Max Reduction %: {(max_reduction/baseline_map)*100:.1f}%'

            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()
