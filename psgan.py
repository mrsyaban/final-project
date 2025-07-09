import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from models.generator import Generator
from models.discriminator import Discriminator
from models.attention import AttentionModel


class PSGAN:
    def __init__(
        self, 
        patch_size=56, 
        image_size=640, 
        device=None, 
        lambda_patch=0.01,
        gamma_adv=0.1,
        patch_area_ratio=0.05,
        distortion_threshold=50
    ):
        """
        Initialize the PS-GAN model.
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.image_size = image_size
        self.lambda_patch = lambda_patch
        self.gamma_adv = gamma_adv
        self.patch_area_ratio = patch_area_ratio
        self.distortion_threshold = distortion_threshold

        self.generator = Generator(input_channels=3, output_channels=3, input_size=patch_size).to(self.device)
        self.discriminator = Discriminator(input_channels=3, input_size=image_size).to(self.device)
        self.attention_model = AttentionModel(image_size=image_size)
        
        # Print model parameter counts
        generator_params = sum(p.numel() for p in self.generator.parameters())
        discriminator_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"Generator parameters: {generator_params:,}")
        print(f"Discriminator parameters: {discriminator_params:,}")


        # Initialize optimizers
        self.d_optimizer = None
        self.g_optimizer = None
        self.steps = 0

            # Store previous weights for tracking changes
        self.prev_g_weights = None
        self.prev_d_weights = None

    def setup_optimizers(self, d_lr=0.0002, g_lr=0.0002):
        """Setup optimizers for training"""
        self.d_optimizer = optim.SGD(self.discriminator.parameters(), lr=d_lr, momentum=0.9)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
        
        self.lr_decay_steps = 900  # Decrease LR every 900 steps
        self.lr_decay_rate = 0.9   # Decrease by 10% (multiply by 0.9)
        
        # Store initial learning rates to track changes
        self.initial_d_lr = d_lr
        self.initial_g_lr = g_lr
    
    def adjust_learning_rate(self):
        """Decrease learning rate by 10% every 900 steps"""
        self.steps += 1
        
        if self.steps % self.lr_decay_steps == 0:
            # Calculate new learning rates
            for param_group in self.d_optimizer.param_groups:
                param_group['lr'] *= self.lr_decay_rate
                
            for param_group in self.g_optimizer.param_groups:
                param_group['lr'] *= self.lr_decay_rate
                
            current_d_lr = self.d_optimizer.param_groups[0]['lr']
            current_g_lr = self.g_optimizer.param_groups[0]['lr']
            
            print(f"Step {self.steps}: Reducing learning rates - D_lr={current_d_lr:.6f}, G_lr={current_g_lr:.6f}")
        

    # --- Loss Components ---
    def gan_loss_fn(self, real_images, fake_images):
        """
        Binary cross entropy loss for GAN training.
        """
                # Real images
        pred_real = self.discriminator(real_images)
        real_labels = torch.ones_like(pred_real, device=self.device)
        loss_real = F.binary_cross_entropy(pred_real, real_labels)

        # Fake images (adversarial images)
        pred_fake = self.discriminator(fake_images)  # Detach to avoid training generator
        fake_labels = torch.zeros_like(pred_fake, device=self.device)
        loss_fake = F.binary_cross_entropy(pred_fake, fake_labels)
        
        total_loss = (loss_real + loss_fake) * 0.5
        return total_loss

    def patch_loss(self, seed_patches, adv_patches):
        """
        Patch distortion loss: L2 norm, penalized only if over threshold (Eq. 6 in PS-GAN paper).
        """
        # distortion = torch.norm(seed_patches - adv_patches, p=2, dim=(1, 2, 3))
        # return distortion.mean()
        distortion = torch.norm(seed_patches - adv_patches, p=2, dim=(1, 2, 3))
        # factor = torch.sigmoid((distortion - self.distortion_threshold) * 0.1)
        # return (distortion * factor).mean()
        return distortion.mean()

    def adv_loss(self, target_model, adv_images, target_labels):
        """
        Target model (YOLO) detection loss: encourage detection suppression.
        """
        yolo_loss = target_model.compute_adv_loss(
            adv_images,
            target_labels, 
            )
        # yolo_loss = target_model.compute_loss(
        #     adv_images,
        #     target_labels, 
        #     confidence_threshold=0.1
        #     )
        
        return yolo_loss
        

    # --- Discriminator and Generator Loss Wrappers ---
    def compute_discriminator_loss(self, real_images, fake_images):
        """
        Discriminator loss = L_real + L_fake
        Corresponds to line 10 in Algorithm 1.
        
        Args:
            real_images: Clean images (ψ_x)
            fake_images: Adversarial images (ψ_δ^G)
        """
        return self.gan_loss_fn(real_images, fake_images.detach())

    def compute_generator_loss(self, adv_patches, seed_patches, adv_images, real_images, target_model, target_labels):
        """
        Generator loss = L_GAN + λ * L_patch + γ * L_adv
        Corresponds to line 15 in Algorithm 1.
        
        Args:
            adv_patches: Generated adversarial patches
            seed_patches: Original seed patches
            adv_images: Adversarial images (patches applied to images)
            target_model: YOLO model for adversarial loss
        """
        
        # GAN loss: fool discriminator
        # pred_fake = self.discriminator(adv_images)
        # real_labels = torch.ones_like(pred_fake, device=self.device)
        # loss_gan = F.binary_cross_entropy(pred_fake, real_labels)

        loss_gan = self.gan_loss_fn(adv_images, real_images)

        # Patch distortion loss
        loss_patch = self.patch_loss(seed_patches, adv_patches)

        # Adversarial loss (YOLO)
        loss_adv = self.adv_loss(target_model, adv_images, target_labels)

        total_loss = self.lambda_patch*loss_gan + self.lambda_patch * loss_patch + self.gamma_adv * loss_adv
        # total_loss = self.lambda_patch * loss_patch + self.gamma_adv * loss_adv

        return total_loss, {
            'gan_loss': loss_gan.item(),
            'patch_loss': loss_patch.item(),
            'adv_loss': loss_adv.item(),
        }

    def _create_adversarial_images(self, target_images, target_labels, adv_patches, hottest_points, keypoints, shape, combine=True):
        """
        Create adversarial images using the updated attention model.
        
        Args:
            target_images: Batch of target images [batch_size, 3, H, W]
            target_labels: Batch of labels [N_boxes, 6]
            adv_patches: Batch of adversarial patches [batch_size, 3, patch_H, patch_W]
        
        Returns:
            adv_images: Adversarial images with patches applied
            patch_masks: Generated patch masks
            patch_sizes: List of patch sizes
        """
        batch_size = target_images.size(0)
        
        with torch.no_grad():
            # attention_maps = self.attention_model.generate_attention_map(target_images, target_labels)

            # # Generate patch masks based on attention maps and labels
            # patch_masks, patch_sizes = self.attention_model.generate_patch_mask(
            #     attention_maps, target_labels, patch_area_ratio=self.patch_area_ratio
            # )

            patch_masks = self.attention_model.generate_patch_mask_v2(
                target_labels, hottest_points, keypoints, shape, patch_area_ratio=self.patch_area_ratio
            )
            
        # Create m² adversarial images by applying each patch to each image
        adv_images_list = []
        
        if combine:
            # Create m² adversarial images by applying each patch to each image
            for i in range(batch_size):  # For each image
                for j in range(batch_size):  # For each patch
                    # Get single image and patch - CLONE to avoid sharing memory
                    single_image = target_images[i:i+1].clone()
                    single_patch = adv_patches[j:j+1]
                    single_mask = patch_masks[i:i+1]  # Use mask from image i
                    
                    # Apply patch j to image i using mask from image i
                    adv_img = self.attention_model.apply_patch(single_image, single_mask, single_patch)
                    adv_images_list.append(adv_img)
        else:
            # Create m adversarial images by applying each patch to its corresponding image
            for i in range(batch_size):
                # Get single image and patch
                single_image = target_images[i:i+1].clone()
                single_patch = adv_patches[i:i+1]
                single_mask = patch_masks[i:i+1]
                
                # Apply patch i to image i using mask from image i
                adv_img = self.attention_model.apply_patch(single_image, single_mask, single_patch)
                adv_images_list.append(adv_img)
        
        adv_images = torch.cat(adv_images_list, dim=0)  # [batch², C, H, W]
        
        
        return adv_images

    def _single_training_step(self, seed_patch_iter, psgan_iter, target_model, k_steps):
        """
        Single training iteration following Algorithm 1 lines 4-15.
        """
        metrics = {
            'd_loss': 0.0,
            'g_loss': 0.0,
            'gan_loss': 0.0,
            'patch_loss': 0.0,
            'adv_loss': 0.0,
        }

        try:
            # Algorithm 1: Lines 4-11 (Discriminator training for k steps)
            d_loss_total = 0.0
            for step in range(k_steps):
                # Line 5: sample minibatch of m images ψ_x = {x1, ..., xm}
                target_images, target_labels, filenames, keypoints, shape, hottest_points, relighting_coeffs = next(psgan_iter)
                
                # Line 6: sample minibatch of m patches ψ_δ = {δ1, ..., δm}
                seed_patches = next(seed_patch_iter)  # Note: seed patch dataloader only returns patches

                # Move to device
                target_images = target_images.to(self.device)
                target_labels = target_labels.to(self.device) if target_labels is not None else None
                seed_patches = seed_patches.to(self.device)

                # Ensure same batch size
                batch_size = min(target_images.size(0), seed_patches.size(0))
                target_images = target_images[:batch_size]
                seed_patches = seed_patches[:batch_size]
                
                # Reindex target_labels to match the truncated batch
                if target_labels is not None and len(target_labels) > 0:
                    # Keep only labels for images in the current batch
                    mask = target_labels[:, 0] < batch_size
                    target_labels = target_labels[mask]
                else:
                    # Create empty labels tensor if no labels
                    target_labels = torch.empty(0, 6).to(self.device)

                # Preprocess patches if needed (convert from [0,1] to [-1,1] if using tanh activation)
                if seed_patches.min() >= 0 and seed_patches.max() <= 1:
                    seed_patches = seed_patches * 2 - 1

                # Line 7: generate minibatch of m adversarial patches ψ_δ^G = {G(δ1), ..., G(δm)}
                
                adv_patches = self.generator(seed_patches)

                # Line 8: obtain attention map M(ψ_x) by Grad-CAM
                # Line 9: construct minibatch of m² adversarial images ψ_δ^G = {xi ⊕ M(xi) ○ δj |i, j = 1, ..., m}
                
                
                adv_images= self._create_adversarial_images(
                    target_images, target_labels, adv_patches, hottest_points, keypoints, shape, combine=True
                )
                
                
                # Create corresponding clean images (m² copies)
                clean_images_list = []
                for i in range(batch_size):  # For each image
                    for j in range(batch_size):  # For each patch
                        clean_images_list.append(target_images[i:i+1])
                
                clean_images = torch.cat(clean_images_list, dim=0)  # [m², C, H, W]

                # Line 10: optimize W_D to max_D L_GAN with G fixed
                self.discriminator.train()
                self.d_optimizer.zero_grad()

                d_loss = self.compute_discriminator_loss(clean_images, adv_images)
                d_loss.backward()
                self.d_optimizer.step()
                
                d_loss_total += d_loss.item()

            # Average discriminator loss over k steps
            metrics['d_loss'] = d_loss_total / k_steps

            # Algorithm 1: Lines 12-15 (Generator training)
            # Line 12: sample minibatch of m images ψ_x = {x1, ..., xm}
            target_images, target_labels, filenames, keypoints, shape, hottest_points, relighting_coeffs = next(psgan_iter)

            # Line 13: sample minibatch of m patches ψ_δ = {δ1, ..., δm}
            seed_patches = next(seed_patch_iter)

            # Move to device and ensure same batch size
            target_images = target_images.to(self.device)
            target_labels = target_labels.to(self.device) if target_labels is not None else None
            seed_patches = seed_patches.to(self.device)

            batch_size = min(target_images.size(0), seed_patches.size(0))
            target_images = target_images[:batch_size]
            seed_patches = seed_patches[:batch_size]

            # Reindex target_labels again
            if target_labels is not None and len(target_labels) > 0:
                mask = target_labels[:, 0] < batch_size
                target_labels = target_labels[mask]
            else:
                target_labels = torch.empty(0, 6).to(self.device)

            # Preprocess patches
            if seed_patches.min() >= 0 and seed_patches.max() <= 1:
                seed_patches = seed_patches * 2 - 1

            # Generate adversarial patches with gradients
            adv_patches = self.generator(seed_patches)

            # Create adversarial images for generator training
            adv_images = self._create_adversarial_images(
                target_images, target_labels, adv_patches, hottest_points, keypoints, shape, combine=False
            )

            # Line 15: optimize W_G to min_G L_GAN + λ L_patch + γ L_adv with D fixed
            self.generator.train()
            self.g_optimizer.zero_grad()

            g_loss, loss_dict = self.compute_generator_loss(
                adv_patches, seed_patches, adv_images, target_images, target_model, target_labels
            )
            g_loss.backward()
            self.g_optimizer.step()

            self.adjust_learning_rate()

            # Update metrics
            metrics['g_loss'] = g_loss.item()
            metrics.update(loss_dict)

            return metrics

        except StopIteration:
            # Return None when dataloaders are exhausted
            return None
            
        except Exception as e:
            print(f"Error in training step: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train_step(
        self, 
        seed_patch_dataloader, 
        psgan_dataloader, 
        epochs=1,
        target_model=None, 
        k_steps=2,
        d_lr=0.0002,
        g_lr=0.0002,
        save_interval=None,
        save_path=None,
        lambda_decay=True,
        verbose=True
    ):
        """
        Full training procedure following Algorithm 1 for multiple epochs.
        
        Args:
            seed_patch_dataloader: DataLoader for seed patches (ψ_δ)
            psgan_dataloader: DataLoader for training images (ψ_x)
            epochs: Number of training epochs
            target_model: YOLO model for adversarial training
            k_steps: Number of discriminator training steps per iteration
            d_lr: Discriminator learning rate
            g_lr: Generator learning rate
            save_interval: Save model every N epochs (None to disable)
            save_path: Path to save models (required if save_interval is set)
            verbose: Print training progress
        """
        if self.d_optimizer is None or self.g_optimizer is None:
            self.setup_optimizers(d_lr, g_lr)

        # Training history
        epoch_metrics = []

        original_lambda_patch = self.lambda_patch

        print(f"Starting training for {epochs} epochs...")
        if lambda_decay:
            print(f"Lambda decay enabled: reducing lambda_patch by 5% every 10 epochs")
            print(f"Initial lambda_patch: {self.lambda_patch:.6f}")
        
        # Algorithm 1: Line 3 - for the number of training epochs do
        for epoch in range(epochs):
            if lambda_decay and (epoch + 1) % 10 == 0:
                self.lambda_patch *= 0.95  # Reduce by 5%
                if verbose:
                    print(f"\nEpoch {epoch + 1}: Lambda decay applied")
                    print(f"New lambda_patch: {self.lambda_patch:.6f}")
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs} (lambda_patch: {self.lambda_patch:.6f})")
            
            # Reset dataloaders for each epoch
            seed_patch_iter = iter(seed_patch_dataloader)
            psgan_iter = iter(psgan_dataloader)
            
            # Track metrics for this epoch
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            epoch_gan_loss = 0.0
            epoch_patch_loss = 0.0
            epoch_adv_loss = 0.0
            step_count = 0
            
            # Training loop for one epoch
            while True:
                # Perform one training step
                step_metrics = self._single_training_step(seed_patch_iter, psgan_iter, target_model, k_steps)
                
                # Check if dataloaders are exhausted
                if step_metrics is None:
                    break
                
                # Accumulate metrics
                epoch_d_loss += step_metrics['d_loss']
                epoch_g_loss += step_metrics['g_loss']
                epoch_gan_loss += step_metrics['gan_loss']
                epoch_patch_loss += step_metrics['patch_loss']
                epoch_adv_loss += step_metrics['adv_loss']
                step_count += 1
                
                if verbose and step_count % 10 == 0:
                    g_weight_change, d_weight_change = self.calculate_weight_changes()
    
                    print(f"  Step {step_count}: D_loss={step_metrics['d_loss']:.4f}, "
                        f"G_loss={step_metrics['g_loss']:.4f}, "
                        f"GAN_loss={step_metrics['gan_loss']:.4f}, "
                        f"Patch_loss={step_metrics['patch_loss']:.4f}, "
                        f"Adv_loss={step_metrics['adv_loss']:.4f}, "
                        f"G_weight_Δ={g_weight_change:.6f}, "
                        f"D_weight_Δ={d_weight_change:.6f}")
            
            # Calculate average metrics for this epoch
            if step_count > 0:
                avg_metrics = {
                    'epoch': epoch + 1,
                    'd_loss': epoch_d_loss / step_count,
                    'g_loss': epoch_g_loss / step_count,
                    'gan_loss': epoch_gan_loss / step_count,
                    'patch_loss': epoch_patch_loss / step_count,
                    'adv_loss': epoch_adv_loss / step_count,
                    'steps': step_count
                }
                epoch_metrics.append(avg_metrics)
                
                if verbose:
                    print(f"Epoch {epoch + 1} Summary:")
                    print(f"  Average D_loss: {avg_metrics['d_loss']:.4f}")
                    print(f"  Average G_loss: {avg_metrics['g_loss']:.4f}")
                    print(f"  Average GAN_loss: {avg_metrics['gan_loss']:.4f}")
                    print(f"  Average Patch_loss: {avg_metrics['patch_loss']:.4f}")
                    print(f"  Average Adv_loss: {avg_metrics['adv_loss']:.4f}")
                    print(f"  Total steps: {step_count}")
            
            # Save model if requested
            if save_interval is not None and save_path is not None and (epoch + 1) % save_interval == 0:
                self.save_models(save_path, epoch + 1)
                if verbose:
                    print(f"  Models saved at epoch {epoch + 1}")
        
        if verbose:
            print(f"\nTraining completed! Total epochs: {epochs}")
            if lambda_decay:
                print(f"Final lambda_patch: {self.lambda_patch:.6f}")
                print(f"Lambda reduction: {((original_lambda_patch - self.lambda_patch) / original_lambda_patch * 100):.2f}%")
        
        # Return the last batch outputs and training history
        return epoch_metrics
    
    def generate_and_save_adversarial_images(self, target_images, target_labels, seed_patches, hottest_points, keypoints, shape, save_dir, prefix="adv_image", original_filenames=None):
        """
        Generate and save adversarial images by applying patches to corresponding images.
        Creates m adversarial images (not m²) by applying each patch to its corresponding image.
        
        Args:
            target_images: Batch of target images [batch_size, 3, H, W]
            target_labels: Batch of labels [N_boxes, 6] or None
            seed_patches: Batch of seed patches [batch_size, 3, patch_H, patch_W]
            save_dir: Directory to save adversarial images
            prefix: Prefix for saved image filenames (used when original_filenames is None)
            original_filenames: List of original filenames (without extension) for each image
        
        Returns:
            adv_images: Generated adversarial images [batch_size, 3, H, W]
            patch_masks: Generated patch masks
            patch_sizes: List of patch sizes
        """
        import torchvision.transforms as transforms
        from PIL import Image
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Move to device
        target_images = target_images.to(self.device)
        target_labels = target_labels.to(self.device) if target_labels is not None else None
        seed_patches = seed_patches.to(self.device)
        
        # Ensure same batch size
        batch_size = min(target_images.size(0), seed_patches.size(0))
        target_images = target_images[:batch_size]
        seed_patches = seed_patches[:batch_size]
        
        # Reindex target_labels to match the truncated batch
        if target_labels is not None and len(target_labels) > 0:
            mask = target_labels[:, 0] < batch_size
            target_labels = target_labels[mask]
        else:
            target_labels = torch.empty(0, 6).to(self.device)
        
        # Preprocess patches if needed (convert from [0,1] to [-1,1] if using tanh activation)
        if seed_patches.min() >= 0 and seed_patches.max() <= 1:
            seed_patches = seed_patches * 2 - 1
        
        # Set generator to evaluation mode
        self.generator.eval()
        
        with torch.no_grad():
            # Generate adversarial patches
            adv_patches = self.generator(seed_patches)
            
            # Create adversarial images (m adversarial images, not m²)
            adv_images = self._create_adversarial_images(
                target_images, target_labels, adv_patches, hottest_points, keypoints, shape, combine=False
            )
        
        # Convert tensors to PIL images and save
        to_pil = transforms.ToPILImage()

        for i in range(batch_size):
            # Determine filename base
            if original_filenames is not None and i < len(original_filenames):
                # Remove extension from original filename if present
                base_name = os.path.splitext(original_filenames[i])[0]
            else:
                base_name = f"{prefix}_{i}"
            
            # Save ONLY adversarial image
            adv_img = adv_images[i].cpu()
            adv_img = torch.clamp((adv_img + 1) / 2, 0, 1)  # Convert from [-1,1] to [0,1] if needed
            adv_pil = to_pil(adv_img)
            adv_path = os.path.join(save_dir, f"{base_name}.png")  # Just save with original filename
            adv_pil.save(adv_path)
        
        print(f"Saved {batch_size} adversarial images to {save_dir}")
        
        return adv_images

    def save_models(self, save_path, epoch):
        """Save generator and discriminator models"""
        os.makedirs(save_path, exist_ok=True)
        
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict() if self.g_optimizer else None,
            'd_optimizer_state_dict': self.d_optimizer.state_dict() if self.d_optimizer else None,
            'epoch': epoch,
            'patch_size': self.patch_size,
            'image_size': self.image_size,
            'lambda_patch': self.lambda_patch,
            'gamma_adv': self.gamma_adv,
            'patch_area_ratio': self.patch_area_ratio,
            'distortion_threshold': self.distortion_threshold,
        }, os.path.join(save_path, f'psgan_epoch_{epoch}.pth'))

    def load_models(self, checkpoint_path):
        """Load generator and discriminator models"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        if checkpoint.get('g_optimizer_state_dict') and self.g_optimizer:
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        if checkpoint.get('d_optimizer_state_dict') and self.d_optimizer:
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        # Load hyperparameters if available
        if 'lambda_patch' in checkpoint:
            self.lambda_patch = checkpoint['lambda_patch']
        if 'gamma_adv' in checkpoint:
            self.gamma_adv = checkpoint['gamma_adv']
        if 'patch_area_ratio' in checkpoint:
            self.patch_area_ratio = checkpoint['patch_area_ratio']
        if 'distortion_threshold' in checkpoint:
            self.distortion_threshold = checkpoint['distortion_threshold']
        
        return checkpoint.get('epoch', 0)
    
    def calculate_weight_changes(self):
        """Calculate the average absolute change in weights since last check"""
        # Get current weights
        current_g_weights = {name: param.data.clone() for name, param in self.generator.named_parameters()}
        current_d_weights = {name: param.data.clone() for name, param in self.discriminator.named_parameters()}
        
        g_change = 0.0
        d_change = 0.0
        
        # Calculate changes if we have previous weights
        if self.prev_g_weights is not None:
            g_changes = []
            for name, curr_w in current_g_weights.items():
                if name in self.prev_g_weights:
                    # Calculate absolute mean change
                    change = torch.abs(curr_w - self.prev_g_weights[name]).mean().item()
                    g_changes.append(change)
            
            if g_changes:
                g_change = sum(g_changes) / len(g_changes)
        
        if self.prev_d_weights is not None:
            d_changes = []
            for name, curr_w in current_d_weights.items():
                if name in self.prev_d_weights:
                    change = torch.abs(curr_w - self.prev_d_weights[name]).mean().item()
                    d_changes.append(change)
            
            if d_changes:
                d_change = sum(d_changes) / len(d_changes)
        
        # Update previous weights
        self.prev_g_weights = current_g_weights
        self.prev_d_weights = current_d_weights
        
        return g_change, d_change