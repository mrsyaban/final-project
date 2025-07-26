import torch
from models.generator import Generator
from datasets.seed_patch_dataset import get_seed_patch_data
# from testing.rizky import show_tensor

import matplotlib.pyplot as plt 

seed_patch_dataloader, seed_patch_dataset = get_seed_patch_data(
    patch_size=32,
    batch_size=32,
    num_workers=8
)
seed_iter = iter(seed_patch_dataloader)
seed_patches = next(seed_iter)

generator = Generator(input_channels=3, output_channels=3, input_size=32).to("cuda")

checkpoint = torch.load("./checkpoints/finalr9_g-1.0_l-0.00387/psgan_epoch_250.pth", map_location="cuda")

from collections import OrderedDict

g_state_dict = OrderedDict()
for k, v in checkpoint['generator_state_dict'].items():
    new_key = k.replace("module.", "") if k.startswith("module.") else k
    g_state_dict[new_key] = v


generator.load_state_dict(g_state_dict)

generator.eval()

seed_patches = seed_patches.to("cuda")

seed_patches_normal = seed_patches * 2 - 1

# adv_patches = generator(seed_patches)
adv_patches_normal = generator(seed_patches_normal)

# Create a figure to display all 32 patches in a grid
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
axes = axes.flatten()

# Convert from normalized range [-1,1] back to [0,1] for display
adv_patches_display = (adv_patches_normal + 1) / 2

# Display each patch in the grid
for i in range(32):
    # Move tensor to CPU, convert to numpy, and rearrange dimensions for matplotlib
    img = adv_patches_display[i].detach().cpu().permute(1, 2, 0).numpy()
    # Plot on corresponding subplot
    axes[i].imshow(img)
    # axes[i].axis('off')  # Hide axes

# plt.tight_layout()
plt.suptitle("Generated Adversarial Patches (train mode)", y=0.95)
plt.savefig("generated_patches_250.png", dpi=300, bbox_inches='tight')
print("Figure saved as generated_patches_250.png")