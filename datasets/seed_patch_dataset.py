import json
import numpy as np
import torch
from torch.utils.data import Dataset
import cairocffi as cairo
import requests
from io import StringIO
import os
import tempfile
from urllib.parse import urlparse
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class QuickDrawPatchDataset(Dataset):
    """Dataset for loading and rendering QuickDraw sketches from Google Cloud Storage URLs."""

    def __init__(self, urls, side=56, transform=None, max_sketches_per_file=None,
                 cache_dir=None, channels=3):
        """
        Args:
            urls (str or list): URL or list of URLs to the NDJSON files containing QuickDraw sketches
                               from Google Cloud Storage.
            side (int): Size of the output raster image (side x side pixels)
            transform (callable, optional): Optional transform to be applied on a sample
            max_sketches_per_file (int, optional): Maximum number of sketches to load from each file
            cache_dir (str, optional): Directory to cache downloaded files. If None, uses a temp directory
            channels (int): Number of channels in the output image (1 or 3)
        """
        self.side = side
        self.transform = transform
        self.channels = channels

        if channels not in [1, 3]:
            raise ValueError("channels must be either 1 or 3")

        # Handle different input types for urls
        if isinstance(urls, str):
            self.urls = [urls]
        else:
            self.urls = urls

        # Set up caching directory
        self.cache_dir = cache_dir
        if self.cache_dir is None:
            self.cache_dir = tempfile.mkdtemp()
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load sketches from all URLs
        self.sketches = []
        self.categories = []  # Store category names for reference

        for url in self.urls:
            category = os.path.splitext(os.path.basename(url))[0]

            # Process the NDJSON file directly from URL
            sketches_from_file = self._load_from_url(url, category, max_sketches_per_file)
            self.sketches.extend(sketches_from_file)


    def _load_from_url(self, url, category, max_sketches):
        """Load sketches from a URL, with optional caching."""
        sketches = []

        # Get filename for caching
        filename = os.path.basename(urlparse(url).path)
        cache_path = os.path.join(self.cache_dir, filename)

        # Check if we have a cached version
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                count = 0
                for line in f:
                    if max_sketches and count >= max_sketches:
                        break

                    data = json.loads(line)
                    drawing = data['drawing']
                    vector_image = [np.array(stroke) for stroke in drawing]

                    sketches.append(vector_image)
                    self.categories.append(category)
                    count += 1
        else:
            # Stream the response to handle large files efficiently
            with requests.get(url, stream=True) as r:
                r.raise_for_status()

                # Save to cache while processing
                with open(cache_path, 'wb') as f:
                    count = 0
                    for line in r.iter_lines():
                        if not line:  # Skip empty lines
                            continue

                        # Cache the line
                        f.write(line + b'\n')

                        if max_sketches and count >= max_sketches:
                            break

                        # Process the line
                        data = json.loads(line)
                        drawing = data['drawing']
                        vector_image = [np.array(stroke) for stroke in drawing]

                        sketches.append(vector_image)
                        self.categories.append(category)
                        count += 1

        return sketches

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        vector_image = self.sketches[idx]
        category = self.categories[idx]  # This is optional, in case you want to know the source

        # Convert the vector image to raster format
        raster = self._vector_to_raster([vector_image])[0]

        # Convert to tensor (normalize to [0, 1])
        tensor = torch.FloatTensor(raster) / 255.0

        # Add channel dimension or convert to 3 channels while keeping black and white
        if self.channels == 1:
            tensor = tensor.unsqueeze(0)  # Shape: (1, side, side)
        else:  # self.channels == 3
            # Repeat the single channel 3 times to create an RGB image
            # while preserving the black and white content
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)  # Shape: (3, side, side)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor  # No label is returned as per requirement

    def _vector_to_raster(self, vector_images, line_diameter=16, padding=16,
                          bg_color=(0,0,0), fg_color=(1,1,1)):
        """Convert vector drawings to raster images using Cairo."""
        original_side = 256.
        side = self.side
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
        ctx = cairo.Context(surface)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_width(line_diameter)

        total_padding = padding * 2. + line_diameter
        new_scale = float(side) / float(original_side + total_padding)
        ctx.scale(new_scale, new_scale)
        ctx.translate(total_padding / 2., total_padding / 2.)

        raster_images = []
        for vector_image in vector_images:
            ctx.set_source_rgb(*bg_color)
            ctx.paint()

            bbox = np.hstack(vector_image).max(axis=1)
            offset = ((original_side, original_side) - bbox) / 2.
            offset = offset.reshape(-1,1)
            centered = [stroke + offset for stroke in vector_image]

            ctx.set_source_rgb(*fg_color)
            for xv, yv in centered:
                ctx.move_to(xv[0], yv[0])
                for x, y in zip(xv, yv):
                    ctx.line_to(x, y)
                ctx.stroke()

            # Extract grayscale image
            data = surface.get_data()
            img = np.frombuffer(data, np.uint8).reshape((side, surface.get_stride() // 4, 4))
            img_gray = 255 - img[:, :, 0]  # Use red channel; invert for white-on-black
            raster_images.append(img_gray)

        return raster_images

def get_seed_patch_data(
  patch_size=56, 
  batch_size=32,
  num_workers=8
):
    # Define transformations
    transform = transforms.Compose([
        # Add any transforms you might need
    ])

    # List of QuickDraw categories from your list
    categories = [
        'airplane',
        'circle',
        'zigzag',
        'sun',
        'star',
        'smiley face',
        'basketball',
        'soccer ball',
        'lightning',
        'snowflake',
        'car'
    ]

    # Create the list of URLs using the Google Cloud Storage path
    base_url = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
    urls = [f"{base_url}{category.replace(' ', '%20')}.ndjson" for category in categories]

    # Create dataset with URLs
    seed_patch_dataset = QuickDrawPatchDataset(
        urls=urls,
        side=patch_size,
        transform=transform,
        max_sketches_per_file=401,
    )

    # Create dataloader
    seed_patch_dataloader = DataLoader(
        seed_patch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Increase worker threads (adjust based on CPU cores)
        pin_memory=True,  # Speed up host to GPU transfers
        persistent_workers=True
    )

    return seed_patch_dataloader, seed_patch_dataset