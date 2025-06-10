import torch
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import os
import folder_paths

class SmartWatermarkNode:
    """
    ComfyUI Node for Smart Watermark Placement using Depth Maps
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image tensor
                "watermark_path": ("STRING", {
                    "default": "",
                    "placeholder": "Path to watermark PNG file"
                }),
                "depth_map": ("IMAGE",),  # Depth map tensor
                "depth_format": (["viridis", "grayscale"], {
                    "default": "viridis"
                }),
                "min_scale": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "max_scale": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
                "opacity": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1
                }),
                "position_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 7,
                    "step": 1,
                    "tooltip": "0=Best position, higher numbers for alternatives"
                }),
                "num_candidates": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 16,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("watermarked_image",)
    FUNCTION = "apply_smart_watermark"
    CATEGORY = "image/watermark"

    def tensor_to_pil(self, tensor):
        """Convert ComfyUI tensor to PIL Image"""
        # ComfyUI tensors are usually [B, H, W, C] with values 0-1
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # Remove batch dimension
        
        # Convert to numpy and scale to 0-255
        np_image = tensor.cpu().numpy()
        if np_image.max() <= 1.0:
            np_image = (np_image * 255).astype(np.uint8)
        
        return Image.fromarray(np_image)

    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to ComfyUI tensor"""
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array and normalize to 0-1
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension [1, H, W, C]
        tensor = torch.from_numpy(np_image).unsqueeze(0)
        
        return tensor

    def ensure_watermark_fits(self, position, watermark_size, image_size):
        """Ensure watermark position doesn't cause cropping"""
        x, y = position
        wm_w, wm_h = watermark_size
        img_w, img_h = image_size
        
        # Calculate maximum allowed positions
        max_x = img_w - wm_w
        max_y = img_h - wm_h
        
        # Adjust position if it would cause cropping
        adjusted_x = max(0, min(x, max_x))
        adjusted_y = max(0, min(y, max_y))
        
        return (adjusted_x, adjusted_y)

    def process_depth_map(self, depth_tensor, format_type):
        """Process depth map based on format type"""
        # Convert tensor to numpy
        if depth_tensor.dim() == 4:
            depth_tensor = depth_tensor.squeeze(0)
        
        depth_np = depth_tensor.cpu().numpy()
        
        if format_type == "viridis":
            # For viridis colormap, we need to extract depth info
            # Typically viridis: blue=low values, yellow=high values
            # We want: blue=background (good for watermark), yellow=objects (avoid)
            
            if depth_np.shape[-1] == 3:  # RGB viridis
                # Convert viridis RGB back to depth values
                # Blue channel is high when depth is low, red channel is high when depth is high
                blue_channel = depth_np[:, :, 2]  # Blue channel
                red_channel = depth_np[:, :, 0]   # Red channel
                
                # Create depth map where low values = background, high values = objects
                depth_map = red_channel - blue_channel + 0.5
                depth_map = np.clip(depth_map, 0, 1)
            else:
                # Single channel, assume normalized depth
                depth_map = depth_np.squeeze()
        
        elif format_type == "grayscale":
            # For grayscale: white=objects (high values), black=background (low values)
            if depth_np.shape[-1] == 3:
                # Convert RGB to grayscale
                depth_map = np.dot(depth_np, [0.299, 0.587, 0.114])
            else:
                depth_map = depth_np.squeeze()
        
        # Ensure depth_map is 2D
        if depth_map.ndim > 2:
            depth_map = depth_map.squeeze()
        
        # Normalize to 0-1 range
        if depth_map.max() > depth_map.min():
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        return depth_map

    def find_best_positions(self, depth_map, watermark_size, image_size, num_positions=8):
        """Find the best positions for watermark placement with fit checking"""
        h, w = depth_map.shape
        img_w, img_h = image_size
        wm_h, wm_w = watermark_size
        
        # Safety check: ensure image dimensions match depth map
        if h != img_h or w != img_w:
            print(f"Warning: Depth map size ({w}x{h}) != Image size ({img_w}x{img_h})")
        
        # Use actual image dimensions for boundary calculations
        h, w = img_h, img_w
        
        if wm_h >= h or wm_w >= w:
            print(f"Warning: Watermark ({wm_w}x{wm_h}) too large for image ({w}x{h})")
            return [(0, 0)]
        
        positions = []
        scores = []
        
        # Create comprehensive grid focusing on edges
        margin = 20
        step = min(40, wm_w//3, wm_h//3)
        
        test_positions = []
        
        # Grid positions with edge priority - ensuring they fit
        for y in range(margin, h - wm_h - margin + 1, step):
            for x in range(margin, w - wm_w - margin + 1, step):
                # Prioritize edge and corner positions
                near_edge = (x < margin + step*3 or x > w - wm_w - margin - step*3 or 
                           y < margin + step*3 or y > h - wm_h - margin - step*3)
                if near_edge:
                    # Ensure position allows watermark to fit completely
                    safe_pos = self.ensure_watermark_fits((x, y), (wm_w, wm_h), (w, h))
                    test_positions.append(safe_pos)
        
        # Always include corners and edge centers - with fit checking
        key_positions = [
            (margin, margin),  # top-left
            (w - wm_w - margin, margin),  # top-right
            (margin, h - wm_h - margin),  # bottom-left
            (w - wm_w - margin, h - wm_h - margin),  # bottom-right
            (w//2 - wm_w//2, margin),  # top-center
            (w//2 - wm_w//2, h - wm_h - margin),  # bottom-center
            (margin, h//2 - wm_h//2),  # left-center
            (w - wm_w - margin, h//2 - wm_h//2),  # right-center
        ]
        
        # Ensure all key positions fit
        for pos in key_positions:
            safe_pos = self.ensure_watermark_fits(pos, (wm_w, wm_h), (w, h))
            test_positions.append(safe_pos)
        
        test_positions = list(set(test_positions))
        
        for x, y in test_positions:
            # Double-check bounds (should be safe now)
            if x + wm_w > w or y + wm_h > h or x < 0 or y < 0:
                continue
            
            # Extract region from depth map
            if y + wm_h <= depth_map.shape[0] and x + wm_w <= depth_map.shape[1]:
                region = depth_map[y:y+wm_h, x:x+wm_w]
            else:
                # Handle depth map size mismatch
                region = depth_map[
                    max(0, min(y, depth_map.shape[0]-1)):min(y+wm_h, depth_map.shape[0]),
                    max(0, min(x, depth_map.shape[1]-1)):min(x+wm_w, depth_map.shape[1])
                ]
            
            if region.size == 0:
                continue
            
            # Score calculation
            mean_depth = np.mean(region)
            depth_variance = np.var(region)
            
            # Prefer low depth values (background) and uniform areas
            background_score = 1.0 - mean_depth
            uniformity_score = 1.0 - min(depth_variance, 1.0)
            contrast_penalty = min(depth_variance * 2, 0.5)
            
            # Bonus for corner/edge positions
            edge_bonus = 0
            if (x < margin + 60 or x > w - wm_w - margin - 60) and \
               (y < margin + 60 or y > h - wm_h - margin - 60):
                edge_bonus = 0.15
            
            final_score = background_score * 0.6 + uniformity_score * 0.3 - contrast_penalty + edge_bonus
            
            positions.append((x, y))
            scores.append(final_score)
        
        if not positions:
            print("Warning: No valid positions found. Using center.")
            center_x = max(0, min(w//2 - wm_w//2, w - wm_w))
            center_y = max(0, min(h//2 - wm_h//2, h - wm_h))
            return [(center_x, center_y)]
        
        # Sort by score (highest first)
        sorted_positions = [pos for _, pos in sorted(zip(scores, positions), reverse=True)]
        return sorted_positions[:num_positions]

    def resize_watermark(self, watermark, target_size, scale_range, image_size):
        """Resize watermark within scale constraints with error handling"""
        try:
            min_scale, max_scale = scale_range
            
            # Validate scale values
            if min_scale <= 0 or max_scale <= 0:
                print(f"Warning: Invalid scale values ({min_scale}, {max_scale}). Using defaults.")
                min_scale, max_scale = 0.8, 1.2
            
            if min_scale > max_scale:
                print(f"Warning: min_scale ({min_scale}) > max_scale ({max_scale}). Swapping values.")
                min_scale, max_scale = max_scale, min_scale
            
            original_size = watermark.size
            img_w, img_h = image_size
            
            # Calculate base scale to fit target size
            scale_w = target_size[0] / original_size[0]
            scale_h = target_size[1] / original_size[1]
            base_scale = min(scale_w, scale_h)
            
            # Apply scale constraints
            min_allowed = base_scale * min_scale
            max_allowed = base_scale * max_scale
            
            # Use maximum allowed scale for better visibility
            final_scale = max_allowed
            
            # Calculate new size
            new_w = int(original_size[0] * final_scale)
            new_h = int(original_size[1] * final_scale)
            
            # Safety check: ensure watermark doesn't exceed image dimensions
            if new_w >= img_w * 0.9 or new_h >= img_h * 0.9:
                print(f"Warning: Watermark too large ({new_w}x{new_h}) for image ({img_w}x{img_h}). Scaling down.")
                # Scale down to fit within 80% of image dimensions
                scale_w_safe = (img_w * 0.8) / original_size[0]
                scale_h_safe = (img_h * 0.8) / original_size[1]
                final_scale = min(scale_w_safe, scale_h_safe)
                new_w = int(original_size[0] * final_scale)
                new_h = int(original_size[1] * final_scale)
            
            # Minimum size check
            min_size = 20
            if new_w < min_size or new_h < min_size:
                print(f"Warning: Watermark too small ({new_w}x{new_h}). Setting minimum size.")
                if new_w < min_size:
                    scale_factor = min_size / new_w
                    new_w = min_size
                    new_h = int(new_h * scale_factor)
                if new_h < min_size:
                    scale_factor = min_size / new_h
                    new_h = min_size
                    new_w = int(new_w * scale_factor)
            
            new_size = (new_w, new_h)
            
            return watermark.resize(new_size, Image.Resampling.LANCZOS)
            
        except Exception as e:
            print(f"Error resizing watermark: {e}. Using original size.")
            return watermark

    def apply_watermark(self, image, watermark, position, opacity):
        """Apply watermark to image"""
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        if watermark.mode != 'RGBA':
            watermark = watermark.convert('RGBA')
        
        # Apply opacity
        watermark_with_opacity = watermark.copy()
        alpha = watermark_with_opacity.split()[-1]
        alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
        watermark_with_opacity.putalpha(alpha)
        
        # Create overlay
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay.paste(watermark_with_opacity, position)
        
        # Composite
        result = Image.alpha_composite(image, overlay)
        return result.convert('RGB')

    def apply_smart_watermark(self, image, watermark_path, depth_map, depth_format, 
                            min_scale, max_scale, opacity, position_index, num_candidates):
        """Main function to apply smart watermark"""
        
        try:
            # Convert inputs
            pil_image = self.tensor_to_pil(image)
            
            # Load watermark
            if not os.path.exists(watermark_path):
                raise ValueError(f"Watermark file not found: {watermark_path}")
            
            watermark = Image.open(watermark_path).convert("RGBA")
            
            # Process depth map
            depth_array = self.process_depth_map(depth_map, depth_format)
            
            # Resize depth map to match image if needed
            if depth_array.shape[:2] != (pil_image.height, pil_image.width):
                depth_array = cv2.resize(depth_array, (pil_image.width, pil_image.height))
            
            # Calculate target watermark size (reasonable proportion)
            img_w, img_h = pil_image.size
            target_size = (max(100, img_w // 6), max(60, img_h // 10))
            
            # Resize watermark
            scale_range = (min_scale, max_scale)
            resized_watermark = self.resize_watermark(watermark, target_size, scale_range, pil_image.size)
            
            # Find best positions
            best_positions = self.find_best_positions(depth_array, resized_watermark.size, pil_image.size, num_candidates)
            
            if not best_positions:
                raise ValueError("No suitable position found for watermark")
            
            # Select position with final safety check
            pos_idx = min(position_index, len(best_positions) - 1)
            chosen_position = best_positions[pos_idx]
            
            # Final verification that watermark fits
            final_position = self.ensure_watermark_fits(chosen_position, resized_watermark.size, pil_image.size)
            if final_position != chosen_position:
                print(f"Position adjusted from {chosen_position} to {final_position} to prevent cropping")
            
            # Apply watermark
            result_image = self.apply_watermark(pil_image, resized_watermark, final_position, opacity)
            
            # Convert back to tensor
            result_tensor = self.pil_to_tensor(result_image)
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"Error in SmartWatermarkNode: {str(e)}")
            # Return original image on error
            return (image,)

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SmartWatermarkNode": SmartWatermarkNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartWatermarkNode": "adiciona marcadagua topzera"
}
