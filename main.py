# main.py
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import wandb
import numpy as np
import random
from tqdm import tqdm
import shutil # For saving images during validation logging & test outputs

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage, ToTensor
import torch.nn.functional as F
from PIL import ImageDraw, ImageFont

# Assuming your custom modules are in a 'src' directory relative to where main.py is run
# Or that your PYTHONPATH is set up correctly.
try:
    import src.utils.conutils as utils
except ImportError:
    class SimpleConsoleHandler(logging.StreamHandler):
        def __init__(self):
            super().__init__()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.setFormatter(formatter)
    utils = None

from src.data.dataloader import SuperResolutionDataset # Your custom dataloader
from src.models.SRResUNet import SRResUNet # Your SRResUNet model

logger = logging.getLogger(__name__) # Module-level logger

def setSeeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def wandb_init(config, run_name):
    logger.info(f"Initializing wandb for project: {config.other.wandb}")
    config_dict = OmegaConf.to_container(config, resolve=True)
    wandb.init(
        project=config.other.wandb,
        config=config_dict,
        name=run_name
    )

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs, config):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(loader, desc=f"Train Epoch {epoch+1}/{num_epochs}", leave=False)
    for batch_idx, (lr_images, hr_images) in enumerate(progress_bar):
        lr_images, hr_images = lr_images.to(device), hr_images.to(device)
        optimizer.zero_grad()
        generated_hr_images = model(lr_images)
        loss = criterion(generated_hr_images, hr_images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * lr_images.size(0)
        progress_bar.set_postfix(loss=loss.item())
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device,
                       epoch, num_epochs, config, exp_dir,
                       log_images_wandb=False):
    model.eval()
    running_loss = 0.0
    progress_bar = tqdm(loader, desc=f"Valid Epoch {epoch+1}/{num_epochs}", leave=False) # Corrected desc
    logged_this_batch = False # Changed variable name for clarity

    for batch_idx, (lr_images, hr_images) in enumerate(progress_bar): # Renamed for clarity
        lr_images, hr_images = lr_images.to(device), hr_images.to(device) # Renamed
        generated_hr_images = model(lr_images) # Renamed
        
        loss = criterion(generated_hr_images, hr_images)
        running_loss += loss.item() * lr_images.size(0)
        progress_bar.set_postfix(loss=loss.item())

        # Prepare images for logging (only for the first batch of the epoch)
        if log_images_wandb and batch_idx == 0 and not logged_this_batch:
            num_images_to_log = min(lr_images.size(0), config.other.get("wandb_log_images_count", 4))
            
            # It's better to work with copies for manipulation if upscaling LR for display
            lr_display = lr_images[:num_images_to_log].cpu()
            gen_display = generated_hr_images[:num_images_to_log].cpu()
            hr_display = hr_images[:num_images_to_log].cpu()

            # If LR and HR/Gen have different sizes, upscale LR for consistent grid display
            # This assumes HR and Gen have the same target size (output of SR model)
            if lr_display.shape[2:] != hr_display.shape[2:]:
                lr_display = F.interpolate(lr_display, size=(hr_display.size(2), hr_display.size(3)), mode='bilinear', align_corners=False)
            
            # The gen_display should already be at the HR target size from the model output.
            # No need to interpolate gen_display again unless its output size is different from hr_display.

            # 1) Make a combined grid: [LR_upscaled_for_display; Gen; GT]
            combined_tensor = torch.cat([lr_display, gen_display, hr_display], dim=0)
            grid_tensor = make_grid(combined_tensor, nrow=num_images_to_log, normalize=True)

            # 2) Convert to PIL for drawing
            pil_image = ToPILImage()(grid_tensor) # Renamed for clarity
            draw = ImageDraw.Draw(pil_image, "RGBA") # Use RGBA for semi-transparent background

            # Compute single-image width/height in the grid
            # Grid has 3 rows (LR, Gen, HR) and 'num_images_to_log' columns
            w_total, h_total = pil_image.size
            w_img = w_total // num_images_to_log 
            h_img = h_total // 3   # 3 rows in the combined grid

            # Try a larger TrueType font
            try:
                # Increase FONT_SIZE significantly for larger labels
                FONT_SIZE = max(15, int(h_img * 0.1)) # Example: 10% of the single image height in the grid, min 15
                font = ImageFont.truetype("arial.ttf", FONT_SIZE)
                # print(f"Loaded Arial font size {FONT_SIZE}")
            except IOError:
                # print(f"Arial.ttf not found or error loading. Using default font.")
                font = ImageFont.load_default() # Fallback
                FONT_SIZE = 10 # Approximate size for default font if needed for calculations

            labels = ["Input LR (disp)", "Generated HR", "Ground Truth HR"]
            PADDING_X = int(w_img * 0.02) # 2% padding
            PADDING_Y = int(h_img * 0.02) # 2% padding
            BG_PAD    = int(FONT_SIZE * 0.2)  # Background padding relative to font size
            STROKE_W  = max(1, int(FONT_SIZE * 0.05)) # Stroke width relative to font size
            STROKE_FILL = (0, 0, 0, 255)  # Black outline, fully opaque

            for col_idx in range(num_images_to_log): # Iterate through columns (each original image sample)
                x_offset = col_idx * w_img # Starting x for this column in the grid
                for row_idx, text_label in enumerate(labels): # Iterate through rows (LR, Gen, HR)
                    y_offset = row_idx * h_img # Starting y for this row in the grid
                    
                    # Position text within each sub-image panel
                    text_pos_x = x_offset + PADDING_X
                    text_pos_y = y_offset + PADDING_Y
                    
                    # Get text bounding box using textbbox for better accuracy
                    # For textbbox, xy is the top-left anchor point of the text.
                    # We provide (0,0) and then translate, or use text_pos directly.
                    # Let's calculate at (0,0) and add offset later for simplicity of bbox calculation.
                    # For stroke, draw.textbbox might require the stroke_width argument.
                    try:
                        text_bbox = draw.textbbox((0, 0), text_label, font=font, stroke_width=STROKE_W)
                    except TypeError: # Older Pillow might not support stroke_width in textbbox
                         text_bbox = draw.textbbox((0, 0), text_label, font=font)

                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]

                    # Draw semi-transparent background rectangle
                    rect_x0 = text_pos_x - BG_PAD
                    rect_y0 = text_pos_y - BG_PAD 
                    rect_x1 = text_pos_x + text_w + BG_PAD
                    rect_y1 = text_pos_y + text_h + BG_PAD
                    
                    draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=(0, 0, 0, 128)) # Semi-transparent black

                    # Draw the text with stroke
                    draw.text(
                        (text_pos_x, text_pos_y), # Actual drawing position
                        text_label,
                        font=font,
                        fill=(255, 255, 255, 255), # White text, fully opaque
                        stroke_width=STROKE_W,
                        stroke_fill=STROKE_FILL
                    )

            # 3) Back to tensor (not strictly necessary if saving PIL image directly, but good for consistency if using wandb.Image)
            # labeled_grid_tensor = ToTensor()(pil_image) # PIL image is already [0,1] if grid_tensor was normalized

            # 4) Log to WandB
            wandb.log({
                f"Epoch_{epoch+1}_Validation_Comparisons": wandb.Image(pil_image) # Log PIL image directly
            }, step=epoch)

            # Optionally save locally
            if config.other.get("save_val_images_local", True): # Default to True as per your code
                img_save_dir = os.path.join(exp_dir, "validation_images") # Renamed save_dir
                os.makedirs(img_save_dir, exist_ok=True)
                pil_image.save(os.path.join(img_save_dir, f"epoch_{epoch+1}_comparisons.png")) # Save the PIL image with labels

            logged_this_batch = True # Ensure logging only once per validation epoch

    epoch_loss = running_loss / len(loader.dataset) # Calculate average loss for the epoch
    return epoch_loss

@torch.no_grad()
def test_model_and_save_outputs(model, test_loader, criterion, device, config, exp_dir):
    model.eval()
    test_loss = 0.0
    recontruct_errors = [] # List to store reconstruction errors
    
    output_dir_lr = os.path.join(exp_dir, "test_outputs", "input_lr")
    output_dir_hr_generated = os.path.join(exp_dir, "test_outputs", "generated_hr")
    output_dir_hr_ground_truth = os.path.join(exp_dir, "test_outputs", "ground_truth_hr") # If GT is available

    os.makedirs(output_dir_lr, exist_ok=True)
    os.makedirs(output_dir_hr_generated, exist_ok=True)
    os.makedirs(output_dir_hr_ground_truth, exist_ok=True)

    logger.info(f"Saving test outputs to: {os.path.join(exp_dir, 'test_outputs')}")
    
    progress_bar = tqdm(test_loader, desc="Testing Model", leave=False)
    image_counter = 0
    for lr_images, hr_images in progress_bar: # Assuming test loader also provides GT HR
        lr_images, hr_images = lr_images.to(device), hr_images.to(device)

        generated_hr_images = model(lr_images)
        
        if criterion: # Calculate loss if criterion is provided
            loss = criterion(generated_hr_images, hr_images)
            test_loss += loss.item() * lr_images.size(0)
            progress_bar.set_postfix(error=loss.item())

        recontruct_error = F.l1_loss(generated_hr_images, hr_images, reduction='mean') if criterion else None
        if recontruct_error is not None:
            #logger.info(f"Reconstruction Error (L1 Loss) for batch: {recontruct_error.item():.6f}")
            recontruct_errors.append(recontruct_error.item())
        else:
            logger.warning("No reconstruction error calculated as criterion is None.")

        # Save each image in the batch
        for i in range(lr_images.size(0)):
            save_image(lr_images[i].cpu(), os.path.join(output_dir_lr, f"lr_img_{image_counter:04d}.png"), normalize=True)
            save_image(generated_hr_images[i].cpu(), os.path.join(output_dir_hr_generated, f"gen_hr_img_{image_counter:04d}.png"), normalize=True)
            save_image(hr_images[i].cpu(), os.path.join(output_dir_hr_ground_truth, f"gt_hr_img_{image_counter:04d}.png"), normalize=True)
            image_counter += 1
            
    if criterion and len(test_loader.dataset) > 0:
        avg_test_loss = test_loss / len(test_loader.dataset)
        logger.info(f"Average Test Loss: {avg_test_loss:.6f}")
        avg_reconstruction_error = np.mean(recontruct_errors) if recontruct_errors else None
        logger.info(f"Average Reconstruction Error: {avg_reconstruction_error:.6f}")
        
        if config.other.wandb:
            wandb.summary["test_loss"] = avg_test_loss
            wandb.summary["avg_reconstruction_error"] = avg_reconstruction_error
    else:
        avg_test_loss = None # No loss calculated

    # save recostruction errors to csv
    dir = os.path.join(exp_dir, "test_outputs")
    os.makedirs(dir, exist_ok=True)
    if recontruct_errors:
        errors_file = os.path.join(dir, "errors.csv")
        with open(errors_file, "w") as f:
            f.write("Index,Reconstruction Error\n")
            for idx, error in enumerate(recontruct_errors):
                f.write(f"{idx},{error:.6f}\n")
        logger.info(f"Saved reconstruction errors to {errors_file}")

    logger.info(f"Saved {image_counter} test image sets.")
    return avg_test_loss


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    if config.other.log_print:
        if utils and hasattr(utils, 'log_printer'):
            console_handler = utils.log_printer()
        else:
            console_handler = SimpleConsoleHandler()
            logger.warning("src.utils.conutils.log_printer not found, using basic console logging.")
        logging.getLogger().addHandler(console_handler)
        logging.getLogger().setLevel(logging.INFO)

    logger.info(f"Selected Dataset - {config.dataset.name}")
    #logger.info("Experiment Configuration:\n%s", OmegaConf.to_yaml(config))
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    exp_dir = hydra_cfg['runtime']['output_dir']
    logger.info(f"Experiment output directory: {exp_dir}")
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f: OmegaConf.save(config, f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    setSeeds(config.other.get("seed", 0))
    if config.other.wandb:
        wandb_init(config, exp_dir.split(os.sep)[-1]) # Use last part of exp_dir as run name
    model_save_dir = os.path.join(exp_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)
    base_dataset_path = config.dataset.path

    logger.info("Setting up the dataloaders.")
    train_dir = os.path.join(config.dataset.path, "train")
    val_dir = os.path.join(config.dataset.path, "val")
    train_lr = os.path.join(train_dir, "low_res")
    train_hr = os.path.join(train_dir, "high_res")
    val_lr = os.path.join(val_dir, "low_res")
    val_hr = os.path.join(val_dir, "high_res")
    
    train_dataset = SuperResolutionDataset(lr_dir=train_lr, hr_dir=train_hr)
    val_dataset = SuperResolutionDataset(lr_dir=val_lr, hr_dir=val_hr)
    if len(train_dataset) == 0: logger.error("Training dataset is empty."); return
    if len(val_dataset) == 0: logger.warning("Validation dataset is empty.")
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=config.training.get("num_workers", 4), pin_memory=device=="cuda")
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=config.training.get("num_workers", 4), pin_memory=device=="cuda")
    logger.info(f"Training dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    logger.info("Initializing the SRResUNet model.")
    model = SRResUNet(in_channels=config.model.in_channels, out_channels=config.model.out_channels, num_filters=config.model.num_filters, num_residuals=config.model.num_residuals, upscale_factor=config.model.upscale_factor).to(device)
    criterion = nn.L1Loss().to(device)
    logger.info(f"Using L1Loss.")
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    logger.info(f"Using Adam optimizer, LR: {config.training.learning_rate}")
    scheduler = None
    if config.training.get("lr_scheduler_enabled", False):
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.training.lr_step_size, gamma=config.training.lr_gamma)
        logger.info(f"Using StepLR scheduler: step_size={config.training.lr_step_size}, gamma={config.training.lr_gamma}")
    else:
        logger.info("No learning rate scheduler.")

    if not config.other.get("run_testing_only", False): # Allow skipping training
        best_val_loss = float('inf')
        logger.info("Starting training loop...")
        for epoch in range(config.training.epochs):
            logger.info(f"--- Epoch {epoch+1}/{config.training.epochs} ---")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config.training.epochs, config)
            logger.info(f"Epoch {epoch+1} - Training Loss: {train_loss:.6f}")
            val_loss = float('inf')
            if len(val_loader) > 0:
                val_loss = validate_one_epoch(model, val_loader, criterion, device, epoch, config.training.epochs, config, exp_dir, log_images_wandb=config.other.wandb)
                logger.info(f"Epoch {epoch+1} - Validation Loss: {val_loss:.6f}")
            else:
                logger.warning(f"Epoch {epoch+1} - Skipping validation.")
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Current Learning Rate: {current_lr:.6e}")
            if config.other.wandb:
                log_dict = {"Epoch": epoch + 1, "Training Loss": train_loss, "Learning Rate": current_lr}
                if val_loss != float('inf'): log_dict["Validation Loss"] = val_loss
                wandb.log(log_dict, step=epoch)
            if scheduler: scheduler.step()
            if val_loss < best_val_loss and len(val_loader) > 0:
                best_val_loss = val_loss
                save_path = os.path.join(model_save_dir, "best_model.pth")
                torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': best_val_loss, 'config': OmegaConf.to_container(config, resolve=True)}, save_path)
                logger.info(f"Saved new best model (Val Loss: {best_val_loss:.6f}) to {save_path}")
        
        final_model_path = os.path.join(model_save_dir, "final_model.pth")
        torch.save({'epoch': config.training.epochs, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'final_train_loss': train_loss, 'final_val_loss': val_loss, 'config': OmegaConf.to_container(config, resolve=True)}, final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
        logger.info("Training completed.")
    else:
        logger.info("Skipping training as run_testing_only is set to True.")


    # --- Testing Phase ---
    if config.other.get("run_testing", False): # Check if testing is enabled in config
        logger.info("--- Starting Testing Phase ---")
        
        # Determine which model to load for testing
        if config.other.run_testing_only:
            model_save_dir = config.other.testing_only_model_path
        model_to_test_path =""
        if config.other.get("use_best_model", True) and os.path.exists(os.path.join(model_save_dir, "best_model.pth")):
            model_to_test_path = os.path.join(model_save_dir, "best_model.pth")
            logger.info(f"Loading best model for testing from: {model_to_test_path}")
        elif os.path.exists(os.path.join(model_save_dir, "final_model.pth")):
            model_to_test_path = os.path.join(model_save_dir, "final_model.pth")
            logger.info(f"Loading final model for testing from: {model_to_test_path}")
        else:
            logger.error("No model found to load for testing. Please train a model first or provide a checkpoint path in config.")
            if config.other.wandb: wandb.finish()
            return

        checkpoint = torch.load(model_to_test_path, map_location=device)
        # Re-initialize model architecture before loading state_dict
        test_model = SRResUNet(in_channels=config.model.in_channels, out_channels=config.model.out_channels, num_filters=config.model.num_filters, num_residuals=config.model.num_residuals).to(device)
        test_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model state loaded from epoch {checkpoint.get('epoch', 'N/A')}.")

        test_dir = os.path.join(config.dataset.path, "test")
        test_dataset_path_lr = os.path.join(test_dir, "low_res")
        test_dataset_path_hr = os.path.join(test_dir, "high_res")

        if not os.path.exists(test_dataset_path_lr) or not os.path.exists(test_dataset_path_hr):
            logger.warning(f"Test dataset path not found ({test_dataset_path_lr} or {test_dataset_path_hr}). Skipping testing.")
        else:
            test_dataset = SuperResolutionDataset(
                lr_dir=test_dataset_path_lr,
                hr_dir=test_dataset_path_hr
            )
            if len(test_dataset) > 0:
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=1, 
                    shuffle=False,
                    num_workers=config.training.get("num_workers", 4), 
                    pin_memory=device=="cuda"
                )
                logger.info(f"Test dataset size: {len(test_dataset)}, Test dataloader batches: {len(test_loader)}")
                test_model_and_save_outputs(test_model, test_loader, criterion if config.other.get("testing_loss", True) else None, device, config, exp_dir)
            else:
                logger.warning("Test dataset is empty. Skipping testing.")
    else:
        logger.info("Testing phase skipped as per configuration.")


    if config.other.wandb:
        if not config.other.run_testing_only:
            wandb.summary["best_validation_loss"] = best_val_loss
        wandb.finish()
    
    logger.info("Experiment finished.")

if __name__ == "__main__":
    main()
