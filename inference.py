import io
import os
import argparse
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from torchvision import transforms
from PIL import Image

from net.models.SUM import SUM, SUM_graph
from net.configs.config_setting import setting_config

from Utils import *
from settings import * 

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

splits_root_folder = "/home/jovyan/Torte/splits/"

def setup(rank, world_size):
    # Initialize the process group for distributed training
    dist.init_process_group(backend="nccl", init_method='env://', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def setup_model(device):
    config = setting_config
    model_cfg = config.model_config

    #if config.network == 'sum':
    model = SUM_graph()(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        )

    state_dict = torch.load('/home/jovyan/SUM-main-graph/best_model_crosscdr_multiGPU_v2_SUM_with_graph_defaultLoss_ANDOSIM_finetuning_enlargedSaliency.pth', map_location="cuda:0")
    state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    print(f"SUM_Model pre-trained is loaded")

    model.to("cuda:0")
    return model
    


def load_and_preprocess_image(img_path):
    image = Image.open(img_path).convert('RGB')
    orig_size = image.size
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image, orig_size


def saliency_map_prediction(img_path, condition, model, device):
    img, orig_size = load_and_preprocess_image(img_path)
    img = img.unsqueeze(0).to(device)
    one_hot_condition = torch.zeros((1, 4), device=device)
    one_hot_condition[0, condition] = 1
    model.eval()
    with torch.no_grad():
        pred_saliency = model(img, one_hot_condition)

    pred_saliency = pred_saliency.squeeze().cpu().numpy()
    return pred_saliency, orig_size



def overlay_heatmap_on_image(original_img, heatmap, alpha=0.25):
    """
    Overlays a heatmap onto the original image.
    
    Args:
    - original_img (numpy array): The original image.
    - heatmap (numpy array): The heatmap to overlay.
    - alpha (float): Transparency of the heatmap.
    
    Returns:
    - overlay_image (numpy array): The final overlaid image.
    """
    # Ensure heatmap is a NumPy array and not None
    if heatmap is None:
        raise ValueError("Heatmap is None. Ensure that a valid heatmap is provided.")

    heatmap = np.array(heatmap, dtype=np.float32)

    # Normalize heatmap to [0, 255]
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)  # Normalize to [0,1]
    heatmap = np.uint8(heatmap * 255)  # Convert to [0,255] uint8

    # Ensure heatmap is single-channel grayscale (CV_8UC1)
    if len(heatmap.shape) == 3 and heatmap.shape[0] == 1:  
        heatmap = heatmap.squeeze(0)  # Remove channel dim if it's (1,H,W)
    elif len(heatmap.shape) == 3 and heatmap.shape[-1] == 3:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if RGB

    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Apply colormap (Ensuring correct input format)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Ensure original image is in correct format
    if original_img.max() <= 1:
        original_img = np.uint8(original_img * 255)  # Convert from [0,1] to [0,255]

    # Convert grayscale to 3-channel if needed
    if len(original_img.shape) == 2:  
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    # Blend the heatmap with the original image
    overlay_image = cv2.addWeighted(original_img, 1, heatmap_color, alpha, 0)

    return overlay_image



def overlay_images(background, heatmap, alpha=0.4, colormap=cv2.COLORMAP_TURBO):
    """ Overlays a heatmap onto a background image with better contrast. """

    # Convert heatmap to NumPy & normalize
    heatmap = np.array(heatmap, dtype=np.float32)
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)  # Normalize to [0,1]
    heatmap = np.uint8(heatmap * 255)  # Convert to [0,255] uint8

    # Ensure heatmap is grayscale
    if len(heatmap.shape) == 3 and heatmap.shape[0] == 1:  
        heatmap = heatmap.squeeze(0)  # Remove channel dim if it's (1,H,W)
    elif len(heatmap.shape) == 3 and heatmap.shape[-1] == 3:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if RGB

    # Apply different colormap to avoid excess blue
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # Convert background from PyTorch tensor if necessary
    if isinstance(background, torch.Tensor):  
        background = background.cpu().numpy()
        background = np.transpose(background, (1, 2, 0))  # Convert (C, H, W) to (H, W, C)

    # Ensure background is in range [0,255] and has 3 channels
    background = np.uint8(background * 255) if background.max() <= 1 else np.uint8(background)
    if len(background.shape) == 2:  
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

    # Blend heatmap & background with controlled transparency
    overlay = cv2.addWeighted(background, 1 - alpha, heatmap, alpha, 0)

    return overlay


import time

def save_and_verify_image(image_tensor, save_path, is_rgb=False, max_retries=5):
    """
    Saves an image tensor as a file and verifies that it is written before proceeding.

    Args:
        image_tensor (torch.Tensor): The image tensor to save.
        save_path (str): The full path where the image should be saved.
        is_rgb (bool): Whether to save the image as RGB.
        max_retries (int): Maximum retries to ensure the file is written.

    Returns:
        bool: True if the image is saved successfully, False otherwise.
    """
    # Save the image
    save_tensor_as_image(image_tensor, save_path, is_rgb)

    # Verify the image is saved
    for _ in range(max_retries):
        import os
        if os.path.exists(save_path):
            return True
        time.sleep(0.2)  # Small delay before retrying

    print(f"⚠️ Warning: Failed to verify image save at {save_path}")
    return False

import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os

def load_and_preprocess_image(image_path, img_size=(256, 256), device="cuda"):
    """
    Reads an RGB image from a given path, preprocesses it, and prepares it as input for a neural network.

    Args:
        image_path (str): Path to the image file.
        img_size (tuple): Target size for resizing (default: (224, 224)).
        device (str): Device to load the image onto ('cuda' or 'cpu').

    Returns:
        torch.Tensor: Preprocessed image ready for a neural network.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Load image using PIL (preserves color channels correctly)
    image = Image.open(image_path).convert("RGB")

    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize(img_size),       # Resize image
        transforms.ToTensor(),             # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained models
    ])

    # Apply transformations
    input_tensor = preprocess(image)

    # Add batch dimension (for NN inference)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # Shape: (1, 3, H, W)

    return input_tensor


def inference(rank, world_size):
    #setup(rank, world_size)

    # Create output directories
    output_path = "folder"
    import os
    os.makedirs(output_path, exist_ok=True)

    stimuli_dir = os.path.join(output_path, "stimuli")
    saliency_dir = os.path.join(output_path, "saliency")
    panoptic_dir = os.path.join(output_path, "panoptic")
    prediction_dir = os.path.join(output_path, "prediction")
    overlay_dir = os.path.join(output_path, "overlay")  # New overlay directory

    for directory in [stimuli_dir, saliency_dir, panoptic_dir, prediction_dir, overlay_dir]:
        os.makedirs(directory, exist_ok=True)

    config = setting_config
    model_cfg = config.model_config

    if config.network == 'sum':
        model = SUM_graph(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            #load_ckpt_path=model_cfg['load_ckpt_path'],
            ).to("cuda:0")

    state_dict = torch.load('/home/jovyan/Torte/SUM-main-graph/SUMGraph_NooSIM_withPretrainedSUM_v5_4.pth', map_location="cuda:0")
    state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print(f"SUM_Model pre-trained is loaded")

    batch_size = 1  # Adjust based on available GPU memory
    inference_dataset = get_datasets_inference(splits_root_folder)
    #train_sampler = DistributedSampler(inference_dataset, num_replicas=world_size, rank=0, shuffle=True, drop_last=True)
    train_loader = DataLoader(dataset=inference_dataset, batch_size=batch_size // world_size, num_workers=0, collate_fn=custom_collate_fn)
    load_dataset = True
    
    device = "cuda:0"

    if load_dataset:
        with torch.no_grad():
            model.eval()
            #for batch_idx, batch in enumerate(tqdm(train_loader, desc="Inference")):
            for idx in range(len(inference_dataset)):

                batch = inference_dataset.__getitem__(idx)
                
                stimuli_cpu, smap_cpu, fmap_cpu, condition_cpu, panoptic_cpu = batch['image'].unsqueeze(0)  , batch['saliency'].unsqueeze(0)  , batch['fixation'].unsqueeze(0)  , batch['label'].unsqueeze(0)  , batch['panoptic'].unsqueeze(0)  

                # Move tensors to device
                stimuli, smap, fmap, condition, panoptic = (
                    stimuli_cpu.to(device, non_blocking=True),
                    smap_cpu.to(device, non_blocking=True),
                    fmap_cpu.to(device, non_blocking=True),
                    condition_cpu.to(device, non_blocking=True),
                    panoptic_cpu.to(device, non_blocking=True)
                )


                if enable_graph:
                    graphs_per_image = []
                    global_features_per_image = []

                    graph_cpu = batch['graph']  
                    #print(f"graph_cpu{graph_cpu}")
                    image_to_graph_indices_cpu = batch['graph_image_indices']
                    #print(f"image_to_graph_indices_cpu:{image_to_graph_indices_cpu}")

                    if enable_global_graph_features:
                        graph_global_features_cpu = batch['graph_global_features']
                        # ✅ Convert list of tensors to a single tensor before indexing
                        if isinstance(graph_global_features_cpu, list):
                            graph_global_features_cpu = torch.stack(graph_global_features_cpu)
                        #print(f"✅ Global Features Shape BEFORE Indexing: {graph_global_features_cpu.shape}") 

                    for i in range(len(stimuli_cpu)):  # Iterate over images in batch
                        graph_indices = (image_to_graph_indices_cpu == i).nonzero(as_tuple=True)[0]  # Get graph indices for this image
                        #print(f"Image {i} has {len(graph_indices)} graphs assigned")

                        # ✅ Extract Graphs for Image
                        if isinstance(graph_cpu, list):
                            graphs_for_image = [graph_cpu[idx] for idx in graph_indices.tolist()]
                        elif isinstance(graph_cpu, torch.Tensor):
                            graphs_for_image = graph_cpu.index_select(0, graph_indices.long())

                        graphs_per_image.append(graphs_for_image)

                        # ✅ Extract Corresponding Global Features
                        if enable_global_graph_features:
                            global_features_for_image = graph_global_features_cpu.index_select(0, graph_indices.long())
                            global_features_per_image.append(global_features_for_image)

                    # ✅ Flatten `graphs_per_image` before batching
                    flat_graphs = [graph for sublist in graphs_per_image for graph in sublist]
                    #print(f"flat_graphs {flat_graphs}")
                    # ✅ Convert to PyG Batch format

                    if len(flat_graphs) == 0:
                        #flat_graphs = None
                        outputs = model(stimuli, condition, None, None)
                        print("Graph is none")
                    else:

                        graph_batches = Batch.from_data_list(flat_graphs)
                        print("Graph is gound",)

                        # ✅ Ensure global feature tensor is not empty before concatenation
                        if global_features_per_image:
                            graph_global_features = torch.cat(global_features_per_image, dim=0)



                        if enable_graph:
                            # Transfer nested_batch to GPU
                            graph_batches_gpu = graph_batches.to(device, non_blocking=True)
                            if enable_global_graph_features:
                                graph_global_features_gpu = graph_global_features.to(device, non_blocking=True)

                        if enable_graph:
                            # Pass stimuli and batched graphs to the model
                            
                            outputs = model(stimuli, condition, graph_batches_gpu, graph_global_features_gpu)
                            #outputs = model(stimuli, condition, graph_batches_gpu, None)

                        else:
                            # Run model inference
                            outputs = model(stimuli, condition)

            
                #if rank == 0:
                # Save all images in batch uniquely
                for i in range(stimuli.shape[0]):  # Loop over batch size
                    #img_id = batch_idx * stimuli.shape[0] + i  # Ensure unique filename per image

                    # Define file paths
                    stimuli_path = os.path.join(stimuli_dir, f"stimuli_{idx}_{i}_{0}.png")
                    saliency_path = os.path.join(saliency_dir, f"saliency_{idx}_{i}_{0}.png")
                    panoptic_path = os.path.join(panoptic_dir, f"panoptic_{idx}_{i}_{0}.png")
                    pred_path = os.path.join(prediction_dir, f"prediction_{idx}_{i}_{0}.png")

                    # Save and verify each image before proceeding
                    save_and_verify_image(stimuli_cpu[i], stimuli_path, True)
                    #if not save_and_verify_image(stimuli_cpu[i], stimuli_path, True):
                    #    continue
                    save_and_verify_image(smap_cpu[i], saliency_path)
                    #if not save_and_verify_image(smap_cpu[i], saliency_path):
                    #    continue
                    save_and_verify_image(panoptic_cpu[i], panoptic_path, True)
                    #if not save_and_verify_image(panoptic_cpu[i], panoptic_path, True):
                    #    continue
                    #if not save_and_verify_image(outputs[i].cpu(), pred_path, False):
                    #    continue
                    save_and_verify_image(outputs[i].cpu(), pred_path, False)

                    # Load the saved stimuli image (ensure it exists)
                    stimuli_img = cv2.imread(stimuli_path)
                    if stimuli_img is None:
                        print(f"⚠️ Error: Failed to load saved stimuli image for batch {idx}, index {i}")
                        continue  # Skip if loading failed
                            
                    # Overlay heatmaps
                    overlay_gt = overlay_heatmap_on_image(stimuli_img, smap_cpu[i].cpu().numpy(), alpha=0.25)
                    overlay_pred = overlay_heatmap_on_image(stimuli_img, outputs[i].cpu().numpy(), alpha=0.25)

                    # Save overlays uniquely
                    cv2.imwrite(os.path.join(overlay_dir, f"overlay_gt_{idx}_{i}_{0}.png"), overlay_gt)
                    cv2.imwrite(os.path.join(overlay_dir, f"overlay_pred_{idx}_{i}_{0}.png"), overlay_pred)
    
    else:
        img = load_and_preprocess_image("/home/jovyan/Torte/inference_input/rgb.png")
        condition = torch.tensor(1)

        # Add a batch dimension (converts to shape [1])
        condition = condition.unsqueeze(0)  


        outputs = model(img, condition)

        pred_path = os.path.join("/home/jovyan/Torte/inference_input/", "prediction_rgb.png")
        save_and_verify_image(outputs.cpu(), pred_path, False)

    print("Inference complete. Results saved in:", output_path)

def main():

    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    '''import warnings
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", module="timm")  # Example for NumPy warnings
    warnings.simplefilter("ignore")
    mp.set_start_method('spawn', force=True)
    mp.spawn(inference, args=(1,), nprocs=1, join=True)
    '''
    inference(1, 1)
if __name__ == "__main__":
    main()
