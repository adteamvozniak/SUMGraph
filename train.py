import copy
import torch

import gc
import time
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from utils.loss_function import SaliencyLoss
from utils.data_process_uni import TrainDataset,ValDataset
    
from net.models.SUM import SUM, SUM_graph
from net.configs.config_setting import setting_config


from torch.utils.tensorboard import SummaryWriter


import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import sys
import os


# Set NCCL environment variables
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_TIMEOUT"] = "3600"  # Increase to 1 hour or more
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils import *
from settings import enable_graph

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")

# Initialize TensorBoard Writer
writer = SummaryWriter(log_dir="runs/training_report_") # id is to be specified

def setup(rank, world_size):
    # Initialize the process group for distributed training
    dist.init_process_group(backend="nccl", init_method='env://', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    print(f"Running on rank {rank} (GPU {rank})")
    setup(rank, world_size)
    batch_size = 10*7 # 75 frames per GPU is max for A100 80GB. It to be adjusted based on the setup
        
    train_dataset, test_dataset, val_dataset = get_datasets(splits_root_folder)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size//world_size, num_workers=7, pin_memory=True, prefetch_factor=9, collate_fn=custom_collate_fn, sampler=train_sampler, drop_last=True)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loaders = DataLoader(dataset = val_dataset, batch_size=batch_size//world_size, num_workers=7, pin_memory=True, prefetch_factor=9, collate_fn=custom_collate_fn, sampler=val_sampler, drop_last=True)

    config = setting_config

    model_cfg = config.model_config

    if enable_graph:
        if config.network == 'sum':
            model = SUM_graph(
                num_classes=model_cfg['num_classes'],
                input_channels=model_cfg['input_channels'],
                depths=model_cfg['depths'],
                depths_decoder=model_cfg['depths_decoder'],
                drop_path_rate=model_cfg['drop_path_rate']
            ).to(rank)

        
        # Load weights on the main process (rank 0) only
        if rank == 0 and model.load_ckpt_path is not None:
            model.load_from()  # Load the pretrained weights from the checkpoint

        if rank== 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total number of parameters in the model: {total_params}")
            state_dict = torch.load('/home/jovyan/Torte/SUM-main-graph/net/pre_trained_weights/sum_model.pth', map_location="cuda:0")
            state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=False)
            print(f"SUM_Model initial is loaded")

            # Get model's existing state_dict keys
            existing_model_keys = set(model.state_dict().keys())
            loaded_state_dict_keys = set(state_dict.keys())

            # Check for missing and unexpected keys manually
            missing_keys = existing_model_keys - loaded_state_dict_keys
            unexpected_keys = loaded_state_dict_keys - existing_model_keys
            loaded_correctly = existing_model_keys.intersection(loaded_state_dict_keys)

            print(f"✅ Successfully loaded {len(loaded_correctly)} existing layers.")

        dist.barrier(device_ids=[rank])
        model = DDP(model, device_ids=[rank])
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    loss_fn = SaliencyLoss() 
    mse_loss = nn.MSELoss()

    # Training and Validation Loop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    num_epochs = 20

    # Early stopping setup
    early_stop_counter = 0
    early_stop_threshold = 4

    global_step = 0  # Track the batch number across epochs

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_sampler.set_epoch(epoch)

        # Training Phase
        model.train()
        metrics = {'loss': [], 'kl': [], 'cc': [], 'sim': [], 'nss': [], 'osim': []}

        first_batch_processed = True  # Track if we've processed the first batch
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):           
            # Measure data loading time
            load_start = time.time()
            stimuli_cpu, smap_cpu, fmap_cpu, condition_cpu, panoptic_cpu = batch['image'], batch['saliency'], batch['fixation'], batch['label'], batch['panoptic']
            if enable_graph:
                graphs_per_image = []
                global_features_per_image = []

                graph_cpu = batch['graph']
                image_to_graph_indices_cpu = batch['graph_image_indices']

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

                # ✅ Convert to PyG Batch format
                graph_batches = Batch.from_data_list(flat_graphs)
                #print(f"✅ graph_batches Created! Contains {len(flat_graphs)} graphs.")
                #print(f"🔹 Sample Graph Data: {graph_batches[0]}")  # Print first graph for debugging
                #print(f"🔹 Batch size: {graph_batches.num_graphs}")  # Number of graphs in batch

                # ✅ Ensure global feature tensor is not empty before concatenation
                if global_features_per_image:
                    #print(f"Global Features Before Cat - Per Image: {[x.shape for x in global_features_per_image]}")
                    graph_global_features = torch.cat(global_features_per_image, dim=0)
                    #print(f"✅ graph_global_features Created! Shape: {graph_global_features.shape}")
                    #print(f"🔹 Sample Feature Vector: {graph_global_features[0]}")  # Print first vector for debugging

                load_time = time.time() - load_start
                if dist.get_rank() == 0:
                    print(f"DataLoader time for Batch {batch_idx}: {load_time:.4f} seconds")
    
                start = time.time()
                # Transfer data to GPU with non-blocking to allow asynchronous transfers if possible
                stimuli, smap, fmap, condition, panoptic = (
                    stimuli_cpu.to(rank, non_blocking=True),
                    smap_cpu.to(rank, non_blocking=True),
                    fmap_cpu.to(rank, non_blocking=True),
                    condition_cpu.to(rank, non_blocking=True),
                    panoptic_cpu.to(rank, non_blocking=True),
                    
                )
                dist.barrier(device_ids=[rank])  # Synchronize across all GPUs for timing consistency
                if enable_graph:
                    # Transfer nested_batch to GPU
                    graph_batches_gpu = graph_batches.to(rank, non_blocking=True)
                    if enable_global_graph_features:
                        graph_global_features_gpu = graph_global_features.to(rank, non_blocking=True)

                    #print(f"✅ Graphs GPU: {graph_batches_gpu.num_graphs}")
                    #print(f"✅ Global Features GPU Shape: {graph_global_features_gpu.shape}")

                dist.barrier(device_ids=[rank])  # Synchronize across all GPUs for timing consistency
                batch_time = time.time() - start
                if dist.get_rank() == 0:
                    print(f"Batch {batch_idx} time on GPU {dist.get_rank()}: {batch_time:.4f} seconds")
            optimizer.zero_grad()

            if enable_graph:
                # Pass stimuli and batched graphs to the model               
                outputs = model(stimuli, condition, graph_batches_gpu, graph_global_features_gpu)
            else:
                outputs = model(stimuli, condition)
            
            # Compute losses
            kl = loss_fn(outputs, smap, loss_type='kldiv')
            cc = loss_fn(outputs, smap, loss_type='cc')
            sim = loss_fn(outputs, smap, loss_type='sim')
            nss = loss_fn(outputs, fmap, loss_type='nss')
            expMSE = loss_fn(smap, outputs, loss_type='expMSE')
            osim = loss_fn(outputs, smap, panoptic, None, loss_type='osim_v4')
            
            loss1 = compute_loss(cc, kl, sim, nss, osim)
            loss2 = mse_loss(outputs, smap)
            loss = loss1+loss2

            # Log loss and accuracy per batch
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            writer.add_scalar("KL/train_batch", kl.item(), global_step)
            writer.add_scalar("SIM/train_batch", sim.item(), global_step)
            writer.add_scalar("oSIM/train_batch", osim.item(), global_step)
            writer.add_scalar("NSS/train_batch", nss.item(), global_step)
            writer.add_scalar("CC/train_batch", cc.item(), global_step)

            loss.backward()
            optimizer.step()

            torch.cuda.synchronize(rank)  # In DDP context

            # Print the loss and batch ID
            print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

            if not np.isnan(loss.item()): metrics['loss'].append(loss.item())
            if not np.isnan(kl.item()): metrics['kl'].append(kl.item())
            if not np.isnan(cc.item()): metrics['cc'].append(cc.item())
            if not np.isnan(sim.item()): metrics['sim'].append(sim.item())
            if not np.isnan(nss.item()): metrics['nss'].append(nss.item())
            if not np.isnan(osim.item()): metrics['osim'].append(osim.item())
            
            global_step += 1  # Increment batch counter
        
        scheduler.step()

        # Calculate mean and std dev for each metric
        for metric in metrics.keys():
            metrics[metric] = (np.mean(metrics[metric]), np.std(metrics[metric]))
        
        # Print training metrics with mean and std dev
        print("Train - " + ", ".join([f"{metric}: {mean:.4f} ± {std:.4f}" for metric, (mean, std) in metrics.items()]))
        
        # cleaning memory
        del loss, outputs, kl, cc, sim, nss, expMSE
        torch.cuda.empty_cache()
        gc.collect()

        # Validation Phase
        model.eval()

        val_metrics = {'loss': [], 'kl': [], 'cc': [], 'sim': [], 'nss': [], 'auc':[], 'osim': [] }

        with torch.no_grad():
            for batch in tqdm(val_loaders, desc=f"Validating"):
                # Print or store batch length as needed
                #print(f"Batch length: {batch_len}")
                stimuli, smap, fmap, condition, panoptic = batch['image'].to(rank, non_blocking=True), batch['saliency'].to(rank, non_blocking=True), batch['fixation'].to(rank, non_blocking=True), batch['label'].to(rank, non_blocking=True), batch['panoptic'].to(rank, non_blocking=True)
                dist.barrier(device_ids=[rank])
                if enable_graph:
                    graphs_per_image = []
                    global_features_per_image = []

                    graph_cpu = batch['graph']
                    image_to_graph_indices_cpu = batch['graph_image_indices']

                    if enable_global_graph_features:
                        graph_global_features_cpu = batch['graph_global_features']
                        # ✅ Convert list of tensors to a single tensor before indexing
                        if isinstance(graph_global_features_cpu, list):
                            graph_global_features_cpu = torch.stack(graph_global_features_cpu)


                    for i in range(len(stimuli_cpu)):  # Iterate over images in batch
                        graph_indices = (image_to_graph_indices_cpu == i).nonzero(as_tuple=True)[0]  # Get graph indices for this image

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

                    # ✅ Convert to PyG Batch format
                    graph_batches = Batch.from_data_list(flat_graphs)
                    #print(f"✅ graph_batches Created! Contains {len(flat_graphs)} graphs.")
                    #print(f"🔹 Sample Graph Data: {graph_batches[0]}")  # Print first graph for debugging
                    #print(f"🔹 Batch size: {graph_batches.num_graphs}")  # Number of graphs in batch

                    # ✅ Ensure global feature tensor is not empty before concatenation
                    if global_features_per_image:
                        graph_global_features = torch.cat(global_features_per_image, dim=0)
                        #print(f"✅ graph_global_features Created! Shape: {graph_global_features.shape}")
                        #print(f"🔹 Sample Feature Vector: {graph_global_features[0]}")  # Print first vector for debugging
    
                    graph_batches_gpu = graph_batches.to(rank, non_blocking=True)
                    graph_global_features_gpu = graph_global_features.to(rank, non_blocking=True)
                    dist.barrier(device_ids=[rank])
                    # Pass stimuli and batched graphs to the model
                    outputs = model(stimuli, condition, graph_batches_gpu, graph_global_features_gpu)
                else:
                    outputs = model(stimuli, condition)

                # Compute losses
                kl = loss_fn(outputs, smap, loss_type='kldiv')
                cc = loss_fn(outputs, smap, loss_type='cc')
                sim = loss_fn(outputs, smap, loss_type='sim')
                nss = loss_fn(outputs, fmap, loss_type='nss')
                osim = loss_fn(outputs, smap, panoptic, None, loss_type='osim')
                    
                loss1 = compute_loss(cc, kl, sim, nss, osim, None)
                loss2 = mse_loss(outputs, smap)
                loss = loss1+loss2

                if not np.isnan(loss.item()): val_metrics['loss'].append(loss.item())
                if not np.isnan(kl.item()): val_metrics['kl'].append(kl.item())
                if not np.isnan(cc.item()): val_metrics['cc'].append(cc.item())
                if not np.isnan(sim.item()): val_metrics['sim'].append(sim.item())
                if not np.isnan(nss.item()): val_metrics['nss'].append(nss.item())
                if not np.isnan(osim.item()): val_metrics['osim'].append(osim.item())

                if rank == 0:
                # Log loss and accuracy per batch
                    writer.add_scalar("Loss/val_batch", loss.item(), global_step)
                    writer.add_scalar("KL/val_batch", kl.item(), global_step)
                    writer.add_scalar("SIM/val_batch", sim.item(), global_step)
                    writer.add_scalar("oSIM/val_batch", osim.item(), global_step)
                    writer.add_scalar("NSS/val_batch", nss.item(), global_step)
                    writer.add_scalar("CC/val_batch", cc.item(), global_step)

            total_val_loss = np.sum(val_metrics['kl'])

            # Check for best model
            if total_val_loss < best_loss:
                print(f"New best model found at epoch {epoch+1}!")
                best_loss = total_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, 'best_model.pth')
                early_stop_counter = 0  # Reset counter after improvement
            else:
                early_stop_counter += 1
                print(f"No improvement in Total Val Loss for {early_stop_counter} epoch(s).")

            # Early stopping check
            if early_stop_counter >= early_stop_threshold:
                print("Early stopping triggered.")
                cleanup()
                break
    cleanup()

def main():
    import os
    import warnings
    import torch
    import torch.multiprocessing as mp

    # Set which GPUs are visible to the script (here, GPUs 0 through 9)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7,8,9"

    # Determine the number of available GPUs
    world_size = torch.cuda.device_count()

    # Set environment variables for distributed training setup
    os.environ['MASTER_ADDR'] = 'localhost'  # Address of the main process
    os.environ['MASTER_PORT'] = '12355'      # Communication port for processes
    os.environ['WORLD_SIZE'] = str(world_size)  # Total number of processes
    os.environ['RANK'] = '0'  # Rank of the main process (usually 0)

    # Suppress most types of warnings to reduce clutter in the output
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", module="timm")  # Suppress warnings from the `timm` module
    warnings.simplefilter("ignore")  # General suppression of warnings

    # Use 'spawn' start method for multiprocessing — safer and more compatible with CUDA
    mp.set_start_method('spawn', force=True)

    # Launch the training function on each GPU as a separate process
    # 'train' should be a function defined elsewhere in your code
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
