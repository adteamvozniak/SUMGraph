import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
from torchvision import transforms
import gc
from tqdm import tqdm

from utils.loss_function import SaliencyLoss
#from utils.data_process_uni import TrainDataset, ValDataset

from net.models.SUM import SUM, SUM_graph

from net.configs.config_setting import setting_config


import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Set NCCL environment variables
os.environ["NCCL_BLOCKING_WAIT"] = "1"
#os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_TIMEOUT"] = "3600"  # Increase to 1 hour or more

sys.path.append(os.path.abspath('/home/jovyan/Torte'))

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils import *
from settings import * 

import warnings
warnings.filterwarnings("ignore")

#CUDA_LAUNCH_BLOCKING=1

enable_graph = True
single_GPU = False
multipleDatasets = False

def setup(rank, world_size):
    # Initialize the process group for distributed training
    dist.init_process_group(backend="nccl", init_method='env://', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def test(rank=0, world_size=1):
    print(f"Running on rank {rank} (GPU {rank})")

    if not single_GPU:
        setup(rank, world_size)

    batch_size = 200*10  #since no gradient is required. It to be set smaller for deepGaze2E

    test_dataset = get_datasets_testing(splits_root_folder) # only testing is requred


    if single_GPU:
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=10, pin_memory=True)

    else:
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size//world_size, num_workers=9, prefetch_factor=5, pin_memory=True, collate_fn=custom_collate_fn, sampler=test_sampler)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = setting_config

    model_cfg = config.model_config

    from collections import Counter
    # Global accumulators
    gt_focus_total = Counter()
    pred_focus_total = Counter()
    n_samples = 0

    if single_GPU:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if config.network == 'sum': 
            model = SUM(
                num_classes=model_cfg['num_classes'],
                input_channels=model_cfg['input_channels'],
                depths=model_cfg['depths'],
                depths_decoder=model_cfg['depths_decoder'],
                drop_path_rate=model_cfg['drop_path_rate'],
                #load_ckpt_path=model_cfg['load_ckpt_path'],
            )
            model.load_from()
            print("Here is the model=",model)
            model.cuda()
    else:
        #dist.barrier()
        model = SUM_graph(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            #load_ckpt_path=model_cfg['load_ckpt_path'],
        ).to(rank)

        if rank == 0:
            print("Model State Dictionary Keys:")
            #for key in model.state_dict().keys():
            #    print(f"Models keys={key}")

        # Load weights on the main process (rank 0) only
        if rank == 0 and model.load_ckpt_path is not None:
            model.load_from()  # Load the pretrained weights from the checkpoint
        if rank == 0:
            #print("Checkpoint State Dictionary Keys:")
            #checkpoint_pretrained = torch.load('/home/jovyan/Torte/SUM-main-graph/best_model_crosscdr_multiGPU_v2_SUM_with_graph_defaultLoss_v3Model.pth', map_location=device)
            #checkpoint_pretrained = torch.load('/home/jovyan/Torte/SUM-main-graph/SUMGraph_NooSIM.pth', map_location=device)
            checkpoint_pretrained = torch.load('/home/jovyan/Torte/SUM-main-graph/SUMGraph_oSIM_fullDataset_v4_1_background_vs_relevant_classification.pth', map_location=device)
            #state_dict = checkpoint_pretrained['model']
            #print(f"We print the dictionary of the checkpoint in DDP")

            #state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in checkpoint_pretrained.items()}

            #for key in state_dict.keys():
            #    print(key)
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded again...")
        dist.barrier()
        model = DDP(model, device_ids=[rank])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params}")

    # Load the pre-trained model weights
    # the model is to be added here
    #exit(1)    
    
    

    #missing_keys, unexpected_keys = model.load_state_dict(checkpoint_pretrained, strict=False)
    #print("Missing keys:", missing_keys)
    #print("Unexpected keys:", unexpected_keys)
    
    
    #model.to('cuda')

    print(f"Trained model loaded !!!!!!!!!!!!!!!")
    # Function for performing validation inference
    first_sample_processed = False
    loss_fn = SaliencyLoss()
    #model.eval()  # Set model to evaluation mode
    test_metrics = {'kl': [], 'cc': [], 'sim': [], 'nss': [], 'auc': [], 'osim': []}
    # Iterate through each validation dataset
    model.eval()
    for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Testing model")):
        if single_GPU:
            stimuli, smap, fmap, condition, panoptic = batch['image'].to(device), batch['saliency'].to(device), batch['fixation'].to(device), batch['label'].to(device), batch['panoptic'].to(devide)
        else:
            #if batch_idx % 50 == 0:
            #torch.cuda.empty_cache()

            # Measure data loading time
            load_start = time.time()
            stimuli_cpu, smap_cpu, fmap_cpu, condition_cpu, panoptic_cpu = batch['image'], batch['saliency'], batch['fixation'], batch['label'], batch['panoptic']
            #dist.barrier()
        
        
        with torch.no_grad():
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



                # Transfer data to GPU with non-blocking to allow asynchronous transfers if possible
                stimuli, smap, fmap, condition, panoptic = (
                    stimuli_cpu.to(rank, non_blocking=True),
                    smap_cpu.to(rank, non_blocking=True),
                    fmap_cpu.to(rank, non_blocking=True),
                    condition_cpu.to(rank, non_blocking=True),
                    panoptic_cpu.to(rank, non_blocking=True)
                )

                if enable_graph:
                    # Transfer nested_batch to GPU
                    graph_batches_gpu = graph_batches.to(rank, non_blocking=True)
                    if enable_global_graph_features:
                        graph_global_features_gpu = graph_global_features.to(rank, non_blocking=True)
                #dist.barrier()
                # Pass stimuli and batched graphs to the model

                # include here that additional logic

                
                if not first_sample_processed:
                    print("🔄 Re-loading model weights after first forward pass...")
                    
                    _ = model(stimuli, condition, graph_batches_gpu, graph_global_features_gpu)
                    print("Dummy run conducted...")


                    if rank == 0:
                        checkpoint_pretrained = torch.load('/home/jovyan/Torte/SUM-main-graph/SUMGraph_oSIM_fullDataset_v4_1_background_vs_relevant_classification.pth', map_location='cuda:0')
                        #state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in checkpoint_pretrained.items()}
                        model.load_state_dict(state_dict=checkpoint_pretrained, strict=True)
                        print(f"Entire model is now loaded...")
                        
                        

                    dist.barrier()

                    # Now broadcast model params from rank 0 to everyone
                    for param in model.parameters():
                        torch.distributed.broadcast(param.data, src=0)
                    first_sample_processed = True  # Ensure this only happens once

                # final

                outputs = model(stimuli, condition, graph_batches_gpu, graph_global_features_gpu)
            else:
                outputs = model(stimuli, condition)           

            # Compute losses
            kl = loss_fn(outputs, smap, loss_type='kldiv')
            cc = loss_fn(outputs, smap, loss_type='cc')
            sim = loss_fn(outputs, smap, loss_type='sim')
            nss = loss_fn(outputs, fmap, loss_type='nss')
            auc = loss_fn(outputs, fmap, loss_type='auc')
            
            osim = loss_fn(outputs, smap, panoptic, loss_type='osim_v4')
            #loss = -2 * cc + 10 * kl - 1 * sim - 1 * nss

            #print(f"kl={kl.item()}, cc={cc.item()}, sim={sim.item()}, nss={nss.item()}, auc={auc.item()}, expMSE={expMSE.item()}, osim={osim.item()},")
            # Accumulate raw metric values
            #test_metrics['loss'].append(loss.item())
            if not np.isnan(kl.item()): test_metrics['kl'].append(kl.item())
            if not np.isnan(cc.item()): test_metrics['cc'].append(cc.item())
            if not np.isnan(sim.item()): test_metrics['sim'].append(sim.item())
            if not np.isnan(nss.item()): test_metrics['nss'].append(nss.item())
            if not np.isnan(osim.item()): test_metrics['osim'].append(osim.item())
            if not np.isnan(auc.item()): test_metrics['auc'].append(auc.item())
            
            #del outputs, osim, sim, kl, cc, nss, auc, expMSE, graph_batches_gpu, graph_global_features_gpu, stimuli, smap, fmap, condition, panoptic
            #if batch_idx % 10 == 0:
            #gc.collect()
            #torch.cuda.empty_cache()


    cpu_metrics = {k: [float(x) for x in v] for k, v in test_metrics.items()}

    del stimuli, smap, fmap, condition, panoptic, outputs, model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    dist.barrier()

    cpu_metrics = {k: [float(x) for x in v] for k, v in test_metrics.items()}

    # Save locally per rank (e.g., rank_0_metrics.csv)
    rank = dist.get_rank()
    df = pd.DataFrame(cpu_metrics)
    df.to_csv(f"SUMGraph_oSIM_rank_{rank}_metrics_v4_osim_1_background_vs_relevant.csv", index=False)


    if not dist.is_initialized() or dist.get_rank() == 0:
        # Read all rank CSVs
        csv_files = glob.glob("SUMgraph_oSIM_rank_*_metrics_v4_osim_1_background_vs_relevant.csv")

        all_metrics = []
        for csv in csv_files:
            df = pd.read_csv(csv)
            all_metrics.append(df)

        # Concatenate all ranks
        df_full = pd.concat(all_metrics, ignore_index=True)

        # Calculate and print GLOBAL means (one value per metric)
        global_means = df_full.mean()

        print("===== GLOBAL METRICS =====")
        for metric, value in global_means.items():
            print(f"{metric}: global mean = {value:.4f}")

def main():
    import os
    import warnings
    
    if single_GPU:
        test()
        print(f"Single GPU mode started...")
    else:
        world_size=torch.cuda.device_count()
        import os
        import warnings

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = '0'
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", module="timm")  # Example for NumPy warnings
        warnings.simplefilter("ignore")
        mp.set_start_method('spawn', force=True)
        mp.spawn(test, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
