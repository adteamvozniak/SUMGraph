import ntpath

import numpy as np

import os
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from pathlib import Path
from torchvision import transforms
import re
data_keys = {'country', 'user_id', 'test_id', 'trial', 'type', 'id', 'file'}
from settings import transform, resolution, debug
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from settings import * 
from PIL import Image
import json
import math
from srcFromUnity import SaliencyMaps_LatestVersion_1 as sm
from srcFromUnity import settings as sm_settings
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
#from .json_extractor import extract_repetitive_and_nested_tags, cal_pdf_v2
#from json_extractor import extract_repetitive_and_nested_tags, cal_pdf
#from gcnn_utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from Graph.utils_skeleton import *
#from torch_geometric.nn import GCNConv
from torch_geometric.nn import GCNConv, GATConv, SAGEConv  # Import PyG Layers
from torch_geometric.data import Data, Batch
import logging
import torch.nn.init as init
import scipy.stats as stats


# Convert PyTorch tensor to NumPy and scale it to 0-255
def save_tensor_as_image(tensor, filename):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()  # Convert to NumPy
    tensor = np.squeeze(tensor)  # Remove batch dimension if necessary

    # Normalize if needed
    if tensor.dtype == np.float32 or tensor.max() <= 1.0:
        tensor = (tensor * 255).astype(np.uint8)

    # Handle grayscale vs RGB
    if tensor.ndim == 2:  # Grayscale
        img = Image.fromarray(tensor, mode='L')
    elif tensor.ndim == 3 and tensor.shape[0] in [1, 3]:  # Channels first (C, H, W)
        img = Image.fromarray(np.moveaxis(tensor, 0, -1))  # Convert to (H, W, C)
    else:
        raise ValueError("Unexpected image shape:", tensor.shape)

    img.save(filename)


def compute_loss(cc, kl, sim, nss, osim=None, wMSE=None, beta=0.1):
    """
    Stabilized loss function using adaptive scaling.

    Args:
        cc, kl, sim, nss, osim: Metric values.
        wMSE: Weighted MSE loss.
        beta: Smoothing factor for moving average normalization.

    Returns:
        Loss value.
    """
    # ✅ Prevent NaN or Inf issues by clamping
    #kl = torch.clamp(kl, min=1e-6, max=10)  # KL can get large, clip it
    #wMSE = torch.clamp(wMSE, min=1e-6, max=10)  # Avoid exploding MSE
    #nss = torch.clamp(nss, min=-5, max=20)  # Keep NSS within a reasonable range

    alpha = 0.85

    '''
    # ✅ Use adaptive weight normalization to balance term scales
    moving_avg_kl = beta * kl + (1 - beta) * kl.detach().mean()  # Smooth KL
    moving_avg_cc = beta * cc + (1 - beta) * cc.detach().mean()
    moving_avg_sim = beta * sim + (1 - beta) * sim.detach().mean()
    moving_avg_nss = beta * nss + (1 - beta) * nss.detach().mean()
    moving_avg_osim = beta * osim + (1 - beta) * osim.detach().mean()

    # ✅ Normalize by moving averages (avoid one term dominating)
    loss1 = (
        -2 * (cc / moving_avg_cc) +
        10 * (kl / moving_avg_kl) -
        1 * (sim / moving_avg_sim) -
        (nss / moving_avg_nss) -
        (osim / moving_avg_osim)
    )

    # ✅ Normalize wMSE contribution
    loss = loss1 + wMSE / (wMSE.detach().mean() + 1e-6)'''

    #kl_norm = kl / (kl.mean().detach() + 1e-6)
    #cc_norm = cc / (cc.mean().detach() + 1e-6)
    #sim_norm = sim / (sim.mean().detach() + 1e-6)
    #nss_norm = nss / (nss.mean().detach() + 1e-6)
    #osim_norm = osim / (osim.mean().detach() + 1e-6)
    #wMSE_norm = wMSE / (wMSE.mean().detach() + 1e-6)
    #print(f"CC: {cc.item()}, kl: {kl.item()}, sim: {sim.item()},  nss: {nss.item()},  osim: {osim.item()}")
    print(f"CC: {cc.item()}, kl: {kl.item()}, sim: {sim.item()},  nss: {nss.item()}")
    #loss1 = -2 * cc_norm + 10 * kl_norm - 1 * sim_norm - 1 * nss_norm - 1 * osim_norm
    if osim is None:
        loss1 = -2 * cc + 10 * kl - 1 * sim - 1 * nss
    else:
        loss1 = -2 * cc + 10 * kl - 1 * sim - 1 * nss - 1*osim
    #loss = alpha*loss1 + (1-alpha)*wMSE
    
    # ✅ Debugging safety: If NaN appears, return a stable default loss
    #if torch.isnan(loss).any() or torch.isinf(loss).any():
    #    print("⚠ Warning: NaN/Inf detected in loss! Returning default loss value.")
    #    loss = torch.tensor(1.0, device=loss.device, dtype=loss.dtype)

    return loss1



def get_datasets_inference(splits_root_folder):
    train_splits, test_splits, val_splits = read_splits(splits_root_folder, ["train.txt", "test.txt", "val.txt"])
    #scan_to_last_folder_v2(root_folder_all_data, train_splits, test_splits, val_splits)

    # Run the search function in parallel
    if not urls_generated:
        De_folder = os.path.join(root_folder_all_data, "Germany")
        #Jp_folder = os.path.join(root_folder_all_data, "JapanTest")
        Jp_folder = os.path.join(root_folder_all_data, "Japan")

        results_test = search_function(test_splits, [De_folder, Jp_folder], max_workers=25)
        write_urls_to_file(splits_root_folder + "final_test_list.txt", results_test)

        train_data, test_data, val_data = read_splits(splits_root_folder, ["final_train_list.txt", "final_test_list.txt", "final_val_list.txt"])
    else:
        if individual_files_are_listed:
            #test_data_images = read_splits(splits_root_folder, ["final_train_list_shorten.txt", "final_test_list_shorten.txt", "final_val_list_shorten.txt"], tests_only=True)
            test_data_images = read_splits(splits_root_folder, ["final_train_list_shorten_backup.txt", "final_test_list_shorten_1000.txt", "final_val_list_shorten_backup.txt"], tests_only=True)
        else:
            print("Splits are ready - reading the data...")
            test_data = read_splits(splits_root_folder, ["final_train_list.txt", "final_test_list.txt", "final_val_list.txt"], tests_only=True)
            print("Data is loaded")

            if debug:
                print("Test_data size=", len(test_data))

            test_data_images = read_all_files_in_all_folders(test_data)
            print("Completed reading all PNG files for testing")


            start_time = time.time()
            test_data_images = sub_sample_with_check(test_data_images, 10)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Time taken for filtering testing dataset: {execution_time} seconds")
            write_urls_to_file(splits_root_folder + "final_test_list_shorten.txt", test_data_images)

    #exit(1)
    if enableVideo:
        test_dataset = SequenceCustomDataLoader(test_data_images, attr_file, seq_length=seq_length)
    else:
        test_dataset = CustomDataLoader(test_data_images, attr_file)

    return test_dataset

def normalized_weighted_loss(cc, kl, sim, nss, osim, wMSE):
    """
    Computes a dynamically weighted loss where each component contributes proportionally,
    without forcing absolute values in the normalization step.

    Args:
        cc, kl, sim, nss, osim, wMSE (torch.Tensor): Individual loss components.

    Returns:
        torch.Tensor: Final balanced loss.
    """
    # Compute total loss sum **without absolute values**
    total_loss = abs(cc) + abs(kl) + abs(sim) + abs(nss) + abs(osim) + 1e-6  # Avoid division by zero

    cc_factor = abs(cc) / total_loss
    kl_factor = abs(kl) / total_loss
    sim_factor = abs(sim) / total_loss
    nss_factor = abs(nss) / total_loss
    osim_factor = abs(osim) / total_loss

    # ✅ Compute the final balanced loss, preserving original signs
    loss = (cc_factor * cc) + (kl_factor * kl) + (sim_factor * sim) + (nss_factor * nss) + (osim_factor * osim) + wMSE


    print(f"CC: {cc.item()}, kl: {kl.item()}, sim: {sim.item()},  nss: {nss.item()},  osim: {osim.item()},  wMSE: {wMSE.item()} ")
    print(f"CC_factor: {cc_factor.item()}, kl_factor: {kl_factor.item()}, sim: {sim_factor.item()},  nss: {nss_factor.item()},  osim: {osim_factor.item()} ")


    return loss



def cal_pdf(values, plot_url=''):
    #print(f"Started ...")
    #print(f"=============")
    #print(f"Values={values}")
    #for i in range(len(values)):
    #    if values[i] > max_visual_distance:
    #        values[i] = max_visual_distance

    values = [value for value in values if value <= max_visual_distance]

    # Calculate the mean and standard deviation
    mean = np.mean(values)
    std_dev = np.std(values)

    # Create a normal distribution using the calculated mean and standard deviation
    normal_dist = stats.norm(mean, std_dev)


    # Calculate the PDF for each value in the list
    pdf_values = normal_dist.pdf(values)

    inverse_log_pdf_values = np.exp(pdf_values)

    #print(f"Details are ready ")
    #print(f"=============")
    return normal_dist, pdf_values, mean, std_dev


def extract_tags_from_json(directory_path, tag):
    extracted_data_for_trial = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as json_file:
                try:
                    data = json.load(json_file)
                    # add code inhere
                    #
                    #

                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {filename}")

    return extracted_data_for_trial


def calculate_distance(point1, point2):
    # Unpack the coordinates
    x1, y1, z1 = point1['x'], point1['y'], point1['z']
    x2, y2, z2 = point2['x'], point2['y'], point2['z']

    # Calculate the distance using the Euclidean distance formula
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    return distance

def extract_repetitive_and_nested_tags(json_file_path, outer_tag, nested_tag_path, nested_tag_path_2):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    values = []
    values_2 = []
    nested_values_gaze = []
    nested_values_intersection = []
    distance = []

    if isinstance(data, dict):
        for item in data.get('Frames', []):
            if outer_tag in item:
                values.append(item[outer_tag])
            nested_tag = item

            for tag in nested_tag_path:
                if tag in nested_tag:
                    nested_tag = nested_tag[tag]
                else:
                    nested_tag = None
                    break
            if nested_tag is not None:
                nested_values_gaze.append(nested_tag)

    if isinstance(data, dict):
        for item in data.get('Frames', []):
            if outer_tag in item:
                values_2.append(item[outer_tag])
            nested_tag = item

            for tag in nested_tag_path_2:
                if tag in nested_tag:
                    nested_tag = nested_tag[tag]
                else:
                    nested_tag = None
                    break
            if nested_tag is not None:
                nested_values_intersection.append(nested_tag)


    if len(nested_values_gaze) == len(nested_values_intersection):
        #print("List are of equal size")
        for point1, point2 in zip(nested_values_gaze, nested_values_intersection):
            distance.append(calculate_distance(point1, point2))
    else:
        #print("List are having diff sizes")
        print(len(nested_values_gaze), len(nested_values_intersection))

    return values, nested_values_gaze, nested_values_intersection, distance

def get_datasetsTest(splits_root_folder):
    #train_splits, test_splits, val_splits = read_splits(splits_root_folder, ["train.txt", "test.txt", "val.txt"])
    #train_splits, test_splits, val_splits = read_splits(splits_root_folder, ["trainTest.txt", "testTest.txt", "valTest.txt"])
    #scan_to_last_folder_v2(root_folder_all_data, train_splits, test_splits, val_splits)
    # Run the search function in parallel

    train_data, test_data, val_data = read_splits(splits_root_folder, ["test_train_list.txt", "test_test_list.txt", "test_val_list.txt"])
    #train_data, test_data, val_data = read_splits(splits_root_folder, ["final_train_list.txt", "final_test_list.txt", "final_val_list.txt"])
    #print("Data is loaded")

    train_data_images = read_all_files_in_all_folders(train_data)
    test_data_images = read_all_files_in_all_folders(test_data)
    val_data_images = read_all_files_in_all_folders(val_data)

    #exit(1)
    if enableVideo:
        train_dataset = SequenceCustomDataLoader(train_data_images, attr_file, seq_length=seq_length)
        test_dataset = SequenceCustomDataLoader(test_data_images, attr_file, seq_length=seq_length)
        val_dataset = SequenceCustomDataLoader(val_data_images, attr_file, seq_length=seq_length)

    else:
        train_dataset = CustomDataLoader(train_data_images, attr_file)
        test_dataset = CustomDataLoader(test_data_images, attr_file)
        val_dataset = CustomDataLoader(val_data_images, attr_file)

    return train_dataset, test_dataset, val_dataset

def get_datasets(splits_root_folder):
    train_splits, test_splits, val_splits = read_splits(splits_root_folder, ["train.txt", "test.txt", "val.txt"])
    #scan_to_last_folder_v2(root_folder_all_data, train_splits, test_splits, val_splits)

    # Run the search function in parallel
    if not urls_generated:
        De_folder = os.path.join(root_folder_all_data, "Germany")
        #Jp_folder = os.path.join(root_folder_all_data, "JapanTest")
        Jp_folder = os.path.join(root_folder_all_data, "Japan")

        results = search_function(train_splits, [De_folder, Jp_folder], max_workers=25)
        write_urls_to_file(splits_root_folder + "final_train_list.txt", results)

        results_test = search_function(test_splits, [De_folder, Jp_folder], max_workers=25)
        write_urls_to_file(splits_root_folder + "final_test_list.txt", results_test)

        results_val = search_function(val_splits, [De_folder, Jp_folder], max_workers=25)
        write_urls_to_file(splits_root_folder + "final_val_list.txt", results_val)

        train_data, test_data, val_data = read_splits(splits_root_folder, ["final_train_list.txt", "final_test_list.txt", "final_val_list.txt"])

    else:
        if individual_files_are_listed:
            train_data_images, test_data_images, val_data_images = read_splits(splits_root_folder, ["final_train_list_shorten.txt", "final_test_list_shorten.txt", "final_val_list_shorten.txt"])
            #train_data_images, test_data_images, val_data_images = read_splits(splits_root_folder, ["final_train_list_shorten_1000.txt", "final_test_list_shorten_1000.txt", "final_val_list_shorten_1000.txt"])
        else:
            print("Splits are ready - reading the data...")
            train_data, test_data, val_data = read_splits(splits_root_folder, ["final_train_list.txt", "final_test_list.txt", "final_val_list.txt"])
            print("Data is loaded")

            #exit(1)
            # end of test section

            # old logic
            #train_data, test_data = train_test_split(all_folders, test_size=0.2, random_state=42)

            if debug:
                print("Train_data size = ", len(train_data))
                print("Test_data size=", len(test_data))
                print("val_data size=", len(val_data))

            #if debug:
            #    print("Dataset sample")

            #print("Starting reading all PNG files for training")
            train_data_images = read_all_files_in_all_folders(train_data)
            print("Completed reading all PNG files for training")
            #print("Starting reading all PNG files for testing")
            test_data_images = read_all_files_in_all_folders(test_data)
            print("Completed reading all PNG files for testing")
            #print("Starting reading all PNG files for validation")
            val_data_images = read_all_files_in_all_folders(val_data)
            print("Completed reading all PNG files for validation")

            start_time = time.time()
            train_data_images = sub_sample_with_check(train_data_images, 10)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Time taken for filtering training dataset: {execution_time} seconds")
            write_urls_to_file(splits_root_folder + "final_train_list_shorten.txt", train_data_images)


            start_time = time.time()
            test_data_images = sub_sample_with_check(test_data_images, 10)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Time taken for filtering testing dataset: {execution_time} seconds")
            write_urls_to_file(splits_root_folder + "final_test_list_shorten.txt", test_data_images)

            start_time = time.time()
            val_data_images = sub_sample_with_check(val_data_images, 10)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Time taken for filtering validation dataset: {execution_time} seconds")
            write_urls_to_file(splits_root_folder + "final_val_list_shorten.txt", val_data_images)


    attr_file = None

    if enableVideo:
        train_dataset = SequenceCustomDataLoader(train_data_images, attr_file, seq_length=seq_length)
        test_dataset = SequenceCustomDataLoader(test_data_images, attr_file, seq_length=seq_length)
        val_dataset = SequenceCustomDataLoader(val_data_images, attr_file, seq_length=seq_length)

    else:
        train_dataset = CustomDataLoader(train_data_images, attr_file)
        test_dataset = CustomDataLoader(test_data_images, attr_file)
        val_dataset = CustomDataLoader(val_data_images, attr_file)

    return train_dataset, test_dataset, val_dataset

def initialize_new_layers(model, missing_keys):
    """ Initializes only new GCNN layers while keeping pretrained ones unchanged. """
    for name, param in model.named_parameters():
        if name in missing_keys:  # Only initialize new layers

            # ✅ Fully Connected Layers for Graph Embeddings (Hidden layers)
            if "graph_fc.weight" in name and param.dim() >= 2:
                init.kaiming_uniform_(param, nonlinearity="relu")  # Good for ReLU-based graph networks

            # ✅ Final Projection Layer (Use Xavier for stability)
            elif "graph_fc_output.weight" in name and param.dim() >= 2:
                init.xavier_uniform_(param)

            # ✅ Xavier Uniform for FC Layer Weights (2D)
            elif "fc" in name and param.dim() >= 2:
                init.xavier_uniform_(param)

            # ✅ Zero Initialization for Biases (1D)
            elif "fc" in name and param.dim() == 1:
                init.zeros_(param)

            # ✅ 1D Convolutional Layers (Conv1d)
            elif "conv" in name and param.dim() == 3:
                init.kaiming_uniform_(param, nonlinearity="relu")

            # ✅ Initialize Attention-Based Graph Layers (GAT, Transformer)
            elif "attn" in name and param.dim() >= 2:
                init.xavier_uniform_(param)

            # ✅ Fusion Gate Initialization (MLP Layer)
            elif "fusion_gate" in name and param.dim() >= 2:
                init.xavier_uniform_(param)

            # ✅ Bias Terms (Ensure Zero Initialization)
            elif "bias" in name:
                init.zeros_(param)

            # ✅ LSTM Initialization (for Set2Set or other LSTM layers)
            elif "lstm" in name:
                if "weight_ih" in name:  # Input-hidden weights
                    init.xavier_uniform_(param)  # Xavier initialization for input weights
                elif "weight_hh" in name:  # Hidden-hidden weights
                    init.orthogonal_(param)  # Orthogonal initialization for recurrent weights
                elif "bias" in name:
                    param.data.fill_(0)  # Zero biases for stability

            # ✅ Default: Xavier for Stability
            elif param.dim() >= 2:
                init.xavier_uniform_(param)

def sub_sample_with_check(images, sampling_rate=3):
    remaining = []

    # Initialize the ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit each file check as a task and store futures
        futures = [executor.submit(check_image, images[i]) for i in range(0, len(images), sampling_rate)]

        # Initialize tqdm with manual update mode
        with tqdm(total=len(futures), desc="Processing images") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    remaining.append(result)
                pbar.update(1)  # Manually update tqdm on each future completion

    print("Length of the array after truncation= ", len(remaining))

    return remaining

def check_image(img_path):
    threshold = 0

    rgb_size = os.path.getsize(img_path)
    depth_size = os.path.getsize(img_path.replace("RGB_2", "Depth_20"))
    panoptic_size = os.path.getsize(img_path.replace("RGB_2", "Panoptic"))

    if (rgb_size > threshold) and (depth_size > threshold) and (panoptic_size > threshold):
        return img_path  # Corrected to return the path of the image
    else:
        return None



def read_splits(split_root_folder, names, tests_only=False, step=1):
    vpn_list_train = []
    vpn_list_test = []
    vpn_list_val = []

    if tests_only:    
        with open(os.path.join(split_root_folder, names[1]), "r") as file:
            for idx, line in enumerate(file):
                if idx % step == 0:  # Select every `step` line (e.g., 5th or 10th)
                    vpn_list_test.append(line.strip())
        
        print(f"Len of test participants (every {step}th):", len(vpn_list_test))
        return vpn_list_test
    else:
        with open(os.path.join(split_root_folder, names[0]), "r") as file:
            for idx, line in enumerate(file):
                if idx % step == 0:
                    vpn_list_train.append(line.strip())

        with open(os.path.join(split_root_folder, names[1]), "r") as file:
            for idx, line in enumerate(file):
                if idx % step == 0:
                    vpn_list_test.append(line.strip())

        with open(os.path.join(split_root_folder, names[2]), "r") as file:
            for idx, line in enumerate(file):
                if idx % step == 0:
                    vpn_list_val.append(line.strip())

        print(f"Len of train participants (every {step}th):", len(vpn_list_train))
        print(f"Len of test participants (every {step}th):", len(vpn_list_test))
        print(f"Len of val participants (every {step}th):", len(vpn_list_val))
        return vpn_list_train, vpn_list_test, vpn_list_val




def process_item(item, folder):
    # Process each item in the train_list individually
    train_list_to_return = []
    if int(item.split('_')[-1]) <= 61:  # DE data
        folder_to_search = os.path.join(folder[0], item)
    else:  # JP data (item > 61)
        folder_to_search = os.path.join(folder[1], item)

    if debug:
        print("Current folder=", folder_to_search)


    for trial in trials:
        start_time = time.time()
        if trial.strip() in os.listdir(folder_to_search.strip()):
            # Extend the train_list_to_return with the result of scan_to_last_folders
            train_list_to_return.extend(scan_to_last_folders(os.path.join(folder_to_search.strip(), trial.strip())))
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds")

    return train_list_to_return

def search_function(train_list, folder, max_workers=25):
    train_list_to_return = []

    # Use ThreadPoolExecutor to parallelize the processing of the train_list
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, item, folder): item for item in tqdm(train_list)}

        # Collect results as the futures complete
        for future in as_completed(futures):
            try:
                result = future.result()
                train_list_to_return.extend(result)  # Add the result to the main list
            except Exception as exc:
                print(f"An error occurred while processing {futures[future]}: {exc}")

    return train_list_to_return


def scan_to_last_folder_v2(directory, train_list, test_list, val_list):
    De_folder = os.path.join(directory, "Germany")
    Jp_folder = os.path.join(directory, "Japan")
    # is not required
    De_list = generate_vpn_list(1, 61)
    Jp_list = generate_vpn_list(62, 121)
    train_list_to_return = search_function(train_list, [De_folder, Jp_folder])
    write_urls_to_file(splits_root_folder + "final_train_list.txt", train_list_to_return)

def generate_vpn_list(x: int, y: int) -> list:
    if x > y:
        raise ValueError("x should be less than or equal to y.")
    return [f"VPN_{i}" for i in range(x, y + 1)]

def scan_to_last_folders(directory, VPN_to_exclude=""):
    # Check if the directory exists

    #print("Splits to exclude=", VPN_to_exclude)

    if not os.path.isdir(directory):
        print(f"The directory '{directory}' does not exist.")
        return

    # Get a list of all items (files and directories) in the current directory
    items = os.listdir(directory)

    # Initialize a list to store the last folders encountered
    last_folders = []

    # Iterate through all items
    for item in items:
        # Join the current item with the directory to get its full path
        full_path = os.path.join(directory, item)
        # Check if the item is a directory
        if os.path.isdir(full_path):
            # Recursively call the function for the subdirectory
            # and get the result (last folders encountered)
            subfolders = scan_to_last_folders(full_path, VPN_to_exclude)
            # If there are no subfolders, add the current directory to the list
            if not subfolders:
                if 'rgb' in full_path.lower() and '@eadir' not in full_path.lower():
                    #and all(vpn.lower() not in full_path.lower() for vpn in VPN_to_exclude)
                    last_folders.append(full_path)
                #elif (VPN_to_exclude.lower() not in full_path.lower()):
                #    last_folders.append(full_path)
            # Otherwise, extend the list with the last folders encountered
            else:
                last_folders.extend(subfolders)

    return last_folders

def write_urls_to_file(file_path, list):
    with open(file_path, 'w') as file:
        for url in list:
            file.write(f"{url}\n")  # Write each URL followed by a newline character

def plot_subplots(container, text=''):
    if debug:
        print("Container length=", )
    list_container = []
    if isinstance(container, ImageFeatures):
        rows = 1
        list_container.extend(plotImage(container.get_rgb_image(), container.get_depth_image(), container.get_seg_image(), container.get_fix_image(), container.get_penalty_matrix_norm(), container.get_fix_binary()))
    elif isinstance(container, list):
        rows = len(container)
        for i in range(rows):
            list_container.append(plotImage(container[i].get_rgb_image(), container[i].get_depth_image(),
                                            container[i].get_seg_image(), container[i].get_fix_image(), container[i].get_penalty_matrix_norm(), container[i].get_fix_binary()))

    if debug:
        print("Rows=", rows)

    fig, axes = plt.subplots(rows, 6, figsize=(20, 5*rows))

    titles = ['RGB', 'DEPTH', 'PANOPTIC', 'SALIENCY', 'PENALTY', 'FIXATION']

    for i in range(rows):
        if debug:
            print("Number of rows=", i)
            print("Dim=", axes.ndim)
        if axes.ndim == 1:
            for j, image in enumerate(list_container):
                if j == 1:
                    axes[j].imshow(image, cmap='gray')
                elif j == 3:
                    axes[j].imshow(image, cmap='gray')
                elif j ==4:
                    axes[j].imshow(image, cmap='gray')
                elif j ==5:
                    axes[j].imshow(image, cmap='binary_r', interpolation='nearest')
                else:
                    axes[j].imshow(image)
                axes[j].set_title(titles[j])
                axes[j].axis('off')
        elif axes.ndim > 1:
            for j, image in enumerate(list_container[i]):
                if debug:
                    print("Columns=", j)
                if j == 1:
                    axes[i, j].imshow(image, cmap='gray')
                elif j == 3:
                    axes[i, j].imshow(image, cmap='gray')
                elif j == 4:
                    axes[i, j].imshow(image, cmap='gray')
                elif j == 5:
                    axes[i, j].imshow(image, cmap='binary_r', interpolation='nearest')
                else:
                    axes[i, j].imshow(image)
                if titles is not None:
                    axes[0, j].set_title(titles[j])
                axes[i, j].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('tight')
    # Save the figure to a file without displaying it
    fig.savefig('images_row.png')  # Save as PNG file


def plotImage(img, depth, pan, gaze, penalty, fixation, text=''):
    # Convert tensor to NumPy array
    image_np = img.cpu().numpy()
    # If the image has a single channel, remove the singleton dimension
    if image_np.shape[0] == 1:
        image_np = np.squeeze(image_np, axis=0)
    # If the image has a channel dimension, rearrange it to last
    if image_np.shape[0] == 3:
        image_np = np.transpose(image_np, (1, 2, 0))


    ###############
    # Convert the tensor to a numpy array
    image_depth = depth.cpu().numpy()


    # If the image has a single channel, remove the singleton dimension
    if image_depth.shape[0] == 1:
        image_depth = np.squeeze(image_depth, axis=0)
    # If the image has a channel dimension, rearrange it to last
    if image_depth.shape[0] == 3:
        image_depth = np.transpose(image_depth, (1, 2, 0))
    ##print(image_depth.shape)

    image_pan = pan.cpu().numpy()
    # If the image has a single channel, remove the singleton dimension
    if image_pan.shape[0] == 1:
        image_pan = np.squeeze(image_pan, axis=0)
    # If the image has a channel dimension, rearrange it to last
    if image_pan.shape[0] == 3:
        image_pan = np.transpose(image_pan, (1, 2, 0))


    image_sal = gaze.cpu().numpy()
    # If the image has a single channel, remove the singleton dimension
    if image_sal.shape[0] == 1:
        image_sal = np.squeeze(image_sal, axis=0)
    # If the image has a channel dimension, rearrange it to last
    if image_sal.shape[0] == 3:
        image_sal = np.transpose(image_sal, (1, 2, 0))

    if isinstance(penalty, torch.Tensor):
        penalty = penalty.cpu().numpy()
        # If the image has a single channel, remove the singleton dimension
    if penalty.shape[0] == 1:
        penalty = np.squeeze(penalty, axis=0)
    # If the image has a channel dimension, rearrange it to last
    if penalty.shape[0] == 3:
        penalty = np.transpose(penalty, (1, 2, 0))

    image_fix = fixation.cpu().numpy()

    image_fix[image_fix > 0] = 1
    return [image_np, image_depth, image_pan, image_sal, penalty, image_fix]


class CustomDataLoader(DataLoader):
    def __init__(self, data, attr_data):
        self.data = data
        self.attr_data = attr_data
        #self.filtered_data = [i for i in range(len(self.data)) if not self.is_fixation_map_empty(i)]

    def is_fixation_map_empty(self, sample):
        """
        Check if the fixation map in the data sample is all zeros.
        Args:
            sample: The ImageFeatures object containing the fixation map.
        Returns:
            True if the fixation map is all zeros, False otherwise.
        """
        fixation_map = sample.get_fix_binary()  # Get the fixation map (binary)
        return torch.sum(fixation_map) == 0  # Check if the fixation map is all zeros

    def is_rgb_image_empty(self, sample):
        """
        Check if the RGB image in the data sample is all zeros.
        Args:
            sample: The ImageFeatures object containing the RGB image.
        Returns:
            True if the RGB image is all zeros, False otherwise.
        """
        rgb_image = sample.get_rgb_image()  # Get the RGB image

        # Convert RGB image to a NumPy array and check if all values are zero
        if isinstance(rgb_image, torch.Tensor):
            return torch.sum(rgb_image) == 0  # For PyTorch tensor
        else:
            return np.all(rgb_image == 0)  # For NumPy arrays or similar types

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        """
        Retrieve a data sample, skipping those with invalid or empty data.
        """
        try:
            while True:  # Loop until a valid sample is found
                logging.debug(f"Processing index {idx}")
                sample = ImageFeatures(self.data[idx], self.attr_data)
                logging.debug(f"Created ImageFeatures object for index {idx}")

                # Check for empty fixation maps
                if self.is_fixation_map_empty(sample):
                    logging.debug(f"Fixation map is empty for index {idx}. Skipping.")
                    idx = (idx + 1) % len(self.data)  # Move to the next index
                    continue

                label = 1  # Assuming label is always 1
                sample_to_return = {
                    'image': sample.get_rgb_image(),        # Get the RGB image
                    'saliency': sample.get_fix_image(),     # Get the saliency map
                    'fixation': sample.get_fix_binary(),    # Get the fixation map (binary)
                    'label': torch.tensor(label),           # Set the label
                    'panoptic': sample.get_seg_image(),     # Get panoptic segmentation
                    #'penalty': penalty_matrix, # Get penalty image
                }
                logging.debug(f"Standard sample keys created for index {idx}")

                # Include graph data if enabled
                if enable_graph:
                    sample_to_return['graph'] = sample.get_graphs()
                    sample_to_return['graph_image_indices'] = torch.tensor(idx)
                    sample_to_return['graph_global_features'] = sample.get_graphs_global_features()
                #logging.debug(f"Returning sample for index {idx}")
                return sample_to_return

        except Exception as e:
            # Log the error and skip to the next index
            #logging.error(f"Skipping index {idx} due to error: {e}")
            # Return the next index instead
            return self.__getitem__((idx + 1) % len(self.data))  # Moves to the next index in a circular fashion

def custom_collate_fn(batch):
    """
    Collate function to handle batches with images, saliency maps, fixation maps, labels,
    panoptic segmentation, and optional graph data.
    """
    images = torch.stack([item['image'] for item in batch])
    saliency_maps = torch.stack([item['saliency'] for item in batch])
    fixation_maps = torch.stack([item['fixation'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    panoptic = torch.stack([item['panoptic'].permute(1, 2, 0) for item in batch])

    
    if enable_graph:
        all_graphs = []
        all_global_features = []
        graph_image_indices = []  # To keep track of which image each graph belongs to

        # Process graphs and global features
        for image_idx, item in enumerate(batch):
            graphs_for_image = item['graph']
            all_graphs.extend(graphs_for_image)  # Collect graphs

            if enable_global_graph_features:
                global_features_for_image = item['graph_global_features']
                all_global_features.extend(global_features_for_image)  # Collect corresponding global features
            
            # Map graph to the corresponding image index
            graph_image_indices.extend([image_idx] * len(graphs_for_image))

        # Stack global features only if enabled
        if enable_global_graph_features and all_global_features:
            global_features = torch.stack(all_global_features)
        else:
            global_features = None  # Set to None if global features are disabled

        # Return the final batch
        return {
            'image': images,
            'saliency': saliency_maps,
            'fixation': fixation_maps,
            'label': labels,
            'panoptic': panoptic,
            'graph': all_graphs,
            'graph_global_features': all_global_features,# if enable_global_graph_features else None,
            'graph_image_indices': torch.tensor(graph_image_indices, dtype=torch.long),

        }
    else:
        return {
            'image': images,
            'saliency': saliency_maps,
            'fixation': fixation_maps,
            'label': labels,
            'panoptic': panoptic
        }

def save_tensor_as_image(tensor, file_name, is_rgb=False):
    # Move tensor to CPU and detach from computation graph
    tensor = tensor.cpu().detach()
    
    # Convert tensor to numpy array
    array = tensor.numpy()
    
    if is_rgb:
        # If it's an RGB image, convert from (C, H, W) to (H, W, C)
        if len(array.shape) == 3 and array.shape[0] == 3:  # (C, H, W)
            array = np.transpose(array, (1, 2, 0))
    else:
        # For grayscale or binary images, ensure it's 2D (H, W)
        array = np.squeeze(array)  # Remove any singleton dimensions
    
    # Normalize the array to the range [0, 255]
    array = (array - array.min()) / (array.max() - array.min() + 1e-8)  # Normalize to [0, 1]
    array = (array * 255).astype(np.uint8)  # Convert to uint8 for image saving
    
    # Convert array to image and save
    img = Image.fromarray(array)
    img.save(file_name)

def create_graph_for_vehicle(nodes, edges, features=None):
    """
    Create a graph representation for a vehicle.

    Args:
    - nodes (torch.Tensor): Node coordinates or features (shape: [num_nodes, node_dim]).
    - edges (torch.Tensor): Edges connecting nodes (shape: [2, num_edges]).
    - features (torch.Tensor, optional): Optional node features (shape: [num_nodes, feature_dim]).
    
    Returns:
    - graph_data (Data): A PyTorch Geometric Data object representing the vehicle's graph.
    """
    # Validate inputs
    #print(f"nodes.dim()={nodes.dim()}, nodes={nodes}")
    if nodes.dim() != 2:
        logging.debug("Expected 'nodes' to be a 2D tensor of shape [num_nodes, node_dim].")
    if edges.dim() != 2 or edges.shape[0] != 2:
        #raise ValueError("Expected 'edges' to be a 2D tensor of shape [2, num_edges].")
        logging.debug("Expected 'edges' to be a 2D tensor of shape [2, num_edges].")
    if features is not None and features.shape[0] != nodes.shape[0]:
        #raise ValueError("The number of features must match the number of nodes.")
        logging.debug("The number of features must match the number of nodes.")
    
    # Create the graph
    if features is not None:
        graph_data = Data(x=features, edge_index=edges, pos=nodes)  # pos represents coordinates (x, y)
        #print(f"Graph created...")
        # Debugging information
        logging.debug(f"Graph created with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges.")
    elif features is None:
        num_nodes = nodes.size(0)
        features = torch.zeros((num_nodes, 1))  # Placeholder with 1 feature per node
        graph_data = Data(x=features, edge_index=edges, pos=nodes)  # pos represents coordinates (x, y)
    else:
        graph_data = Data(edge_index=edges, pos=nodes)  # If no features, only positions and edges
    
    return graph_data

    

class ImageFeatures(Dataset):
    def __init__(self, image_path, attributes_path):
        self.image_path = image_path
        self.attr_path = attributes_path
        self.path_components = self.extract_path_components()
        self.country = self.extract_country()
        self.user_id = self.extract_userid()
        self.test_id = self.extract_test_id()
        self.trial = self.extract_trial()
        self.type = self.extract_type()
        self.file_number = self.extract_file_number()
        self.rgb_image = self.extract_rgb_image(self.image_path, False, True)
        self.seg_image = self.extract_other_image("Panoptic")
        self.fix_image, self.fix_binary = self.extract_fixations_gen_saliency(self.image_path)
        if enable_graph:
            if self.test_id == 'Zebra':
                file_name = f"{self.test_id}_{self.trial}_{self.user_id.replace('VPN_', '')}_1_Zebra.json"
            else:
                file_name = f"{self.test_id}_{self.trial}_{self.user_id.replace('VPN_', '')}_1_Crossing.json"
            file_path = os.path.join(
                root_folder_all_data, self.country, self.user_id, self.test_id, self.trial, file_name
            )
            if os.path.exists(file_path):
                self.car_data = self.extract_car_data_from_json(file_path, self.file_number)
            else:
                #print(f"File does not exist: {file_path}, trying next in-line")
                if self.test_id == 'Zebra':
                    file_name = f"{self.test_id}_{self.trial}_{self.user_id.replace('VPN_', '')}_2_Zebra.json"
                else:
                    file_name = f"{self.test_id}_{self.trial}_{self.user_id.replace('VPN_', '')}_2_Crossing.json"
                file_path = os.path.join(root_folder_all_data, self.country, self.user_id, self.test_id, self.trial, file_name)
                if os.path.exists(file_path):
                    #print(f"File path={file_path}")
                    self.car_data = self.extract_car_data_from_json(file_path, self.file_number)
                else:
                    print(f"File does not exist: {file_path}")
            
            self.graphs, self.graphs_global_features = self.extract_skeletons(self.image_path, self.file_number)


    def get_graphs(self):
        return self.graphs

    def get_graphs_global_features(self):
        return self.graphs_global_features

    def scale_skeleton_coordinates(self, skeleton, orig_width, orig_height, target_width, target_height):
        """
        Scales the x, y coordinates of each node in the skeleton from the original image size
        to the target image size. Nodes with [-1, -1] (i.e., no detection) are left unchanged.
        
        Args:
        - skeleton: A dictionary of keypoints with coordinates as values (e.g., {"1": [x, y], ...}).
        - orig_width: Original image width.
        - orig_height: Original image height.
        - target_width: Target image width.
        - target_height: Target image height.
        
        Returns:
        - A new dictionary with scaled coordinates.
        """
        scaled_skeleton = {}
        
        if not isinstance(skeleton, dict):
            print(f"Skipping frame due to unexpected format: {skeleton}")

        for key, coords in skeleton.items():
            if coords != [-1, -1]:  # Only scale if valid coordinates exist
                x, y = coords
                scaled_x = x * (target_width / orig_width)
                scaled_y = y * (target_height / orig_height)
                scaled_skeleton[key] = [scaled_x, scaled_y]
            else:
                scaled_skeleton[key] = coords  # Keep [-1, -1] as it is
        
        return scaled_skeleton


    def pack_graphs_for_image(self, vehicles_graphs):
        """
        Pack graphs of all vehicles in an image into a single batch for processing.

        Args:
        - vehicles_graphs: List of Data objects representing each vehicle's graph.
        
        Returns:
        - batched_graphs: A PyTorch Geometric Batch object containing all vehicle graphs.
        """
        # Use PyTorch Geometric's Batch class to pack the graphs
        batched_graphs = Batch.from_data_list(vehicles_graphs)
        
        return batched_graphs

    
    def fix_and_load_json(self, file_path):
        try:
            # Read the file as a raw string
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check and fix missing brackets
            open_braces = content.count("{")
            close_braces = content.count("}")
            open_brackets = content.count("[")
            close_brackets = content.count("]")
            
            # Fix missing braces or brackets at the end
            if open_brackets > close_brackets:
                content += "]" * (open_brackets - close_brackets)
            if open_braces > close_braces:
                content += "}" * (open_braces - close_braces)
            
            # Parse the fixed JSON
            return json.loads(content)
        
        except json.JSONDecodeError as e:
            print(f"Failed to fix and parse JSON file {file_path}: {e}")
            return None


    def extract_car_data_from_json(self, json_file_path, target_frame_number):
        """
        Reads a JSON file, searches for the target frame number, 
        and extracts information about cars in that frame.

        Args:
        - json_file_path (str): Path to the JSON file.
        - target_frame_number (int): The frame number to search for.

        Returns:
        - car_data (dict): A dictionary with car IDs as keys and extracted attributes as values.

        {
            "car_1": {
                "distanceToEgo": 46.32717514038086,
                "speed": 2.742044687271118,
                "isMovingTowardsEgo": true
            },
            "car_3": {
                "distanceToEgo": 12.446002006530762,
                "speed": 2.761911630630493,
                "isMovingTowardsEgo": false
            }
        }

        """
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            print(f"Error: File not found: {json_file_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON: {e}")
            return None

        # Ensure the JSON structure contains frames
        if "frames" not in data or not isinstance(data["frames"], list):
            print(f"Error: JSON file does not have a valid 'frames' structure.")
            return None

        # Search for the target frame
        car_data = {}
        for frame in data["frames"]:
            if int(frame.get("frameNumber")) == int(target_frame_number):
                # Process all cars in the frame
                for car in frame.get("cars", []):
                    car_id = car.get("name")
                    if car_id:
                        car_data[car_id] = {
                            "distanceToEgo": car.get("distanceToEgo"),
                            "speed": car.get("speed"),
                            "isMovingTowardsEgo": car.get("isMovingTowardsEgo"),
                        }
                break  # Stop searching once the frame is found

        if not car_data:
            #print(f"Warning: No data found for frame number {target_frame_number} within file {json_file_path}")
            logging.debug(f"Warning: No data found for frame number {target_frame_number} within file {json_file_path}")
        
        #print(f"Car`s data from json={car_data}")
        return car_data

    def extract_skeletons(self, path, frame_number):
        directory_url = os.path.dirname(path)
        parent_folder_url = os.path.dirname(directory_url)

        json_file = os.path.join(parent_folder_url, "keypoints.json")

        #print(f"########################## open image Path={path}, frame_number={frame_number} ####################")
        if self.get_country() == "Germany":
            orig_width = 1920
            orig_height = 1080
        elif self.get_country() == "Japan":
            orig_width = 1280
            orig_height = 720

        target_width = resolution[0]
        target_height = resolution[1]

        feature_size = len(CAR_KEYPOINTS_24) + 2  # One-hot + coordinates

        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as file:
                    data = json.load(file)
            except EOFError:
                print(f"Error: Unexpected end of file while reading data = {path}, {frame_number}")
            except Exception as e:
                print(f"Error in file {json_file}: {e}")
                print("Fixing the broken file")
                data_tmp = fix_and_load_json(file)
                print("Json file fixed")
                data = json.loads(data_tmp)

            
            if frame_number in data:
                # Check if data[frame_number] is a dictionary
                if not isinstance(data[frame_number], dict):
                    print(f"Frame {frame_number} data is not a dictionary. Returning dummy graph.")
                    dummy_graph = Data(
                        x=torch.empty((0, feature_size), dtype=torch.float),  # Correct feature size
                        edge_index=torch.empty((2, 0), dtype=torch.long),  # No edges
                        pos=torch.empty((0, 2), dtype=torch.float)  # No positions
                    )
                    return dummy_graph

                frame_info = {}
                all_vehicle_graphs = []
                all_vehicle_global_features = []
                

                if not isinstance(data[frame_number], dict):
                    print(f"Skipping frame due to unexpected format: {data[frame_number]}")
                for car_id, keypoints in data[frame_number].items():
                    if not isinstance(keypoints, dict):
                        print(f"Skipping car_id {car_id} due to unexpected format: {type(keypoints)}")
                        continue
                    
                    #print(f"Car`s ID={car_id}")
                    valid_keypoints = {kp_id: coords for kp_id, coords in keypoints.items() if coords != [-1, -1]}
                    valid_coords = [coords for kp_id, coords in keypoints.items() if coords != [-1, -1]]

                    #print(f"valid_keypoints{valid_keypoints}, valid_coordinates{valid_coords}")

                    if transform:
                        valid_keypoints_coord = self.scale_skeleton_coordinates(
                            valid_keypoints, orig_width, orig_height, target_width, target_height
                        )
                        

                    valid_edges = [
                        (u, v) for u, v in SKELETON_24_MAPPING if u in valid_keypoints and v in valid_keypoints
                    ]

                    # Convert valid edges into a tensor
                    edge_index = torch.tensor(valid_edges, dtype=torch.long).t() if valid_edges else torch.empty((2, 0), dtype=torch.long)


                    frame_info[car_id] = [valid_keypoints, valid_keypoints_coord, valid_edges]


                    # Create node features
                    node_features = []
                    for kp_id, coords in valid_keypoints_coord.items():
                        keypoint_type = torch.zeros(len(CAR_KEYPOINTS_24), dtype=torch.float)
                        keypoint_type[int(kp_id) - 1] = 1.0  # One-hot encoding
                        normalized_coords = torch.tensor(coords, dtype=torch.float)
                        node_features.append(torch.cat((keypoint_type, normalized_coords)))

                    node_features = torch.stack(node_features) if node_features else torch.empty((0, feature_size))

                    if enable_global_graph_features:
                        # Add global attributes for the car
                        
                        graph_attrs = self.car_data.get(car_id, {
                            "distanceToEgo": 0.0,  # Default values if attribute is missing
                            "speed": 0.0,
                            "isMovingTowardsEgo": torch.tensor(float(False)),
                        })

                        #print(f"Car`s global settings={graph_attrs}")

                        max_distance = 100
                        max_speed = 50
                        # Normalize `distanceToEgo` and `speed`
                        distance_normalized = graph_attrs["distanceToEgo"] / max_distance  # Replace max_distance with an appropriate value
                        speed_normalized = graph_attrs["speed"] / max_speed               # Replace max_speed with an appropriate value

                        # ✅ Convert `isMovingTowardsEgo` into a single binary value (not one-hot)
                        moving_towards_binary = torch.tensor([float(graph_attrs["isMovingTowardsEgo"])], dtype=torch.float)

                        # Combine into a single global feature vector
                        global_features = torch.cat([
                            torch.tensor([distance_normalized, speed_normalized, moving_towards_binary], dtype=torch.float)
                        ])

                    # Create graph
                    valid_coords = torch.tensor(list(valid_keypoints_coord.values()), dtype=torch.float)
                    #print(f"Valid coordinates={valid_coords}")
                    graph = create_graph_for_vehicle(valid_coords, edge_index, node_features)
                    
                    #print(f"Graph = {graph.x},  {graph.edge_index},  {graph.pos} ")
                    
                    if enable_global_graph_features:
                        all_vehicle_global_features.append(global_features)  # Attach global features as an attribute
                    all_vehicle_graphs.append(graph)

                assert len(all_vehicle_graphs) == len(all_vehicle_global_features), f"Mismatch! Graphs: {len(all_vehicle_graphs)}, Attributes: {len(all_vehicle_global_features)}"
                #print(f"Final count: Graphs = {len(all_vehicle_graphs)}, Attributes = {len(all_vehicle_global_features)}")

                return all_vehicle_graphs, all_vehicle_global_features

        dummy_graph = Data(
            x=torch.empty((0, feature_size), dtype=torch.float),  # Correct feature size
            edge_index=torch.empty((2, 0), dtype=torch.long),  # No edges
            pos=torch.empty((0, 2), dtype=torch.float)  # No positions
        )
        dummy_global_features = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)
        return [dummy_graph], [dummy_global_features]

    def get_fix_binary(self):
        self.fix_binary[self.fix_binary > 0] = 1
        return self.fix_binary

    def get_penalty_matrix(self):
        return self.penalty_matrix

    def get_penalty_matrix_norm(self):
        return self.penalty_matrix_norm

    def extract_regularization_terms(self):
        # Ensure the binary mask is truly binary (0 or 1)
        #print(f"Entered the final penalty step")
        if debug:
            print('Fix image shape = ', self.fix_image.shape)

        if debug:
            #print("Binary mask shape=",binary_mask_tensor.shape)
            print("Depth_image.shape=", self.depth_image.shape)
            print("Depth_image.shape=", self.depth_image)

        depth_image = (self.depth_image).cpu().numpy()

        # Step 2: Flatten the 2D tensor to a 1D tensor
        depth_image_tmp = depth_image

        transformed_values_to_pdf = self.log_normal_dist.pdf(depth_image_tmp)

        transformed_values_to_pdf_norm = (transformed_values_to_pdf - np.min(transformed_values_to_pdf)) / (np.max(transformed_values_to_pdf) - np.min(transformed_values_to_pdf))

        if debug:
            print(f"Mean: {self.mean}")
            print(f"Standard Deviation: {self.std_dev}")
            x = np.linspace(min(depth_image_tmp.flatten()) - 3 * self.std_dev, max(depth_image_tmp.flatten()) + 3 * self.std_dev, 1000)
            y = self.log_normal_dist.pdf(x)

            plt.plot(x, y, label='LogNormal')
            plt.scatter(depth_image, transformed_values_to_pdf, color='red', label='PDF_Values')
            plt.title('Log Normal Distribution and PDF Values')
            plt.xlabel('Values')
            plt.ylabel('Probability Density')
            plt.legend()
            plt.xlim(0, 50)
            plot_url = '/home/jovyan/Torte/data_samples/'
            # Save the plot to a file without displaying it
            plt.savefig(os.path.join(plot_url, 'normal_distribution_plot_extra.png'))
            # Clear the figure after saving to avoid any conflicts if more plots are created later
            plt.clf()

        if debug:
            print("Stored under=", os.path.join(root_folder, self.get_trial()))
        if debug:
            print("Contenct of depth 2 pdf values=", transformed_values_to_pdf)

        # return penalty matrix for saliency pixel
        return transformed_values_to_pdf, transformed_values_to_pdf_norm

    def extract_raw_data(self):
        # add Germany subfolder
        #print(f"Started calculation of log normal for vision ")
        url = ""
        if self.get_country() == "Germany":
            url = os.path.join(root_folder_src, "Germany")
        elif self.get_country() == "Japan":
            url = os.path.join(root_folder_src, "Japan")
            url = os.path.join(url, "Experiment")
        #
        # add VPN user

        def format_folder_name(folder_name):
            # Extract digits from the folder name using regular expression

            match = re.match(r'([A-Za-z_]+)(\d+)', folder_name)

            if match:
                # Extract the string and digits parts
                string_part = match.group(1)

            match = re.search(r'(\d+)', folder_name)

            if match:
                # Extract the digit as a string
                digit_str = match.group(1)
                # Convert the digit string to an integer
                digit = int(digit_str)
                # Format the digit with a leading zero if it is between 1 and 9
                formatted_digit = f'{digit:02}'
                return string_part+formatted_digit
            else:
                return None

        self.user_id = format_folder_name(self.get_userid())

        url = os.path.join(url, self.get_userid())

        if debug:
            print("Url=", url)
        # Check if the folder exists
        if not os.path.exists(url):
            print("Folder does not exist.")
        # List all folders in the specified path
        folders = [folder for folder in os.listdir(url) if os.path.isdir(os.path.join(url, folder))]
        if debug:
            print("Folders=", folders)
        # Select a folder except for the "Mocap" folder
        selected_folder = [folder for folder in folders if folder != "Mocap"]

        if debug:
            print("Remaining folder is=", selected_folder)

        url = os.path.join(url, selected_folder[0])

        if debug:
            print("URL=", url)
        if not os.path.exists(url):
            print("Folder does not exist.")

            # List all files in the specified folder
        all_files = [f for f in os.listdir(url) if os.path.isfile(os.path.join(url, f))]

        exclude_patterns = ["notrial", "_trial_F_", "_trial_E_"]
        # Exclude files containing any of the specified patterns
        excluded_files = [f for f in all_files if not any(pattern in f for pattern in exclude_patterns)]

        if debug:
            print("Final list of files = ", excluded_files)
            print(self.get_testid(), self.get_trial())
        # Select files containing both specified patterns
        selected_file = [f for f in excluded_files if "_"+self.get_testid() in f and "_"+self.get_trial() in f]

        if debug:
            print("The finalname of interest is = ",selected_file )

        json_file_url = os.path.join(url, selected_file[0])
        #print("Json file=", json_file_url)
        # parse url/file with raw data and collect avg details
        #print(f"Calculating pdfs for samples")
        extracted_values, extracted_nested_values_gaze_origin, extracted_nested_values_intersection, distance = extract_repetitive_and_nested_tags(
            json_file_url, outer_tag,
            nested_tag_path_gazeOrigin, nested_tag_path_pos3d)

        
        #print(f"Calculating pdfs for samples")
        #print(f"Disntances={distance}")
        log_normal_dist, pdf_values, mean, std_dev = cal_pdf(distance, plot_url=os.path.join(root_folder, self.get_trial()))
        #print(f"Calculated log normal for vision ")
        #print(f"=====================")
        return log_normal_dist, pdf_values, mean, std_dev


    def get_fix_image(self):
        return self.fix_image

    def get_gender(self):
        return self.gender


    def get_vp_group(self):
        return self.vp_group

    def get_sex_orient(self):
        return self.sexual_orient

    def get_age(self):
        return self.age_years

    def get_weight(self):
        return self.weight_kg

    def get_height(self):
        return self.height_cm

    def get_glasses(self):
        return self.glasses_lenses

    def get_driv_exp(self):
        return self.act_driving_exp_in_years

    def get_acc_ext_as_ped(self):
        return self.accident_exp_as_ped

    def get_exp_vr(self):
        return self.prior_exp_vr

    def extract_study_attributes(self, filename):
        if debug:
            print("Search filename=", filename)
        df = pd.read_excel(filename)
        user_id_column = df.columns[0]
        if debug:
            print("Search column =", user_id_column)


        id_only= re.sub(' ', '', re.sub(r'\D', '', self.get_userid()))
        if debug:
            print("ID = ", id_only)

        additional_subjects_attr = df[df['VPN'] == int(id_only)]
        #print("Gender=",additional_subjects_attr[df['VP Group']])

        if not additional_subjects_attr.empty:
            # Access information in the first matching row (assuming there's only one match)
            first_matching_row = additional_subjects_attr.iloc[0]

            self.gender = first_matching_row['gender']
            self.vp_group = first_matching_row['VP Group']
            self.sexual_orient = first_matching_row['sexual orientation']
            self.age_years = first_matching_row['age in years']
            self.weight_kg = first_matching_row['weight in kg']
            self.height_cm = first_matching_row['height in cm']
            self.glasses_lenses = first_matching_row['glasses/lenses']
            self.act_driving_exp_in_years = first_matching_row['active driving experience in years']
            self.accident_exp_as_ped = first_matching_row['accidents experienced as a pedestrian']
            self.prior_exp_vr = first_matching_row['prior experience with VR']

            if debug:
                print("Attr = ", additional_subjects_attr)
                print(self.gender)
                print(self.prior_exp_vr)


    def set_gender(self, value):
        self.gender = value

    def extract_fixations_gen_saliency(self, file_name):

        # Remove the basename from the URL
        directory_url = os.path.dirname(file_name)
        parent_folder_url = os.path.dirname(directory_url)

        #debug = True

        if debug:
            print("FixationFile name = ", self.get_testid())
        
        if self.get_country() == "Germany":
            json_file = os.path.join(parent_folder_url, "fixations"+self.get_testid() + ".json")
        elif self.get_country() == "Japan":
            json_file = os.path.join(parent_folder_url, "fixations_withMemory_"+self.get_testid() + ".json")

        
        #print("Json file=",json_file)
        
        if os.path.exists(json_file):
            gaze = []
            try:
                with open(json_file, 'r') as file:
                    # Load the JSON data from the file
                    data = json.load(file)
            except Exception as e:
                print(f"Error in file {json_file}: {e}")
            try:
                memoryActive = data["salMemoryActive"]
                fixations = data["fixationsMem"]
            except:
                print("No sal memory is activated")
                memoryActive = False
                fixations = data["fixations"]

            trialId = data["trialID"]
            participant = data["participantID"]
            res = data["resolution"]

            if transform:
                scale_factor_x = resolution[0]/res['x']
                scale_factor_y = resolution[1]/res['y']
                scale_factors_width_height = [scale_factor_x, scale_factor_y]
                #if debug:
                #    print("Scale factors = ", scale_factors_width_height)
            else:
                scale_factors_width_height = [1, 1]

            if transform:
                res_x = resolution[0]
                res_y = resolution[1]
            else:
                res_x = res['x']
                res_y = res['y']

            dva1 = res_x / sm_settings.horizontalFOV
            dva2 = res_y / sm_settings.verticalFOV

            color_map = plt.cm.get_cmap('binary')
            reversed_color_map = color_map.reversed()

            gaze_img = ''

            gaze_fix = ''

            for fi in fixations:
                frame_nr = fi["frameNumber"]

                if int(frame_nr) == int(self.file_number):
                    if debug:
                        print("Frame number=", frame_nr)
                    if memoryActive:
                        # for i in fi["pos2dWithMemory"]:
                        #    print("Sal content =",i["x"], i["y"])
                        gaze_img, gaze_fix = sm.increaseSal_3_memory((int(res_x), int(res_y)), file_name, parent_folder_url, fi["pos2dWithMemory"], dva1, dva2, scale_factors_width_height)
                        #if debug:
                        #    print("gaze_img",gaze_img)

                    else:
                        if ((fi["pos2d"]["x"] >= 0.0) and (fi["pos2d"]["y"] >= 0.0) and (fi["pos2d"]["x"] < res_x) and (
                                fi["pos2d"]["y"] < res_y)):
                           gaze_img, gaze_fix = sm.increaseSal_3((int(res_x), int(res_y)), file_name, parent_folder_url, fi["pos2d"], dva1, dva2, scale_factors_width_height)
                        else:
                            empty_img = np.zeros((int(res_y), int(res_x)), np.uint8)
                            #plt.imsave(save_loc + file_name, empty_img, cmap=reversed_color_map)
                            #print(f"frame nr {frame_nr} not valid")
                            return torch.tensor(empty_img), torch.tensor(empty_img)


            if not isinstance(gaze_img, np.ndarray):
                #print(type(gaze_img), "Frame number=", frame_nr, "json_file=", json_file)
                gaze_img = np.zeros((int(res_y), int(res_x), 1))
                gaze_fix = np.zeros((int(res_y), int(res_x), 1))
            else:
                gaze_img = np.mean(gaze_img[..., :3], axis=-1)
                # Reshape to (240, 320, 1)
                gaze_img = gaze_img[..., np.newaxis]
                gaze_img = np.transpose(gaze_img, (2, 0, 1))
                ##### This to be checked

            if debug:
                print("Original size of fixation = ", gaze_img.shape)

            return torch.tensor(gaze_img, dtype=torch.float32), torch.tensor(gaze_fix, dtype=torch.float32)

    def get_fixations(self):
        if debug:
            print("Number of 1s=", np.sum(self.fix_binary > 0.))
        return self.fixations

    def extract_path_components(self):
        if debug:
            print((self.image_path).split("/"))
        if os.path.isfile(Path(self.image_path)):
            #print((self.image_path).split("/"))
            return (self.image_path).split("/")

    def extract_country(self):
        return self.path_components[3]

    def get_country(self):
        return self.country

    def extract_userid(self):
        return self.path_components[4]

    def get_userid(self):
        return self.user_id

    def extract_test_id(self):
        return self.path_components[5]

    def get_testid(self):
        return self.test_id

    def extract_trial(self):
        return self.path_components[6]

    def get_trial(self):
        return self.trial

    def extract_type(self):
        return self.path_components[7]

    def get_type(self):
        return self.type

    def extract_file_number(self):
        if debug:
            print("Filename=",os.path.splitext(os.path.basename(self.image_path))[0])
        return os.path.splitext(os.path.basename(self.image_path))[0]

    def get_file_number(self):
        return self.file_number

    def extract_rgb_image(self, image, depth=False, transform=False):

        try:
            if depth:
                # Open the RGB image using PIL
                image_rgb = Image.open(image)

                if debug:
                    print("Depth image #of channels=", image_rgb.size)

                image_grayscale = image_rgb.convert('L')

                if debug:
                    print("Depth image converted #of channels=", image_grayscale.size)
                    print("Depth image content #of channels=", image_grayscale)
                # Invert the depth information
                if invert:
                    max_intensity = 255  # Assuming 8-bit depth
                    image_grayscale = Image.eval(image_grayscale, lambda x: max_intensity - x)
            else:
                rgb_image = Image.open(image)

            if transform:
                if depth:
                    resized_image = image_grayscale.resize((resolution[0], resolution[1]), resample=Image.NEAREST)
                    image_np = np.array(resized_image)

                    # Convert to a PyTorch tensor
                    rgb_image = torch.from_numpy(image_np).float()
                else:
                    # Apply transformations if needed
                    transform = transforms.Compose([
                        transforms.Resize((resolution[1], resolution[0])),
                        transforms.ToTensor(),
                        # Add more transforms as needed
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

                if depth:
                    return rgb_image
                else:
                    return transform(rgb_image)
            else:
                return rgb_image
        except FileNotFoundError:
            if depth:
                return np.zeros(resolution[0], resolution[1], dtype=np.float32)
            else:
                return np.zeros((resolution[0], resolutionp[1], 3), dtype=np.uint32)
            print("File does not exist, thus, the creation of the class skipped!")
            raise

    def get_rgb_image(self):
        return self.rgb_image

    def extract_other_image(self, name=""):
        search_term = self.path_components[7]

        #was  search_term = self.path_components[7]
        if debug:
            print("Entire term=", self.path_components)
            print("Search term=", search_term)
            print("New term=", name)

        if name == "Depth":
            name = "Depth_20"

        new_image_path = str(self.image_path).replace(str(search_term), name)
        if debug:
            print("Depth image =",new_image_path)
        if os.path.exists(new_image_path):
            if name == "Depth_20":
                return self.extract_rgb_image(new_image_path, True, True)
            elif name == "Panoptic":
                return self.extract_rgb_image(new_image_path, False, True)
            else:
                print("No name specified")
        else:
            print(f"No file '{name}' found under '{new_image_path}'. Verify location of the file!!!" )

    def get_depth_image(self):
        return self.depth_image

    def get_seg_image(self):
        return self.seg_image


class SequenceCustomDataLoader(Dataset):
    def __init__(self, image_list, att_list, seq_length):
        self.sequence_length = seq_length
        self.image_list = image_list
        self.att_list = att_list

    def __len__(self):
        return len(self.image_list) - self.sequence_length+1

    def __getitem__(self, idx):
        sequence = []
        for i in range(self.sequence_length):
            img_path = self.image_list[idx+i]
            imgContainer = ImageFeatures(img_path, attributes_path=self.att_list)
            sequence.append(imgContainer)

        #sequence = torch.stack(sequence)
        return sequence


def read_all_folders(self, root_folder):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    if debug:
        print(subfolders)
    return subfolders

def get_file_id(filename):
    # Assuming the file ID is before the first underscore in the filename
    return int(filename.split('_')[0])


def intersction_of_lists(list1, list2):

    set1 = set(list1)
    set2 = set(list2)
    overlap = set1.intersection(set2)

    #if debug:
    #    print("Overlaped list=", list(overlap))
    return sorted(list(overlap))

from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def read_all_files_in_all_folders(folderList, load=False):
    images = []

    # Define a nested function to process each folder individually
    def process_folder(folder):
        folder_images = []
        parent_folder, target_folder = os.path.split(folder)

        alternative_check = os.path.join(parent_folder, "Depth_20")
        alternative_check_2 = os.path.join(parent_folder, "Panoptic")

        if not os.path.exists(alternative_check):
            # Return a tuple with an error message and empty list
            return f"Missing folder: {alternative_check}", []

        alt_img_tmp = [filename for filename in sorted(os.listdir(alternative_check)) if filename.endswith(".png")]
        alt_img_tmp_2 = [filename for filename in sorted(os.listdir(alternative_check_2)) if filename.endswith(".png")]
        img_tmp = [filename for filename in sorted(os.listdir(folder)) if filename.endswith(".png")]

        intersection_list = intersction_of_lists(alt_img_tmp, img_tmp)
        intersection_list_2 = intersction_of_lists(alt_img_tmp_2, intersection_list)

        folder_images = [os.path.join(folder, i) for i in intersection_list_2]

        return None, folder_images  # Return None as error and list of images found

    # Open error log file once outside the thread pool
    with open(output_missing_file_path, 'a') as error_log:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Submit each folder to the executor
            future_to_folder = {executor.submit(process_folder, folder): folder for folder in folderList}

            for future in as_completed(future_to_folder):
                folder = future_to_folder[future]
                try:
                    error, folder_images = future.result()
                    if error:
                        error_log.write(f"{error}\n")
                    images.extend(folder_images)
                except Exception as e:
                    print(f"An error occurred while processing {folder}: {e}")

    #print("Size of image container =", len(images))

    if debug:
        print("Size of image container =", len(images))
        with open(splits_root_folder + "total_number_of_img.txt", 'a') as file:
            file.write(str(len(images)) + '\n')

    return images



