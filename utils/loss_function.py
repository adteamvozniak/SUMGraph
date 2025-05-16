import torch as t
import torch.nn as nn
import numpy as np
from skimage.transform import resize
import torch.nn.functional as F

from scipy.stats import pearsonr

class SaliencyLoss(nn.Module):
    def __init__(self):
        super(SaliencyLoss, self).__init__()

    def forward(self, preds, labels, panoptic ='', penalty ='', loss_type='cc'):
        losses = []
        if loss_type == 'cc':
            for i in range(labels.shape[0]):  # labels.shape[0] is batch size
                loss = loss_CC(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'kldiv':
            for i in range(labels.shape[0]):
                loss = loss_KLdiv(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'sim':
            for i in range(labels.shape[0]):
                loss = loss_similarity(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'nss':
            for i in range(labels.shape[0]):
                loss = loss_NSS(preds[i], labels[i])
                losses.append(loss)

        elif loss_type == 'auc':
            for i in range(labels.shape[0]):
                loss = AUC_Judd(preds[i], labels[i])
                loss_tensor = t.tensor(loss, dtype=t.float64, device="cuda:0")  # Convert to tensor
                losses.append(loss_tensor)

        elif loss_type == 'fac_loss':
            for i in range(labels.shape[0]):
                loss = fac_loss(labels[i], preds[i], panoptic[i])
                losses.append(loss)

        elif loss_type == 'fac_metric':
            for i in range(labels.shape[0]):
                loss = fac_metric(labels[i], preds[i], panoptic[i])
                losses.append(loss)

        elif loss_type == 'expMSE':
            for i in range(labels.shape[0]):
                loss = weighted_mse_loss(labels[i], preds[i])
                losses.append(loss)

        elif loss_type == 'osim':
            for i in range(labels.shape[0]):
                #loss = loss_osim_v2(preds[i], labels[i], panoptic[i], penalty[i])
                loss = loss_osim_v2(preds[i], labels[i], panoptic[i])
                losses.append(loss)

        elif loss_type == 'osim_v3':
            for i in range(labels.shape[0]):
                #loss = loss_osim_v2(preds[i], labels[i], panoptic[i], penalty[i])
                loss = loss_osim_v3(preds[i], labels[i], panoptic[i])
                losses.append(loss)

        elif loss_type == 'osim_v4':
            for i in range(labels.shape[0]):
                #loss = loss_osim_v2(preds[i], labels[i], panoptic[i], penalty[i])
                loss = loss_osim_v4(preds[i], labels[i], panoptic[i])
                losses.append(loss)

        elif loss_type == 'w_mse':
            for i in range(labels.shape[0]):
                #print(f"DEBUG: preds.shape = {preds.shape}, labels.shape = {labels.shape}, penalty.shape = {penalty.shape}")
                #print(f"DEBUG: type of penalty = {type(penalty)}")
                #print(f"DEBUG: penalty content = {penalty}")  # This will help identify if it's empty or incorrect
                loss = w_mse_loss(preds[i], labels[i], panoptic[i], penalty[i])
                losses.append(loss)
            

        return t.stack(losses).mean(dim=0, keepdim=True)

def extract_masks_with_color(panoptic_segmentation, is_torch=True):
    """
    Extracts individual binary masks and their corresponding RGB colors from a panoptic segmentation map.
    Returns a list of (mask, color_rgb) tuples.
    """
    masks_with_colors = []

    if is_torch:
        if panoptic_segmentation.dim() == 3 and panoptic_segmentation.shape[0] == 3:
            panoptic_segmentation = panoptic_segmentation.permute(1, 2, 0)
        elif panoptic_segmentation.shape[-1] != 3:
            raise ValueError("Expected the last dimension of `panoptic_segmentation` to be 3 for RGB channels.")

        unique_entities = t.unique(panoptic_segmentation.view(-1, 3), dim=0)

        for color in unique_entities:
            color = color.view(1, 1, 3)
            mask = (panoptic_segmentation == color).all(dim=-1).float()
            masks_with_colors.append((mask, tuple(color.view(-1).tolist())))
    
    else:
        import numpy as np
        if panoptic_segmentation.shape[-1] != 3:
            raise ValueError("Expected the last dimension of `panoptic_segmentation` to be 3 for RGB channels.")

        unique_entities = np.unique(panoptic_segmentation.reshape(-1, 3), axis=0)

        for color in unique_entities:
            mask = np.all(panoptic_segmentation == color, axis=-1).astype(np.float32)
            masks_with_colors.append((mask, tuple(color.tolist())))

    return masks_with_colors

def w_mse_loss(y_pred, y_true, panoptic, penalty_matrix):
    """
    Computes a pixel-wise weighted MSE loss using a penalty matrix.

    Args:
        y_pred (torch.Tensor): Predicted saliency map (B, 1, H, W)
        y_true (torch.Tensor): Ground truth saliency map (B, 1, H, W)
        penalty_matrix (torch.Tensor): Weighting matrix (1, 1, H, W) or (B, 1, H, W)

    Returns:
        torch.Tensor: Weighted MSE loss
    """
    # ✅ Remove singleton channel dimension → (B, H, W)
    #y_pred = y_pred.squeeze(1)
    #y_true = y_true.squeeze(1)
   
    if penalty_matrix.ndim == 2:  # (H, W) → (1, H, W) to match batch size
        penalty_matrix = penalty_matrix.unsqueeze(0)

    # 🔥 Debugging step: Check if all shapes match
        #print(f"DEBUG: y_pred shape = {y_pred.shape}, y_true shape = {y_true.shape}, penalty_matrix shape = {penalty_matrix.shape}")

    assert y_pred.shape == penalty_matrix.shape, f"Penalty shape mismatch: {penalty_matrix.shape} vs {y_pred.shape}"


    penalty_matrix = t.clamp(penalty_matrix, min=0, max=1)

    assert y_pred.shape == y_true.shape, f"Shape mismatch: y_pred {y_pred.shape}, y_true {y_true.shape}"
    assert penalty_matrix.shape[-2:] == y_pred.shape[-2:], f"Penalty shape mismatch: {penalty_matrix.shape} vs {y_pred.shape}"

    # Compute squared error
    squared_error = (y_true - y_pred) ** 2

    # Apply pixel-wise penalty
    weighted_error = penalty_matrix * squared_error

    def gaussian_smooth(tensor, kernel_size=5, sigma=2.0):
        kernel = t.exp(-t.linspace(-(kernel_size//2), kernel_size//2, kernel_size)**2 / (2*sigma**2))
        kernel = kernel / kernel.sum()  # Normalize kernel
        kernel_2d = kernel[:, None] * kernel[None, :]
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size).to(tensor.device)
        tensor = tensor.float()  # Convert to float32
        kernel_2d = kernel_2d.float()  # Ensure kernel is also float32
        kernel_2d = kernel_2d.to(dtype=tensor.dtype, device=tensor.device)
        return F.conv2d(tensor, kernel_2d, padding=kernel_size//2, groups=1)

    # 🔥 Smooth the weighted MSE loss map
    smoothed_loss = gaussian_smooth(weighted_error)

    # Normalize by the sum of penalty weights (avoid division by zero)
    norm_factor = penalty_matrix.sum(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
    #loss = weighted_error.sum(dim=(-1, -2), keepdim=True) / norm_factor
    loss = smoothed_loss.sum(dim=(-1, -2), keepdim=True) / norm_factor

    # Reduce across batch
    return loss.mean()

def weighted_mse_loss(gt_saliency, pred_saliency):
    """
    Compute a weighted Mean Squared Error (MSE) loss between ground truth and predicted saliency maps,
    with weights based on exp(-pred_saliency).

    Args:
    - gt_saliency (torch.Tensor): Ground truth saliency map.
    - pred_saliency (torch.Tensor): Predicted saliency map.

    Returns:
    - torch.Tensor: Weighted MSE loss.
    """
    # Ensure inputs are of the same shape
    assert gt_saliency.shape == pred_saliency.shape, "Input tensors must have the same shape."
    
    # Compute the element-wise squared difference
    squared_diff = (pred_saliency - gt_saliency) ** 2
    
    # Apply the exponential weighting term
    weighting = t.exp(-pred_saliency)
    weighted_squared_diff = squared_diff * weighting
    
    # Return the mean of the weighted squared differences
    loss = weighted_squared_diff.mean()
    return loss

def loss_KLdiv(pred_map, gt_map):
    eps = 2.2204e-16
    pred_map = pred_map / t.sum(pred_map)
    gt_map = gt_map / t.sum(gt_map)
    div = t.sum(t.mul(gt_map, t.log(eps + t.div(gt_map, pred_map + eps))))
    return div


def loss_CC(pred_map, gt_map):
    gt_map_ = (gt_map - t.mean(gt_map))
    pred_map_ = (pred_map - t.mean(pred_map))
    cc = t.sum(t.mul(gt_map_, pred_map_)) / t.sqrt(t.sum(t.mul(gt_map_, gt_map_)) * t.sum(t.mul(pred_map_, pred_map_)))
    return cc


def loss_similarity(pred_map, gt_map):
    gt_map = (gt_map - t.min(gt_map)) / (t.max(gt_map) - t.min(gt_map))
    gt_map = gt_map / t.sum(gt_map)

    pred_map = (pred_map - t.min(pred_map)) / (t.max(pred_map) - t.min(pred_map))
    pred_map = pred_map / t.sum(pred_map)

    diff = t.min(gt_map, pred_map)
    score = t.sum(diff)

    return score

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def is_pedestrian_color(color_rgb, pedestrian_rgbs):
    return color_rgb in pedestrian_rgbs

def is_vehicle_color(color_rgb):
    r, g, b = color_rgb
    return r == 0 and g == 0 and b > 0

def loss_osim_v3(pred_map, gt_map, panoptic):
    min_mask_size = 20

    # Convert pedestrian hex colors to RGB tuples
    pedestrian_hex_colors = [
        "#ff0033", "#e61740", "#cc1439", "#cc2949", "#b32440", 
        "#b3364f", "#992e43", "#993d50", "#803342", "#80404d"
    ]
    pedestrian_rgbs = {hex_to_rgb(c) for c in pedestrian_hex_colors}

    def is_relevant(color_rgb):
        return color_rgb in pedestrian_rgbs or is_vehicle_color(color_rgb)

    # Lists to store relevant and irrelevant sums
    relevant_gt, relevant_pred = [], []
    irrelevant_gt, irrelevant_pred = [], []

    # Must return list of (mask, color_rgb) tuples
    masks_with_colors = extract_masks_with_color(panoptic, is_torch=True)

    for mask, color_rgb in masks_with_colors:
        if mask.sum().item() < min_mask_size:
            continue
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        gt_values = gt_map[mask > 0]
        pred_values = pred_map[mask > 0]

        sum_gt = t.sum(gt_values)
        sum_pred = t.sum(pred_values)

        if is_relevant(color_rgb):
            relevant_gt.append(sum_gt)
            relevant_pred.append(sum_pred)
        else:
            irrelevant_gt.append(sum_gt)
            irrelevant_pred.append(sum_pred)

    def safe_similarity(pred_list, gt_list):
        if not pred_list or not gt_list:
            return t.tensor(0.0, device=gt_map.device, dtype=gt_map.dtype)
        pred = t.stack(pred_list)
        gt = t.stack(gt_list)
        sim = 1 - t.abs(pred - gt) / (pred + gt + 1e-6)
        return sim.mean()

    loss_relevant = 1 - safe_similarity(relevant_pred, relevant_gt)
    loss_irrelevant = 1 - safe_similarity(irrelevant_pred, irrelevant_gt)

    # Combine losses equally for now
    total_loss = (loss_relevant + loss_irrelevant) / 2

    return total_loss




def loss_osim_v4(pred_map, gt_map, panoptic):
    min_mask_size = 20

    # Define relevant object color codes
    pedestrian_hex_colors = [
        "#ff0033", "#e61740", "#cc1439", "#cc2949", "#b32440", 
        "#b3364f", "#992e43", "#993d50", "#803342", "#80404d"
    ]
    pedestrian_rgbs = {hex_to_rgb(c) for c in pedestrian_hex_colors}

    def is_vehicle_color(rgb):
        r, g, b = rgb
        return r == 0 and g == 0 and b > 0

    def is_relevant(rgb):
        return rgb in pedestrian_rgbs or is_vehicle_color(rgb)

    # Relevant masks will be kept separate
    relevant_gt, relevant_pred = [], []

    # Irrelevant will be grouped into a single mask
    irrelevant_mask = None

    masks_with_colors = extract_masks_with_color(panoptic, is_torch=True)

    for mask, color_rgb in masks_with_colors:
        if mask.sum().item() < min_mask_size:
            continue
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if is_relevant(color_rgb):
            gt_values = gt_map[mask > 0]
            pred_values = pred_map[mask > 0]
            relevant_gt.append(t.sum(gt_values))
            relevant_pred.append(t.sum(pred_values))
        else:
            # Merge all irrelevant masks together
            if irrelevant_mask is None:
                irrelevant_mask = mask.clone()
            else:
                irrelevant_mask += mask

    # Handle irrelevant mask if it exists
    if irrelevant_mask is not None and t.sum(irrelevant_mask > 0).item() >= min_mask_size:
        gt_values = gt_map[irrelevant_mask > 0]
        pred_values = pred_map[irrelevant_mask > 0]
        relevant_gt.append(t.sum(gt_values))
        relevant_pred.append(t.sum(pred_values))

    # Compute overall similarity
    if not relevant_gt or not relevant_pred:
        print("⚠ Warning: No valid masks found! Returning zero loss.")
        return t.tensor(0.0, device=gt_map.device, dtype=gt_map.dtype)

    pred_tensor = t.stack(relevant_pred)
    gt_tensor = t.stack(relevant_gt)

    sim = 1 - t.abs(pred_tensor - gt_tensor) / (pred_tensor + gt_tensor + 1e-6)
    osim_loss = 1 - sim.mean()

    return osim_loss

def loss_osim_v2(pred_map, gt_map, panoptic, penalty=None):
    sum_mask_gt = []
    sum_mask_pred = []
    min_mask_size = 20

    masks = extract_masks(panoptic, True)  # Ensure this function does not return empty results
    for mask in masks:
        mask_size = t.sum(mask > 0).item()
        if mask_size < min_mask_size:
            continue  # Skip small masks
        
        # Ensure mask has the correct shape
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        gt_values = gt_map[mask > 0]
        pred_values = pred_map[mask > 0]

        # Sum saliency values within the mask and store them
        sum_mask_gt.append(t.sum(gt_values))
        sum_mask_pred.append(t.sum(pred_values))

    # ✅ Handle the case where no valid masks exist
    if not sum_mask_gt or not sum_mask_pred:
        print("⚠ Warning: No valid masks found! Returning zero loss.")
        return t.tensor(0.0, device=gt_map.device, dtype=gt_map.dtype)  # Safe fallback

    # Convert lists to tensors
    sum_mask_gt = t.stack(sum_mask_gt) 
    sum_mask_pred = t.stack(sum_mask_pred)

    # ✅ Check for NaNs before computing loss
    if t.isnan(sum_mask_gt).any() or t.isnan(sum_mask_pred).any():
        print(f"⚠ Warning: NaN detected in masks! sum_mask_gt={sum_mask_gt}, sum_mask_pred={sum_mask_pred}")
        return t.tensor(0.0, device=gt_map.device, dtype=gt_map.dtype)  # Safe fallback

    # Compute object-based similarity using the SIM function
    loss = loss_similarity(sum_mask_pred, sum_mask_gt)

    # ✅ Final NaN check before returning
    if t.isnan(loss):
        print(f"⚠ Warning: NaN detected in computed loss! Returning zero.")
        return t.tensor(0.0, device=gt_map.device, dtype=gt_map.dtype)

    return loss


def loss_osim(pred_map, gt_map, panoptic):
    sum_mask_gt = []
    sum_mask_pred = []
    min_mask_size=20
    masks = extract_masks(panoptic, True)
    for mask in masks:
 
        mask_size = t.sum(mask > 0).item()
        if mask_size < min_mask_size:
            continue  # Skip small masks
        # Extract saliency values within the mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        gt_values = gt_map[mask > 0]
        pred_values = pred_map[mask > 0]
        # Sum saliency values within the mask and store them
        sum_mask_gt.append(t.sum(gt_values))
        sum_mask_pred.append(t.sum(pred_values))
    # Convert lists to tensors for similarity computation
    sum_mask_gt = t.stack(sum_mask_gt) if sum_mask_gt else t.tensor([0.0], device=gt_saliency_map.device)
    sum_mask_pred = t.stack(sum_mask_pred) if sum_mask_pred else t.tensor([0.0], device=pred_saliency_map.device)
    
    # Compute object-based similarity using the SIM function
    
    loss = loss_similarity(sum_mask_pred, sum_mask_gt)
    return loss

def loss_NSS(pred_map, fix_map):
    '''ground truth here is a fixation map'''

    pred_map_ = (pred_map - t.mean(pred_map)) / t.std(pred_map)

    # Convert the fixation map to a binary mask
    fix_map_binary = fix_map > 0

    score = t.mean(t.masked_select(pred_map_, fix_map_binary))
    return score


def normalize(x, method='standard', axis=None):
    # TODO: Prevent divided by zero if the map is flat
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


def AUC_Judd(saliency_map, fixation_map, jitter=True):
    saliency_map = saliency_map.cpu().numpy() if saliency_map.is_cuda else saliency_map.numpy()
    fixation_map = fixation_map.cpu().numpy() > 0.5 if fixation_map.is_cuda else (fixation_map.numpy() > 0.5)
    # If there are no fixations to predict, return NaN
    if not np.any(fixation_map):
        print('No fixations to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        #print(f"Saliency_map.shape={saliency_map.shape} versus fixation_map.shape={fixation_map.shape}")
        if saliency_map.ndim == 3 and saliency_map.shape[0] == 1:
            saliency_map = np.squeeze(saliency_map)  # Convert [1, 256, 256] to [256, 256]
        else:
            saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='reflect')
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        # Generate random numbers in the same shape as saliency_map as float64
        random_values = np.random.rand(*saliency_map.shape).astype(np.float64)
        saliency_map = saliency_map.astype(np.float64) + random_values * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds) + 2)
    fp = np.zeros(len(thresholds) + 2)
    tp[0] = 0
    tp[-1] = 1
    fp[0] = 0
    fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh)  # Total number of saliency map values above threshold
        tp[k + 1] = (k + 1) / float(n_fix)  # Ratio of saliency map values at fixation locations above threshold
        fp[k + 1] = (above_th - k - 1) / float(n_pixels - n_fix)  # Ratio of other saliency map values above threshold
    return np.trapz(tp, fp)  # y, x



def fac_metric(gt_saliency, pred_saliency, panoptic_segmentation, alpha=0.33, beta=0.33, min_mask_size=25, min_mask_weight=1e-6):
    """
    Compute Fixation Alignment Consistency (FAC) metric using a minimum mask size.
    Aggregates the object-level consistency scores to give an overall metric.

    Parameters:
    - gt_saliency: 2D array representing the ground truth saliency map.
    - pred_saliency: 2D array representing the predicted saliency map.
    - panoptic_segmentation: 3D array representing the panoptic segmentation (3-channel RGB).
    - alpha: Weight for FOO.
    - beta: Weight for OAR.
    - gamma: Weight for SC.
    - min_mask_size: Minimum number of non-zero pixels in a mask to include it in the calculation.
    - min_mask_weight: Minimum weight to consider a mask in the aggregation.

    Returns:
    - Final FAC score as a float between 0 and 1.
    """
    # Normalize the weights to sum up to 1
    weight_sum = alpha + beta
    alpha /= weight_sum
    beta /= weight_sum


    masks = extract_masks(panoptic_segmentation)
    fac_score = 0
    weights_sum = 0

    # Collect individual scores for visualization
    foo_scores, oar_scores, sc_scores = [], [], []

    for i, mask in enumerate(masks):
        # Check if the mask size is below the minimum threshold
        mask_size = np.count_nonzero(mask)
        if mask_size < min_mask_size:
            continue  # Skip masks that are too small

        # Extract GT and Pred values for the current mask
        gt_values = gt_saliency[mask > 0]
        pred_values = pred_saliency[mask > 0]

        # Compute individual metrics
        foo = compute_foo_metric(mask, gt_saliency, pred_saliency, is_torch=True)
        oar = compute_oar(mask, gt_saliency, pred_saliency, is_torch=True)


        foo_scores.append(foo)
        oar_scores.append(oar)

        # Calculate a mask weight based on GT saliency
        mask_weight = np.sum(gt_values) / (np.sum(gt_saliency) + 1e-8)

        # Only consider masks with a significant weight
        if mask_weight < min_mask_weight:
            continue

        weights_sum += mask_weight

        # Calculate FAC for the current mask
        fac_score += mask_weight * (alpha * foo + beta * oar)

    # Normalize the final FAC score by the total weight
    if weights_sum > 0:
        fac_score /= weights_sum
    else:
        print("Warning: No significant masks found for FAC calculation.")

    return np.clip(fac_score, 0, 1)

def fac_loss(gt_saliency, pred_saliency, panoptic_segmentation, alpha=0.45, beta=0.45, gamma=0.1, min_mask_size=50,
                min_mask_weight=1e-6):
    """
    Compute Fixation Alignment Consistency (FAC) metric using a minimum mask size.
    Aggregates the object-level consistency scores to give an overall metric.

    Parameters:
    - gt_saliency: 2D array (numpy or torch) representing the ground truth saliency map.
    - pred_saliency: 2D array (numpy or torch) representing the predicted saliency map.
    - panoptic_segmentation: 3D array (numpy or torch) representing the panoptic segmentation (3-channel RGB).
    - alpha: Weight for FOO.
    - beta: Weight for OAR.
    - gamma: Weight for SC.
    - min_mask_size: Minimum number of non-zero pixels in a mask to include it in the calculation.
    - min_mask_weight: Minimum weight to consider a mask in the aggregation.

    Returns:
    - Final FAC score as a float between 0 and 1.
    """
    # Detect input type and set corresponding operations
    is_torch = isinstance(gt_saliency, t.Tensor) and isinstance(pred_saliency, t.Tensor) and isinstance(
        panoptic_segmentation, t.Tensor)

    # Normalize the weights to sum up to 1
    weight_sum = alpha + beta + gamma
    alpha /= weight_sum
    beta /= weight_sum
    gamma /= weight_sum

    masks = extract_masks(panoptic_segmentation, is_torch)
    fac_score = 0
    weights_sum = 0

    for i, mask in enumerate(masks):
        # Check if the mask size is below the minimum threshold
        mask_size = mask.sum() if is_torch else np.count_nonzero(mask)
        if mask_size < min_mask_size:
            continue  # Skip masks that are too small

        # Extract GT and Pred values for the current mask
        gt_values = gt_saliency[mask > 0]
        pred_values = pred_saliency[mask > 0]

        # Compute individual metrics
        foo = compute_foo(gt_values, pred_values, is_torch)
        oar = compute_oar(mask, gt_saliency, pred_saliency, is_torch)
        sc = compute_spatial_consistency(gt_values, pred_values, is_torch)

        # Calculate a mask weight based on GT saliency
        mask_weight = (gt_values.sum() / (gt_saliency.sum() + 1e-8)) if is_torch else np.sum(gt_values) / (
                    np.sum(gt_saliency) + 1e-8)

        # Only consider masks with a significant weight
        if mask_weight < min_mask_weight:
            continue

        weights_sum += mask_weight

        # Calculate FAC for the current mask
        fac_score += mask_weight * (alpha * foo + beta * oar + gamma * sc)

    # Normalize the final FAC score by the total weight
    if weights_sum > 0:
        fac_score /= weights_sum
    else:
        print("Warning: No significant masks found for FAC calculation.")

    return t.clamp(fac_score, 0, 1) if is_torch else np.clip(fac_score, 0, 1)


def fac_loss(gt_saliency, pred_saliency, panoptic_segmentation, alpha=0.45, beta=0.45, gamma=0.1, min_mask_size=50, min_mask_weight=1e-6):
    """
    Compute Fixation Alignment Consistency (FAC) metric using a minimum mask size.
    Aggregates the object-level consistency scores to give an overall metric.

    Parameters:
    - gt_saliency: 2D torch tensor representing the ground truth saliency map.
    - pred_saliency: 2D torch tensor representing the predicted saliency map.
    - panoptic_segmentation: 3D torch tensor representing the panoptic segmentation (3-channel RGB).
    - alpha: Weight for FOO.
    - beta: Weight for OAR.
    - gamma: Weight for SC.
    - min_mask_size: Minimum number of non-zero pixels in a mask to include it in the calculation.
    - min_mask_weight: Minimum weight to consider a mask in the aggregation.

    Returns:
    - Final FAC score as a float between 0 and 1.
    """

    gt_saliency = gt_saliency.squeeze(0)
    
    pred_saliency = pred_saliency.squeeze(0)
    # Normalize the weights to sum up to 1
    weight_sum = alpha + beta + gamma
    alpha /= weight_sum
    beta /= weight_sum
    gamma /= weight_sum

    masks = extract_masks(panoptic_segmentation, is_torch=True)
    fac_score = 0
    weights_sum = 0

    for mask in masks:
        # Check if the mask size is below the minimum threshold
        mask_size = mask.sum()
        if mask_size < min_mask_size:
            continue  # Skip masks that are too small

        # Extract GT and Pred values for the current mask
        gt_values = gt_saliency[mask > 0]
        pred_values = pred_saliency[mask > 0]

        # Compute individual metrics
        foo = compute_foo(gt_values, pred_values, is_torch=True)
        oar = compute_oar(mask, gt_saliency, pred_saliency, is_torch=True)
        sc = compute_spatial_consistency(gt_values, pred_values, is_torch=True)

        # Calculate a mask weight based on GT saliency
        mask_weight = gt_values.sum() / (gt_saliency.sum() + 1e-8)

        # Only consider masks with a significant weight
        if mask_weight < min_mask_weight:
            continue

        weights_sum += mask_weight

        # Calculate FAC for the current mask
        fac_score += mask_weight * (alpha * foo + beta * oar + gamma * sc)

    # Normalize the final FAC score by the total weight
    if weights_sum > 0:
        fac_score /= weights_sum
    else:
        print("Warning: No significant masks found for FAC calculation.")

    return t.clamp(fac_score, 0, 1)

def extract_masks(panoptic_segmentation, is_torch=True):
    masks = []
    
    if is_torch:
        
        #print("panoptic shape =", panoptic_segmentation.shape)

        # Reshape to be compatible for color comparison (assumes panoptic_segmentation has a color dimension of size 3)
        # Ensure the tensor is channel-last [H, W, 3] for processing
        if panoptic_segmentation.dim() == 3 and panoptic_segmentation.shape[0] == 3:
            # Permute to [H, W, 3]
            panoptic_segmentation = panoptic_segmentation.permute(1, 2, 0)
        elif panoptic_segmentation.shape[-1] != 3:
            raise ValueError("Expected the last dimension of `panoptic_segmentation` to be 3 for RGB channels.")

        unique_entities = t.unique(panoptic_segmentation.view(-1, 3), dim=0)

        for color in unique_entities:
            # Reshape color to (1, 1, 3) for broadcasting if necessary
            color = color.view(1, 1, 3)
            
            # Check if reshaping is required to match dimensions properly
            if panoptic_segmentation.dim() == 3 and color.dim() == 3:
                # Perform the element-wise comparison and obtain mask
                mask = (panoptic_segmentation == color).all(dim=-1).float()
            else:
                raise ValueError("Dimension mismatch after reshaping; check input shapes.")
            
            masks.append(mask)
    else:
        # Assuming panoptic_segmentation is a NumPy array with the last dimension as color channels
        import numpy as np
        
        if panoptic_segmentation.shape[-1] != 3:
            raise ValueError("Expected the last dimension of `panoptic_segmentation` to be 3 for RGB channels.")
        
        unique_entities = np.unique(panoptic_segmentation.reshape(-1, 3), axis=0)

        for color in unique_entities:
            mask = np.all(panoptic_segmentation == color, axis=-1).astype(np.float32)
            masks.append(mask)
    
    return masks


def compute_spatial_consistency(gt_values, pred_values, is_torch):
    gt_values = gt_values.flatten()
    pred_values = pred_values.flatten()

    if is_torch:
        if gt_values.numel() == 0 or pred_values.numel() == 0:
            return t.tensor(0.0)
        if t.all(gt_values == 0) and t.all(pred_values == 0):
            return t.tensor(1.0)
        if gt_values.std() == 0 or pred_values.std() == 0:
            return t.tensor(0.0)

        correlation = t.corrcoef(t.stack((gt_values, pred_values)))[0, 1].clamp(min=0)
        normalized_diff = (gt_values - pred_values).abs().mean() / (gt_values.mean() + 1e-8)
        diff_penalty = 1 - min(1, normalized_diff)
        sc = 0.5 * correlation + 0.5 * diff_penalty
        return sc.clamp(0, 1)

    else:
        if len(gt_values) == 0 or len(pred_values) == 0:
            return 0
        if np.all(gt_values == 0) and np.all(pred_values == 0):
            return 1
        if np.std(gt_values) == 0 or np.std(pred_values) == 0:
            return 0

        correlation, _ = pearsonr(gt_values, pred_values)
        correlation = max(0, correlation)
        normalized_diff = np.mean(np.abs(gt_values - pred_values)) / (np.mean(gt_values) + 1e-8)
        diff_penalty = 1 - min(1, normalized_diff)
        sc = 0.5 * correlation + 0.5 * diff_penalty
        return np.clip(sc, 0, 1)

def compute_foo(gt_values, pred_values, is_torch):
    if is_torch:
        if gt_values.sum() == 0:
            return t.tensor(0.0)

        overlap = t.min(gt_values, pred_values)
        overlap_ratio = overlap.sum() / (gt_values.sum() + 1e-8)
        abs_diff = (gt_values - pred_values).abs()
        penalty = 1 - (abs_diff.sum() / (gt_values.sum() + 1e-8))

        foo = 0.5 * overlap_ratio + 0.5 * penalty
        return foo.clamp(0, 1)
    else:
        if np.sum(gt_values) == 0:
            return 0

        overlap = np.minimum(gt_values, pred_values)
        overlap_ratio = np.sum(overlap) / (np.sum(gt_values) + 1e-8)
        abs_diff = np.abs(gt_values - pred_values)
        penalty = 1 - (np.sum(abs_diff) / (np.sum(gt_values) + 1e-8))

        foo = 0.5 * overlap_ratio + 0.5 * penalty
        return np.clip(foo, 0, 1)

def compute_foo_metric(mask, gt_values, pred_values, is_torch):
    if is_torch:
        """
        Computes Fixation Object Overlap (FOO) between GT and Pred values within an object mask.
        Measures the overlap by calculating the proportion of overlap between GT and Pred saliency values.
        Penalizes large differences in saliency values.
        """
        # Filter GT and Pred values within the mask area
        gt_values = gt_values[mask > 0]
        pred_values = pred_values[mask > 0]

        if t.sum(gt_values) == 0:
            return t.tensor(0.0, device=gt_values.device)  # Avoid division by zero if no saliency in GT within the mask

        # Calculate the proportion of overlapping saliency values within the mask area
        overlap = t.minimum(gt_values, pred_values)
        overlap_ratio = t.sum(overlap) / (t.sum(gt_values) + 1e-8)

        foo = t.clamp(overlap_ratio, min=0, max=1)
        return foo
    else:
        if np.sum(gt_values) == 0:
            return 0

        overlap = np.minimum(gt_values, pred_values)
        overlap_ratio = np.sum(overlap) / (np.sum(gt_values) + 1e-8)

        foo = overlap_ratio
        return np.clip(foo, 0, 1)

def compute_oar(mask, gt_saliency, pred_saliency, is_torch):
    if is_torch:
        object_gt_saliency_sum = gt_saliency[mask > 0].sum()
        object_pred_saliency_sum = pred_saliency[mask > 0].sum()

        if object_gt_saliency_sum == 0:
            return t.tensor(0.0)

        return t.clamp(object_pred_saliency_sum / (object_gt_saliency_sum + 1e-8), 0, 1)
    else:
        object_gt_saliency_sum = np.sum(gt_saliency[mask > 0])
        object_pred_saliency_sum = np.sum(pred_saliency[mask > 0])

        if object_gt_saliency_sum == 0:
            return 0

        return np.clip(object_pred_saliency_sum / (object_gt_saliency_sum + 1e-8), 0, 1)