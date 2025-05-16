from .vmamba import VSSM
from .vmamba_graph import VSSM_graph
#from .vmamba_graph_resize import VSSM_graph
import torch
from torch import nn


class SUM(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.salu_mamba = VSSM(in_chans=input_channels,
                               num_classes=num_classes,
                               depths=depths,
                               depths_decoder=depths_decoder,
                               drop_path_rate=drop_path_rate,
                               )

    def forward(self, x, condition):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.salu_mamba(x, condition)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits

    def load_from(self):
        if self.load_ckpt_path is not None:
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']

            # Load the state dictionary with strict=False
            self.salu_mamba.load_state_dict(pretrained_dict, strict=False)

            # Custom loading logic for pretrained layers, if needed
            model_dict = self.salu_mamba.state_dict()
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            self.salu_mamba.load_state_dict(model_dict, strict=False)



class SUM_graph(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        # Initialize the VSSM model, which supports graph and attribute fusion
        self.salu_mamba = VSSM_graph(in_chans=input_channels,
            num_classes=num_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x, condition, vehicles_graphs=None, attributes=None):
        """
        Forward pass of the SUM model, including graph and attribute data for fusion.

        Args:
            x (torch.Tensor): Input image tensor.
            condition (torch.Tensor): Condition vector for modulation.
            vehicles_graphs (torch_geometric.data.Batch, optional): Batched vehicle graphs.
            attributes (torch.Tensor, optional): Attributes matrix for each vehicle (e.g., speed, color).
            
        Returns:
            torch.Tensor: Final output after passing through the SUM model.
        """
        # Handle grayscale input by repeating channels
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Forward pass through the VSSM model with graph and attribute fusion
        logits = self.salu_mamba(x, condition, vehicles_graphs=vehicles_graphs, attributes=attributes)
        
        # Apply sigmoid if the model is binary (num_classes=1)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits

    def load_from(self):
        """
        Load weights from a checkpoint, with custom mapping for layer names
        to match the current model's structure.
        """
        if self.load_ckpt_path is not None:
            model_checkpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = model_checkpoint['model']

            # Load the state dictionary with strict=False
            self.salu_mamba.load_state_dict(pretrained_dict, strict=False)

            # Custom loading logic for pretrained layers
            model_dict = self.salu_mamba.state_dict()
            pretrained_odict = model_checkpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                # Map encoder layers to decoder layers for upsampling path
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v

            # Filter out keys that do not match and update the model dictionary
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            self.salu_mamba.load_state_dict(model_dict, strict=False)