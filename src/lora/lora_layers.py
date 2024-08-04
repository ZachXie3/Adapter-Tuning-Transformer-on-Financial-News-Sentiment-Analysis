# -----------------------------
# referenced from https://github.com/alexriggio/BERT-LoRA-TensorRT/tree/main
#
# -----------------------------
import math
from typing import Tuple

import torch
import torch.nn as nn


class LinearLoRA(nn.Module):
    """
    A low-rank adapted linear layer. 

    Args:
        in_dim: int = An integer representing the input dimension of the linear layer 
        out_dim: int = An integer representing the output dimension of the linear layer
        r: int = An integer representing the rank of the low-rank approximated matrices
        lora_alpha: int = An integer representing the numerator of the scaling constant alpha / r 
        lora_dropout: float = A float between 0 and 1 representing the dropout probability      
    """       
    
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        r: int = 8, 
        lora_alpha: int = 16, 
        lora_dropout: float = 0.1,    
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout) 
        
        # Check that the rank is at least 1
        assert r > 0, "Variable 'r' is not greater than zero. Choose a rank of 1 or greater."
            
        # recreate the linear layer and freeze it (the actual weight values will be copied in outside of this class)
        self.pretrained = nn.Linear(in_dim, out_dim, bias=True)
        self.pretrained.weight.requires_grad = False

        # create the low-rank A matrix and initialize with same method as in Hugging Face PEFT library
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # create the low-rank B matrix and initialize to zero 
        self.lora_B = nn.Linear(r, out_dim, bias=False)
        nn.init.constant_(self.lora_B.weight, 0)

        # scaling constant
        self.scaling = self.lora_alpha / self.r
                        
    def forward(self, x):
        pretrained_out = self.pretrained(x)
        lora_out = self.lora_dropout(x)
        lora_out = self.lora_A(lora_out)
        lora_out = self.lora_B(lora_out)
        lora_out = lora_out * self.scaling        
        return pretrained_out + lora_out
    
    
def freeze_model(model):
    """Freezes all layers except the LoRa modules and classifier."""
    for name, param in model.named_parameters():
        if "lora" not in name and "classifier" not in name:
            param.requires_grad = False


def unfreeze_model(model):
    """Unfreezes all parameters in a model by setting requires_grad to True."""
    for name, param in model.named_parameters():
        param.requires_grad = True

            
def create_lora(module, r, lora_dropout, lora_alpha):
    """Converts a linear module to a LoRA linear module"""
    k, d = module.weight.shape
    lora = LinearLoRA(
        in_dim=d, 
        out_dim=k, 
        r=r, 
        lora_dropout=lora_dropout, 
        lora_alpha=lora_alpha)

    with torch.no_grad():
        lora.pretrained.weight.copy_(module.weight)
        lora.pretrained.bias.copy_(module.bias) 

    return lora


def merge_lora_linear(module):
    """Merge LoRA module with pretrained linear moddule."""
    k, d = module.pretrained.weight.shape
    linear = nn.Linear(d, k, bias=True)
    
    with torch.no_grad():
        linear.weight.copy_(module.pretrained.weight + (module.lora_B.weight @ module.lora_A.weight) * module.scaling)
        linear.bias.copy_(module.pretrained.bias)
        
    return linear


def add_lora_layers(
    model,  
    module_names: Tuple=("query", "value"), 
    r: int=8, 
    lora_alpha: float=16,
    lora_dropout: float=0.1, 
    module_types: Tuple=(nn.Linear,)
):
    """
        Replaces chosen linear modules with LoRA equivalents. 
     
        Args:
            model: torch.nn.Module = The PyTorch model to be used
            module_names: Tuple = A tuple containing the names of the linear layers to replace
                Ex. ("query") to replace the linear modules with "query" in the name --> bert.encoder.layer.0.attention.self.query
            r: int = rank of the low-rank approximated matrices
            lora_alpha: int = numerator of the scaling constant alpha / r 
            lora_dropout: float = dropout probability, between 0 and 1
        """                     
    
    
    # disable dropout in frozen layers
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    # replace linear modules with lora modules
    for name, module in model.named_children():
        if isinstance(module, module_types) and name in module_names:
            lora_layer = create_lora(module, r=r, lora_dropout=lora_dropout, lora_alpha=lora_alpha)
            setattr(model, name, lora_layer)                  
        else:
            add_lora_layers(module, module_names, r, lora_dropout, lora_alpha)
        

def merge_lora_layers(model, module_names: Tuple=("query", "value"), dropout=0.1):
    """
        Merge LoRA modules with original linear modules in the model. 
   
        Args:
            model: torch.nn.Module = The PyTorch model to be used
            module_names: Tuple = A tuple containing the names of the LoRA layers to replace
                Ex. ("query") to replace the LoRA modules with "query" in the name --> bert.encoder.layer.0.attention.self.query
            r: int = 
            dropout: float = dropout probability, between 0 and 1   
        """                     
    # enable dropout in frozen layers
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout
    # replace chosen linear modules with lora modules
    for name, module in model.named_children():
        if name in module_names and hasattr(module, "pretrained"):
            linear_layer = merge_lora_linear(module)
            setattr(model, name, linear_layer)                  
        else:
            merge_lora_layers(module, module_names=module_names, dropout=0.1)
                         