"""
Viralmap
Version: v1.0
ADAPT (2025)
"""

# // imports
import torch
import torch.nn as nn
from transformers import AutoModel

# // base model
class VMAPBase(nn.Module):
    """
    Viralmap v1.0 model
    """
    def __init__(self, model_name:str="facebook/esm2_t33_650M_UR50D", num_labels:int=10):
        super().__init__()
        
        # // transformer
        self.esm        = AutoModel.from_pretrained(model_name)
        hidden_dim_flag = self.esm.config.hidden_size

        # // classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim_flag, hidden_dim_flag // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim_flag // 2, num_labels))
        
    def forward(self, input_ids, attention_mask=None):
        # // esm embeddings
        outputs = self.esm(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            return_dict    = True)
        hidden_states_flag = outputs.last_hidden_state
        
        # // classifier
        logits = self.classifier(hidden_states_flag)
        
        # // 
        return logits 
        





