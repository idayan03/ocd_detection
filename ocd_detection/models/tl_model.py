import torch
import argparse
import numpy as np
import torch.nn as nn

from typing import Any, Dict
from efficientnet_pytorch import EfficientNet

class TLModel(nn.Module):
    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        # self.input_dims = np.prod(data_config["input_dims"])
        self.input_dims = data_config['input_dims']
        self.num_classes = len(data_config["mapping"])

        # transfer learning (pretrained = true)
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b0')
        # freeze layers by using eval()
        self.feature_extractor.eval()
        # freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        n_sizes = self._get_conv_output(self.input_dims)
        # Modify classifier accordingly
        self.classifier = nn.Linear(n_sizes, self.num_classes)

    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x