from typing import Dict, Optional, Union, Tuple

import torch
from torch import nn
from transformers import XLMRobertaPreTrainedModel, XLMRobertaModel


class XLMRobertaClassificationHead(nn.Module):
    """Head for toxic classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.fc2 = nn.Linear(2 * config.hidden_size, 1)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x


class XLMRobertaForToxicClassification(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.classifier = XLMRobertaClassificationHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        return logits
