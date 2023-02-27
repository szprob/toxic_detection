import os
from typing import Dict, Optional

import torch
from torch import nn

from toxic_detection.module_utils import PreTrainedModule
from toxic_detection.text_detection.bert.embedding import Embeddings
from toxic_detection.text_detection.bert.encoder import Encoder
from toxic_detection.text_detection.bert.nn_utils import get_pad_mask


class BERT(PreTrainedModule, nn.Module):

    """Modeling bert.

    Attributes:
        config (Optional[Dict],optional):
            Modeling config.
            Defaults to None.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
    ) -> None:
        PreTrainedModule.__init__(self)
        nn.Module.__init__(self)

        if config is None:
            config = {}
        self.config = config

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Dict):
        self._config = config
        self.vocab_size = config.get("vocab_size", 50000)
        self.hidden_size = config.get("hidden_size", 512)
        self.num_heads = config.get("num_heads", 8)
        self.maxlen = config.get("maxlen", 512)
        self.n_layers = config.get("n_layers", 8)
        self.embed = Embeddings(
            self.vocab_size, maxlen=self.maxlen, hidden_size=self.hidden_size
        )
        self.encoders = Encoder(
            d_model=self.hidden_size, num_heads=self.num_heads, n_layers=self.n_layers
        )
        # Can also have linear and tanh here
        # This implemention has no nsp block.
        self.apply(self._init_params)

    def _init_params(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self, inputs: torch.Tensor, segment_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function of bert.

        Args:
            inputs (torch.Tensor):
                The index of words. shape:(b,l)
            segment_ids (Optional[torch.Tensor], optional):
                The index of segments.
                This arg is not usually used.
                Defaults to None.

        Returns:
            torch.Tensor:
                BERT result. shape:(b,l,d)
        """
        x = self.embed(inputs, segment_ids)
        attn_mask = get_pad_mask(inputs)
        x = self.encoders(x, attn_mask)
        # can have h_pooled hera: fc(x[:,0])
        return x

    def load(self, model: str) -> None:
        """Load  state dict from local model path or dict.

        Args:
            model (str):
                Model file need to be loaded.
                A string, the path of a pretrained model.

        Raises:
            ValueError: str model should be a path!
        """

        if isinstance(model, str):
            if os.path.isdir(model):
                self._load_from_dir(model)
            elif os.path.isfile(model):
                dir = os.path.join(self._tmpdir.name, "toxic_detection_bert")
                if os.path.exists(dir):
                    pass
                else:
                    os.mkdir(dir)
                self._unzip2dir(model, dir)
                self._load_from_dir(dir)
            else:
                raise ValueError("""str model should be a path!""")

        else:
            raise ValueError("""str model should be a path!""")

    def _load_from_dir(self, model_dir: str) -> None:
        """Set model params from `model_file`.

        Args:
            model_dir (str):
                Dir containing model params.
        """
        model_files = os.listdir(model_dir)

        # config
        if "config.pkl" not in model_files:
            raise FileNotFoundError("""config should in model dir!""")

        config = self._load_pkl(os.path.join(model_dir, "config.pkl"))
        self.config = config

        # model
        if "model.pkl" not in model_files:
            raise FileNotFoundError("""model should in model dir!""")

        self.load_state_dict(
            torch.load(os.path.join(model_dir, "model.pkl"), map_location="cpu")
        )
        self.eval()
