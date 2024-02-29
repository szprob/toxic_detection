import os
from typing import Dict, Optional

import torch

from toxic_detection.module_utils import PreTrainedModule
from .nets import XLMRobertaForToxicClassification
from transformers import XLMRobertaTokenizer


class Detector(PreTrainedModule):
    """Toxic detector .

    Attributes:
        config (Optional[Dict],optional):
            Modeling config.
            Defaults to None.
    """

    def __init__(
        self,
        *,
        config: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = {}
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def load(self, model: str) -> None:
        """Load  state dict from huggingface repo or local model path or dict.

        Args:
            model (str):
                Model file need to be loaded.
                Can be either:
                    path of a pretrained model.
                    model repo.

        Raises:
            ValueError: str model should be a path!
        """
        if model in self._PRETRAINED_LIST:
            model = self.download(model)
        if isinstance(model, str):
            if os.path.isdir(model):
                self._load_from_dir(model)
            elif os.path.isfile(model):
                dir = os.path.join(self._tmpdir.name, "toxic_detection_roberta")
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
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_dir)
        self.model = (
            XLMRobertaForToxicClassification.from_pretrained(model_dir)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def detect(self, text:str) -> Dict[str, float]:
        
        data = self.tokenizer.batch_encode_plus(
            [text],
            truncation=True,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
        )
        out = self.model(
            input_ids=data["input_ids"].to(self.device),
            attention_mask=data["attention_mask"].to(self.device),
        )
        toxic_score = torch.sigmoid(out).detach().cpu()[0].tolist()
        toxic_score = [round(s, 2) for s in toxic_score]
        res = {'toxic':toxic_score[0]}

        return res
