import os
from typing import Dict, Optional

import torch

from toxic_detection.module_utils import PreTrainedModule
from .classification_model import Classifier
from .tokenization.tokenization import (
    Tokenizer as BertTokenizer,
)
from .utils import convert_to_unicode


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

        self._tokenizer = BertTokenizer(maxlen=self._maxlen)
        self._classifier = Classifier(self.config)

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Dict):
        self._config = config
        self._vocab_size = config.get("vocab_size", 50000)
        self._hidden_size = config.get("hidden_size", 512)
        self._num_heads = config.get("num_heads", 8)
        self._maxlen = config.get("maxlen", 512)
        self._n_layers = config.get("n_layers", 8)
        self._tag_num = config.get("tag_num", 6)
        self._tags = config.get(
            "tags",
            [
                "toxic",
                "severe_toxic",
                "obscene",
                "threat",
                "insult",
                "identity_hate",
            ],
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def classifier(self):
        return self._classifier

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

        # classifier
        if "classifier.pkl" not in model_files:
            raise FileNotFoundError("""classifier should in model dir!""")

        self._classifier = Classifier(self._config)
        self._classifier.load_state_dict(
            torch.load(os.path.join(model_dir, "classifier.pkl"), map_location="cpu")
        )
        self._classifier.eval()

        # bert_tokenizer
        self._tokenizer.load(os.path.join(model_dir, "bert_tokenizer.pkl"))

    def detect(self, text: str) -> Dict[str, float]:
        """Scoring the input text.

        Args:
            input (str):
                Text input.

        Returns:
            Dict[str,float]:
                The toxic score of the input .
        """
        text = convert_to_unicode(text)
        input = self._tokenizer.encode_tensor(
            text, maxlen=self.config.get("maxlen", 512)
        ).view(1, -1)
        toxic_score = self._classifier.score(input).view(-1).tolist()
        toxic_score = [round(s, 2) for s in toxic_score]
        res = dict(
            zip(
                self._tags,
                toxic_score,
            )
        )

        return res
