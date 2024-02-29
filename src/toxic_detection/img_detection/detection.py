import os
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from torchvision import transforms

from .classification_model import Classifier
from .img_utils import read_im, get_pieces_from_img, get_max_dict
from ..module_utils import PreTrainedModule


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
        self._config = config
        self._in_features = config.get("in_features", 2048)
        self._tag_num = config.get("tag_num", 2)
        self._tags = config.get("tags", ["obscene"])

        self._classifier = Classifier(self.config)
        self._trans = transforms.Compose(
            [
                # transforms.ToPILImage()
                transforms.Resize(256),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: Dict):
        self._config = config
        self._in_features = config.get("in_features", 2048)
        self._tag_num = config.get("tag_num", 2)
        self._tags = config.get("tags", ["obscene"])

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
                dir = os.path.join(self._tmpdir.name, "img_detection")
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
        config = self._load_pkl(os.path.join(model_dir, "config.pkl"))
        self.config = config
        self._classifier = Classifier(config)
        self._classifier.load_state_dict(
            torch.load(os.path.join(model_dir, "classifier.pkl"), map_location="cpu")
        )
        self._classifier.eval()

    def _transform(self, input: Image.Image) -> torch.Tensor:
        out = self._trans(input).view(1, 3, 224, 224).float()
        return out

    def _score(self, input: torch.Tensor) -> List[float]:
        """Scoring the input image."""
        toxic_score = self._classifier.score(input)
        toxic_score = [round(s, 3) for s in toxic_score][1:]
        return toxic_score

    def _detect(self, input: Image.Image) -> Dict:

        input = self._transform(input)
        toxic_score = self._score(input)

        out = dict(
            zip(
                self._tags,
                toxic_score,
            )
        )
        return out

    def detect(
        self, input: Union[str, bytes, Image.Image], multi_pieces: bool = True
    ) -> Dict:
        """Detects toxic contents from image `input`.

        Args:
            input (Union[str,bytes,Image.Image]):
                Image path of bytes.

        Raises:
            ValueError:
                `input` should be a str or bytes!

        Returns:
            Dict:
                Pattern as  Dict[str,float]
        """
        im = read_im(input)

        score = self._detect(im)
        max_score = max(score.values())
        if max_score > 0.75:
            return score
        elif max_score > 0.45 and multi_pieces:
            score = self._multi_pieces_img_detect(im)
            return score
        else:
            return score

    def _multi_pieces_img_detect(self, im):
        ims = get_pieces_from_img(im)
        score = None
        for i in ims:
            image_toxic_score = self._detect(i)
            if score is None:
                score = image_toxic_score
            else:
                score = get_max_dict(score, image_toxic_score)
        return score
