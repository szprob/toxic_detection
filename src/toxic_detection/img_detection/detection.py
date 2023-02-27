import os
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from torchvision import transforms

from toxic_detection.img_detection.classification_model import Classifier
from toxic_detection.img_detection.img_utils import read_im
from toxic_detection.module_utils import PreTrainedModule


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
        """Load  state dict from local model path .

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

    def _transform(self, input: Union[str, bytes, Image.Image]) -> torch.Tensor:
        """Transforms image to torch tensor.

        Args:
            input (Union[str,bytes,Image.Image]):
                Image .

        Raises:
            ValueError:
                `input` should be a str or bytes!

        Returns:
            torch.Tensor:
                Transformed torch tensor.
        """

        im = read_im(input)
        out = self._trans(im).view(1, 3, 224, 224).float()
        return out

    def _score(self, input: torch.Tensor) -> List[float]:
        """Scoring the input image."""
        toxic_score = self._classifier.score(input)
        toxic_score = [round(s, 3) for s in toxic_score][1:]
        return toxic_score

    def detect(self, input: Union[str, bytes, Image.Image]) -> Dict:
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

        im = self._transform(input)
        toxic_score = self._score(im)

        out = dict(
            zip(
                self._tags,
                toxic_score,
            )
        )
        return out
