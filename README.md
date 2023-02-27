# toxic_detection

文本/图像的恶意检测.

交互的demo部署到了huggingface,连接:
https://huggingface.co/spaces/szzzzz/toxic_detection

## 安装使用

```shell
git clone git@github.com:szprob/toxic_detection.git
cd toxic_detection
python setup.py install --user
```

## 模型

图像部分使用了resnet50.

模型百度云地址：https://pan.baidu.com/s/1tJABYK92zIgGONwQvRjv7A ,提取码：qewg

文本部分使用了bert,训练了16m和51m两种参数量的模型.

16m百度云地址：https://pan.baidu.com/s/1W8JdKHHguWvi9DVvVkz2bQ ,提取码：qewg

51m百度云地址：https://pan.baidu.com/s/1mU0FWQ3gdSgmzen6X-3nDQ ,提取码：qewg


## 文本检测

文本目前只做了英文.
类别和kaggle toxic comment detection任务一致.
包括"toxic","severe_toxic","obscene","threat","insult","identity_hate".
使用方法如下:

```python
from toxic_detection import TextToxicDetector

model = TextToxicDetector()
model.load(model_path)

model.detect("fuccck you.")
```

## 图像检测
图像只有两个类别,为"obscene","discomfort".
使用方法如下:

```python
from toxic_detection import ImgToxicDetector

model = ImgToxicDetector()
model.load(model_path)

model.detect(img_path)
```
