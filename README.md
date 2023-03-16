# toxic_detection

文本/图像的恶意检测.

交互的demo部署到了huggingface:
https://huggingface.co/spaces/szzzzz/toxic_detection

## 安装使用

```shell
git clone git@github.com:szprob/toxic_detection.git
cd toxic_detection
python setup.py install --user
```

或者直接

```shell
pip install git+https://github.com/szprob/toxic_detection.git
```

## 模型

预训练模型全部开源,可以直接下载,也可以直接在代码中读取远端模型.

图像部分使用了resnet50.

模型百度云地址：https://pan.baidu.com/s/1tJABYK92zIgGONwQvRjv7A ,提取码：qewg

huggingface : https://huggingface.co/szzzzz/toxic_detection_res50

文本部分使用了bert,训练了16m和51m两种参数量的模型.

16m百度云地址：https://pan.baidu.com/s/1W8JdKHHguWvi9DVvVkz2bQ ,提取码：qewg

huggingface : https://huggingface.co/szzzzz/text_detect_bert_16m

51m百度云地址：https://pan.baidu.com/s/1mU0FWQ3gdSgmzen6X-3nDQ ,提取码：qewg

huggingface : https://huggingface.co/szzzzz/text_detect_bert_51m


## 文本检测

文本目前只做了英文.
类别和kaggle toxic comment detection任务一致.
包括"toxic","severe_toxic","obscene","threat","insult","identity_hate".
使用方法如下:

```python
from toxic_detection import TextToxicDetector

model = TextToxicDetector()

# 如果模型down到了本地
model.load(model_path)
# 也可以直接使用远端
model.load('szzzzz/text_detect_bert_16m')

# 模型预测
result = model.detect("fuccck you.")
'''
result
{'toxic': 0.94,
 'severe_toxic': 0.03,
 'obscene': 0.59,
 'threat': 0.02,
 'insult': 0.44,
 'identity_hate': 0.05}
'''

```

## 图像检测
图像只有两个类别,为"obscene","discomfort".
使用方法如下:

```python
from toxic_detection import ImgToxicDetector

model = ImgToxicDetector()
# 如果模型down到了本地
model.load(model_path)
# 也可以直接使用远端
model.load('szzzzz/toxic_detection_res50')

result = model.detect(img_path)
'''
result
{'obscene': 0.22,
 'discomfort': 0.93,}
'''

```
