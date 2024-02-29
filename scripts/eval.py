from toxic_detection import TextToxicDetector

model = TextToxicDetector()
model.load('szzzzz/xlm-roberta-base-text-toxic')

# 模型预测
result = model.detect("fuccck you.")
result

from toxic_detection import BertTextToxicDetector

model = BertTextToxicDetector()
model.load('szzzzz/text_detect_bert_16m')

# 模型预测
result = model.detect("fuccck you.")
result


from toxic_detection import ImgToxicDetector

model = ImgToxicDetector()
model.load('szzzzz/toxic_detection_res50')
result = model.detect(url)
result


