# Toxic Detection
Toxic detection for text or image.

## toxic detection for text
This module takes care of detecting toxic contents of a given text.

```python
from  import BertToxicDetector

model = BertToxicDetector()
model.from_pretrained()

model.detect("i like free fire.")
```
