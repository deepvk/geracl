# GeRaCl: General Rapid text Classifier

**GeRaCl** is an open‑source **framework** for building, training, and evaluating efficient zero‑shot text classifiers on top of any BERT‑like sentence-encoder. It is inspired by the [GLiNER](https://github.com/urchade/GLiNER/tree/main) framework.

### ✨ Why GeRaCl?

| Feature                        | What it means for you                                                                             |
| ------------------------------ | ------------------------------------------------------------------------------------------------- |
| **Zero‑shot by design**        | Classify with **arbitrary** label sets that you decide at run‑time — just pass a list of strings. |
| **One forward pass**           | As fast as ordinary text classification; no pairwise loops like in NLI‑based approaches.          |
| **Model‑agnostic**             | Works with any Hugging Face sentence-encoder.                                                     |
| **155 M reference checkpoint** | A lean [baseline](https://huggingface.co/deepvk/GeRaCl-USER2-base) (155M parameters) that beats much larger sentence‑encoders (300-500M parameters). |
| **All‑in‑one toolkit**         | Training/eval scripts, HF Hub and WandB integration.                                              |


### 🚀 Quick Start

Clone and install directly from GitHub:

```bash
git clone https://github.com/deepvk/geracl
cd geracl

pip install -r requirements.txt
```

Verify your installation:

```python
import geracl
print(geracl.__version__)
```

### 🧑‍💻 Usage Examples

#### Single classification scenario

```python
from transformers import AutoTokenizer
from geracl import GeraclHF, ZeroShotClassificationPipeline

model = GeraclHF.from_pretrained('deepvk/GeRaCl-USER2-base').to('cuda').eval()
tokenizer  = AutoTokenizer.from_pretrained('deepvk/GeRaCl-USER2-base')

pipe = ZeroShotClassificationPipeline(model, tokenizer, device="cuda")

text = "Утилизация катализаторов: как неплохо заработать"
labels = ["экономика", "происшествия", "политика", "культура", "наука", "спорт"]
result = pipe(text, labels, batch_size=1)[0]

print(labels[result])
```

#### Multiple classification scenarios

```python
from transformers import AutoTokenizer
from geracl import GeraclHF, ZeroShotClassificationPipeline

model = GeraclHF.from_pretrained('deepvk/GeRaCl-USER2-base').to('cuda').eval()
tokenizer  = AutoTokenizer.from_pretrained('deepvk/GeRaCl-USER2-base')

pipe = ZeroShotClassificationPipeline(model, tokenizer, device="cuda")

texts = [
  "Утилизация катализаторов: как неплохо заработать",
  "Мне не понравился этот фильм."
]
labels = [
  ["экономика", "происшествия", "политика", "культура", "наука", "спорт"],
  ["нейтральный", "позитивный", "негативный"]
]
results = pipe(texts, labels, batch_size=2)

for i in range(len(labels)):
    print(labels[i][results[i]])
```
