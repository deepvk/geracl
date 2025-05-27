# GeRaCl: General Rapid text Classifier

**GeRaCl** is an efficient, zero-shot text classification model inspired by the [GLiNER](https://github.com/urchade/GLiNER/tree/main) framework. It shows comparable performace to popular sentence-encoder models that have less than 1B parameters while having only 155M parameters. Also, it is more efficient than most popular NLI-tuned zero-shot classifiers because GeRaCl performs classification in a single forward pass.

### 🚀 Quick Start

Clone and install directly from GitHub:

```bash
git clone https://github.com/deepvk/zero-shot-classification
cd GeRaCl

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
results = pipe(texts, labels, batch_size=1)

for i in range(len(labels)):
    print(labels[i][results[i]])
```
