## 1. Импорт библиотек

```python
import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
```

- os: Для работы с операционной системой, например, установки переменных окружения.
- pandas: Библиотека для работы с табличными данными, загрузки и обработки CSV-файлов.
- transformers: Используется для загрузки токенизаторов и моделей BERT, а также управления обучением через Trainer.
- sklearn.model_selection: Разделение данных на обучающие и тестовые выборки.
- sklearn.metrics: Оценка качества модели через метрики accuracy и f1_score.
- torch: Фреймворк для работы с данными и нейросетями.

## 2. Отключение WandB

```python
os.environ["WANDB_DISABLED"] = "true"
```

- Отключает интеграцию с Weights and Biases (инструмент для отслеживания экспериментов). Может мешать, если используется Google Colab.

## 3. Загрузка данных
```python
data = pd.read_csv('labeled.csv')
```

- Загружает датасет из CSV-файла с комментариями и их метками (comment и toxic).


## 4. Преобразование меток
```python
data['toxic'] = data['toxic'].apply(lambda x: 1 if x > 0 else 0)
```
- Преобразует метки toxic в бинарный формат:
  - 1: токсичный комментарий.
  - 0: нетоксичный комментарий.


## 5. Разделение данных
```python
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['comment'], data['toxic'], test_size=0.2, random_state=42
)
```

- Делит данные на тренировочные и тестовые (валидационные) в пропорции 80/20.
- random_state=42: Фиксирует результат разделения для воспроизводимости.


## 6. Инициализация токенизатора

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)
```
-  Загружает предварительно обученный токенизатор BERT.
- truncation=True: Обрезает текст до максимальной длины.
- padding=True: Дополняет текст до максимальной длины.
- max_length=128: Ограничивает длину текстов 128 токенами.

## 7. Класс для PyTorch Dataset
```python
class ToxicityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

```
- Определяет пользовательский датасет для обучения PyTorch.
- __getitem__: Возвращает токенизированный текст и метку по индексу.
- __len__: Возвращает общее число примеров.


## 8. Создание объектов датасетов
```python
train_dataset = ToxicityDataset(train_encodings, train_labels.to_list())
val_dataset = ToxicityDataset(val_encodings, val_labels.to_list())
```
- Преобразует данные в формат PyTorch для обучения.


## 9. Инициализация модели
```python
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

```

- Загружает предварительно обученную модель BERT для классификации.
- num_labels=2: Указывает, что модель должна предсказывать 2 класса (токсичный/нетоксичный).


## 10. Параметры обучения
```python
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy="epoch"
)
```

- num_train_epochs=2: Количество эпох обучения.
- per_device_train_batch_size=8: Размер батча для тренировки.
- per_device_eval_batch_size=8: Размер батча для валидации.
- warmup_steps=500: Количество шагов для разогрева.
- weight_decay=0.01: Снижение веса для регуляризации.
- eval_strategy="epoch": Оценка модели после каждой эпохи.

## 11. Функция вычисления метрик
```python
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}
```

- accuracy: Доля правильных предсказаний.
- f1: Усредненная F1-метрика.


## 12. Инициализация Trainer
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)
```
- Trainer: Высокоуровневый класс для обучения и валидации модели.
