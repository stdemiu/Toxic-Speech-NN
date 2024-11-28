

import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch

# Отключение WandB
os.environ["WANDB_DISABLED"] = "true"

# Загрузка данных
data = pd.read_csv('labeled.csv')

# Преобразование меток для задачи бинарной классификации
data['toxic'] = data['toxic'].apply(lambda x: 1 if x > 0 else 0)  # Преобразование в формат 0 и 1

# Разделение данных на тренировочную и тестовую выборки
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['comment'], data['toxic'], test_size=0.2, random_state=42
)

# Инициализация токенизатора и модели BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# Преобразование данных в PyTorch Dataset
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

train_dataset = ToxicityDataset(train_encodings, train_labels.to_list())
val_dataset = ToxicityDataset(val_encodings, val_labels.to_list())

# Инициализация модели BERT для бинарной классификации
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

# Параметры обучения
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

# Функция для вычисления метрик
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# Инициализация Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Обучение модели
trainer.train()

# Оценка модели
trainer.evaluate()

# Сохранение модели и токенизатора
trainer.save_model("./saved_toxicity_model")
tokenizer.save_pretrained("./saved_toxicity_model")