from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Загрузка модели и токенизатора
model = BertForSequenceClassification.from_pretrained("./saved_toxicity_model")
tokenizer = BertTokenizer.from_pretrained("./saved_toxicity_model")

def predict_toxicity(text):
    # Токенизация текста
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Отключение подсчета градиентов
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Получение вероятностей и предсказания
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()
    
    return {"toxic": prediction, "probabilities": probabilities.tolist()}

test_text = "А вот я бы на твоем месте..."
result = predict_toxicity(test_text)
print(test_text)
print(f"Prediction: {'Токсик!' if result['toxic'] == 1 else 'Не токсик!'}")
print(f"Probabilities: {result['probabilities']}")
