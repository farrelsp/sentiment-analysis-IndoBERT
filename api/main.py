from fastapi import FastAPI
import time
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer

app = FastAPI()

model_path = "C:/Users/user\Desktop/Portfolio/Sentiment Analysis/model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

i2w = {0: 'positive', 1: 'neutral', 2: 'negative'}


class Item(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/sentiment/predict/")
async def predict(review: Item):
  start = time.time()
  
  # Tokenize the text
  inputs = tokenizer.encode(review.text)
  inputs = torch.LongTensor(inputs).view(1, -1).to(model.device)

  # Inference
  logits = model(inputs)[0]
  label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()
  probability = F.softmax(logits, dim=-1).squeeze()[label].item()

  end = time.time()
  duration = end-start

  result = {
     "text": review.text,
     "prediction": i2w[label],
     "probability": probability,
     "duration": duration
  }

  return result