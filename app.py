from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import re
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
from fastapi.templating import Jinja2Templates

# Configurar dispositivo (CPU o GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el modelo de análisis de sentimientos preentrenado en español
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Inicializar el pipeline de análisis de sentimientos
sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Crear la aplicación FastAPI
app = FastAPI()

# Configurar Jinja2 para renderizar HTML
templates = Jinja2Templates(directory="templates")

# Montar archivos estáticos (JavaScript, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Modelo de datos para la entrada
class SentimentRequest(BaseModel):
    text: str

# Función para limpiar el texto
def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9ñÑáéíóúÁÉÍÓÚ\s]', '', text)  # Eliminar caracteres especiales
    return text

# Ruta principal para servir la interfaz web
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Ruta de análisis de sentimientos
@app.post("/api/sentiment/")
async def analyze_sentiment(request: SentimentRequest):
    cleaned_text = clean_text(request.text)
    sentiment_output = sentiment_pipeline(cleaned_text)[0]
    sentiment_label = sentiment_output['label']
    sentiment_score = sentiment_output['score']
    
    try:
        sentiment_blob = TextBlob(cleaned_text).translate(from_lang='es', to='en').sentiment
        polarity = round(sentiment_blob.polarity, 2)  # Redondear a 2 decimales
        subjectivity = round(sentiment_blob.subjectivity, 2)  # Redondear a 2 decimales
    except Exception as e:
        polarity, subjectivity = 0.0, 0.0

    return {
        
        "bert_sentiment": {
            "label": sentiment_label,
            "confidence": sentiment_score
        },
        "textblob_sentiment": {
            "polarity": polarity,
            "subjectivity": subjectivity
        }
    }
