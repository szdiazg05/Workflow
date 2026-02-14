from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import GOOGLE_API_KEY
import json

def extraer_texto(response):
    """Extrae solo el texto limpio"""
    content = response.content
    if isinstance(content, dict):
        return content.get('text', str(content))
    if isinstance(content, list) and content:
        first = content[0]
        return first.get('text', str(first)) if isinstance(first, dict) else str(first)
    return str(content)

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
)

class Estilo_respuesta(BaseModel):
    Respuesta_original: str = Field(description="Respuesta original sin cambios")
    Respuesta_directa: str = Field(description="Versión concisa y directa")
    Respuesta_divertida: str = Field(description="Versión con humor")

def stylist(Respuesta_original: str) -> dict:

    system_prompt = """Genera 3 versiones de la respuesta. Responde SOLO con JSON válido."""
    user_prompt = f"""Respuesta original:{Respuesta_original}

Genera un JSON con esta estructura EXACTA:
{{"Respuesta_original": "copia exacta de la respuesta original",
  "Respuesta_directa": "version concisa, maximo 2 oraciones",
  "Respuesta_divertida": "version con humor y emojis, manten la info correcta"}}"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    json_text = extraer_texto(response)
    json_text = json_text.replace('```json', '').replace('```', '').strip()
    
    data = json.loads(json_text)
    if isinstance(data, list):
        data=data[0]if data else {}

    styled = Estilo_respuesta(**data)
    
    return styled.model_dump()

if __name__ == "__main__": 
    test = "Los 5 estados más calientes en agosto 2021 fueron: Baja California Sur (32°C), Sonora (31°C), Sinaloa (30°C), Colima (30°C) y Nayarit (29°C)."
    result = stylist(test)  
    print(f"Original:{result['Respuesta_original']}")
    print(f"Conciso:{result['Respuesta_directa']}")
    print(f"Gracioso:{result['Respuesta_divertida']}")