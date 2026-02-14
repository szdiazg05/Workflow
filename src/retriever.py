import os
import pandas as pd
from PIL import Image
import google.generativeai as genai 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import GOOGLE_API_KEY

def extraer_texto(response):
    content = response.content
    if isinstance(content, dict):
        return content.get('text', str(content))
    if isinstance(content, list) and content:
        first = content[0]
        return first.get('text', str(first)) if isinstance(first, dict) else str(first)
    return str(content)

llm_langchain = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
)
genai.configure(api_key=GOOGLE_API_KEY)
model_vision = genai.GenerativeModel('gemini-flash-latest')

def load_file_content(filename: str):
    """Carga el contenido del archivo"""
    filepath = os.path.join("data", filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No existe: {filepath}")
    
    if filename.endswith(".csv"):
        df = pd.read_csv(filepath)
        context = f"""Datos del CSV:
{df.head(10).to_string()}

- Total de filas: {len(df)}
- Columnas: {', '.join(df.columns)}
- Periodo: desde {df['PERIODO'].min()} hasta {df['PERIODO'].max()}"""
        return context, "text"
    
    elif filename.endswith(".txt"):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if len(content) > 4000:
            content = content[:4000]
        return content, "text"
    
    elif filename.endswith((".jpg", ".png")):
        return filepath, "image"
    
    else:
        raise ValueError(f"Tipo no soportado: {filename}")


def retriever(user_question: str, filename: str) -> str:
    """Genera respuesta basada en el archivo usando LangChain"""
    
    content, content_type = load_file_content(filename)
    
    if content_type == "image":
        img = Image.open(content)
        prompt = f"""Analiza esta imagen y responde:

{user_question}

REGLA: Si la información NO está en la imagen, di: "Esta informacion no esta disponible en la imagen"."""
        
        response = model_vision.generate_content([prompt, img])
        return response.text
    
    else:
        system_prompt = f"""Eres un asistente experto. Responde UNICAMENTE basandote en el siguiente contexto.

CONTEXTO:
{content}

REGLAS CRÍTICAS:
- SOLO usa información del contexto proporcionado
- Si la pregunta no puede responderse con el contexto, di claramente: "Esta información no está disponible en los datos proporcionados"."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_question)
        ]
        
        response = llm_langchain.invoke(messages)
        return extraer_texto(response)
    
if __name__ == "__main__":
    answer = retriever("¿Top 3 estados más calientes en enero 1985?", "datos_clima_mexico.csv")
    print(f"Respuesta: {answer[:200]}")