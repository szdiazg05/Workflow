from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
)

def extraer_texto(response):
    content = response.content
    if isinstance(content, dict):
        return content.get('text', str(content))
    if isinstance(content, list) and content:
        return content[0].get('text', str(content[0])) if isinstance(content[0], dict) else str(content[0])
    return str(content)

def router(user_question: str) -> str:
    """Decide qué archivo usar"""
    
    prompt = f"""Eres un enrutador. Decide qué archivo usar.

Archivos:
1. "datos_clima_mexico.csv" - Datos de clima
2. "GPT-41_PromptingGuide.txt" - Guía GPT-4.1
3. "maiz_info.jpg" - Maíz México

Pregunta: {user_question}

Responde SOLO el nombre del archivo, nada mas."""
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_question)
    ] 
    response = llm.invoke(messages)
    
    filename = extraer_texto(response).strip().replace('"', '').replace("'", "")
    
    valid_files = [
        "datos_clima_mexico.csv",
        "GPT-41_PromptingGuide.txt",
        "maiz_info.jpg"
    ]
    
    if filename not in valid_files:
        for valid_file in valid_files:
            if valid_file in filename:
                filename = valid_file
                break
        else:
            raise ValueError(f"Archivo no reconocido: {filename}.Por favor reformula tu pregunta.")
    
    return filename

if __name__ == "__main__":
    result = router("¿Temperatura máxima en agosto?")
    print(f"Archivo seleccionado:{result}")