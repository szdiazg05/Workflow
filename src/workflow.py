from src.router import router
from src.retriever import retriever
from src.stylist import stylist

def run_workflow(user_question: str) -> dict:
    """Ejecuta el flujo completo del agente.
    
    Pasos:
    1. Router: Decide qué archivo usar
    2. Retriever: Lee archivo y genera respuesta
    3. Stylist: Reformatea en 3 tonos
    
    Args:
        user_question: Pregunta del usuario
        
    Returns:
        Diccionario con todos los resultados
    """
    selected_file = router(user_question)
    original_answer = retriever(user_question, selected_file)
    styled_responses = stylist(original_answer)

    result = {"selected_file": selected_file,**styled_responses}

    return result

if __name__ == "__main__":
    test_questions = [
        "¿Por qué el huitlacoche es superior en nutrientes al maíz tradicional?",
        "¿Cuáles fueron las 3 temperaturas mínimas en diciembre de 2025?",
        "¿Algún consejo para promptear a GPT-5?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        
        result = run_workflow(question)
        
        print(f"Archivo:{result['selected_file']}")
        print(f"Original:{result['original_answer']}")
        print(f"Conciso:{result['dry_answer']}")
        print(f"Gracioso:{result['funny_answer']}")