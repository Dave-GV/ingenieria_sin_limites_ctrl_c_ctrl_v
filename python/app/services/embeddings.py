import numpy as np
from sentence_transformers import SentenceTransformer

model = None

# Función para cargar modelo
def load_model():
    """
    Retorna modelo SentenceTransformer según el nombre del modelo, en este caso se utilizará sentence-transformers/all-MiniLM-L6-v2
    """
    global model

    if not model:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    return model

# Función para generar embeddings
def generate_embeddings(text):
    """
    Entrada: list
    Output: numpy.ndarray
    Si la lista de entrada solamente es un elemento entonces es el np.ndarray es de la forma (384,).
    Si son N elementos entonces es de la forma (N, 384).
    """
    if isinstance(text, str):
        text = [text]

    model = load_model()
    embeddings = model.encode(text).astype('float32')
    return embeddings

# Obtener similitud coseno
def cosine_similarity(a, b):
    """
    Los valores de entrada son dos np.ndarray y se retorna un float
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    return float(np.dot(a, b))