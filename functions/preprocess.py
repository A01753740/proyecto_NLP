import pandas as pd

def load_data(file_path):
    """Carga los datos del archivo CSV."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Preprocesa los datos:
    - Elimina entradas vacías.
    - Normaliza espacios.
    """
    data = data.dropna(subset=['reviewText'])
    
    data['reviewText'] = data['reviewText'].str.strip()
    data = data['reviewText']
    
    return data