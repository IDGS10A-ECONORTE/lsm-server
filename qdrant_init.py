from qdrant_client import QdrantClient, models
# Usamos http_models para compatibilidad, aunque PayloadSchemaType fue traído a la raíz.
from qdrant_client.http import models as http_models 
from qdrant_client.http.models import PayloadSchemaType 
import numpy as np
import json
import os

# --- CONFIGURACIÓN DE CONEXIÓN Y ESTRUCTURA ---
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME = "lsm_signs"
VECTOR_DIMENSION = 60
DICTIONARY_FILE = "lsm_dictionary_data.txt" # Nombre del archivo generado por el recorder

# --- FUNCIÓN DE CARGA DE DATOS ---

def load_data_from_txt():
    """
    Lee el archivo TXT línea por línea, parsea cada línea como JSON y retorna
    una lista de diccionarios, lista para ser insertada en Qdrant.
    """
    if not os.path.exists(DICTIONARY_FILE):
        print(f"❌ ERROR: Archivo '{DICTIONARY_FILE}' no encontrado.")
        print("Por favor, usa 'lsm_offline_recorder.py' para generar datos primero.")
        return None

    data_list = []
    try:
        with open(DICTIONARY_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Cada línea es un objeto JSON que contiene sign_name, difficulty y vector
                    data_list.append(json.loads(line))
        
        print(f"✅ Datos cargados: {len(data_list)} registros encontrados en '{DICTIONARY_FILE}'.")
        return data_list
    
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: El archivo '{DICTIONARY_FILE}' contiene JSON inválido en la línea. {e}")
        return None
    except Exception as e:
        print(f"❌ Error al leer el archivo de datos: {e}")
        return None


# --- FUNCIÓN DE INICIALIZACIÓN ---

def initialize_qdrant_collection():
    """
    Se conecta a Qdrant, elimina la colección, la recrea con 60D y carga los datos del TXT.
    """
    print(f"Intentando conectar a Qdrant en {QDRANT_HOST}:{QDRANT_PORT}...")
    
    # Cargar los datos del archivo de texto primero
    raw_data_points = load_data_from_txt()
    if raw_data_points is None or not raw_data_points:
        return # Salir si no hay datos válidos

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    try:
        # 1. Eliminar la colección existente si la hay
        if client.collection_exists(collection_name=COLLECTION_NAME):
            print(f"La colección '{COLLECTION_NAME}' ya existe. Eliminando...")
            client.delete_collection(collection_name=COLLECTION_NAME) 
        
        print(f"Creando la colección '{COLLECTION_NAME}' con {VECTOR_DIMENSION} dimensiones.")

        # 2. Definir CONFIGURACIÓN DE OPTIMIZACIÓN como diccionario (SOLUCIÓN AL ERROR DE VALIDACIÓN)
        optimizers_dict = {
            "default_segment_number": 2, 
            "deleted_threshold": 0.2,
            "vacuum_min_vector_number": 100,
            "flush_interval_sec": 5,
            "max_optimization_threads": 1, 
        }

        # 3. CREAR LA COLECCIÓN
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_DIMENSION, 
                distance=models.Distance.COSINE
            ),
            optimizers_config=optimizers_dict 
        )
        print("Colección creada con éxito.")
        
        # 4. Preparar e insertar los puntos desde el archivo TXT
        points = []
        for data in raw_data_points:
            # Usamos el 'id' del registro (aunque sea simple)
            point_id = data.get("id")
            
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=data["vector"],
                    payload={
                        "sign_name": data["sign_name"], 
                        "difficulty": data["difficulty"]
                    } 
                )
            )

        print(f"Insertando {len(points)} puntos de referencia...")
        
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True 
        )
        
        # 5. Crear índice para el campo de dificultad
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="difficulty",
            field_type=PayloadSchemaType.KEYWORD
        )
        print("Índice 'difficulty' creado para filtrado rápido.")
        
        print(f"✅ Inicialización de Qdrant completa. Total de puntos: {client.count(collection_name=COLLECTION_NAME, exact=True).count}")

    except Exception as e:
        print(f"❌ Error al inicializar la colección de Qdrant. Asegúrate de que el contenedor esté corriendo.")
        print(f"ERROR: {e}")

if __name__ == "__main__":
    initialize_qdrant_collection()