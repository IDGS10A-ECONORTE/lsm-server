from qdrant_client import QdrantClient, models
from qdrant_client.http import models as http_models
from qdrant_client.http.models import PayloadSchemaType 
import numpy as np

# --- CONFIGURACIÓN DE CONEXIÓN Y ESTRUCTURA ---
QDRANT_HOST = "localhost" 
QDRANT_PORT = 6333
COLLECTION_NAME = "lsm_signs"

# 20 landmarks * 3 coordenadas (x, y, z) = 60 DIMENSIONES
VECTOR_DIMENSION = 60

# --- DICCIONARIO BASE (SIMULANDO LA ESTRUCTURA QUE TE FUNCIONA) ---
# Hemos unificado la estructura para asegurar que cada punto tenga un nivel de dificultad.
# En la realidad, las claves "HOLA", "LSM", etc., DEBEN tener un nivel de dificultad asignado.
# Aquí asumimos que los que no lo tienen son por defecto "MEDIO" o "FÁCIL".

MOCK_SIGNS_RAW = {
    # Formato simple (se asume DIFICULTAD por defecto)
    "HOLA": {"vector": np.random.rand(VECTOR_DIMENSION).tolist(), "difficulty": "MEDIO"},
    "LSM": {"vector": np.random.rand(VECTOR_DIMENSION).tolist(), "difficulty": "MEDIO"},
    "GRACIAS": {"vector": np.random.rand(VECTOR_DIMENSION).tolist(), "difficulty": "MEDIO"},
    "JUGAR": {"vector": np.random.rand(VECTOR_DIMENSION).tolist(), "difficulty": "DIFÍCIL"},
    "A": {"vector": np.random.rand(VECTOR_DIMENSION).tolist(), "difficulty": "FÁCIL"},
    "B": {"vector": np.random.rand(VECTOR_DIMENSION).tolist(), "difficulty": "FÁCIL"},
}

# --- FUNCIÓN DE INICIALIZACIÓN ---

def initialize_qdrant_collection():
    """
    Se conecta a Qdrant, elimina la colección y la recrea con el esquema de 60D.
    """
    print(f"Intentando conectar a Qdrant en {QDRANT_HOST}:{QDRANT_PORT}...")
    
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
        
        # 4. Preparar y insertar los puntos (Maneja ambos formatos: simple y anidado)
        points = []
        point_id_counter = 0

        for sign_name, data in MOCK_SIGNS_RAW.items():
            
            # --- LÓGICA DE UNIFICACIÓN DE ESTRUCTURA ---
            if 'payload' in data:
                # Caso de estructura anidada (como "AZUL")
                vector_data = data['vector']
                payload_data = data['payload']
            else:
                # Caso de estructura simple (como "HOLA")
                vector_data = data['vector']
                payload_data = {
                    "sign_name": sign_name,
                    "difficulty": data.get("difficulty", "MEDIO") # Usar el valor o "MEDIO" por defecto
                }
            # ----------------------------------------

            points.append(
                models.PointStruct(
                    id=point_id_counter,
                    vector=vector_data,
                    payload=payload_data 
                )
            )
            point_id_counter += 1

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
        print(f"❌ Error al inicializar la colección de Qdrant. Asegúrate de que el contenedor esté corriendo y el puerto 6333 esté libre.")
        print(f"ERROR: {e}")

if __name__ == "__main__":
    initialize_qdrant_collection()