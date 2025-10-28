from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# --- CONFIGURACIÓN ---
QDRANT_HOST = "localhost" 
QDRANT_PORT = 6333
COLLECTION_NAME = "lsm_signs"
# Ajusta el ID. Si 'A' fue el 5to punto insertado (después de HOLA, LSM, GRACIAS, JUGAR), su ID es 4.
POINT_ID_TO_RETRIEVE = 0

def retrieve_sign_by_id(point_id: int):
    """
    Se conecta a Qdrant y recupera un punto (registro) específico usando su ID.
    """
    print(f"Intentando conectar a Qdrant en {QDRANT_HOST}:{QDRANT_PORT}...")
    
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            print(f"❌ ERROR: La colección '{COLLECTION_NAME}' no existe.")
            return

        print(f"✅ Conexión exitosa. Buscando punto con ID: {point_id}...")
        
        # Usamos client.retrieve() para obtener el punto exacto por ID
        records = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],      # Se espera una lista de IDs
            with_payload=True,   # Incluir el sign_name y la dificultad
            with_vectors=True    # Incluir el vector de 60D
        )

        # 3. Mostrar los resultados
        if records:
            record = records[0]
            print("\n--- REGISTRO ENCONTRADO ---")
            print(f"  ID: {record.id}")
            print(f"  Seña: {record.payload.get('sign_name')}")
            print(f"  Dificultad: {record.payload.get('difficulty')}")
            print(f"  Vector (60D): {record.vector[:3]}... [{len(record.vector)} dimensiones]")
            print("-" * 30)
        else:
            print(f"\n⚠️ Advertencia: No se encontró ningún punto con ID {point_id}.")

    except Exception as e:
        print(f"\n❌ ERROR de conexión o consulta: {e}")

if __name__ == "__main__":
    retrieve_sign_by_id(POINT_ID_TO_RETRIEVE)