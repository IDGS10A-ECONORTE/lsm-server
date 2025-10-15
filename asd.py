from qdrant_client import QdrantClient

# --- CONFIGURACIÓN DE CONEXIÓN ---
QDRANT_HOST = "localhost" 
QDRANT_PORT = 6333
COLLECTION_NAME = "lsm_signs"

def query_three_records():
    """
    Se conecta a Qdrant y consulta los primeros 3 registros de la colección 'lsm_signs'.
    """
    print(f"Intentando conectar a Qdrant en {QDRANT_HOST}:{QDRANT_PORT}...")
    
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # 1. Verificar si la colección existe
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            print(f"❌ ERROR: La colección '{COLLECTION_NAME}' no existe. Por favor, ejecuta 'qdrant_init.py' primero.")
            return

        print(f"✅ Conexión exitosa. Consultando los primeros 3 registros de '{COLLECTION_NAME}'...")
        
        # 2. Usar el método scroll para obtener una muestra de los datos
        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=3,                 # Limitar a 3 registros
            with_payload=True,       # Incluir la dificultad y el sign_name
            with_vectors=False       # Omitir el vector de 60D para una salida limpia
        )

        # 3. Mostrar los resultados
        if records:
            print("\n--- REGISTROS ENCONTRADOS (3 Primeros) ---")
            for record in records:
                # Mostrar el ID y el contenido del Payload
                print(f"  ID: {record.id}")
                print(f"  Signo: {record.payload.get('sign_name')}")
                print(f"  Dificultad: {record.payload.get('difficulty')}")
                # Mostrar el payload completo para el caso de 'AZUL' u otros campos
                print(f"  Payload Completo: {record.payload}")
                print("-" * 30)
        else:
            print("\n⚠️ Advertencia: La colección está vacía.")

    except Exception as e:
        print(f"\n❌ ERROR de conexión o consulta: {e}")
        print("Asegúrate de que el contenedor Docker de Qdrant esté corriendo.")

if __name__ == "__main__":
    query_three_records()