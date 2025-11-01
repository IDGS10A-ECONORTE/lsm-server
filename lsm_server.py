import asyncio
import websockets
import json
import base64
import numpy as np
import cv2
import mediapipe as mp
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

# --- CONFIGURACIÓN GLOBAL ---
QDRANT_HOST = "localhost" # Usar "qdrant" si usas Docker Compose
QDRANT_PORT = 6333
QDRANT_COLLECTION = "lsm_signs"

SOCKET_IP = "0.0.0.0"
PORT = 7777 # Usando el puerto que especificaste

# Inicializar MediaPipe Hand Solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Inicializar Cliente Qdrant (debe ser accesible al iniciar)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Diccionario para mantener las conexiones activas y el estado de los jugadores
CONNECTED_PLAYERS = {}

# ----------------------------------------------------------------------
# FUNCIONES DE NORMALIZACIÓN (60 DIMENSIONES, X/Y/Z, TAMAÑO Y LATERALIDAD)
# ----------------------------------------------------------------------

def get_normalized_hand_vector(results):
    """
    Procesa los resultados de MediaPipe para crear un vector de 60 dimensiones (X, Y, Z normalizados).
    Aplica: Centrado, Escalado por Tamaño (Distancia L0-L9), y Mirroring (Reflexión en X).
    """
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None, None

    # Asumimos una sola mano para el vector
    landmarks = results.multi_hand_landmarks[0].landmark
    handedness = results.multi_handedness[0].classification[0].label # 'Left' o 'Right'

    # 1. Centrado (Traslación): Usamos la muñeca (L0) como origen.
    wrist = landmarks[0]
    relative_landmarks = []
    for lm in landmarks:
        # Calcular coordenadas relativas (X', Y', Z')
        relative_landmarks.append((lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z))

    # 2. Escalado (Ajuste de Tamaño): Distancia L0 a L9 (base del dedo corazón).
    l9 = relative_landmarks[9]
    d_ref = np.sqrt(l9[0]**2 + l9[1]**2 + l9[2]**2)
    
    # Manejar caso de división por cero (ej. mano muy cerca o detección pobre)
    if d_ref < 1e-6: 
        return None, None
        
    final_vector = []
    
    # 3. Escalado y Reflexión (Mirroring)
    # Iteramos desde L1 hasta L20 (excluyendo L0 - muñeca)
    for i in range(1, 21): 
        x_prime, y_prime, z_prime = relative_landmarks[i]

        x_scaled = x_prime / d_ref
        y_scaled = y_prime / d_ref
        z_scaled = z_prime / d_ref
        
        # Aplicar Mirroring (Reflexión en X para mano izquierda - zurdos)
        if handedness == 'Left':
            x_scaled *= -1
            
        final_vector.extend([x_scaled, y_scaled, z_scaled])

    # El vector final es de 20 landmarks * 3 coordenadas = 60 dimensiones.
    return final_vector, handedness

# ----------------------------------------------------------------------
# LÓGICA DE VALIDACIÓN DE QDRANT
# ----------------------------------------------------------------------

def validate_sign_against_qdrant(vector, target_sign):
    """
    Busca el vector normalizado en Qdrant para validar si es el signo objetivo.
    """
    if vector is None or target_sign is None:
        return False, "Error de procesamiento o no hay objetivo."
    
    try:
        # 1. Definir el filtro (solo buscar el signo objetivo actual)
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="sign_name", 
                    match=models.MatchValue(value=target_sign.upper())
                )
            ]
        )
        
        # 2. Búsqueda de Vecinos Más Cercanos Aproximados (ANN)
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            query_filter=query_filter, 
            limit=1 
        )

        # 3. Evaluación del Resultado
        # Nota: Puedes ajustar el umbral 0.80 (80%) según la precisión de tu entrenamiento
        if search_result and search_result[0].score > 0.80: 
            # La seña más cercana del target deseado tiene una alta similitud
            return True, f"¡Seña Correcta! Similitud: {search_result[0].score:.2f}"
        else:
            # No se encontró una coincidencia con el umbral de similitud
            return False, "Seña incorrecta o similitud insuficiente."

    except UnexpectedResponse as e:
        print(f"Error Qdrant (Colección): {e}")
        return False, f"Error Qdrant: Colección '{QDRANT_COLLECTION}' no lista o dimensiones incorrectas."
    except Exception as e:
        print(f"Error grave en la consulta a Qdrant: {e}")
        return False, "Error interno del servidor al consultar DB."

# ----------------------------------------------------------------------
# BUCLE PRINCIPAL DEL WEBSOCKET (Manejo de Conexión y Datos)
# ----------------------------------------------------------------------

async def process_player_image(websocket):
    """Maneja la conexión de un jugador y el ciclo de juego."""
    
    player_id = len(CONNECTED_PLAYERS) + 1
    CONNECTED_PLAYERS[websocket] = {'player_id': player_id, 'target_sign': 'A', 'is_active': True}
    print(f"[Conexión] Jugador {player_id} conectado.")
    
    try:
        # Envía el estado inicial al cliente
        await websocket.send(json.dumps({
            "status": "CONNECTED", 
            "player_id": player_id, 
            "target": CONNECTED_PLAYERS[websocket]['target_sign'],
            "message": "Esperando comandos o imágenes..."
        }))

        async for message in websocket:
            
            # Intento de deserialización del mensaje JSON
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                print(f"Error JSON de Jugador {player_id}. Mensaje descartado.")
                continue

            # 1. COMANDO DE CONTROL
            if data.get('command') == 'SET_TARGET':
                new_target = data.get('sign', 'A').upper()
                CONNECTED_PLAYERS[websocket]['target_sign'] = new_target
                print(f"Jugador {player_id}: Objetivo establecido: {new_target}")
                
                await websocket.send(json.dumps({"status": "TARGET_SET", "target": new_target}))
                continue
            
            # 2. PROCESAMIENTO DE IMAGEN (Ciclo de juego)
            elif data.get('type') == 'image' and CONNECTED_PLAYERS[websocket]['target_sign'] != 'NONE':
                
                is_correct = False
                feedback = "Señal no detectada o error de procesamiento."
                target_sign = CONNECTED_PLAYERS[websocket]['target_sign']

                # Bloque try/except para el procesamiento de imagen
                try:
                    # Deserialización y decodificación
                    img_bytes = base64.b64decode(data['image_data'])
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    # Verificar validez de la imagen
                    if frame is None or frame.size == 0:
                        feedback = "Error de decodificación (Frame nulo o incompleto)."
                    else:
                        # Procesar y obtener el vector 60D
                        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                        hand_vector, handedness = get_normalized_hand_vector(results)

                        # Validar contra Qdrant
                        is_correct, feedback = validate_sign_against_qdrant(hand_vector, target_sign)

                except Exception as proc_e:
                    # Capturar cualquier fallo de procesamiento (ej. MediaPipe/Numpy/CV)
                    print(f"ERROR DE PROCESAMIENTO JUGADOR {player_id}: {proc_e.__class__.__name__}")
                    feedback = f"Error de Visión: {proc_e.__class__.__name__}"


                # 3. RESPONDER AL CLIENTE (Movido fuera del try/except anidado)
                await websocket.send(json.dumps({
                    "result": is_correct,
                    "feedback": feedback,
                    "target": target_sign,
                }))
            
    except websockets.exceptions.ConnectionClosedOK:
        print(f"[Desconexión] Jugador {player_id} desconectado limpiamente.")
    except Exception as e:
        print(f"[Error Fatal] Jugador {player_id} forzado a desconectar: {e}")
    finally:
        # Limpieza de la conexión
        if websocket in CONNECTED_PLAYERS:
            del CONNECTED_PLAYERS[websocket]
        print(f"Jugadores activos restantes: {len(CONNECTED_PLAYERS)}")


async def main():
    """Función principal para iniciar el servidor WebSocket."""
    # Verificar la conexión inicial a Qdrant antes de iniciar el servidor WS
    try:
        qdrant_client.get_collections()
        print("✅ Conexión inicial a Qdrant exitosa.")
    except Exception as e:
        print(f"❌ ADVERTENCIA: No se pudo conectar a Qdrant en {QDRANT_HOST}:{QDRANT_PORT}. El servidor iniciará, pero las validaciones fallarán. Error: {e}")
        
    async with websockets.serve(process_player_image, SOCKET_IP, PORT):
        print("-------------------------------------------------------")
        print(f"Servidor WebSocket LSM iniciado en ws://{SOCKET_IP}:{PORT}")
        print("-------------------------------------------------------")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServidor detenido por el usuario.")
    except Exception as e:
        print(f"El servidor falló al iniciar: {e}")