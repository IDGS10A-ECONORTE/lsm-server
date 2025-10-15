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
# Si Python corre en Docker Compose, usar "qdrant". Si es local, usar "localhost".
QDRANT_HOST = "localhost" 
QDRANT_PORT = 6333
QDRANT_COLLECTION = "lsm_signs"
PORT = 7777

# Inicializar MediaPipe Hand Solutions
# Nota: Hands.process() necesita una imagen RGB, cv2.imdecode da BGR.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Inicializar Cliente Qdrant (fuera del async main para evitar recreación)
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
    
    # Manejar caso de división por cero (ej. mano no detectada o detección pobre)
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
        # Esto estandariza todas las manos a una perspectiva de "Mano Derecha"
        if handedness == 'Left':
            x_scaled *= -1
            
        final_vector.extend([x_scaled, y_scaled, z_scaled])

    # El vector final es de 20 landmarks * 3 coordenadas = 60 dimensiones.
    return final_vector, handedness

# ----------------------------------------------------------------------
# LÓGICA DE VALIDACIÓN Y WEBSOCKET
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
            query_filter=query_filter, # Aplicar el filtro del minijuego/objetivo
            limit=1 
        )

        # 3. Evaluación del Resultado
        if search_result and search_result[0].score > 0.80: # Usar un umbral de similitud (ej. 80%)
            return True, f"¡Seña Correcta! Similitud: {search_result[0].score:.2f}"
        else:
            # Si el target no se encuentra con suficiente similitud
            return False, "Seña incorrecta o similitud insuficiente."

    except UnexpectedResponse as e:
        # Esto ocurre si la colección no existe o la dimensión es incorrecta (60D vs la colección)
        return False, f"Error Qdrant (Colección): Verifica la inicialización."
    except Exception as e:
        print(f"Error grave en la consulta a Qdrant: {e}")
        return False, "Error interno del servidor."


async def process_player_image(websocket, path):
    """Maneja la conexión de un jugador y el ciclo de juego."""
    
    # Asignar ID y estado inicial al jugador
    player_id = len(CONNECTED_PLAYERS) + 1
    CONNECTED_PLAYERS[websocket] = {'player_id': player_id, 'target_sign': 'A', 'is_active': True}
    print(f"[Conexión] Jugador {player_id} conectado.")
    
    try:
        await websocket.send(json.dumps({
            "status": "CONNECTED", 
            "player_id": player_id, 
            "message": "Esperando comandos de Godot..."
        }))

        async for message in websocket:
            data = json.loads(message)
            
            # 1. COMANDO DE CONTROL (Desde Godot)
            if data.get('command') == 'SET_TARGET':
                # Establece el signo que el jugador debe realizar
                new_target = data.get('sign', 'A').upper()
                CONNECTED_PLAYERS[websocket]['target_sign'] = new_target
                print(f"Jugador {player_id}: Objetivo establecido: {new_target}")
                
                await websocket.send(json.dumps({"status": "TARGET_SET", "target": new_target}))
                continue
            
            # 2. PROCESAMIENTO DE IMAGEN (Ciclo de juego)
            elif data.get('type') == 'image' and CONNECTED_PLAYERS[websocket]['target_sign'] != 'NONE':
                
                # Deserializar la imagen (Base64)
                img_bytes = base64.b64decode(data['image_data'])
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                # Procesar la imagen (Convertir BGR a RGB para MediaPipe)
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Obtener el vector de seña normalizado (60D)
                hand_vector, handedness = get_normalized_hand_vector(results)

                # Validar el vector contra el signo objetivo en Qdrant
                target_sign = CONNECTED_PLAYERS[websocket]['target_sign']
                is_correct, feedback = validate_sign_against_qdrant(hand_vector, target_sign)

                # 3. RESPONDER A GODOT
                await websocket.send(json.dumps({
                    "result": is_correct,
                    "feedback": feedback,
                    "target": target_sign,
                    "hand_used": handedness # Útil para debug en Godot
                }))
            
    except websockets.exceptions.ConnectionClosedOK:
        print(f"[Desconexión] Jugador {player_id} desconectado limpiamente.")
    except json.JSONDecodeError:
        print(f"Error JSON de Jugador {player_id}")
    except Exception as e:
        print(f"[Error] Jugador {player_id} desconectado por error: {e}")
    finally:
        # Limpieza de la conexión
        if websocket in CONNECTED_PLAYERS:
            del CONNECTED_PLAYERS[websocket]
        print(f"Jugadores activos restantes: {len(CONNECTED_PLAYERS)}")


async def main():
    """Función principal para iniciar el servidor WebSocket."""
    # Enlazar al host 0.0.0.0 para acceso desde el exterior (útil en Docker)
    async with websockets.serve(process_player_image, "0.0.0.0", PORT):
        print("-------------------------------------------------------")
        print(f"Servidor WebSocket LSM iniciado en ws://0.0.0.0:{PORT}")
        print("-------------------------------------------------------")
        await asyncio.Future() # Corre indefinidamente

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServidor detenido por el usuario.")
    except Exception as e:
        print(f"El servidor falló al iniciar: {e}")