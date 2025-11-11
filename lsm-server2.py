import asyncio
import websockets
import json
import base64
import numpy as np
import cv2
import os
import mediapipe as mp
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

# --- CONFIGURACIÓN GLOBAL ---
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "lsm_signs"
VECTOR_DIMENSION_PER_HAND = 60 # 20 landmarks * 3 ejes (X, Y, Z)

SOCKET_IP = "0.0.0.0"
PORT = 7777 

# Inicializar MediaPipe Hand Solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # <--- CAMBIO CLAVE: SOPORTE PARA 2 MANOS
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Inicializar Cliente Qdrant 
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Diccionario para mantener las conexiones activas y el estado de los jugadores
CONNECTED_PLAYERS = {}

# ----------------------------------------------------------------------
# FUNCIONES DE NORMALIZACIÓN (120 DIMENSIONES CON DOMINANCIA)
# ----------------------------------------------------------------------

def get_normalized_hand_vector(results, player_dominance):
    """
    Procesa los resultados de MediaPipe para normalizar la mano(s) detectada(s)
    y crear un vector unificado de 120 dimensiones.
    
    >>> CAMBIO: La reflexión X ahora se aplica a cualquier mano 'Left',
    >>> independientemente de la dominancia del jugador.

    Retorna (hand_vector_120d, hands_detected_count)
    """
    right_vector_60d = np.zeros(VECTOR_DIMENSION_PER_HAND, dtype=np.float32)
    left_vector_60d = np.zeros(VECTOR_DIMENSION_PER_HAND, dtype=np.float32)
    hands_detected_count = 0

    if not results.multi_hand_landmarks:
        return None, 0

    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
        
        # 1. Obtener los 21 puntos (x, y, z)
        points = []
        for landmark in hand_landmarks.landmark:
            points.extend([landmark.x, landmark.y, landmark.z])
        vector_63d = np.array(points, dtype=np.float32)

        # 2. Normalización de Traslación (Centrado en la Muñeca L0)
        wrist_x, wrist_y, wrist_z = vector_63d[0], vector_63d[1], vector_63d[2]
        for j in range(0, 63, 3):
            vector_63d[j] -= wrist_x    # X
            vector_63d[j+1] -= wrist_y  # Y
            vector_63d[j+2] -= wrist_z  # Z

        # 3. Normalización de Escala (L0 a L9)
        scale_factor = np.linalg.norm(vector_63d[27:30])
        if scale_factor < 1e-6:
            continue
        vector_63d /= scale_factor
        hand_vector_60d = vector_63d[3:] # Eliminar Muñeca (es [0,0,0])
        
        # 4. Determinar la lateralidad
        handedness = results.multi_handedness[i].classification[0].label
        
        # 5. Aplicar la Unificación de Lateralidad (LÓGICA REQUERIDA)
        # Reflejar el eje X solo si MediaPipe detectó la mano como 'Left'.
        # Esto unifica TODAS las señas a una perspectiva de mano derecha (estándar).
        if handedness == 'Left':
            # Reflejar el eje X (cada tercer elemento)
            for k in range(0, len(hand_vector_60d), 3):
                hand_vector_60d[k] *= -1

        hands_detected_count += 1
        
        # 6. Asignar el vector 60D al slot correcto (Mano derecha o izquierda original)
        if handedness == 'Right':
            right_vector_60d = hand_vector_60d
        elif handedness == 'Left':
            left_vector_60d = hand_vector_60d

    # El vector final es [Derecha_60D, Izquierda_60D] (120D)
    final_vector_120d = np.concatenate((right_vector_60d, left_vector_60d)).tolist()
    
    # Nota: El parámetro player_dominance ya no afecta la reflexión,
    # pero se mantiene en el diccionario de CONNECTED_PLAYERS y en la llamada
    # para futuras validaciones que sí podrían usarlo.
    
    return final_vector_120d, hands_detected_count

# ----------------------------------------------------------------------
# FUNCIONES DE VALIDACIÓN Y BÚSQUEDA QDRANT (MODIFICADA)
# ----------------------------------------------------------------------

def validate_sign_against_qdrant(vector_120d, target_sign):
    """
    Busca el vector de 120 dimensiones en Qdrant.

    Retorna (is_correct, feedback_message, score_percentage).
    """
    if vector_120d is None or target_sign is None:
        return False, "Error de procesamiento o no hay objetivo.", 0.0
    
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
        # Qdrant realiza la búsqueda de similitud (coseno) de forma precisa con vectores de 120D.
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector_120d,
            query_filter=query_filter, 
            limit=1 
        )

        # 3. Evaluación del Resultado
        score = search_result[0].score if search_result else 0.0
        score_percent = score * 100.0 
        
        if score > 0.98: 
            return True, f"¡Seña Correcta! Similitud: {score_percent:.1f}%", score_percent
        else:
            return False, f"Similitud Insuficiente: {score_percent:.1f}%", score_percent

    except UnexpectedResponse as e:
        print(f"Error Qdrant (Colección): {e}")
        return False, f"Error Qdrant: Colección '{QDRANT_COLLECTION}' no lista.", 0.0
    except Exception as e:
        print(f"Error grave en la consulta a Qdrant: {e}")
        return False, "Error interno del servidor al consultar DB.", 0.0

# ----------------------------------------------------------------------
# FUNCIÓN SÍNCRONA DE PROCESAMIENTO DE IMAGEN (EJECUCIÓN BLOQUEANTE)
# ----------------------------------------------------------------------

def _perform_sign_validation(image_bytes, target_sign, player_dominance):
    """
    Maneja la decodificación, el procesamiento CV/MediaPipe y la validación Qdrant.
    """
    is_correct = False
    feedback = "Señal no detectada o error de procesamiento."
    score_percent = 0.0

    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("No se pudo decodificar el frame de la imagen.")
        
        # MediaPipe trabaja en RGB, cv2 por defecto está en BGR
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        hand_vector_120d, hands_detected_count = get_normalized_hand_vector(results, player_dominance)

        if hand_vector_120d and hands_detected_count > 0:
            is_correct, feedback, score_percent = validate_sign_against_qdrant(hand_vector_120d, target_sign)
        else:
            feedback = "Mano(s) no detectada(s) por MediaPipe."
            
    except Exception as proc_e:
        print(f"Error de procesamiento síncrono: {proc_e}")
        feedback = f"Error de Visión: {proc_e.__class__.__name__}"
    
    return is_correct, feedback, score_percent

# ----------------------------------------------------------------------
# BUCLE PRINCIPAL DEL WEBSOCKET (ASÍNCRONO)
# ----------------------------------------------------------------------

async def process_player_image(websocket):
    player_id = hex(id(websocket))
    # Inicializar con dominance 'RIGHT' por defecto
    CONNECTED_PLAYERS[websocket] = {"id": player_id, "target_sign": "NONE", "dominance": "RIGHT"} 
    print(f"[Conexión] Nuevo jugador {player_id} conectado. Dominancia por defecto: RIGHT")
    
    loop = asyncio.get_event_loop()
    
    await websocket.send(json.dumps({
        "status": "CONNECTED",
        "player_id": player_id,
        "message": f"Conexión establecida. ID: {player_id}. Dominancia: RIGHT"
    }))

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                print(f"Error JSON de {player_id}")
                continue

            if 'player_id' in data:
                CONNECTED_PLAYERS[websocket]['id'] = data['player_id']
                player_id = data['player_id']

            # 1. MENSAJE DE CONFIGURACIÓN DE OBJETIVO (set_target)
            if (
                data.get('type') == 'set_target' and 'sign' in data
            ) or (
                data.get('command', '').upper() == 'SET_TARGET' and 'sign' in data
            ):
                new_target = str(data['sign']).upper()
                CONNECTED_PLAYERS[websocket]['target_sign'] = new_target
                print(f"[Juego] Jugador {player_id} objetivo fijado a: {new_target}")
                
                await websocket.send(json.dumps({
                    "status": "TARGET_SET",
                    "player_id": CONNECTED_PLAYERS[websocket]['id'],
                    "target": new_target,
                }))

            # 2. NUEVO MENSAJE DE CONFIGURACIÓN DE DOMINANCIA
            elif (
                data.get('type') == 'player_config' and 'dominance' in data
            ) or (
                data.get('command', '').upper() == 'SET_DOMINANCE' and 'dominance' in data
            ):
                new_dominance = str(data['dominance']).upper()
                if new_dominance in ['LEFT', 'RIGHT']:
                    CONNECTED_PLAYERS[websocket]['dominance'] = new_dominance
                    print(f"[Config] Jugador {player_id} dominancia fijada a: {new_dominance}")
                    await websocket.send(json.dumps({
                        "status": "DOMINANCE_SET",
                        "player_id": CONNECTED_PLAYERS[websocket]['id'],
                        "dominance": new_dominance,
                    }))
                else:
                     await websocket.send(json.dumps({
                        "status": "ERROR",
                        "message": "Dominancia inválida. Use 'LEFT' o 'RIGHT'."
                    }))

            # 3. PROCESAMIENTO DE IMAGEN
            elif data.get('type') == 'image' and CONNECTED_PLAYERS[websocket]['target_sign'] != 'NONE':
                
                target_sign = CONNECTED_PLAYERS[websocket]['target_sign']
                player_dominance = CONNECTED_PLAYERS[websocket]['dominance'] # Obtener dominancia

                try:
                    encoded_image = data.get('image_data') or data.get('data')
                    if not encoded_image: raise ValueError("Mensaje de imagen sin datos.")

                    if isinstance(encoded_image, str):
                        img_bytes = base64.b64decode(encoded_image.encode('utf-8'))
                    else:
                        img_bytes = base64.b64decode(encoded_image)
                        
                except Exception as decode_e:
                    # ... (Manejo de error de decodificación)
                    await websocket.send(json.dumps({
                        "player_id": CONNECTED_PLAYERS[websocket]['id'],
                        "result": False, "feedback": f"Error de Datos: {decode_e.__class__.__name__}", "target": target_sign, "score": 0.0,
                    }))
                    continue
                    
                # --- Ejecución Síncrona Lenta en Hilo (Desbloqueante) ---
                is_correct, feedback, score_percent = await loop.run_in_executor(
                    None, 
                    _perform_sign_validation, 
                    img_bytes, 
                    target_sign,
                    player_dominance # Enviar el parámetro de dominancia
                )

                # 4. RESPONDER AL CLIENTE
                await websocket.send(json.dumps({
                    "player_id": CONNECTED_PLAYERS[websocket]['id'],
                    "result": is_correct,
                    "feedback": feedback,
                    "target": target_sign,
                    "score": score_percent, 
                }))

            # 5. MENSAJE DE PAUSA/INACTIVIDAD
            elif data.get('type') == 'stop_target' or data.get('command', '').upper() == 'STOP_TARGET':
                # ... (Resto del código) ...
                CONNECTED_PLAYERS[websocket]['target_sign'] = 'NONE'
                print(f"[Juego] Jugador {player_id} objetivo detenido.")
                
                await websocket.send(json.dumps({
                    "status": "TARGET_STOPPED",
                    "player_id": CONNECTED_PLAYERS[websocket]['id'],
                    "target": "NONE",
                }))
            
            else:
                await websocket.send(json.dumps({
                    "status": "UNKNOWN_COMMAND",
                    "message": "Comando no reconocido o target no fijado."
                }))

    except websockets.exceptions.ConnectionClosedOK:
        print(f"[Desconexión] Jugador {player_id} desconectado limpiamente.")
    except Exception as e:
        print(f"[Error Fatal] Jugador {player_id} forzado a desconectar: {e}")
    finally:
        if websocket in CONNECTED_PLAYERS:
            del CONNECTED_PLAYERS[websocket]
        print(f"Jugadores activos restantes: {len(CONNECTED_PLAYERS)}")


async def main():
    try:
        qdrant_client.get_collections()
        print("✅ Conexión inicial a Qdrant exitosa.")
    except Exception as e:
        print(f"❌ ADVERTENCIA: No se pudo conectar a Qdrant. Error: {e}")
        
    async with websockets.serve(process_player_image, SOCKET_IP, PORT):
        print("-------------------------------------------------------")
        print(f"Servidor WebSocket LSM (Dual Hand) iniciado en ws://{SOCKET_IP}:{PORT}")
        print("-------------------------------------------------------")
        await asyncio.Future() 


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServidor detenido por el usuario.")
    except Exception as e:
        print(f"\nError en la ejecución principal: {e}")