import asyncio
import websockets
import json
import base64
import numpy as np
import cv2
import os
import mediapipe as mp
import random # Importado para la nueva función de asignación
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

# --- CONFIGURACIÓN GLOBAL ---
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = "lsm_signs"

SOCKET_IP = "0.0.0.0"
PORT = 7777 # Puerto del WebSocket

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
    Procesa los resultados de MediaPipe para normalizar la mano detectada
    y crear un vector de 60 dimensiones (20 landmarks * 3 coordenadas).
    
    Retorna (hand_vector_60d, handedness)
    """
    if not results.multi_hand_landmarks:
        return None, None

    # Asumimos una sola mano (max_num_hands=1 en la configuración)
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # 1. Obtener los 21 puntos (x, y, z)
    points = []
    for landmark in hand_landmarks.landmark:
        points.extend([landmark.x, landmark.y, landmark.z])

    # El vector inicial es de 63 dimensiones (21 * 3)
    vector_63d = np.array(points, dtype=np.float32)

    # 2. Normalización de Traslación (Centrado en la Muñeca)
    # El punto 0 es la muñeca (wrist)
    wrist_x, wrist_y, wrist_z = vector_63d[0], vector_63d[1], vector_63d[2]
    
    # Restar las coordenadas de la muñeca a todos los puntos (X, Y, Z)
    for i in range(0, 63, 3):
        vector_63d[i] -= wrist_x    # X
        vector_63d[i+1] -= wrist_y  # Y
        vector_63d[i+2] -= wrist_z  # Z

    # 3. Normalización de Escala (Dividir por la distancia de la muñeca al dedo medio)
    # Usamos el punto 9 (MCP del Dedo Medio) para escala
    # vector_63d[27], [28], [29] son X, Y, Z del punto 9 (índices 27-29)
    # Nota: Como ya están centrados, son las distancias respecto a la muñeca.
    scale_factor = np.linalg.norm(vector_63d[27:30])
    
    # Prevenir división por cero si la mano está colapsada o no se detectó bien
    if scale_factor < 1e-6:
        return None, None 

    # Aplicar el factor de escala a todos los puntos, excepto la muñeca (ya es [0,0,0])
    vector_63d /= scale_factor
    
    # 4. Eliminar el punto de la Muñeca (es [0,0,0] después de la normalización)
    # El vector final es de 60 dimensiones (20 * 3)
    hand_vector_60d = vector_63d[3:] 

    # 5. Determinar la lateralidad (handedness)
    handedness = results.multi_handedness[0].classification[0].label
    
    if handedness == 'Left':
        # Reflejar el eje X (cada tercer elemento, empezando por el primero)
        # Esto convierte la seña zurda a la perspectiva diestra
        for i in range(0, len(hand_vector_60d), 3):
            hand_vector_60d[i] *= -1

    return hand_vector_60d.tolist(), handedness

# ----------------------------------------------------------------------
# FUNCIONES DE LÓGICA DE JUEGO (NUEVAS)
# ----------------------------------------------------------------------
def get_signs_by_difficulty_from_qdrant(difficulty_level):
    """
    Recupera una lista de 'sign_name' desde Qdrant filtrando por difficulty_level.
    
    Esta función es síncrona y BLOQUEANTE, pero la usaremos con run_in_executor.
    """
    signs = []
    
    # Si la dificultad es 'ANY', no aplicamos filtro
    if difficulty_level.upper() == 'ANY':
        query_filter = None
    else:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="difficulty_level", 
                    match=models.MatchValue(value=difficulty_level.upper())
                )
            ]
        )

    try:
        # Usamos scroll para recuperar todos los puntos que coincidan con el filtro.
        # Solo solicitamos los campos 'sign_name' del payload.
        all_points, _ = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=query_filter,
            with_payload=True,
            limit=500, # Límite razonable para una búsqueda rápida de todos los signos
            with_vectors=False, # No necesitamos el vector, solo el payload
            payload_selector=models.PayloadSelectorInclude(
                include=["sign_name"]
            )
        )

        for point in all_points:
            # Añadir el nombre de la seña si existe y no está ya en la lista
            sign_name = point.payload.get("sign_name")
            if sign_name and sign_name not in signs:
                signs.append(sign_name)
                
    except Exception as e:
        print(f"Error al consultar Qdrant para dificultad {difficulty_level}: {e}")
        # En caso de error, retorna una lista vacía para evitar fallos.
        return []

    return signs

def assign_target_sign(difficulty_level="EASY", loop=None):
    """
    Asigna una seña objetivo aleatoria basada en la dificultad, consultando Qdrant.
    
    IMPORTANTE: Si se llama de forma asíncrona, debe usarse con run_in_executor.
    """
    
    # Esta función ahora es un simple wrapper que llama a la función de Qdrant.
    signs = get_signs_by_difficulty_from_qdrant(difficulty_level)

    if signs:
        return random.choice(signs)
    else:
        print(f"Advertencia: No se encontraron señas para la dificultad '{difficulty_level}'.")
        return None

# ----------------------------------------------------------------------
# FUNCIONES DE VALIDACIÓN Y BÚSQUEDA QDRANT (Umbral Ajustado)
# ----------------------------------------------------------------------

def validate_sign_against_qdrant(vector, target_sign):
    """
    Busca el vector normalizado en Qdrant para validar si es el signo objetivo.
    
    Retorna (is_correct, feedback_message, score_percentage).
    """
    if vector is None or target_sign is None:
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
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector,
            query_filter=query_filter, 
            limit=1 
        )

        # 3. Evaluación del Resultado
        score = search_result[0].score if search_result else 0.0
        score_percent = score * 100.0 
        
        # Umbral requerido: mayor a 94% (score > 0.94)
        if score > 0.94: 
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

def _perform_sign_validation(image_bytes, target_sign):
    """
    Maneja la decodificación, el procesamiento CV/MediaPipe y la validación Qdrant.
    Esta función es síncrona (bloqueante) y debe ejecutarse en un hilo secundario.
    
    Retorna: (is_correct, feedback, score_percent)
    """
    is_correct = False
    feedback = "Señal no detectada o error de procesamiento."
    score_percent = 0.0

    try:
        # Decodificación de la imagen de bytes a un frame de OpenCV
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("No se pudo decodificar el frame de la imagen.")
        
        # Procesar y obtener el vector 60D
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        hand_vector, handedness = get_normalized_hand_vector(results)

        # Si se detectó una mano, validar contra Qdrant y OBTENER EL SCORE
        if hand_vector:
            is_correct, feedback, score_percent = validate_sign_against_qdrant(hand_vector, target_sign)
        else:
            feedback = "Mano no detectada por MediaPipe."
            
    except Exception as proc_e:
        # Manejo de errores durante el procesamiento (decodificación, MediaPipe)
        print(f"Error de procesamiento síncrono: {proc_e}")
        feedback = f"Error de Visión: {proc_e.__class__.__name__}"
    
    return is_correct, feedback, score_percent

# ----------------------------------------------------------------------
# BUCLE PRINCIPAL DEL WEBSOCKET (ASÍNCRONO)
# ----------------------------------------------------------------------

async def process_player_image(websocket):
    """
    Maneja la conexión WebSocket y procesa los mensajes del cliente de forma asíncrona.
    """
    player_id = hex(id(websocket))
    CONNECTED_PLAYERS[websocket] = {"id": player_id, "target_sign": "NONE"}
    print(f"[Conexión] Nuevo jugador {player_id} conectado.")
    
    loop = asyncio.get_event_loop()
    
    # Enviar mensaje de bienvenida con ID al nuevo cliente
    await websocket.send(json.dumps({
        "status": "CONNECTED",
        "player_id": player_id,
        "message": f"Conexión establecida. ID: {player_id}"
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

            # 1. MENSAJE DE CONFIGURACIÓN (Fijar objetivo manual)
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
                    "target": new_target,
                }))
            
            # 1.B. NUEVO: ASIGNAR OBJETIVO POR DIFICULTAD
            elif data.get('type') == 'assign_target' and 'difficulty' in data:
                difficulty = str(data['difficulty'])
                
                # Ejecutar la función de asignación de seña (que consulta Qdrant) en un hilo
                new_target = await loop.run_in_executor(
                    None, 
                    assign_target_sign, # La función a ejecutar síncronamente
                    difficulty          # Argumento para la función
                )
                
                if new_target:
                    CONNECTED_PLAYERS[websocket]['target_sign'] = new_target
                    print(f"[Juego] Jugador {player_id} objetivo asignado a: {new_target} (Dificultad: {difficulty})")
                    
                    await websocket.send(json.dumps({
                        "status": "TARGET_ASSIGNED",
                        "target": new_target,
                        "difficulty": difficulty,
                    }))
                else:
                    await websocket.send(json.dumps({
                        "status": "ERROR",
                        "message": f"Nivel de dificultad '{difficulty}' inválido o no hay señas en Qdrant.",
                    }))

            # 2. PROCESAMIENTO DE IMAGEN (Ciclo de juego)
            elif data.get('type') == 'image' and CONNECTED_PLAYERS[websocket]['target_sign'] != 'NONE':
                
                # --- Preparación Asíncrona Rápida (Decodificación Base64) ---
                target_sign = CONNECTED_PLAYERS[websocket]['target_sign']
                
                try:
                    encoded_image = data.get('image_data') or data.get('data')
                    if not encoded_image:
                        raise ValueError("Mensaje de imagen sin datos.")

                    if isinstance(encoded_image, str):
                        # Convertir a bytes para que sea el mismo formato que el 'else'
                        img_bytes = base64.b64decode(encoded_image.encode('utf-8'))
                    else:
                        img_bytes = base64.b64decode(encoded_image)
                        
                except Exception as decode_e:
                    print(f"Error de decodificación en {player_id}: {decode_e}")
                    feedback = f"Error de Datos: {decode_e.__class__.__name__}"
                    await websocket.send(json.dumps({
                        "result": False, "feedback": feedback, "target": target_sign, "score": 0.0,
                    }))
                    continue # Pasar al siguiente mensaje
                    
                # --- Ejecución Síncrona Lenta en Hilo (Desbloqueante) ---
                is_correct, feedback, score_percent = await loop.run_in_executor(
                    None, 
                    _perform_sign_validation, 
                    img_bytes, 
                    target_sign
                )

                # 3. RESPONDER AL CLIENTE
                await websocket.send(json.dumps({
                    "result": is_correct,
                    "feedback": feedback,
                    "target": target_sign,
                    "score": score_percent, 
                }))

            # 4. MENSAJE DE PAUSA/INACTIVIDAD (Detener el juego)
            elif data.get('type') == 'stop_target' or data.get('command', '').upper() == 'STOP_TARGET':
                CONNECTED_PLAYERS[websocket]['target_sign'] = 'NONE'
                print(f"[Juego] Jugador {player_id} objetivo detenido.")
                
                await websocket.send(json.dumps({
                    "status": "TARGET_STOPPED",
                    "target": "NONE",
                }))
            
            # Otros tipos de mensajes no definidos
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
        await asyncio.Future() # Mantiene el servidor en ejecución


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServidor detenido por el usuario.")
    except Exception as e:
        print(f"\nError en la ejecución principal: {e}")