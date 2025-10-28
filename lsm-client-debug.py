import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import websockets
import asyncio
import json
import base64
from PIL import Image, ImageTk

# --- CONFIGURACIÓN DEL CLIENTE ---
# Asegúrate de que el puerto coincida con lsm_server.py
WEBSOCKET_URL = "ws://localhost:7777"
CAMERA_INDEX = 1
FPS_LIMIT = 15 # Limitar el envío a aproximadamente 15 frames por segundo
FRAME_INTERVAL_MS = int(1000 / FPS_LIMIT)

class GameClientApp:
    def __init__(self, master):
        self.master = master
        master.title("LSM Game Client (Simulación Godot)")
        
        # Estado de conexión
        self.ws_connected = False
        self.websocket = None
        
        # Cámara
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        
        # --- UI ELEMENTS ---
        self.master.geometry("750x600")
        
        # 1. Marco de Video
        self.video_frame = tk.Frame(master)
        self.video_frame.pack(padx=10, pady=5)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # 2. Etiqueta de Estado de Conexión
        self.conn_label = tk.Label(master, text="Desconectado...", fg="red", font=("Arial", 10))
        self.conn_label.pack(pady=5)
        
        # 3. Etiqueta de Seña Objetivo y Feedback
        self.target_label = tk.Label(master, text="Objetivo: (No establecido)", fg="gray", font=("Arial", 14))
        self.target_label.pack(pady=5)

        # 4. Etiqueta de Resultado de la Seña
        self.result_label = tk.Label(master, text="Esperando Seña...", fg="blue", font=("Arial", 18, "bold"))
        self.result_label.pack(pady=10)

        # 5. Botón de Control (Ejemplo: Obtener nuevo objetivo)
        self.target_button = tk.Button(master, text="Solicitar Seña 'A'", command=lambda: self.send_command('SET_TARGET', 'A'))
        self.target_button.pack(pady=10)
        
        # Iniciar la conexión WebSocket
        self.master.after(100, self.start_websocket)
        # Iniciar el bucle de la cámara
        self.update_frame()

    # ---------------------------------
    # --- CONEXIÓN Y RECEPCIÓN WS ---
    # ---------------------------------

    def start_websocket(self):
        """Inicia el cliente WebSocket de forma asíncrona."""
        asyncio.run_coroutine_threadsafe(self.connect_and_receive(), asyncio.get_event_loop())

    async def connect_and_receive(self):
        """Maneja la conexión y el bucle de recepción de datos."""
        try:
            # Intentar conectar
            self.websocket = await websockets.connect(WEBSOCKET_URL)
            self.ws_connected = True
            self.conn_label.config(text=f"Conectado a {WEBSOCKET_URL}", fg="green")
            
            # Enviar el primer comando para obtener un objetivo por defecto
            await self.send_command_async('SET_TARGET', 'HOLA')

            # Bucle de Recepción: Escuchar continuamente las respuestas del servidor
            while self.ws_connected:
                message = await self.websocket.recv()
                self.process_server_response(message)

        except ConnectionRefusedError:
            self.conn_label.config(text="ERROR: Conexión rechazada (Servidor inactivo)", fg="red")
        except Exception as e:
            self.conn_label.config(text=f"ERROR de conexión WS: {e}", fg="red")
        finally:
            self.ws_connected = False
            self.websocket = None

    def process_server_response(self, message):
        """Actualiza la interfaz con la respuesta del lsm_server."""
        try:
            response = json.loads(message)

            if response.get("status") == "NEW_TARGET" or response.get("status") == "TARGET_SET":
                # Si el servidor establece un nuevo objetivo
                self.target_label.config(text=f"Objetivo Actual: {response.get('target')}", fg="black")
                self.result_label.config(text="¡Haz la Seña!", fg="blue")

            elif "result" in response:
                # Si el servidor envía un resultado de validación
                is_correct = response["result"]
                feedback = response.get("feedback", "Procesando...")
                
                if is_correct:
                    self.result_label.config(text=f"✅ ¡CORRECTO! ({feedback})", fg="green")
                else:
                    self.result_label.config(text=f"❌ INCORRECTO: {feedback}", fg="red")

        except json.JSONDecodeError:
            print("Error al decodificar la respuesta JSON.")
        except Exception as e:
            print(f"Error al procesar respuesta: {e}")

    # ---------------------------------
    # --- ENVÍO DE DATOS WS ---
    # ---------------------------------

    def send_command(self, command, sign):
        """Función síncrona para enviar comandos desde la UI (Botón)."""
        if self.ws_connected:
            asyncio.run_coroutine_threadsafe(self.send_command_async(command, sign), asyncio.get_event_loop())
        else:
            self.conn_label.config(text="¡Desconectado! Reintentando...", fg="red")
            self.start_websocket()

    async def send_command_async(self, command, sign):
        """Envía un comando de control (ej. SET_TARGET) al servidor."""
        if self.websocket and self.ws_connected:
            message = json.dumps({"command": command, "sign": sign})
            await self.websocket.send(message)

    async def send_frame_data(self, image_data):
        """Envía la imagen codificada en Base64 al servidor."""
        if self.websocket and self.ws_connected:
            message = json.dumps({"type": "image", "image_data": image_data.decode('utf-8')})
            await self.websocket.send(message)
            
    # ---------------------------------
    # --- CAPTURA DE CÁMARA ---
    # ---------------------------------

    def update_frame(self):
        """Bucle principal para la cámara y el envío de datos."""
        ret, frame = self.cap.read()
        if ret:
            # Invertir el frame para el efecto espejo de la webcam
            frame = cv2.flip(frame, 1)
            
            # 1. Procesar y enviar el frame al servidor (solo si está conectado)
            if self.ws_connected:
                # Codificar el frame a JPEG o PNG y luego a Base64
                _, buffer = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50]) # Compresión para velocidad
                base64_data = base64.b64encode(buffer)
                
                # Enviar de forma asíncrona
                asyncio.run_coroutine_threadsafe(self.send_frame_data(base64_data), asyncio.get_event_loop())

            # 2. Actualizar la UI (Dibujar en Tkinter)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        # Repetir el bucle con un retraso (controla los FPS de la UI y el envío)
        self.master.after(FRAME_INTERVAL_MS, self.update_frame)

def main():
    # Configurar el bucle de eventos de asyncio para Tkinter
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    def run_tk(root, loop):
        """Corre el bucle de Tkinter y el bucle de eventos de asyncio en paralelo."""
        try:
            root.mainloop()
        finally:
            loop.stop()

    def run_asyncio(loop):
        """Corre el bucle de eventos de asyncio."""
        loop.run_forever()

    root = tk.Tk()
    app = GameClientApp(root)

    # Configurar el cierre de la ventana
    def on_closing():
        if app.cap.isOpened():
            app.cap.release()
        if app.websocket:
            asyncio.run_coroutine_threadsafe(app.websocket.close(), loop)
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Ejecutar Tkinter en el hilo principal y asyncio en un hilo separado
    import threading
    threading.Thread(target=lambda: run_asyncio(loop), daemon=True).start()
    run_tk(root, loop)

if __name__ == "__main__":
    main()