import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import websockets
import asyncio
import json
import base64
from PIL import Image, ImageTk
import threading

# --- CONFIGURACIÓN DEL CLIENTE ---
WEBSOCKET_URL = "ws://localhost:7777"
# Usar 0 si es la cámara por defecto, o el índice correcto
CAMERA_INDEX = 0
FPS_LIMIT = 15 
FRAME_INTERVAL_MS = int(1000 / FPS_LIMIT)

# Lista de señas de prueba (simula los sign_name de tu colección Qdrant)
TARGET_SIGNS_LIST = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "L", "M", "N", "O", "R", "S", "T", "U", "V", "W" ,"Y"] 

# Usamos una variable global para el loop de asyncio que se ejecuta en el hilo secundario
GLOBAL_ASYNC_LOOP = None 

class GameClientApp:
    def __init__(self, master):
        self.master = master
        master.title("LSM Game Client (Simulación Godot)")
        
        # Estado de conexión
        self.ws_connected = False
        self.websocket = None
        
        # Estado del índice de seña
        self.sign_index = -1 
        
        # Cámara
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
             messagebox.showerror("Error de Cámara", "No se pudo abrir la cámara.")
             master.destroy()
             return
        
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

        # 5. Botón de Control (Avance de Índice)
        self.target_button = tk.Button(master, text="INICIAR (Siguiente Seña)", command=self.advance_sign_index, bg="#4CAF50", fg="white", font=("Arial", 12))
        self.target_button.pack(pady=10)
        
        # Iniciar la conexión WebSocket
        self.master.after(100, self.start_websocket)
        # Iniciar el bucle de la cámara
        self.update_frame()

    def advance_sign_index(self):
        """Avanza al siguiente índice de seña y envía el nuevo SET_TARGET."""
        
        self.sign_index = (self.sign_index + 1) % len(TARGET_SIGNS_LIST)
        next_sign = TARGET_SIGNS_LIST[self.sign_index]
        
        self.target_button.config(text=f"Siguiente Seña: {next_sign} (Índice: {self.sign_index})")
        
        # Enviar el comando al servidor
        self.send_command('SET_TARGET', next_sign)

    # ---------------------------------
    # --- CONEXIÓN Y RECEPCIÓN WS ---
    # ---------------------------------

    def start_websocket(self):
        """Inicia el cliente WebSocket de forma asíncrona usando el loop global."""
        # Se asegura que la corutina se ejecute en el hilo secundario (ASYNC_LOOP)
        if GLOBAL_ASYNC_LOOP and not GLOBAL_ASYNC_LOOP.is_running():
             # Si el loop no está corriendo, intentamos iniciarlo
             threading.Thread(target=self._run_connect, daemon=True).start()
        elif GLOBAL_ASYNC_LOOP:
             # Si el loop ya está corriendo, programamos la conexión
             asyncio.run_coroutine_threadsafe(self.connect_and_receive(), GLOBAL_ASYNC_LOOP)

    def _run_connect(self):
        """Función síncrona helper para ejecutar connect_and_receive en el hilo."""
        global GLOBAL_ASYNC_LOOP
        if GLOBAL_ASYNC_LOOP and GLOBAL_ASYNC_LOOP.is_running():
            asyncio.run_coroutine_threadsafe(self.connect_and_receive(), GLOBAL_ASYNC_LOOP)
        else:
             # Si el hilo no está listo, intentamos de nuevo en un momento
             self.master.after(500, self.start_websocket)


    async def connect_and_receive(self):
        """Maneja la conexión y el bucle de recepción de datos."""
        
        # Bucle para reintentar la conexión de manera persistente
        while True:
            try:
                # Intentar conectar
                self.websocket = await websockets.connect(WEBSOCKET_URL)
                self.ws_connected = True
                self.master.after(0, lambda: self.conn_label.config(text=f"Conectado a {WEBSOCKET_URL}", fg="green"))
                print(f"WS Cliente: Conexión exitosa a {WEBSOCKET_URL}")
                
                # Al conectar, solicitar el primer objetivo
                self.advance_sign_index() 

                # Bucle de Recepción: Escuchar continuamente las respuestas
                while self.ws_connected:
                    message = await self.websocket.recv()
                    # Ejecutar la actualización de la UI en el hilo principal de Tkinter
                    self.master.after(0, lambda m=message: self.process_server_response(m))

            except ConnectionRefusedError:
                self.master.after(0, lambda: self.conn_label.config(text="ERROR: Conexión rechazada (Servidor inactivo)", fg="red"))
                await asyncio.sleep(3) # Esperar antes de reintentar
            except Exception as e:
                self.master.after(0, lambda: self.conn_label.config(text=f"ERROR de conexión WS: {e}", fg="red"))
                print(f"WS Cliente: Error en bucle de recepción: {e}")
                await asyncio.sleep(3) # Esperar antes de reintentar
            finally:
                self.ws_connected = False
                self.websocket = None


    def process_server_response(self, message):
        """Actualiza la interfaz con la respuesta del lsm_server."""
        try:
            response = json.loads(message)

            # Caso: el servidor asignó meta o cambió target
            if response.get("status") in ["NEW_TARGET", "TARGET_SET", "TARGET_ASSIGNED"]:
                target = response.get("target", '(Desconocido)')
                self.target_label.config(text=f"Objetivo Actual: {target}", fg="black")
                self.result_label.config(text="¡Haz la Seña!", fg="blue")

            # Caso: el servidor respondió con validación
            elif "result" in response:
                is_correct = response["result"]
                feedback = response.get("feedback", "Procesando...")
                target = response.get('target', '(Error)')

                if is_correct:
                    self.result_label.config(
                        text=f"✅ ¡CORRECTO! ({feedback})",
                        fg="green"
                    )
                else:
                    self.result_label.config(
                        text=f"❌ INCORRECTO: {feedback}",
                        fg="red"
                    )

            else:
                print("Respuesta desconocida:", response)

        except Exception as e:
            print("Error en process_server_response:", e)


    # ---------------------------------
    # --- ENVÍO DE DATOS WS ---
    # ---------------------------------

    def send_command(self, command, sign):
        """Función síncrona para enviar comandos desde la UI (Botón)."""
        # Usamos el loop global
        if self.ws_connected and GLOBAL_ASYNC_LOOP:
            asyncio.run_coroutine_threadsafe(self.send_command_async(command, sign), GLOBAL_ASYNC_LOOP)
        else:
            self.conn_label.config(text="¡Desconectado! Intentando reconexión automática...", fg="red")

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
            frame = cv2.flip(frame, 1)
            
            if self.ws_connected and GLOBAL_ASYNC_LOOP:
                # Ejecutar el envío de datos en el loop asíncrono
                _, buffer = cv2.imencode('.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50]) 
                base64_data = base64.b64encode(buffer)
                
                asyncio.run_coroutine_threadsafe(self.send_frame_data(base64_data), GLOBAL_ASYNC_LOOP)

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        self.master.after(FRAME_INTERVAL_MS, self.update_frame)

def main():
    global GLOBAL_ASYNC_LOOP

    # 1. Configurar el bucle de eventos de asyncio para el hilo secundario
    try:
        GLOBAL_ASYNC_LOOP = asyncio.get_running_loop()
    except RuntimeError:
        GLOBAL_ASYNC_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(GLOBAL_ASYNC_LOOP)

    def run_asyncio(loop):
        """Corre el bucle de eventos de asyncio en el hilo secundario."""
        print("ASYNCIO: Loop iniciado en hilo secundario.")
        loop.run_forever()

    # 2. Iniciar el loop de asyncio en un hilo separado
    threading.Thread(target=lambda: run_asyncio(GLOBAL_ASYNC_LOOP), daemon=True).start()
    
    # 3. Iniciar la aplicación Tkinter en el hilo principal
    root = tk.Tk()
    app = GameClientApp(root)

    # 4. Configurar el cierre de la ventana
    def on_closing():
        if app.cap.isOpened():
            app.cap.release()
        if app.websocket:
            # Cerrar el websocket de forma asíncrona y luego detener el loop
            try:
                future = asyncio.run_coroutine_threadsafe(app.websocket.close(), GLOBAL_ASYNC_LOOP)
                future.result(timeout=1)
            except:
                pass
        
        # Detener el loop de asyncio y la aplicación Tkinter
        GLOBAL_ASYNC_LOOP.call_soon_threadsafe(GLOBAL_ASYNC_LOOP.stop)
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()

if __name__ == "__main__":
    main()