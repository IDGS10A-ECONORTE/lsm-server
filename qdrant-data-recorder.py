import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import json
import os
import time # Necesario para el delay

# --- CONFIGURACIÓN GLOBAL ---
CAMERA_INDEX = 1
VECTOR_DIMENSION = 60 # 20 landmarks * 3 ejes (X, Y, Z)
OUTPUT_FILENAME = "lsm_dictionary_data2.txt"
CAPTURE_DELAY_SECONDS = 3 # Retardo de 3 segundos antes de la captura

# Inicialización de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class OfflineSignRecorderApp:
    def __init__(self, master):
        self.master = master
        master.title("LSM Data Recorder")
        
        # Inicializar MediaPipe y Cámara
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7)
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.last_results = None 
        self.point_counter = 0
        self.capturing = False # Bandera para controlar el proceso de captura

        # --- CONFIGURACIÓN DE LA INTERFAZ ---
        self.master.geometry("750x650") # Ajustamos un poco la geometría
        
        # 1. Marco para el Video
        self.video_frame = tk.Frame(master)
        self.video_frame.pack(padx=10, pady=10)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # 2. Marco para Controles
        self.control_frame = tk.Frame(master)
        self.control_frame.pack(pady=10)
        
        # Entrada de Nombre de Seña
        tk.Label(self.control_frame, text="Nombre de la Seña:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.sign_name_entry = tk.Entry(self.control_frame, width=20)
        self.sign_name_entry.grid(row=0, column=1, padx=5, pady=5)

        # Selección de Dificultad
        tk.Label(self.control_frame, text="Dificultad:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.difficulty_var = tk.StringVar(master)
        self.difficulty_var.set("FÁCIL")
        self.difficulty_options = ["FÁCIL", "MEDIO", "DIFÍCIL"]
        self.difficulty_menu = tk.OptionMenu(self.control_frame, self.difficulty_var, *self.difficulty_options)
        self.difficulty_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # Botón de Guardar
        self.save_button = tk.Button(self.control_frame, text="INICIAR CAPTURA (3s)", command=self.start_countdown, bg="#008080", fg="white", font=("Arial", 10, "bold"))
        self.save_button.grid(row=2, column=0, columnspan=2, pady=15)
        
        # Etiqueta para la cuenta regresiva y estado
        self.countdown_label = tk.Label(master, text="", fg="orange", font=("Arial", 12, "bold"))
        self.countdown_label.pack()

        # Contador de Puntos
        self.status_label = tk.Label(master, text=f"Archivo: {OUTPUT_FILENAME} | Puntos: 0", fg="blue")
        self.status_label.pack()
        
        self._load_initial_count()
        self.update_frame()

    def _load_initial_count(self):
        """Intenta contar los puntos existentes en el archivo de salida."""
        try:
            if os.path.exists(OUTPUT_FILENAME):
                with open(OUTPUT_FILENAME, 'r') as f:
                    self.point_counter = sum(1 for line in f if line.strip())
            self.status_label.config(text=f"Archivo: {OUTPUT_FILENAME} | Puntos: {self.point_counter}", fg="blue")
        except Exception:
            self.status_label.config(text="Error leyendo el archivo.", fg="red")

    # --- LÓGICA DE NORMALIZACIÓN (60D) ---
    def get_normalized_60d_vector(self, landmark_list, handedness_label):
        """
        Calcula el vector normalizado de 60 dimensiones (X, Y, Z de 20 puntos)
        con centrado, escalado y mirroring.
        """
        landmarks = landmark_list.landmark
        
        # 1. Centrado (Traslación): Muñeca (L0)
        wrist = landmarks[0]
        relative_landmarks = []
        for lm in landmarks:
            relative_landmarks.append((lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z))

        # 2. Escalado (Tamaño): Distancia L0 a L9
        l9 = relative_landmarks[9]
        d_ref = np.sqrt(l9[0]**2 + l9[1]**2 + l9[2]**2)
        
        if d_ref < 1e-6: 
            return None 
            
        final_vector = []
        
        # 3. Escalado y Reflexión (Mirroring)
        # Iteramos de L1 a L20 (excluyendo L0)
        for i in range(1, 21): 
            x_prime, y_prime, z_prime = relative_landmarks[i]

            x_scaled = x_prime / d_ref
            y_scaled = y_prime / d_ref
            z_scaled = z_prime / d_ref
            
            # Aplicar Mirroring (Reflexión en X para mano izquierda)
            if handedness_label == 'Left':
                x_scaled *= -1
                
            final_vector.extend([x_scaled, y_scaled, z_scaled])

        return final_vector

    # --- LÓGICA DE CUENTA REGRESIVA Y GRABACIÓN ---
    def start_countdown(self):
        """
        Inicia la cuenta regresiva de 3 segundos antes de la captura.
        """
        if self.capturing: # Evitar múltiples inicios de captura
            return

        sign_name = self.sign_name_entry.get().strip().upper()
        if not sign_name:
            messagebox.showwarning("Advertencia", "Por favor, ingresa un nombre para la seña.")
            return
            
        self.capturing = True
        self.save_button.config(state=tk.DISABLED, text="Preparando...") # Deshabilitar botón
        self.countdown(CAPTURE_DELAY_SECONDS)

    def countdown(self, remaining):
        """
        Actualiza la etiqueta de cuenta regresiva.
        """
        if remaining > 0:
            self.countdown_label.config(text=f"Capturando en {remaining}...", fg="orange")
            self.master.after(1000, self.countdown, remaining - 1)
        else:
            self.countdown_label.config(text="¡CAPTURA!", fg="green")
            self.master.after(500, self.record_and_save_action) # Pequeño delay para ver "¡CAPTURA!"
            self.master.after(1000, lambda: self.countdown_label.config(text="", fg="orange")) # Limpiar después

    def record_and_save_action(self):
        """
        Ejecuta la acción de guardar después del delay.
        """
        sign_name = self.sign_name_entry.get().strip().upper()
        difficulty = self.difficulty_var.get().upper()
        
        if not self.last_results or not self.last_results.multi_hand_landmarks:
            messagebox.showwarning("Advertencia", "No se detectó ninguna mano. Intenta acercar la mano a la cámara.")
            self.reset_capture_state()
            return

        hand_landmarks = self.last_results.multi_hand_landmarks[0]
        handedness = self.last_results.multi_handedness[0].classification[0].label
        
        vector_60d = self.get_normalized_60d_vector(hand_landmarks, handedness)

        if vector_60d and len(vector_60d) == VECTOR_DIMENSION:
            data_record = {
                "id": self.point_counter, 
                "sign_name": sign_name,
                "difficulty": difficulty,
                "vector": vector_60d,
            }
            
            try:
                with open(OUTPUT_FILENAME, 'a') as f:
                    f.write(json.dumps(data_record) + "\n")
                
                self.point_counter += 1
                self.status_label.config(text=f"✅ '{sign_name}' ({difficulty}) GUARDADA. Puntos: {self.point_counter}", fg="green")
                self.sign_name_entry.delete(0, tk.END)
                
            except Exception as e:
                messagebox.showerror("Error de Archivo", f"Fallo al escribir en el archivo {OUTPUT_FILENAME}: {e}")
        else:
            self.status_label.config(text="❌ Error de procesamiento del vector (Dimensión incorrecta o D_ref 0).", fg="red")
        
        self.reset_capture_state()

    def reset_capture_state(self):
        """Restablece el estado de captura y habilita el botón."""
        self.capturing = False
        self.save_button.config(state=tk.NORMAL, text=f"INICIAR CAPTURA ({CAPTURE_DELAY_SECONDS}s)")


    # --- LÓGICA DE CÁMARA Y VIDEO ---
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            self.last_results = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            if self.last_results.multi_hand_landmarks:
                for hand_landmarks in self.last_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            img = img.resize((640, 480), Image.Resampling.LANCZOS) 
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        self.master.after(10, self.update_frame)


def main():
    root = tk.Tk()
    app = OfflineSignRecorderApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [app.cap.release(), root.destroy()])
    root.mainloop()

if __name__ == "__main__":
    main()