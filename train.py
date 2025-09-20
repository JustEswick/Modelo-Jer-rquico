# ==============================================================================
# SECCIÓN DE IMPORTACIONES
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR # Planificador para la tasa de aprendizaje
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm # Para las barras de progreso
from pathlib import Path # Para manejar rutas de archivos de forma moderna

# Importamos nuestras clases personalizadas de los otros archivos
from dataset import StrokesDataset, collate_fn
from model import HierarchicalDrawingModel

# ==============================================================================
# HIPERPARÁMETROS Y CONFIGURACIÓN GLOBAL
# ==============================================================================
# --- Rutas de Archivos ---
JSON_FILE = 'pikachu_hierarchical.json'
MODEL_SAVE_PATH = 'hierarchical_model.pth'
FRAMES_DIR = 'frames'

# --- Parámetros de Entrenamiento ---
NUM_EPOCHS = 10000
LEARNING_RATE = 0.0015 # Un valor inicial robusto
BATCH_SIZE = 1 # Nuestro dataset solo tiene 1 dibujo, así que el lote es de 1

# --- Parámetros de Visualización ---
PLOT_EVERY = 50 # Guardar una imagen del progreso cada 50 épocas

# --- Pesos de la Función de Pérdida ---
# Aumenta la importancia de aprender a levantar el lápiz (p)
PEN_STATE_LOSS_WEIGHT = 5.0 
# Controla la fuerza de la penalización por desplazamiento (drift)
CENTROID_LOSS_WEIGHT = 0.01

# ==============================================================================
# CONFIGURACIÓN DEL DIRECTORIO DE SALIDA
# ==============================================================================
# Aseguramos que el directorio para guardar los frames exista antes de empezar.
# `pathlib` es una forma moderna y segura de manejar rutas de archivos.
frame_dir = Path.cwd() / FRAMES_DIR
frame_dir.mkdir(exist_ok=True) # `exist_ok=True` evita un error si la carpeta ya existe.
print(f"Directorio de frames asegurado: {frame_dir}")


# ==============================================================================
# FUNCIÓN DE VISUALIZACIÓN
# ==============================================================================
def visualize_reconstruction(model: nn.Module, dataloader: DataLoader, device: torch.device, epoch: int, scale_factor: float):
    """
    Genera y guarda una imagen que compara el dibujo original con la reconstrucción del modelo.
    Incluye un paso de re-centrado para una visualización estable.
    """
    model.eval() # Ponemos el modelo en modo de evaluación (desactiva dropout, etc.)
    with torch.no_grad(): # Desactivamos el cálculo de gradientes para esta sección, es más rápido.
        
        # Obtenemos el único lote de datos de nuestro dataloader
        for data, mask in dataloader:
            data, mask = data.to(device), mask.to(device)
            
            # El modelo genera su reconstrucción
            reconstruction = model(data, mask)

            # --- PREPARACIÓN DE DATOS PARA GRAFICAR ---
            # Movemos los tensores a la CPU y los convertimos a arrays de NumPy
            original_deltas_padded = data.squeeze(0).cpu().numpy()
            recon_deltas_padded = reconstruction.squeeze(0).cpu().numpy()
            mask_np = mask.squeeze(0).cpu().numpy()

            # Usamos la máscara para filtrar solo los puntos reales, eliminando el padding
            flat_original_deltas = np.vstack([original_deltas_padded[i, :int(mask_np[i].sum())] for i in range(mask_np.shape[0])])
            flat_recon_deltas = np.vstack([recon_deltas_padded[i, :int(mask_np[i].sum())] for i in range(mask_np.shape[0])])

            # Convertimos deltas (movimientos relativos) a coordenadas absolutas con cumsum
            original_coords = np.cumsum(flat_original_deltas[:, :2] * scale_factor, axis=0)
            recon_coords = np.cumsum(flat_recon_deltas[:, :2] * scale_factor, axis=0)

            # --- PASO DE ESTABILIZACIÓN ---
            # Re-centramos ambos dibujos en el origen (0,0) para una comparación justa y estable.
            original_coords -= np.mean(original_coords, axis=0)
            recon_coords -= np.mean(recon_coords, axis=0)
            
            # --- PLOTEO CON MATPLOTLIB ---
            plt.figure(figsize=(8, 8))
            plt.plot(original_coords[:, 0], -original_coords[:, 1], 'b-', alpha=0.7, label='Original') # Original en azul
            plt.plot(recon_coords[:, 0], -recon_coords[:, 1], 'r-', alpha=0.7, label='Reconstrucción') # Reconstrucción en rojo
            plt.title(f'Época {epoch}')
            plt.legend()
            plt.axis('equal') # Asegura que la escala de los ejes X e Y sea la misma
            plt.xticks([]) # Oculta los números de los ejes
            plt.yticks([])
            
            plt.savefig(f'{FRAMES_DIR}/epoch_{epoch:06d}.png') # Guarda la imagen
            plt.close() # Cierra la figura para liberar memoria
    
    model.train() # IMPORTANTE: Volvemos a poner el modelo en modo de entrenamiento

# ==============================================================================
# FUNCIÓN DE PÉRDIDA DE CENTROIDE
# ==============================================================================
def calculate_centroid_loss(deltas_padded: torch.Tensor, mask: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """
    Calcula la pérdida por desplazamiento del centroide.
    Esta función usa operaciones de PyTorch para asegurar que los gradientes fluyan.
    """
    # Des-normalizamos los deltas para trabajar en el espacio de coordenadas real
    deltas_unscaled = deltas_padded[:, :, :, :2] * scale_factor
    
    # Aplanamos los deltas reales usando la máscara
    all_coords = []
    for i in range(deltas_unscaled.shape[0]): # Iteramos sobre el lote (aunque solo haya 1)
        flat_deltas = torch.cat([
            deltas_unscaled[i, j, :int(mask[i, j].sum()), :] 
            for j in range(mask.shape[1])
        ])
        
        # Calculamos coordenadas absolutas con cumsum de PyTorch
        coords = torch.cumsum(flat_deltas, dim=0)
        all_coords.append(coords)

    # Calculamos el centroide (punto medio) de todas las coordenadas
    centroid = torch.mean(all_coords[0], dim=0)
    
    # La pérdida es la distancia al cuadrado del centroide al origen.
    # El objetivo es que este valor sea lo más cercano a cero posible.
    loss = torch.sum(centroid.pow(2))
    return loss

# ==============================================================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ==============================================================================
if __name__ == "__main__":
    # --- 1. CONFIGURACIÓN INICIAL ---
    # Selecciona el dispositivo (GPU si está disponible, si no, CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- 2. PREPARACIÓN DE DATOS ---
    # Creamos una instancia de nuestro Dataset personalizado
    dataset = StrokesDataset(JSON_FILE)
    # El DataLoader se encarga de preparar los lotes de datos usando nuestro collate_fn
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # --- 3. INICIALIZACIÓN DEL MODELO Y HERRAMIENTAS DE ENTRENAMIENTO ---
    # Creamos una instancia de nuestro modelo y la movemos al dispositivo seleccionado
    model = HierarchicalDrawingModel().to(device)
    # Definimos la función de pérdida para la reconstrucción
    reconstruction_criterion = nn.MSELoss()
    # Definimos el optimizador que ajustará los pesos del modelo
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # weight_decay es una regularización para evitar overfitting
    # Definimos el scheduler para ajustar dinámicamente la tasa de aprendizaje
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.5) # Decaimiento del LR cada 5000 épocas.
    
    print("--- ¡Comenzando el Entrenamiento! ---")
    model.train() # Ponemos el modelo en modo de entrenamiento

    # --- 4. BUCLE PRINCIPAL DE ENTRENAMIENTO ---
    for epoch in range(NUM_EPOCHS):
        # Creamos una barra de progreso para la época actual
        pbar = tqdm(dataloader, desc=f"Época {epoch+1}/{NUM_EPOCHS}")
        
        for data, mask in pbar:
            # Mover los datos del lote al dispositivo
            data, mask = data.to(device), mask.to(device)
            
            # 1. Limpiar gradientes: Reinicia los gradientes del paso anterior.
            optimizer.zero_grad()
            
            # 2. Paso Adelante (Forward Pass): El modelo genera su predicción.
            reconstructed_data = model(data, mask)
            
            # --- 3. CÁLCULO DE LA PÉRDIDA TOTAL ---
            mask_expanded = mask.unsqueeze(-1) # Preparamos la máscara para la pérdida
            
            # 3a. Pérdida de las coordenadas (dx, dy)
            loss_coords = reconstruction_criterion(
                reconstructed_data[:, :, :, :2] * mask_expanded, 
                data[:, :, :, :2] * mask_expanded
            )
            # 3b. Pérdida del estado del lápiz (p)
            loss_pen_state = reconstruction_criterion(
                reconstructed_data[:, :, :, 2:] * mask_expanded,
                data[:, :, :, 2:] * mask_expanded
            )
            # 3c. Pérdida del desplazamiento del centroide (drift)
            loss_drift = calculate_centroid_loss(reconstructed_data, mask, dataset.scale_factor)
            
            # 3d. Combinamos todas las pérdidas en una sola, aplicando los pesos
            total_loss = loss_coords + (PEN_STATE_LOSS_WEIGHT * loss_pen_state) + (CENTROID_LOSS_WEIGHT * loss_drift)
            
            # 4. Paso Atrás (Backward Pass): PyTorch calcula los gradientes (backpropagation).
            total_loss.backward()
            
            # 5. Recorte de Gradientes: Previene que los gradientes se vuelvan demasiado grandes.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 6. Actualización de Pesos: El optimizador ajusta el modelo usando los gradientes.
            optimizer.step()
            
            # Actualizamos la información en la barra de progreso
            pbar.set_postfix(
                loss=f'{total_loss.item():.6f}', 
                coords=f'{loss_coords.item():.6f}', 
                pen=f'{loss_pen_state.item():.6f}',
                drift=f'{loss_drift.item():.6f}'
            )

        # 7. Actualización del Scheduler: Se llama una vez por época.
        scheduler.step()

        # Generamos una imagen del progreso en los intervalos definidos
        if (epoch + 1) % PLOT_EVERY == 0:
            visualize_reconstruction(model, dataloader, device, epoch + 1, dataset.scale_factor)

    # --- 5. FIN DEL ENTRENAMIENTO ---
    print("\n--- ¡Entrenamiento Finalizado! ---")
    # Guardamos los pesos aprendidos por el modelo en un archivo
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Modelo guardado en {MODEL_SAVE_PATH}")