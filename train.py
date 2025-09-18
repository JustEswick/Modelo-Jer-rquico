import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Importamos nuestras clases personalizadas
from dataset import StrokesDataset, collate_fn
from model import HierarchicalDrawingModel

# --- HIPERPARÁMETROS Y CONFIGURACIÓN ---
JSON_FILE = 'pikachu_hierarchical.json'
MODEL_SAVE_PATH = 'hierarchical_model.pth'
FRAMES_DIR = 'frames'
NUM_EPOCHS = 10000
LEARNING_RATE = 0.0001
PLOT_EVERY = 500 # Guardar una imagen del progreso cada 500 épocas

def deltas_to_coordinates(delta_strokes, scale_factor):
    """Convierte una lista de trazos en formato delta a coordenadas absolutas para dibujar."""
    all_coords = []
    start_point = np.array([0, 0])
    for stroke in delta_strokes:
        # Des-normalizar
        stroke[:, :2] *= scale_factor
        # Calcular suma acumulada para obtener coordenadas absolutas
        coords = np.cumsum(stroke[:, :2], axis=0) + start_point
        all_coords.append(coords)
        start_point = coords[-1] # El siguiente trazo empieza donde terminó este
    return all_coords

def visualize_reconstruction(model, dataloader, device, epoch, scale_factor):
    """
    VERSIÓN NUEVA: Incluye re-centrado post-proceso para eliminar el drift.
    """
    model.eval() # Modo evaluación
    with torch.no_grad():
        for data, mask in dataloader:
            data, mask = data.to(device), mask.to(device)
            reconstruction = model(data, mask)

            # --- PREPARACIÓN DE DATOS ---
            # Movemos todo a CPU y convertimos a numpy
            original_deltas_padded = data.squeeze(0).cpu().numpy()
            recon_deltas_padded = reconstruction.squeeze(0).cpu().numpy()
            mask_np = mask.squeeze(0).cpu().numpy()

            flat_original_deltas = np.vstack([original_deltas_padded[i, :int(mask_np[i].sum())] for i in range(mask_np.shape[0])])
            flat_recon_deltas = np.vstack([recon_deltas_padded[i, :int(mask_np[i].sum())] for i in range(mask_np.shape[0])])

            original_coords = np.cumsum(flat_original_deltas[:, :2] * scale_factor, axis=0)
            recon_coords = np.cumsum(flat_recon_deltas[:, :2] * scale_factor, axis=0)

            # --- PASO DE ESTABILIZACIÓN: RE-CENTRAR AMBOS DIBUJOS ANTES DE PLOTEAR ---
            # Se calcula el centro de cada dibujo y se le resta para forzarlo al origen.
            original_coords -= np.mean(original_coords, axis=0)
            recon_coords -= np.mean(recon_coords, axis=0)
            
            # --- PLOTEO ---
            plt.figure(figsize=(8, 8))
            plt.plot(original_coords[:, 0], -original_coords[:, 1], 'b-', alpha=0.7, label='Original')
            plt.plot(recon_coords[:, 0], -recon_coords[:, 1], 'r-', alpha=0.7, label='Reconstrucción')
            plt.title(f'Época {epoch}')
            plt.legend()
            plt.axis('equal')
            plt.xticks([])
            plt.yticks([])
            
            if not os.path.exists(FRAMES_DIR):
                os.makedirs(FRAMES_DIR)
            plt.savefig(f'{FRAMES_DIR}/epoch_{epoch:06d}.png')
            plt.close()
    
    model.train() # Volver al modo de entrenamiento
if __name__ == "__main__":
    # --- 1. CONFIGURACIÓN INICIAL ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # --- 2. DATOS ---
    dataset = StrokesDataset(JSON_FILE)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # --- 3. MODELO, CRITERIO Y OPTIMIZADOR ---
    model = HierarchicalDrawingModel().to(device)
    criterion = nn.MSELoss() # Error Cuadrático Medio, bueno para regresión de coordenadas
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("--- ¡Comenzando el Entrenamiento! ---")
    model.train() # Ponemos el modelo en modo de entrenamiento

    # --- 4. BUCLE DE ENTRENAMIENTO ---
    for epoch in range(NUM_EPOCHS):
        # Usamos tqdm para una barra de progreso
        pbar = tqdm(dataloader, desc=f"Época {epoch+1}/{NUM_EPOCHS}")
        
        for data, mask in pbar:
            data, mask = data.to(device), mask.to(device)
            
            # a. Poner gradientes a cero
            optimizer.zero_grad()
            
            # b. Forward pass (el modelo hace su predicción)
            reconstructed_data = model(data, mask)
            
            # c. Calcular la pérdida (solo en los puntos reales, no en el padding)
            # Expandimos la máscara para que tenga la misma última dimensión que los datos
            mask_expanded = mask.unsqueeze(-1)
            loss = criterion(reconstructed_data * mask_expanded, data * mask_expanded)
            
            # d. Backward pass (PyTorch calcula cómo ajustar los pesos)
            loss.backward()
            
            # e. Optimizer step (se actualizan los pesos)
            optimizer.step()
            
            # Actualizamos la barra de progreso con la pérdida actual
            pbar.set_postfix(loss=f'{loss.item():.6f}')

        # --- 5. VISUALIZACIÓN PERIÓDICA ---
        if (epoch + 1) % PLOT_EVERY == 0:
            print(f"\nGenerando visualización para la época {epoch+1}...")
            visualize_reconstruction(model, dataloader, device, epoch + 1, dataset.scale_factor)

    print("\n--- ¡Entrenamiento Finalizado! ---")
    
    # --- 6. GUARDAR EL MODELO ENTRENADO ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Modelo guardado en {MODEL_SAVE_PATH}")