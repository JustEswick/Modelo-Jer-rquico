import torch
import numpy as np
import json
import matplotlib.pyplot as plt

# Importamos las clases necesarias de nuestros otros archivos
from model import HierarchicalDrawingModel
from dataset import StrokesDataset, collate_fn # Necesitamos el collate_fn para procesar el dato
from torch.utils.data import DataLoader

# --- CONFIGURACIÓN ---
MODEL_PATH = 'hierarchical_model.pth' # El archivo de pesos que generó train.py
JSON_FILE = 'pikachu_hierarchical.json'
OUTPUT_IMAGE = 'pikachu_final_reconstruction.png'

def plot_reconstruction(model, dataloader, device):
    """
    Carga datos, genera una reconstrucción con el modelo entrenado y la grafica.
    Utiliza la lógica de graficado CORREGIDA.
    """
    model.eval()
    with torch.no_grad():
        # Extraer el factor de escala del dataset
        scale_factor = dataloader.dataset.scale_factor
        
        # Obtener el único lote de datos
        data, mask = next(iter(dataloader))
        data, mask = data.to(device), mask.to(device)

        # Generar la reconstrucción final
        reconstruction = model(data, mask)

        # Mover todo a CPU y NumPy para procesar y graficar
        original_deltas_padded = data.squeeze(0).cpu().numpy()
        recon_deltas_padded = reconstruction.squeeze(0).cpu().numpy()
        mask_np = mask.squeeze(0).cpu().numpy()
        
        # Aplanar todos los deltas reales en una única secuencia
        flat_original_deltas = np.vstack([original_deltas_padded[i, :int(mask_np[i].sum())] for i in range(mask_np.shape[0])])
        flat_recon_deltas = np.vstack([recon_deltas_padded[i, :int(mask_np[i].sum())] for i in range(mask_np.shape[0])])

        # Convertir la secuencia completa de deltas a coordenadas absolutas
        original_coords = np.cumsum(flat_original_deltas[:, :2] * scale_factor, axis=0)
        recon_coords = np.cumsum(flat_recon_deltas[:, :2] * scale_factor, axis=0)
        
        # Graficar el resultado final
        print("Generando gráfico final...")
        plt.figure(figsize=(10, 10))
        plt.plot(original_coords[:, 0], -original_coords[:, 1], 'b-', linewidth=2, label='Original')
        plt.plot(recon_coords[:, 0], -recon_coords[:, 1], 'r-', linewidth=2, label='Reconstrucción del Modelo')
        plt.title('Reconstrucción Final del Modelo Entrenado')
        plt.legend()
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(OUTPUT_IMAGE, dpi=300)
        plt.close()
        print(f"¡Gráfico guardado en {OUTPUT_IMAGE}!")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instanciar la arquitectura del modelo (vacía por ahora)
    model = HierarchicalDrawingModel().to(device)
    
    # 2. Cargar los pesos entrenados desde el archivo .pth
    print(f"Cargando pesos del modelo desde {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # 3. Cargar los datos originales para dárselos al modelo
    dataset = StrokesDataset(JSON_FILE)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    # 4. Llamar a la función para generar y graficar la reconstrucción
    plot_reconstruction(model, dataloader, device)