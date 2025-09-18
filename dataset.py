import json
import torch
from torch.utils.data import Dataset, DataLoader

class StrokesDataset(Dataset):
    """
    Dataset para cargar los datos de trazos jerárquicos desde un archivo JSON.
    """
    def __init__(self, json_file):
        print(f"Cargando datos desde {json_file}...")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.scale_factor = data['scale_factor']
        # Convertimos cada trazo a un tensor de PyTorch
        self.strokes = [torch.tensor(s, dtype=torch.float32) for s in data['strokes']]
        print("Datos cargados exitosamente.")

    def __len__(self):
        # Tenemos un único dibujo en nuestro dataset
        return 1

    def __getitem__(self, idx):
        # Ignoramos el índice y siempre devolvemos el dibujo completo
        return self.strokes

def collate_fn(batch):
    """
    Función de empaquetado (collate) que maneja el padding.
    'batch' es una lista de muestras. En nuestro caso, será una lista con un solo
    elemento: la lista completa de trazos de nuestro dibujo.
    Ej: [ [stroke1_tensor, stroke2_tensor, ...] ]
    """
    # 1. Extraer la lista de trazos del lote (batch)
    all_strokes_list = batch[0]
    num_strokes = len(all_strokes_list)

    # 2. Encontrar la longitud del trazo más largo para el padding
    max_len = max(s.shape[0] for s in all_strokes_list)
    
    # 3. Crear tensores de ceros para los datos y la máscara
    # El tensor de datos tendrá la forma (num_trazos, max_puntos_por_trazo, 3)
    padded_data = torch.zeros(num_strokes, max_len, 3)
    # La máscara nos dirá qué partes son datos reales (1) vs padding (0)
    mask = torch.zeros(num_strokes, max_len)

    # 4. Rellenar los tensores con nuestros datos
    for i, stroke in enumerate(all_strokes_list):
        seq_len = stroke.shape[0]
        padded_data[i, :seq_len, :] = stroke
        mask[i, :seq_len] = 1.0
        
    # Añadimos una dimensión de lote (batch dimension) al principio
    # La forma final será (1, num_trazos, max_puntos_por_trazo, 3)
    return padded_data.unsqueeze(0), mask.unsqueeze(0)


if __name__ == "__main__":
    JSON_FILE = 'pikachu_hierarchical.json'
    
    # --- DEMOSTRACIÓN DE USO ---
    
    # 1. Crear una instancia del Dataset
    dataset = StrokesDataset(json_file=JSON_FILE)
    
    # 2. Crear una instancia del DataLoader
    # Usamos batch_size=1 porque nuestro dataset solo tiene un dibujo.
    # El collate_fn se encargará de procesar la lista de trazos de ese dibujo.
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, # No tiene sentido barajar un solo elemento
        collate_fn=collate_fn
    )
    
    # 3. Extraer el primer (y único) lote de datos
    # El bucle se ejecutará solo una vez
    for batch_data, batch_mask in dataloader:
        print("\n--- ¡Lote de datos procesado! ---")
        print(f"Forma del tensor de datos: {batch_data.shape}")
        print(f"Forma del tensor de máscara: {batch_mask.shape}")
        
        # Verifiquemos el contenido de la máscara para un trazo
        # (ej. el primer trazo)
        first_stroke_mask = batch_mask[0, 0, :]
        num_real_points = int(first_stroke_mask.sum().item())
        
        print(f"\nEjemplo de máscara para el primer trazo:")
        print(f"  - Longitud total en el tensor (con padding): {len(first_stroke_mask)}")
        print(f"  - Número de puntos reales en el trazo: {num_real_points}")
        print(f"  - Contenido de la máscara (primeros 10 puntos): {first_stroke_mask[:10].int().tolist()}")
        if len(first_stroke_mask) > num_real_points:
             print(f"  - Contenido de la máscara (últimos 10 puntos, mostrando el padding): {first_stroke_mask[-10:].int().tolist()}")