import json
import numpy as np
import matplotlib.pyplot as plt

JSON_FILE = 'pikachu_hierarchical.json'
OUTPUT_IMAGE = 'verification_plot.png'

def verify_data_centering(json_file):
    """
    Carga los datos del JSON y los grafica para verificar si están centrados.
    """
    print(f"Verificando el archivo: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)

    strokes_data = data['strokes']
    scale_factor = data['scale_factor']

    # Aplanamos todos los deltas en una única secuencia
    flat_deltas = np.array([point for stroke in strokes_data for point in stroke])

    # Convertimos a coordenadas absolutas
    coords = np.cumsum(flat_deltas[:, :2] * scale_factor, axis=0)

    # Graficamos
    plt.figure(figsize=(8, 8))
    plt.plot(coords[:, 0], -coords[:, 1], 'g-') # Dibujo en verde
    plt.title('Verificación de Centrado de Datos')
    plt.axhline(0, color='black', linewidth=0.5) # Eje X
    plt.axvline(0, color='black', linewidth=0.5) # Eje Y
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(OUTPUT_IMAGE)
    plt.close()
    
    print(f"Gráfico de verificación guardado en {OUTPUT_IMAGE}")

if __name__ == "__main__":
    verify_data_centering(JSON_FILE)