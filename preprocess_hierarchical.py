import json
import numpy as np
from svgpathtools import svg2paths, Path
from tqdm import tqdm

def get_points_from_svg(svg_file, density=1.0, jump_threshold=15.0):
    """Extrae puntos de un SVG y los agrupa en trazos."""
    print(f"Cargando y procesando {svg_file}...")
    paths, attributes = svg2paths(svg_file)
    all_strokes = []
    current_stroke = []
    last_point = np.array([0, 0])

    for path in tqdm(paths, desc="Procesando Trazados SVG"):
        num_samples = int(path.length() * density)
        if num_samples < 2:
            continue
        points = [path.point(i / (num_samples - 1)) for i in range(num_samples)]
        for p in points:
            current_point = np.array([p.real, p.imag])
            if np.linalg.norm(current_point - last_point) > jump_threshold:
                if len(current_stroke) > 1:
                    all_strokes.append(np.array(current_stroke))
                current_stroke = []
            current_stroke.append(current_point)
            last_point = current_point
    if len(current_stroke) > 1:
        all_strokes.append(np.array(current_stroke))
    print(f"Se encontraron {len(all_strokes)} trazos en total.")
    return all_strokes

def center_strokes(strokes):
    """
    NUEVA FUNCIÓN: Centra el dibujo completo en el origen (0,0).
    """
    # Juntamos todos los puntos de todos los trazos en una sola matriz
    all_points = np.vstack(strokes)
    # Calculamos el punto medio (centro de masa)
    mean_point = np.mean(all_points, axis=0)
    print(f"Centro original del dibujo: ({mean_point[0]:.2f}, {mean_point[1]:.2f}). Centrando en (0,0)...")
    
    # Restamos el punto medio de cada punto en cada trazo
    centered_strokes = [stroke - mean_point for stroke in strokes]
    return centered_strokes

def strokes_to_deltas(strokes):
    """Convierte coordenadas absolutas a formato delta (Stroke-3)."""
    all_deltas = []
    last_point = np.array([0, 0])
    for stroke in tqdm(strokes, desc="Convirtiendo a Deltas"):
        deltas = np.diff(stroke, axis=0, prepend=last_point.reshape(1, 2))
        pen_states = np.zeros((len(deltas), 1))
        pen_states[-1, 0] = 1
        stroke_data = np.concatenate([deltas, pen_states], axis=1)
        all_deltas.append(stroke_data)
        last_point = stroke[-1]
    return all_deltas

def normalize_strokes(strokes_data):
    """Normaliza los deltas usando su desviación estándar."""
    all_deltas = np.vstack([stroke[:, :2] for stroke in strokes_data])
    scale_factor = np.std(all_deltas)
    print(f"Factor de normalización (std dev): {scale_factor:.4f}")
    normalized_strokes = []
    for stroke in strokes_data:
        normalized_stroke = stroke.copy()
        normalized_stroke[:, :2] /= scale_factor
        normalized_strokes.append(normalized_stroke.tolist())
    return normalized_strokes, scale_factor

if __name__ == "__main__":
    SVG_FILE = 'pikachu.svg'
    OUTPUT_FILE = 'pikachu_hierarchical.json'
    POINT_DENSITY = 1.2
    JUMP_THRESHOLD = 20.0
    
    # 1. Extraer trazos (coordenadas absolutas)
    strokes_absolute = get_points_from_svg(SVG_FILE, POINT_DENSITY, JUMP_THRESHOLD)
    
    # 2. *** NUEVO PASO: Centrar el dibujo ***
    centered_strokes = center_strokes(strokes_absolute)
    
    # 3. Convertir a formato delta (Stroke-3)
    strokes_deltas = strokes_to_deltas(centered_strokes)
    
    # 4. Normalizar los datos
    normalized_data, scale = normalize_strokes(strokes_deltas)
    
    # 5. Guardar todo en un archivo JSON
    final_output = {'strokes': normalized_data, 'scale_factor': scale}
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_output, f)
    print("-" * 30)
    print(f"¡Éxito! Datos CENTRADOS guardados en {OUTPUT_FILE}")