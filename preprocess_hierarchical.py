import json
import numpy as np
from svgpathtools import svg2paths, Path
from tqdm import tqdm

def get_points_from_svg(svg_file, density=1.0, jump_threshold=15.0):
    """
    Extrae puntos de un SVG y los agrupa en trazos.
    Un "salto" grande en la distancia entre puntos consecutivos indica un nuevo trazo.
    """
    print(f"Cargando y procesando {svg_file}...")
    paths, attributes = svg2paths(svg_file)
    
    all_strokes = []
    current_stroke = []
    
    # Origen de coordenadas inicial
    last_point = np.array([0, 0])

    for path in tqdm(paths, desc="Procesando Trazados SVG"):
        # Muestreamos más puntos en trazados más largos para mantener la densidad
        num_samples = int(path.length() * density)
        if num_samples < 2:
            continue

        points = [path.point(i / (num_samples - 1)) for i in range(num_samples)]
        
        for p in points:
            current_point = np.array([p.real, p.imag])
            
            # Si el salto es muy grande, consideramos que es un nuevo trazo
            if np.linalg.norm(current_point - last_point) > jump_threshold:
                if len(current_stroke) > 1:
                    all_strokes.append(np.array(current_stroke))
                current_stroke = []

            current_stroke.append(current_point)
            last_point = current_point

    # Añadir el último trazo que quedó en el buffer
    if len(current_stroke) > 1:
        all_strokes.append(np.array(current_stroke))
        
    print(f"Se encontraron {len(all_strokes)} trazos en total.")
    return all_strokes

def strokes_to_deltas(strokes):
    """
    Convierte una lista de trazos (coordenadas absolutas) a formato Stroke-3
    (delta_x, delta_y, p_end_of_stroke).
    """
    all_deltas = []
    
    # Empezamos desde el origen (0,0)
    last_point = np.array([0, 0])
    
    for stroke in tqdm(strokes, desc="Convirtiendo a Deltas"):
        # El primer punto del trazo es relativo al último punto del trazo anterior
        deltas = np.diff(stroke, axis=0, prepend=last_point.reshape(1,2))
        
        # Creamos el estado 'p' (0 para punto intermedio, 1 para final de trazo)
        pen_states = np.zeros((len(deltas), 1))
        pen_states[-1, 0] = 1 # Marcamos el último punto del trazo
        
        # Concatenamos (dx, dy, p)
        stroke_data = np.concatenate([deltas, pen_states], axis=1)
        all_deltas.append(stroke_data)
        
        # El último punto de este trazo es el origen para el siguiente
        last_point = stroke[-1]
        
    return all_deltas

def normalize_strokes(strokes_data):
    """
    Normaliza los datos dividiendo por la desviación estándar.
    """
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
    SVG_FILE = 'pikachu.svg'      # <-- Asegúrate de tener este archivo
    OUTPUT_FILE = 'pikachu_hierarchical.json'
    
    # --- PARÁMETROS ---
    # Aumenta este valor para obtener curvas más suaves (más puntos)
    POINT_DENSITY = 1.2
    # Distancia en píxeles para considerar un "salto" de lápiz
    JUMP_THRESHOLD = 20.0
    
    # 1. Extraer trazos del SVG
    strokes_absolute = get_points_from_svg(
        SVG_FILE, 
        density=POINT_DENSITY, 
        jump_threshold=JUMP_THRESHOLD
    )
    
    # 2. Convertir a formato delta (Stroke-3)
    strokes_deltas = strokes_to_deltas(strokes_absolute)
    
    # 3. Normalizar los datos
    normalized_data, scale = normalize_strokes(strokes_deltas)
    
    # 4. Guardar todo en un archivo JSON
    final_output = {
        'strokes': normalized_data,
        'scale_factor': scale
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_output, f)
        
    print("-" * 30)
    print(f"¡Éxito! Datos guardados en {OUTPUT_FILE}")
    print(f"Número de trazos: {len(normalized_data)}")
    print("Ejemplo del primer trazo (primeros 3 puntos):")
    for point in normalized_data[0][:3]:
        print(f"  (dx={point[0]:.4f}, dy={point[1]:.4f}, p={int(point[2])})")