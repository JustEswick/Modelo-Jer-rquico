import torch
import torch.nn as nn

class HierarchicalDrawingModel(nn.Module):
    def __init__(self, point_dim=3, stroke_embedding_dim=128, conductor_hidden_dim=256, musician_hidden_dim=512, nhead=4, num_conductor_layers=2):
        super().__init__()
        
        self.point_dim = point_dim
        self.stroke_embedding_dim = stroke_embedding_dim
        self.conductor_hidden_dim = conductor_hidden_dim
        self.musician_hidden_dim = musician_hidden_dim
        
        # --- 1. Codificador de Trazos (StrokeEncoder) ---
        # Lee los puntos de un trazo y devuelve su estado oculto final como resumen.
        self.stroke_encoder = nn.GRU(
            input_size=point_dim, 
            hidden_size=stroke_embedding_dim,
            batch_first=True
        )
        
        # --- 2. El Conductor (Conductor) ---
        # Un Transformer Encoder que procesa la secuencia de resúmenes de trazos.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=stroke_embedding_dim, 
            nhead=nhead, 
            dim_feedforward=conductor_hidden_dim,
            batch_first=True
        )
        self.conductor = nn.TransformerEncoder(encoder_layer, num_layers=num_conductor_layers)
        
        # --- 3. El Músico (MusicianDecoder) ---
        # Un GRU que genera los puntos de un trazo a partir de una instrucción.
        self.musician_decoder = nn.GRU(
            input_size=point_dim,
            hidden_size=musician_hidden_dim,
            batch_first=True
        )
        
        # Capa de proyección para conectar el Conductor con el Músico
        self.conductor_to_musician = nn.Linear(stroke_embedding_dim, musician_hidden_dim)
        
        # Capa de salida final para generar cada punto
        self.output_layer = nn.Linear(musician_hidden_dim, point_dim)

    def forward(self, strokes_batch, mask):
        # strokes_batch shape: (batch, num_strokes, max_points, 3)
        # mask shape: (batch, num_strokes, max_points)
        
        batch_size, num_strokes, max_points, _ = strokes_batch.shape
        
        # --- Paso 1: Codificar cada trazo en un embedding ---
        # Aplanamos los trazos para procesarlos todos a la vez con el GRU
        strokes_flat = strokes_batch.view(batch_size * num_strokes, max_points, self.point_dim)
        # El StrokeEncoder devuelve la salida de cada paso y el estado oculto final.
        # Solo nos interesa el estado oculto final (el "resumen").
        _, hidden_states = self.stroke_encoder(strokes_flat)
        # hidden_states shape: (1, batch*num_strokes, embedding_dim)
        
        # Reorganizamos los embeddings a su forma secuencial
        stroke_embeddings = hidden_states.squeeze(0).view(batch_size, num_strokes, self.stroke_embedding_dim)
        # stroke_embeddings shape: (batch, num_strokes, embedding_dim)

        # --- Paso 2: El Conductor procesa la secuencia de trazos ---
        # Creamos una máscara para el Transformer para que ignore los trazos de padding
        stroke_mask = mask.any(dim=2) # Un trazo es real si tiene al menos un punto real
        instructions = self.conductor(stroke_embeddings, src_key_padding_mask=~stroke_mask)
        # instructions shape: (batch, num_strokes, embedding_dim)

        # --- Paso 3: El Músico genera los puntos para cada trazo ---
        # Proyectamos las instrucciones para que coincidan con el tamaño del Músico
        musician_initial_hidden = self.conductor_to_musician(instructions)
        # musician_initial_hidden shape: (batch, num_strokes, musician_hidden_dim)
        
        # Preparamos la entrada para el decodificador (usaremos los puntos reales como input)
        # y su estado oculto inicial.
        musician_input = strokes_flat
        musician_initial_hidden_flat = musician_initial_hidden.view(batch_size * num_strokes, self.musician_hidden_dim).unsqueeze(0)
        
        # Pasamos todo por el decodificador
        musician_output, _ = self.musician_decoder(musician_input, musician_initial_hidden_flat)
        
        # Proyectamos la salida del Músico para obtener los puntos reconstruidos
        reconstructed_points_flat = self.output_layer(musician_output)
        
        # Reorganizamos la salida a su forma original
        reconstructed_strokes = reconstructed_points_flat.view(batch_size, num_strokes, max_points, self.point_dim)
        
        return reconstructed_strokes

if __name__ == "__main__":
    # --- DEMOSTRACIÓN DE USO ---
    # Creamos datos falsos para probar la arquitectura
    BATCH_SIZE = 1
    NUM_STROKES = 10
    MAX_POINTS = 50
    POINT_DIM = 3
    
    dummy_data = torch.randn(BATCH_SIZE, NUM_STROKES, MAX_POINTS, POINT_DIM)
    dummy_mask = torch.ones(BATCH_SIZE, NUM_STROKES, MAX_POINTS)
    
    # Creamos una instancia del modelo
    model = HierarchicalDrawingModel()
    print("Modelo creado exitosamente.")
    print(model)
    
    # Pasamos los datos falsos a través del modelo
    output = model(dummy_data, dummy_mask)
    
    print("\n--- ¡Prueba de Flujo de Datos Exitosa! ---")
    print(f"Forma de entrada: {dummy_data.shape}")
    print(f"Forma de salida:   {output.shape}")
    
    # Verificamos que las formas de entrada y salida coincidan
    assert dummy_data.shape == output.shape
    print("\n¡Las formas de entrada y salida coinciden!")