# ==============================================================================
# SECCIÓN DE IMPORTACIONES
# ==============================================================================
import torch
import torch.nn as nn
import math

# ==============================================================================
# CLASE DE AYUDA: CODIFICACIÓN POSICIONAL (PositionalEncoding)
# ==============================================================================
# PROPÓSITO: Los Transformers, por sí mismos, no entienden el orden de una secuencia.
# Esta clase inyecta información sobre la posición de cada punto en el trazo.
# ANALOGÍA: Es como numerar las páginas de un libro. Sin los números,
# las páginas son solo un montón de texto desordenado. Con ellos, se puede leer
# la historia en el orden correcto.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Crea un vector de posiciones (0, 1, 2, ...)
        position = torch.arange(max_len).unsqueeze(1)
        # Crea la escala de frecuencia para las funciones seno y coseno
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Inicializa el tensor de codificación posicional
        pe = torch.zeros(max_len, 1, d_model)
        
        # Aplica las funciones seno y coseno en dimensiones alternas
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # `register_buffer` guarda el tensor como parte del estado del modelo, pero
        # no como un parámetro que deba ser entrenado.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Añade la codificación posicional al tensor de entrada.
        x: Tensor, forma [seq_len, batch_size, embedding_dim]
        """
        # Suma la codificación posicional al embedding de entrada
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# ==============================================================================
# CLASE PRINCIPAL DEL MODELO: HierarchicalDrawingModel
# ==============================================================================
class HierarchicalDrawingModel(nn.Module):
    """
    La arquitectura completa de nuestro modelo jerárquico.
    Contiene al Codificador, al Conductor y al Músico.
    """
    def __init__(self, point_dim: int = 3, stroke_embedding_dim: int = 256, musician_hidden_dim: int = 256, nhead: int = 4, num_conductor_layers: int = 2, num_musician_layers: int = 2):
        super().__init__()
        
        # --- Dimensiones y Parámetros de la Arquitectura ---
        self.point_dim = point_dim # Dimensión de cada punto (dx, dy, p)
        self.stroke_embedding_dim = stroke_embedding_dim # Dimensión del "resumen" de cada trazo
        self.musician_hidden_dim = musician_hidden_dim # Dimensión interna del Músico
        
        # --- 1. El Codificador de Trazos (StrokeEncoder) ---
        # ANALOGÍA: El líder de una sección de la orquesta. Lee la partitura de un
        # solo músico (un trazo) y la resume en una idea principal.
        self.stroke_encoder = nn.GRU(
            input_size=point_dim, 
            hidden_size=stroke_embedding_dim,
            batch_first=True # Indica que la dimensión del lote va primero
        )
        
        # --- 2. El Conductor (Conductor) ---
        # ANALOGÍA: El Director de Orquesta. Recibe las ideas de todas las secciones
        # y las contextualiza para crear una visión global de la obra.
        conductor_encoder_layer = nn.TransformerEncoderLayer(
            d_model=stroke_embedding_dim, 
            nhead=nhead, # Número de "cabezas de atención"
            # CORRECCIÓN: Usar * 2 es más estándar y consistente con el decoder
            dim_feedforward=stroke_embedding_dim * 2, 
            batch_first=True
        )
        self.conductor = nn.TransformerEncoder(conductor_encoder_layer, num_layers=num_conductor_layers)
        
        # --- 3. El Músico (MusicianDecoder) ---
        # ANALOGÍA: El Solista virtuoso. Recibe una instrucción contextualizada del
        # Director y la ejecuta a la perfección para generar un trazo detallado.
        
        # Capa de entrada: Proyecta los puntos (dim 3) a la dimensión interna del modelo.
        self.musician_input_embedding = nn.Linear(point_dim, musician_hidden_dim)
        # Módulo de codificación posicional para el Músico
        self.positional_encoder = PositionalEncoding(d_model=musician_hidden_dim)
        
        # La capa principal del Músico: un Transformer Decoder
        musician_decoder_layer = nn.TransformerDecoderLayer(
            d_model=musician_hidden_dim,
            nhead=nhead,
            dim_feedforward=musician_hidden_dim * 2,
            batch_first=True
        )
        self.musician_decoder = nn.TransformerDecoder(musician_decoder_layer, num_layers=num_musician_layers)
        
        # Aserción: Nos aseguramos de que las dimensiones de comunicación entre
        # el Conductor y el Músico sean las mismas.
        assert stroke_embedding_dim == musician_hidden_dim, "Conductor y Músico deben tener la misma dimensión"

        # Capa de salida: Proyecta la salida del Músico de vuelta a la dimensión de un punto (3).
        self.output_layer = nn.Linear(musician_hidden_dim, point_dim)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Genera una máscara causal para el Transformer Decoder.
        Esto evita que el modelo "vea el futuro" al predecir el siguiente punto.
        """
        # Crea una matriz triangular superior de unos, y la convierte a booleano.
        # True indica una posición que será ignorada (enmascarada).
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, strokes_batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        El "paso adelante": define cómo fluyen los datos a través del modelo.
        """
        batch_size, num_strokes, max_points, _ = strokes_batch.shape
        device = strokes_batch.device

        # --- Paso 1 y 2: Codificador de Trazos y Conductor ---
        # Aplanamos los lotes y los trazos para procesarlos todos a la vez.
        strokes_flat = strokes_batch.view(batch_size * num_strokes, max_points, self.point_dim)
        # El Codificador procesa los trazos y nos quedamos con su estado oculto final (el "resumen").
        _, hidden_states = self.stroke_encoder(strokes_flat)
        # Reorganizamos los resúmenes en el formato de secuencia de trazos.
        stroke_embeddings = hidden_states.squeeze(0).view(batch_size, num_strokes, self.stroke_embedding_dim)
        
        # El Conductor procesa la secuencia de resúmenes para obtener instrucciones contextualizadas.
        stroke_padding_mask = ~mask.any(dim=2) # Máscara para los trazos de padding
        instructions = self.conductor(stroke_embeddings, src_key_padding_mask=stroke_padding_mask)

        # --- Paso 3: El Músico (Transformer Decoder) ---
        N = batch_size * num_strokes # El tamaño del "lote" aplanado para el Músico

        # Preparamos la "memoria" (las instrucciones del Conductor) para el Músico.
        memory = instructions.view(N, 1, self.stroke_embedding_dim)
        
        # Preparamos la secuencia objetivo (target, o `tgt`) que el Músico debe aprender a generar.
        tgt = strokes_flat
        tgt_embedded = self.musician_input_embedding(tgt)
        # El Transformer estándar necesita la forma [seq_len, batch, features], así que permutamos.
        tgt_permuted = tgt_embedded.permute(1, 0, 2)
        tgt_with_pe = self.positional_encoder(tgt_permuted).permute(1, 0, 2) # Añadimos pos. encoding y volvemos a permutar

        # Creamos las máscaras que el Músico necesita
        tgt_causal_mask = self._generate_causal_mask(max_points, device) # Máscara para no ver el futuro
        tgt_padding_mask = ~mask.view(N, max_points).bool() # Máscara para el padding de puntos
        memory_padding_mask = stroke_padding_mask.view(N, 1) # Máscara para el padding de la memoria

        # Pasamos todo por el Músico (Transformer Decoder)
        musician_output = self.musician_decoder(
            tgt=tgt_with_pe,
            memory=memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask
        )
        
        # Proyectamos la salida para obtener los puntos reconstruidos
        reconstructed_points_flat = self.output_layer(musician_output)
        
        # Reorganizamos la salida a su forma original de lote jerárquico
        reconstructed_strokes = reconstructed_points_flat.view(batch_size, num_strokes, max_points, self.point_dim)
        
        return reconstructed_strokes

# ==============================================================================
# BLOQUE DE PRUEBA
# ==============================================================================
if __name__ == "__main__":
    """
    Este bloque solo se ejecuta cuando corres `python model.py` directamente.
    Sirve para verificar que las dimensiones de los tensores son correctas y que
    no hay errores en el flujo de datos del modelo.
    """
    BATCH_SIZE = 2
    NUM_STROKES = 10
    MAX_POINTS = 50
    POINT_DIM = 3
    EMBEDDING_DIM = 256
    
    dummy_data = torch.randn(BATCH_SIZE, NUM_STROKES, MAX_POINTS, POINT_DIM)
    dummy_mask = torch.ones(BATCH_SIZE, NUM_STROKES, MAX_POINTS)
    dummy_mask[0, -2:, :] = 0
    dummy_mask[1, -4:, :] = 0
    
    model = HierarchicalDrawingModel(stroke_embedding_dim=EMBEDDING_DIM, musician_hidden_dim=EMBEDDING_DIM)
    print("Modelo Transformer creado exitosamente.")
    
    output = model(dummy_data, dummy_mask)
    
    print("\n--- ¡Prueba de Flujo de Datos Exitosa! ---")
    print(f"Forma de entrada: {dummy_data.shape}")
    print(f"Forma de salida:   {output.shape}")
    assert dummy_data.shape == output.shape
    print("\n¡Las formas de entrada y salida coinciden!")