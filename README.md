Instrucciónes de pasos para ejecutar el programa correctamente
#Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\activate
#Generar carpeta "frames"
# Instalar dependencias
#instalar uv
pip install uv

#instalar numpy svgpathtools tqdm matplotlib
uv pip install numpy svgpathtools tqdm matplotlib
#instalar pytorch para poder usar GPU
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

#Preprocesar el SVG para crear los datos de entrenamiento
python preprocess_hierarchical.py
# Cargar y procesar los datos 
python dataset.py
# Entrenar el modelo
python train.py
# Generar una visualziación final de los resultados obtenidos
python generate_final_plot.py
