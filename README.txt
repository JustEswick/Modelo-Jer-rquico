#Primero crear y activar entorno virtual

#Crear el entorno virtual
python -m venv venv
#Activar el entorno virtual
.\venv\Scripts\activate

#Crea una carpeta llamada "frames"

# Instalar dependencias

#Comienza instalando uv
pip install uv

#Instala el resto de dependencias (numpy svgpathtools tqdm matplotlib)
uv pip install numpy svgpathtools tqdm matplotlib

#Instala pytorch por separado, importante revisar la versión de cuda necesaria, si es de generación 20/30/40... usar la version cu121
#si no se tiene una tarjeta nvidia, por defecto usara el CPU
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

--------------Flujo de ejecución de los scripts---------------------
#Preprocesar el SVG para crear los datos de entrenamiento
python preprocess_hierarchical.py

# Cargar y procesar los datos 
python dataset.py

# Entrenar el modelo
python train.py

# Generar una visualziación final de los resultados obtenidos
python generate_final_plot.py
----------------------------------------------------------------------
# El script verify_data carga y dibuja los datos del archivo JSON
# asi se visualiza si el archivo está siendo generado correctamente
python verify_data.py

#El script model.py sirve para comprobar que la entrada es correcta
python model.py