# eye
Experimental



# Side Eye Audio Extractor

Herramienta para extraer audio de videos usando la técnica **Side Eye** (micrófono visual basado en Rolling Shutter + OIS).

## Instalación

pip install -r requirements.txt

## Uso básico

python sideeye_extractor.py video.mp4

## Opciones recomendadas

# Mejor calidad (más lento)
python sideeye_extractor.py video.mp4 -s 4 --ois 0.35 -o audio_recuperado.wav

# Modo rápido (sin OIS)
python sideeye_extractor.py video.mp4 --no-ois --no-vis

# Analizar un archivo WAV generado
python sideeye_extractor.py --analyze audio_recuperado.wav

## Parámetros principales

-s, --step → Submuestreo de líneas (4-6 recomendado)
--ois → Peso del OIS (0.25-0.40)
--no-ois → Desactiva OIS (más rápido)
--no-vis → Sin gráficos
-o, --output → Nombre del archivo WAV de salida

**python sideeye_extractor.py video_sideeye.mp4 -o audio.wav -s 4 --ois 0.35 --lowcut 80**

## Ayuda

python sideeye_extractor.py -h

## Cómo obtener buenos resultados

Graba un video con la cámara del teléfono (preferiblemente trasera).
Pon un altavoz reproduciendo audio cerca del teléfono.
Usa resolución alta y 30 o 60 FPS.
Prueba primero con música o voz clara.

## Limitaciones

Limitaciones

La calidad depende mucho de la vibración transmitida al teléfono.
Funciona mejor con altavoces electrónicos que con voz humana directa.

