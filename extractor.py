#!/usr/bin/env python3
"""
Side Eye Audio Extractor - Versión Final Mejorada
Extrae audio de videos usando Rolling Shutter + OIS (técnica Side Eye)

Uso:
    python sideeye_extractor.py video.mp4
    python sideeye_extractor.py video.mp4 -o audio.wav -s 4 --ois 0.35
    python sideeye_extractor.py video.mp4 --no-ois --no-vis
"""

import cv2
import numpy as np
from scipy import signal
from scipy.io.wavfile import write
from scipy.signal import wiener
import matplotlib.pyplot as plt
import os
import argparse
import time
from pathlib import Path

# Intentar importar tqdm (opcional)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm no instalado. Instala con: pip install tqdm")


def extract_side_eye_advanced(video_path, 
                             output_wav="sideeye_recovered.wav",
                             step=5, 
                             ois_weight=0.32, 
                             lowcut=65, 
                             highcut=1900,
                             use_ois=True, 
                             visualize=True,
                             target_fs=48000):
    """
    Extrae audio de un video usando la técnica Side Eye
    
    Args:
        video_path: Ruta del archivo de video
        output_wav: Archivo WAV de salida
        step: Submuestreo de líneas (4-8, menor=mejor calidad)
        ois_weight: Peso de la señal OIS (0.25-0.40)
        lowcut: Frecuencia mínima del filtro (Hz)
        highcut: Frecuencia máxima del filtro (Hz)
        use_ois: Activar detección de OIS por Optical Flow
        visualize: Mostrar gráficos
        target_fs: Frecuencia de muestreo del audio final
    """
    
    # Validar archivo
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No se encuentra el archivo: {video_path}")
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se puede abrir el video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print("\n" + "="*65)
    print("🎙️  SIDE EYE AUDIO EXTRACTOR")
    print("="*65)
    print(f"📹 {Path(video_path).name}")
    print(f"   • Resolución: {width}x{height}")
    print(f"   • FPS: {fps:.2f}")
    print(f"   • Frames: {total_frames}")
    print(f"   • Duración: {duration:.1f}s")
    print(f"\n⚙️  Configuración:")
    print(f"   • STEP: {step}")
    print(f"   • OIS: {'Activado' if use_ois else 'Desactivado'}")
    if use_ois:
        print(f"   • OIS weight: {ois_weight:.2f}")
    print(f"   • Filtro: {lowcut}-{highcut} Hz")
    print("="*65)
    
    start_time = time.time()
    
    # ============================
    # PROCESAMIENTO DE FRAMES
    # ============================
    
    rolling_signals = []
    ois_motion = []
    frame_count = 0
    prev_gray = None
    p0 = None
    
    # Configuración Optical Flow
    feature_params = dict(maxCorners=100, 
                         qualityLevel=0.3, 
                         minDistance=7, 
                         blockSize=7)
    lk_params = dict(winSize=(15, 15), 
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    print("\n📊 Procesando frames...")
    
    # Barra de progreso
    if HAS_TQDM:
        pbar = tqdm(total=total_frames, unit="frame", desc="Progreso")
    else:
        pbar = None
        print("   Progreso: ", end='')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Progreso (sin tqdm)
        if pbar is None and frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"\r   Progreso: {frame_count}/{total_frames} ({progress:.1f}%)", end='')
        
        # Convertir a grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float32) / 255.0
        
        # === 1. ROLLING SHUTTER ===
        for y in range(0, height, step):
            line = gray_float[y, :]
            intensity = np.mean(line)
            variance = np.var(line)
            gradient = np.mean(np.abs(np.diff(line)))
            value = 0.45 * intensity + 0.35 * variance + 0.20 * gradient
            rolling_signals.append(value)
        
        # === 2. OIS - OPTICAL FLOW ===
        if use_ois and prev_gray is not None:
            gray_uint8 = (gray_float * 255).astype(np.uint8)
            
            if p0 is not None and len(p0) > 10:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray_uint8, 
                                                      p0, None, **lk_params)
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                if len(good_new) > 8:
                    dx = np.mean(good_new[:, 0] - good_old[:, 0])
                    dy = np.mean(good_new[:, 1] - good_old[:, 1])
                    magnitude = np.sqrt(dx**2 + dy**2)
                    ois_motion.append(magnitude * 70)
                else:
                    ois_motion.append(0)
            else:
                ois_motion.append(0)
        
        # Actualizar para siguiente frame
        prev_gray = (gray_float * 255).astype(np.uint8)
        if use_ois and (p0 is None or len(p0) < 20):
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        
        # Actualizar tqdm
        if pbar is not None:
            pbar.update(1)
    
    # Cerrar barra de progreso
    if pbar is not None:
        pbar.close()
    else:
        print(f"\r   Progreso: {total_frames}/{total_frames} (100.0%)")
    
    cap.release()
    processing_time = time.time() - start_time
    print(f"\n✅ Procesados {frame_count} frames en {processing_time:.1f}s")
    
    # ============================
    # POST-PROCESAMIENTO
    # ============================
    
    print("\n🎛️  Procesando señal...")
    
    rolling_signal = np.array(rolling_signals, dtype=np.float64)
    effective_fs = fps * (height / step)
    print(f"   • Muestras: {len(rolling_signal):,}")
    print(f"   • Fs efectiva: {effective_fs:.0f} Hz")
    
    # Combinar con OIS
    if use_ois and len(ois_motion) > 0:
        ois_resampled = signal.resample(np.array(ois_motion), len(rolling_signal))
        combined = rolling_signal * (1 - ois_weight) + ois_resampled * ois_weight
        print(f"   • OIS activado (peso: {ois_weight:.2f})")
    else:
        combined = rolling_signal
        if use_ois:
            print(f"   • OIS: sin datos detectados")
        else:
            print(f"   • OIS: desactivado por usuario")
    
    # Detrend
    combined = signal.detrend(combined)
    
    # Filtro pasa-banda
    nyquist = effective_fs / 2
    highcut_actual = min(highcut, nyquist * 0.9)
    
    b, a = signal.cheby1(6, 0.5, [lowcut/nyquist, highcut_actual/nyquist], btype='band')
    filtered = signal.filtfilt(b, a, combined)
    
    # Reducción de ruido
    filtered = wiener(filtered, mysize=9)
    
    # Normalizar
    filtered = filtered / np.max(np.abs(filtered)) * 0.95
    
    # Upsampling
    upsampled = signal.resample(filtered, int(len(filtered) * target_fs / effective_fs))
    
    # Calcular SNR aproximado
    noise = combined - filtered
    snr_db = 20 * np.log10(np.std(filtered) / (np.std(noise) + 1e-10))
    
    # Guardar audio
    write(output_wav, target_fs, upsampled.astype(np.float32))
    
    # ============================
    # VISUALIZACIÓN
    # ============================
    
    if visualize:
        print("   • Generando gráficos...")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Rolling Shutter Raw
        axes[0].plot(rolling_signal[:8000], color='blue', alpha=0.7, linewidth=0.5)
        axes[0].set_title('Rolling Shutter - Señal Cruda', fontsize=12)
        axes[0].set_ylabel('Intensidad')
        axes[0].grid(True, alpha=0.3)
        
        # Señal filtrada
        axes[1].plot(filtered[:8000], color='darkorange', linewidth=0.5)
        axes[1].set_title('Señal Filtrada (Audio Recuperado)', fontsize=12)
        axes[1].set_ylabel('Amplitud')
        axes[1].grid(True, alpha=0.3)
        
        # Espectro de frecuencia
        freq, psd = signal.welch(filtered, effective_fs, nperseg=1024)
        axes[2].semilogy(freq, psd, color='red', linewidth=0.8)
        axes[2].set_title('Espectro de Frecuencias', fontsize=12)
        axes[2].set_xlabel('Frecuencia (Hz)')
        axes[2].set_ylabel('Densidad Espectral')
        axes[2].grid(True, alpha=0.3)
        axes[2].axvline(300, color='gray', linestyle='--', alpha=0.6, label='Voz (300 Hz)')
        axes[2].axvline(3400, color='gray', linestyle='--', alpha=0.4, label='Límite voz')
        axes[2].legend()
        
        plt.tight_layout()
        plot_file = Path(output_wav).stem + "_analysis.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"   • Gráfico guardado: {plot_file}")
        plt.show()
    
    # ============================
    # RESULTADOS
    # ============================
    
    audio_duration = len(upsampled) / target_fs
    file_size = os.path.getsize(output_wav) / 1024
    
    print("\n" + "="*65)
    print("🎉 ¡PROCESO COMPLETADO!")
    print("="*65)
    print(f"📁 Audio: {output_wav}")
    print(f"   • Duración: {audio_duration:.2f}s")
    print(f"   • Frecuencia: {target_fs} Hz")
    print(f"   • Tamaño: {file_size:.1f} KB")
    print(f"📊 SNR estimado: {snr_db:.1f} dB", end="")
    
    if snr_db > 15:
        print(" 🟢 Excelente")
    elif snr_db > 10:
        print(" 🟡 Aceptable")
    elif snr_db > 5:
        print(" 🟠 Bajo")
    else:
        print(" 🔴 Muy bajo")
    
    print("="*65)
    print("🎵 Reproduce el archivo WAV para escuchar el resultado.")
    
    return upsampled


# ============================
# FUNCIÓN PARA ANALIZAR AUDIO
# ============================

def analyze_audio(wav_path):
    """Analiza un archivo WAV generado por la herramienta"""
    from scipy.io.wavfile import read
    
    if not os.path.exists(wav_path):
        print(f"❌ No se encuentra: {wav_path}")
        return
    
    fs, data = read(wav_path)
    duration = len(data) / fs
    
    print("\n" + "="*65)
    print("📊 ANÁLISIS DE AUDIO")
    print("="*65)
    print(f"📁 Archivo: {wav_path}")
    print(f"   • Duración: {duration:.2f}s")
    print(f"   • Frecuencia: {fs} Hz")
    print(f"   • Muestras: {len(data):,}")
    print(f"   • Pico máximo: {np.max(np.abs(data)):.4f}")
    print(f"   • RMS: {np.sqrt(np.mean(data**2)):.4f}")
    
    if len(data) > 1024:
        freq, psd = signal.welch(data.astype(np.float64), fs, nperseg=1024)
        max_freq = freq[np.argmax(psd)]
        print(f"   • Frecuencia dominante: {max_freq:.1f} Hz")
        
        # Energía en banda de voz
        voice_mask = (freq >= 80) & (freq <= 3400)
        voice_energy = np.sum(psd[voice_mask])
        total_energy = np.sum(psd)
        voice_ratio = voice_energy / (total_energy + 1e-10) * 100
        print(f"   • Energía en banda voz: {voice_ratio:.1f}%")
    print("="*65)


# ============================
# INTERFAZ DE COMANDOS
# ============================

def main():
    parser = argparse.ArgumentParser(
        description="Side Eye Audio Extractor - Extrae audio de videos usando micrófono visual",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s video.mp4
  %(prog)s video.mp4 -o audio.wav -s 4 --ois 0.35
  %(prog)s video.mp4 --no-ois --no-vis
  %(prog)s --analyze audio.wav
        """
    )
    
    # Argumentos principales
    parser.add_argument('video', nargs='?', help='Ruta del archivo de video (MP4)')
    parser.add_argument('-o', '--output', default='sideeye_recovered.wav',
                       help='Archivo WAV de salida (default: sideeye_recovered.wav)')
    
    # Parámetros de procesamiento
    parser.add_argument('-s', '--step', type=int, default=5,
                       help='Submuestreo de líneas (4-8, menor=mejor) (default: 5)')
    parser.add_argument('--ois', type=float, default=0.32,
                       help='Peso del OIS (0.25-0.40) (default: 0.32)')
    parser.add_argument('--lowcut', type=int, default=65,
                       help='Frecuencia mínima del filtro (Hz) (default: 65)')
    parser.add_argument('--highcut', type=int, default=1900,
                       help='Frecuencia máxima del filtro (Hz) (default: 1900)')
    parser.add_argument('--target-fs', type=int, default=48000,
                       help='Frecuencia de muestreo del audio (default: 48000)')
    
    # Opciones
    parser.add_argument('--no-ois', action='store_true',
                       help='Desactivar OIS (más rápido, menos preciso)')
    parser.add_argument('--no-vis', action='store_true',
                       help='Desactivar visualización de gráficos')
    
    # Modo análisis
    parser.add_argument('--analyze', metavar='FILE',
                       help='Analizar un archivo WAV generado')
    
    args = parser.parse_args()
    
    # Modo análisis
    if args.analyze:
        analyze_audio(args.analyze)
        return
    
    # Modo normal
    if not args.video:
        parser.print_help()
        return
    
    try:
        extract_side_eye_advanced(
            args.video,
            output_wav=args.output,
            step=args.step,
            ois_weight=args.ois,
            lowcut=args.lowcut,
            highcut=args.highcut,
            use_ois=not args.no_ois,
            visualize=not args.no_vis,
            target_fs=args.target_fs
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


# ============================
# PUNTO DE ENTRADA
# ============================

if __name__ == "__main__":
    main()
