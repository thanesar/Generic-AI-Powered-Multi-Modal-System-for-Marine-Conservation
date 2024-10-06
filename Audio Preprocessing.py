import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import butter, filtfilt

def load_audio(audio_path, sample_rate=22050):
    """Loads the audio file and resamples it to the specified sample rate."""
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    print(f"Loaded audio with sample rate: {sr}")
    return audio, sr

def reduce_noise(audio, sr, lowcut=50.0, highcut=3000.0):
    """Applies a bandpass filter to reduce noise outside the frequency range of interest."""
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype="band")
    filtered_audio = filtfilt(b, a, audio)
    return filtered_audio

def compute_stft(audio, sr, n_fft=2048, hop_length=512):
    """Computes the Short-Time Fourier Transform (STFT) of the audio."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    stft_magnitude, _ = librosa.magphase(stft)
    return stft_magnitude

def apply_clahe_on_spectrogram(spectrogram):
    """Applies CLAHE to the spectrogram to enhance contrast."""
    # Convert the spectrogram to 8-bit grayscale for CLAHE processing
    spectrogram_uint8 = np.uint8(255 * (spectrogram / np.max(spectrogram)))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_spectrogram = clahe.apply(spectrogram_uint8)
    return enhanced_spectrogram

def display_spectrogram(spectrogram, sr, hop_length):
    """Displays the spectrogram using librosa's display functionality."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.amplitude_to_db(spectrogram, ref=np.max),
        sr=sr,
        hop_length=hop_length,
        y_axis='log',
        x_axis='time'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()

def save_spectrogram(spectrogram, output_path):
    """Saves the enhanced spectrogram as an image file."""
    plt.imsave(output_path, spectrogram, cmap='gray')
    print(f"Spectrogram saved at: {output_path}")

def preprocess_audio(audio_path, output_path, sample_rate=22050, lowcut=50.0, highcut=3000.0):
    """Full pipeline for loading, filtering, computing STFT, applying CLAHE, and saving the spectrogram."""
    # Load audio
    audio, sr = load_audio(audio_path, sample_rate=sample_rate)

    # Reduce noise with bandpass filter
    filtered_audio = reduce_noise(audio, sr, lowcut=lowcut, highcut=highcut)

    # Compute STFT to get the spectrogram
    spectrogram = compute_stft(filtered_audio, sr)

    # Apply CLAHE for contrast enhancement
    enhanced_spectrogram = apply_clahe_on_spectrogram(spectrogram)

    # Display the spectrogram
    display_spectrogram(enhanced_spectrogram, sr, hop_length=512)

    # Save the spectrogram as an image
    save_spectrogram(enhanced_spectrogram, output_path)

# Parameters
audio_path = 'your_audio_file.wav'  # Path to the underwater audio file
output_path = 'output/enhanced_spectrogram.png'  # Output path for the saved spectrogram
sample_rate = 22050  # Resampling rate
lowcut = 50.0  # Lower cutoff frequency for noise reduction
highcut = 3000.0  # Upper cutoff frequency for noise reduction

# Run the preprocessing pipeline
preprocess_audio(audio_path, output_path, sample_rate, lowcut, highcut)
