import os
import numpy as np
import scipy.io.wavfile as wav

# We will try to import librosa and soundfile, but provide a robust pure numpy/scipy fallback
# in case the installation is incomplete or libraries aren't available.
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

def load_audio(file_path, target_sr=16000):
    """
    Loads an audio file and resamples it to target_sr.
    """
    if HAS_LIBROSA:
        try:
            y, sr = librosa.load(file_path, sr=target_sr)
            return y, sr
        except Exception as e:
            pass
            
    if HAS_SOUNDFILE:
        try:
            y, sr = sf.read(file_path)
            # If multi-channel, convert to mono
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            # Simple decimation/interpolation if sample rate differs
            if sr != target_sr:
                num_samples = int(len(y) * target_sr / sr)
                y = np.interp(np.linspace(0, len(y), num_samples), np.arange(len(y)), y)
            return y, target_sr
        except Exception as e:
            pass
            
    # Fallback to scipy.io.wavfile
    sr, y = wav.read(file_path)
    # Convert integer types to float32
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648.0
    elif y.dtype == np.uint8:
        y = (y.astype(np.float32) - 128.0) / 128.0
        
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
        
    if sr != target_sr:
        num_samples = int(len(y) * target_sr / sr)
        y = np.interp(np.linspace(0, len(y), num_samples), np.arange(len(y)), y)
        sr = target_sr
        
    return y, sr

def extract_features_1d(y, sr, n_mfcc=13):
    """
    Extracts a 1D feature vector for traditional machine learning models (Random Forest).
    Includes MFCC means/stds, spectral centroid, spectral rolloff, chroma, and zero crossing rate.
    """
    if HAS_LIBROSA:
        # Extract features using librosa
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        
        # Combine summary statistics (mean & standard deviation)
        feature_vector = np.concatenate([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(delta_mfccs, axis=1),
            np.std(delta_mfccs, axis=1),
            np.mean(delta2_mfccs, axis=1),
            np.std(delta2_mfccs, axis=1),
            [np.mean(spectral_centroid), np.std(spectral_centroid)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
            [np.mean(zcr), np.std(zcr)],
            [np.mean(rms), np.std(rms)]
        ])
        return feature_vector
    else:
        # Fallback manual calculation using basic numpy STFT for portability
        # We calculate a simple short-time Fourier transform
        n_fft = 512
        hop_length = 256
        
        # Windowing
        frames = []
        for i in range(0, len(y) - n_fft, hop_length):
            frames.append(y[i:i+n_fft] * np.hanning(n_fft))
        
        if len(frames) == 0:
            return np.zeros(n_mfcc * 6 + 8, dtype=np.float32)
            
        frames = np.array(frames) # shape (n_frames, n_fft)
        
        # Magnitude spectrum
        fft_out = np.fft.rfft(frames, axis=1)
        magnitude = np.abs(fft_out) # shape (n_frames, n_fft//2 + 1)
        
        # Spectral Centroid
        freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
        centroid = np.sum(magnitude * freqs, axis=1) / (np.sum(magnitude, axis=1) + 1e-10)
        
        # Zero Crossing Rate
        zcr = np.mean(np.abs(np.diff(np.sign(y))))
        
        # Simple energy (RMS)
        rms = np.sqrt(np.mean(frames**2, axis=1))
        
        # Log spectrogram mean and std as surrogate for MFCCs in fallback mode
        # Bin frequencies into n_mfcc bands
        band_size = magnitude.shape[1] // n_mfcc
        pseudo_mfccs = []
        for i in range(n_mfcc):
            band_magnitude = magnitude[:, i*band_size : (i+1)*band_size]
            pseudo_mfccs.append(np.log1p(np.mean(band_magnitude, axis=1)))
        pseudo_mfccs = np.array(pseudo_mfccs) # shape (n_mfcc, n_frames)
        
        feature_vector = np.concatenate([
            np.mean(pseudo_mfccs, axis=1),
            np.std(pseudo_mfccs, axis=1),
            # Deltas approximated by difference
            np.mean(np.diff(pseudo_mfccs, prepend=0, axis=1), axis=1),
            np.std(np.diff(pseudo_mfccs, prepend=0, axis=1), axis=1),
            # Delta-deltas
            np.mean(np.diff(pseudo_mfccs, prepend=0, append=0, axis=1)[:, :-1], axis=1),
            np.std(np.diff(pseudo_mfccs, prepend=0, append=0, axis=1)[:, :-1], axis=1),
            [np.mean(centroid), np.std(centroid)],
            [np.mean(centroid), np.std(centroid)], # duplicate spectral rolloff surrogate
            [zcr, zcr*0.1],
            [np.mean(rms), np.std(rms)]
        ])
        return feature_vector

def extract_features_2d(y, sr, n_mfcc=20, max_pad=128):
    """
    Extracts a 2D feature matrix (e.g. MFCC spectrogram over time) for deep learning (CNN).
    The matrix is padded or truncated to a fixed time steps (max_pad).
    """
    if HAS_LIBROSA:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        if mfccs.shape[1] < max_pad:
            # Pad
            pad_width = max_pad - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate
            mfccs = mfccs[:, :max_pad]
        return mfccs
    else:
        # Fallback to pseudo-MFCCs 2D grid
        n_fft = 512
        hop_length = int(len(y) / max_pad) if len(y) > max_pad else 256
        
        frames = []
        for i in range(0, len(y) - n_fft, hop_length):
            frames.append(y[i:i+n_fft] * np.hanning(n_fft))
            if len(frames) >= max_pad:
                break
                
        # Pad frames if needed
        while len(frames) < max_pad:
            frames.append(np.zeros(n_fft))
            
        frames = np.array(frames[:max_pad])
        fft_out = np.fft.rfft(frames, axis=1)
        magnitude = np.abs(fft_out)
        
        band_size = magnitude.shape[1] // n_mfcc
        pseudo_mfccs = []
        for i in range(n_mfcc):
            band_magnitude = magnitude[:, i*band_size : (i+1)*band_size]
            pseudo_mfccs.append(np.log1p(np.mean(band_magnitude, axis=1)))
            
        return np.array(pseudo_mfccs) # shape (n_mfcc, max_pad)

def process_file(file_path):
    """
    Helper function to load file and return both 1D and 2D features.
    """
    y, sr = load_audio(file_path)
    feat_1d = extract_features_1d(y, sr)
    feat_2d = extract_features_2d(y, sr)
    return feat_1d, feat_2d
