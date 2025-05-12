import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# part d 

#  i 

# 29) Initialise the files
fs, x = wavfile.read('C:/Users/User/Documents/ASP Coursework/Question 3/x_noise.wav')
_, y_l = wavfile.read('C:/Users/User/Documents/ASP Coursework/Question 3/y_l_noise.wav')
_, y_h = wavfile.read('C:/Users/User/Documents/ASP Coursework/Question 3/y_h_noise.wav')


# 30) set variables 
nfft = 1024
length = (len(x) // nfft) * nfft
x = x[:length].astype(float)
y_l = y_l[:length].astype(float)
y_h = y_h[:length].astype(float)

x_seg = x.reshape(-1, nfft)
y_l_seg = y_l.reshape(-1, nfft)
y_h_seg = y_h.reshape(-1, nfft)

# 31) Establlsh the H1 estimator
def compute_h1_frf(x_seg, y_seg, nfft):
    Sxy_sum = 0
    Sxx_sum = 0
    for xi, yi in zip(x_seg, y_seg):
        X = np.fft.fft(xi, n=nfft)[:nfft//2 + 1]
        Y = np.fft.fft(yi, n=nfft)[:nfft//2 + 1]
        Sxy_sum += Y * np.conj(X)
        Sxx_sum += X * np.conj(X)
    H1 = Sxy_sum / (Sxx_sum + 1e-10)
    return H1

# 32) Compute the FRFs
H1_low = compute_h1_frf(x_seg, y_l_seg, nfft)
H1_high = compute_h1_frf(x_seg, y_h_seg, nfft)

# 33) Reconstruct the full spectrum
def full_spectrum(H_half):
    return np.concatenate([H_half, np.conj(H_half[-2:0:-1])])

H1_low_full = full_spectrum(H1_low)
H1_high_full = full_spectrum(H1_high)

# 34) Compute the impulse responses
h_low = np.fft.ifft(H1_low_full)
h_high = np.fft.ifft(H1_high_full)

# 35) Check if they are real or complex
is_real_low = np.isreal(h_low).all()
is_real_high = np.isreal(h_high).all()
max_real_low = np.max(np.abs(np.real(h_low)))
max_imag_low = np.max(np.abs(np.imag(h_low)))
max_real_high = np.max(np.abs(np.real(h_high)))
max_imag_high = np.max(np.abs(np.imag(h_high)))

{
    "low_pass": {
        "is_real": is_real_low,
        "max_real": max_real_low,
        "max_imag": max_imag_low
    },
    "high_pass": {
        "is_real": is_real_high,
        "max_real": max_real_high,
        "max_imag": max_imag_high
    }
}






# ii



# Check if low-pass impulse response is real
print("Low-pass IR is real:", np.isreal(h_low).all())
print("High-pass IR is real:", np.isreal(h_high).all())

# If not real, print max real and imaginary magnitudes
if not np.isreal(h_low).all():
    print("Low-pass IR – max real:", np.max(np.abs(np.real(h_low))))
    print("Low-pass IR – max imag:", np.max(np.abs(np.imag(h_low))))

if not np.isreal(h_high).all():
    print("High-pass IR – max real:", np.max(np.abs(np.real(h_high))))
    print("High-pass IR – max imag:", np.max(np.abs(np.imag(h_high))))




#  iii

# Truncate to multiple of nfft
nfft = 1024
length = (len(x) // nfft) * nfft
x = x[:length].astype(float)
y_l = y_l[:length].astype(float)
y_h = y_h[:length].astype(float)

x_segs = x.reshape(-1, nfft)
y_l_segs = y_l.reshape(-1, nfft)
y_h_segs = y_h.reshape(-1, nfft)

# H1 estimation
def compute_h1_frf(x_segs, y_segs, nfft):
    Sxy_sum, Sxx_sum = 0, 0
    for xi, yi in zip(x_segs, y_segs):
        X = np.fft.fft(xi, n=nfft)[:nfft//2 + 1]
        Y = np.fft.fft(yi, n=nfft)[:nfft//2 + 1]
        Sxy_sum += Y * np.conj(X)
        Sxx_sum += X * np.conj(X)
    return Sxy_sum / (Sxx_sum + 1e-10)

H1_low = compute_h1_frf(x_segs, y_l_segs, nfft)
H1_high = compute_h1_frf(x_segs, y_h_segs, nfft)

# Full spectrum reconstruction
def full_spectrum(H_half):
    return np.concatenate([H_half, np.conj(H_half[-2:0:-1])])

H1_low_full = full_spectrum(H1_low)
H1_high_full = full_spectrum(H1_high)

# Inverse FFT to get impulse responses
h_low = np.fft.ifft(H1_low_full)
h_high = np.fft.ifft(H1_high_full)

# Calculate lengths and return info
len_low = len(h_low)
len_high = len(h_high)
freq_vector_length = nfft // 2 + 1
frf_vector_length = len(H1_low)

(len_low, freq_vector_length, frf_vector_length), (len_high, freq_vector_length, frf_vector_length)







# iv 

# 36) Plotting the full for low pass filter
plt.figure(figsize=(10, 4))
plt.stem(np.real(h_low), linefmt='red', markerfmt='ro', basefmt='black')
plt.title('Impulse Response – Low-pass Filter (Full Length)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()


# 37) Plotting the first 200 samples for low pass filter 
plt.figure(figsize=(10, 4))
plt.stem(np.real(h_low[:200]), linefmt='red', markerfmt='ro', basefmt='black')
plt.title('Impulse Response – Low-pass Filter (First 200 Samples)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()


# 38) Plotting the full for high pass filte

plt.figure(figsize=(10, 4))
plt.stem(np.real(h_high), linefmt='blue', markerfmt='bo', basefmt='black')
plt.title('Impulse Response – High-pass Filter (Full Length)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# 39) Plotting the first 200 samples for high pass filter 

plt.figure(figsize=(10, 4))
plt.stem(np.real(h_high[:200]), linefmt='blue', markerfmt='bo', basefmt='black')
plt.title('Impulse Response – High-pass Filter (First 200 Samples)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show() 


# v 

#written answer on the document


'''


'''



# part e 

# i 

# 40) Define local paths 
base_path = "C:/Users/User/Documents/ASP Coursework/Question 3/"

# 41) Define file references and target frequencies
signals = {
    "100 Hz": {
        "input": "x_sin100.wav",
        "low": "y_l_sin100.wav",
        "high": "y_h_sin100.wav",
        "freq": 100
    },
    "3 kHz": {
        "input": "x_sin3k.wav",
        "low": "y_l_sin3k.wav",
        "high": "y_h_sin3k.wav",
        "freq": 3000
    },
    "7 kHz": {
        "input": "x_sin7k.wav",
        "low": "y_l_sin7k.wav",
        "high": "y_h_sin7k.wav",
        "freq": 7000
    }
}

#  42) Store the results
frequencies = []
rms_ratios_db_low = []
rms_ratios_db_high = []

#  43) Compute RMS and dB ratios
for label, files in signals.items():
    fs, x = wavfile.read(base_path + files["input"])
    _, y_l = wavfile.read(base_path + files["low"])
    _, y_h = wavfile.read(base_path + files["high"])

    # 44) Convert values into float and then align lengths
    min_len = min(len(x), len(y_l), len(y_h))
    x = x[:min_len].astype(float)
    y_l = y_l[:min_len].astype(float)
    y_h = y_h[:min_len].astype(float)

    # 45) RMS calculations
    rms_x = np.sqrt(np.mean(x**2))
    rms_l = np.sqrt(np.mean(y_l**2))
    rms_h = np.sqrt(np.mean(y_h**2))

    # 46) Convert them to dB
    db_l = 20 * np.log10(rms_l / rms_x)
    db_h = 20 * np.log10(rms_h / rms_x)

    frequencies.append(files["freq"])
    rms_ratios_db_low.append(db_l)
    rms_ratios_db_high.append(db_h)

# 47) Print all of the results
for f, db_l, db_h in zip(frequencies, rms_ratios_db_low, rms_ratios_db_high):
    print(f"{f} Hz:")
    print(f"  Low-pass output/input ratio: {db_l:.2f} dB")
    print(f"  High-pass output/input ratio: {db_h:.2f} dB\n")



# 48) Function to compute H1 FRF
def compute_h1(x_segs, y_segs, nfft):
    Sxy_sum = 0
    Sxx_sum = 0
    for xi, yi in zip(x_segs, y_segs):
        X = np.fft.fft(xi, n=nfft)[:nfft//2 + 1]
        Y = np.fft.fft(yi, n=nfft)[:nfft//2 + 1]
        Sxy_sum += Y * np.conj(X)
        Sxx_sum += X * np.conj(X)
    return Sxy_sum / (Sxx_sum + 1e-10)

# 49) Compte FRFs and convert into dB
H_low = compute_h1(x_segs, y_l_segs, nfft)
H_high = compute_h1(x_segs, y_h_segs, nfft)
freq = np.linspace(0, fs / 2, nfft // 2 + 1)
H_low_db = 20 * np.log10(np.abs(H_low))
H_high_db = 20 * np.log10(np.abs(H_high))

# 50) Substitute in the measured RMS dB values from e(i)
sin_freqs = [100, 3000, 7000]
rms_db_low = [-0.02, -10.32, -23.54]
rms_db_high = [-52.14, -3.16, -0.60]

# 51) Plot the FRFs and RMS points
plt.figure(figsize=(10, 5))
plt.plot(freq, H_low_db, label='FRF Low-pass', color='red')
plt.plot(freq, H_high_db, label='FRF High-pass', color='blue')

# 52) Overlay the RMS ratiso
plt.scatter(sin_freqs, rms_db_low, color='darkred', marker='o', label='Measured RMS (Low-pass)')
plt.scatter(sin_freqs, rms_db_high, color='darkblue', marker='x', label='Measured RMS (High-pass)')


# 53) Plot the graph with the axis titles and title. 
plt.title("FRFs with Measured Sinusoidal RMS Response Overlay")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
















