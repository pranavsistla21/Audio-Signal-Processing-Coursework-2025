# part a



# 1) import required modules 

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram, get_window


# 2)	Declare the file path, so the correct .wav file is open 

file_path = 'C:/Users/User/Documents/ASP Coursework/Question 1/1234567890_py.wav'
sample_rate, data = wav.read(file_path)


# 3)	Declare variables for the sample rate, the duration (total time of the signal)  and the  time axis so that they can be plotted 

duration = len(data) / sample_rate
time = np.linspace(0, duration, len(data)) 
# 4)	Plot the graph setting the title, the axis headings and picking the colour.  

plt.figure(figsize=(12, 4))
plt.plot(time, data)
plt.plot(time, data, color='red')
plt.title("Whole Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True) 
plt.tight_layout()
plt.show()


# 5)	Set a start time and an end time. 

start_time = 0.0
end_time = 0.1

# 6)	Use indexes to declare the shortened section

start_index = int(start_time * sample_rate)
end_index = int(end_time * sample_rate)
zoom_segment = data[start_index:end_index]
zoom_time = time[start_index:end_index]

# 7)	Plot the section with the same axis titles and colour as the previous graph. 

plt.figure(figsize=(12, 4))
plt.plot(zoom_time, zoom_segment, color='red')
plt.title("Zoomed-in Signal to The First Tone (0.0s to 0.1s)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

'''


'''


# part b 
#           (i)

# 8) Normalise the data

if data.dtype != np.float32:
    data = data / np.max(np.abs(data))

# 9) Get first 0.5s segment (for tone '1')
segment_duration = 0.5  # seconds
segment_samples = int(segment_duration * sample_rate)
segment = data[:segment_samples]

#  10) Compute and setting up the DFT plot without padding 
N = len(segment)
freqs = fftfreq(N, d=1/sample_rate)
dft = fft(segment)
magnitude = np.abs(dft)
positive_freqs = freqs[:N//2]
positive_magnitude = magnitude[:N//2]

# 11) Convert to dB
epsilon = 1e-10
magnitude_db = 20 * np.log10(positive_magnitude + epsilon)

# 12) plotting the magnitude vs frequenc graph (not padded)

plt.figure(figsize=(12, 4))
plt.plot(positive_freqs, magnitude_db, color='red')
plt.title("DFT Magnitude Spectrum (0.5s Segment, No Padding)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True)
plt.tight_layout()
plt.show()

#  13) Identify dominant frequencies (peaks above 50% of max)
threshold = np.max(positive_magnitude) * 0.5
dominant_freqs_original = positive_freqs[positive_magnitude > threshold]

print(dominant_freqs_original)



# part (b)(ii)

#  14) padding until the segment has a length of 1 second

padded_length = sample_rate  # 1 sec = 44100 samples
segment_padded = np.zeros(padded_length)
segment_padded[:N] = segment

# 15)   Computing the  DFT
dft_padded = fft(segment_padded)
freqs_padded = fftfreq(padded_length, d=1/sample_rate)
magnitude_padded = np.abs(dft_padded)
positive_freqs_padded = freqs_padded[:padded_length//2]
positive_magnitude_padded = magnitude_padded[:padded_length//2]
magnitude_padded_db = 20 * np.log10(positive_magnitude_padded + epsilon)


# 16)  Plotting the new graph for the padded segment

plt.figure(figsize=(12, 4))
plt.plot(positive_freqs_padded, magnitude_padded_db, color='red')
plt.title("DFT Magnitude Spectrum (Zero-Padded to 1s)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True)
plt.tight_layout()
plt.show()

#  17)  To identify the  dominant frequencies in new zero-padded version
threshold_padded = np.max(positive_magnitude_padded) * 0.5
dominant_freqs_padded = positive_freqs_padded[positive_magnitude_padded > threshold_padded]


# 18) Print the dominant frequencies before and after padding.

print(dominant_freqs_padded)


'''



'''

# part c

# 19) Normalising the data
if data.dtype != np.float32:
    data = data / np.max(np.abs(data))

# 20) Declaring the window and overlap parameters
window_duration = 0.1  # 100 ms
window_size = int(window_duration * sample_rate)
window = get_window('hann', window_size)
overlap = window_size // 2

# 21) Ensuring the nfft >= window_size to the next power of 2 
nfft = 2**int(np.ceil(np.log2(window_size)))

# 22) Comptue the  spectrogram
frequencies, times, Sxx = spectrogram(
    data,
    fs=sample_rate,
    window=window,
    nperseg=window_size,
    noverlap=overlap,
    nfft=nfft,
    scaling='density',
    mode='magnitude'
)

# 23) Convert magnitude to dB
epsilon = 1e-10
Sxx_db = 20 * np.log10(Sxx + epsilon)

# 24) Plot spectrogram as required with the axis titles and title
plt.figure(figsize=(14, 6))
plt.pcolormesh(times, frequencies, Sxx_db, shading='gouraud', cmap='Reds')
plt.title("Spectrogram (100 ms Hann Window, 50% Overlap)")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.colorbar(label='Magnitude [dB]')
plt.ylim([0, 2000])  
plt.tight_layout()
plt.show()



'''


'''

# 25) create a function that takes the window size (in ms) as an input and also sets a colour. 

def plot_spectrogram(window_ms, color_map='Reds'):
    window_duration = window_ms / 1000  # convert to seconds
    window_size = int(window_duration * sample_rate)
    
    # 26) set the inital conditions 
    
    overlap = window_size // 2
    nfft = 2**int(np.ceil(np.log2(window_size)))  # ensure nfft >= window_size
    window = get_window('hann', window_size)

    # 27) Compute the spectrogram amd declare the required value
    freqs, times, Sxx = spectrogram(
        data,
        fs=sample_rate,
        window=window,
        nperseg=window_size,
        noverlap=overlap,
        nfft=nfft,
        scaling='density',
        mode='magnitude'
    )

    # 28) Convert to dB
    epsilon = 1e-10
    Sxx_db = 20 * np.log10(Sxx + epsilon)

    # 29) configure the plot with the axis and the title, as required. 
    plt.figure(figsize=(14, 6))
    plt.pcolormesh(times, freqs, Sxx_db, shading='gouraud', cmap=color_map)
    plt.title(f"Spectrogram ({window_ms} ms Hann Window, 50% Overlap)")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.ylim([0, 2000])  # Focus on DTMF frequency range
    plt.colorbar(label='Magnitude [dB]')
    plt.tight_layout()
    plt.show()

# 30) part a of the question is to plot for 50 ms, so call upon the function with window_ms = 50

plot_spectrogram(50)

# 31) part b of the question is to plot for 200 ms, so call upon the function with window_ms = 50

plot_spectrogram(200)



'''



'''


# part e


# 32) Define the sample rate and duration
sample_rate = 44100
tone_duration = 1.0  # 1 second per tone
t = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)

# 33) DTMF frequency mapping
dtmf_freqs = {
    '3': (697, 1477),
    '0': (941, 1336),
    '4': (770, 1209)
}

# 34) Synthesise the three tones
signal = np.concatenate([
    np.sin(2 * np.pi * dtmf_freqs[d][0] * t) + np.sin(2 * np.pi * dtmf_freqs[d][1] * t)
    for d in ['3', '0', '4']
])

# 35) Normalise the signal
signal = signal / np.max(np.abs(signal))

# 36) Create the spectrogram amd set up the window size.
window_ms = 100
window_size = int(window_ms / 1000 * sample_rate)
overlap = window_size // 2
nfft = 2**int(np.ceil(np.log2(window_size)))
window = get_window('hann', window_size)

# 37) Compute spectrogram, declaring all the required values. 
freqs, times, Sxx = spectrogram(
    signal,
    fs=sample_rate,
    window=window,
    nperseg=window_size,
    noverlap=overlap,
    nfft=nfft,
    scaling='density',
    mode='magnitude'
)

# 38) Convert the values intoto dB
epsilon = 1e-10
Sxx_db = 20 * np.log10(Sxx + epsilon)

# 39)	Plot the spectrogram with the required axis and title. 
plt.figure(figsize=(14, 6))
plt.pcolormesh(times, freqs, Sxx_db, shading='gouraud', cmap='Reds')
plt.title("Spectrogram of Synthesized Signal [3, 0, 4]")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.colorbar(label="Magnitude [dB]")
plt.ylim([0, 2000])
plt.tight_layout()
plt.show()








