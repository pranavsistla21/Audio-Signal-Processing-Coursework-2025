#part a
# i 

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


# 1) Load in the signals (I used throwaway variables here as they aren’t needed anywhere else)

fs, x = wavfile.read('C:/Users/User/Documents/ASP Coursework/Question 3/x_noise.wav')
_, y_l = wavfile.read('C:/Users/User/Documents/ASP Coursework/Question 3/y_l_noise.wav')
_, y_h = wavfile.read('C:/Users/User/Documents/ASP Coursework/Question 3/y_h_noise.wav')


# 2) Ensure the signals are the same length, so you can do the coherence calculation

nfft = 1024
length = (len(x) // nfft) * nfft
x = x[:length]
y_l = y_l[:length]
y_h = y_h[:length]


# 3) Reshape the signals into segments 
x_seg = x.reshape(-1, nfft)
y_l_seg = y_l.reshape(-1, nfft)
y_h_seg = y_h.reshape(-1, nfft)


# 4) Define the coherence function (using FFT)

def compute_coherence(x_seg, y_seg, fs, nfft):
    Pxy_avg = 0
    Pxx_avg = 0
    Pyy_avg = 0
    for xi, yi in zip(x_seg, y_seg):
        X = np.fft.fft(xi, n=nfft)[:nfft//2 + 1]
        Y = np.fft.fft(yi, n=nfft)[:nfft//2 + 1]
        Pxx = np.abs(X) ** 2
        Pyy = np.abs(Y) ** 2
        Pxy = X * np.conj(Y)
        Pxy_avg += Pxy
        Pxx_avg += Pxx
        Pyy_avg += Pyy
    Pxy_avg /= len(x_seg)
    Pxx_avg /= len(x_seg)
    Pyy_avg /= len(y_seg)
    coherence = np.abs(Pxy_avg)**2 / (Pxx_avg * Pyy_avg + 1e-10)
    freq = np.linspace(0, fs / 2, nfft // 2 + 1)
    return freq, coherence

# 5) Run the functions with the segments we have created 

f, coherence_low = compute_coherence(x_seg, y_l_seg, fs, nfft)
_, coherence_high = compute_coherence(x_seg, y_h_seg, fs, nfft)


# 6) Plot the frequency vector spacing

plt.figure(figsize=(10, 5))
plt.plot(f, coherence_low, label='Low-pass Output', color='red')
plt.plot(f, coherence_high, label='High-pass Output', color='blue')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence γ²(f)')
plt.title('Coherence between x[n] and y_l[n], y_h[n]')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()




# ii

# 7) input the known values 

fs = 48000
delta_f = 46.875

# 8) Calculate the number of FFT points using the formula and find the frequency vector 

nfft = int(fs / delta_f)
freq_vector = np.linspace(0, fs / 2, nfft // 2 + 1)

# 9) Print all the information that has been solved.
print("FFT points:", nfft)
print("Length of frequency vector:", len(freq_vector))
print("First 10 frequency values (Hz):", freq_vector[:10])



# iii

# Written answer that I have put on the document. 


# iv 


# Set FFT length and segment length
nfft = 1024
length = (len(x) // nfft) * nfft

# Truncate to a multiple of nfft and reshape
x = x[:length].astype(float)
y_l = y_l[:length].astype(float)
y_h = y_h[:length].astype(float)

x_segs = x.reshape(-1, nfft)
y_l_segs = y_l.reshape(-1, nfft)
y_h_segs = y_h.reshape(-1, nfft)

# Define function to compute coherence
def compute_coherence(x_segs, y_segs, fs, nfft):
    Pxy_sum, Pxx_sum, Pyy_sum = 0, 0, 0
    for xi, yi in zip(x_segs, y_segs):
        X = np.fft.fft(xi, n=nfft)[:nfft//2+1]
        Y = np.fft.fft(yi, n=nfft)[:nfft//2+1]
        Pxy_sum += Y * np.conj(X)
        Pxx_sum += X * np.conj(X)
        Pyy_sum += Y * np.conj(Y)
    Pxy_avg = Pxy_sum / len(x_segs)
    Pxx_avg = Pxx_sum / len(x_segs)
    Pyy_avg = Pyy_sum / len(x_segs)
    coherence = np.abs(Pxy_avg)**2 / (Pxx_avg * Pyy_avg + 1e-10)
    freq = np.linspace(0, fs / 2, nfft // 2 + 1)
    return freq, coherence

# Compute coherence for both outputs
f, coh_low = compute_coherence(x_segs, y_l_segs, fs, nfft)
_, coh_high = compute_coherence(x_segs, y_h_segs, fs, nfft)

# Plot the coherence functions
plt.figure(figsize=(10, 5))
plt.plot(f, coh_low, color='red', label='Coherence: x_noise vs y_l_noise')
plt.plot(f, coh_high, color='blue', label='Coherence: x_noise vs y_h_noise')
plt.title('Magnitude-Squared Coherence for White Noise Input')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence γ²(f)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# v 





'''



'''

# part b

# i 

# Note all the same values from earlier (a (ii)) are still carrying on. 


# 10) Create a H1 estimator function
def compute_h1(x_seg, y_seg, fs, nfft):
    
    # 11) INitialise 'average' variables for power spectral density (Sxx) and  Cross spectral density (Sxy) )
    
    Sxy_avg = 0
    Sxx_avg = 0
    
    # 12) Create a loop that goes through each segment and computes the FFTs, the spectral quantities and then accumulates the total power inputs both for input and cross
    
    for xi, yi in zip(x_seg, y_seg):
        X = np.fft.fft(xi, n= nfft)[:nfft//2 + 1]
        Y = np.fft.fft(yi, n= nfft)[:nfft//2 + 1]
        Sxx = X * np.conj(X)
        Sxy = Y * np.conj(X)
        Sxy_avg += Sxy
        Sxx_avg += Sxx
    Sxy_avg = Sxy_avg/ len(x_seg)
    Sxx_avg = Sxx_avg/ len(x_seg)
    
    # 13) Compute the H1 estimate 
    H1 = Sxy_avg / (Sxx_avg + 1e-10)
    
    #14) generate the frequency vector 
    freq = np.linspace(0, fs / 2, nfft // 2 + 1)
    
    
    print ("Frequency is", freq)
    
    return freq, H1

# 15) Calculate the  FRFs as asked, using our function 

f, H1_low = compute_h1(x_seg, y_l_seg, fs, nfft)
_, H1_high = compute_h1(x_seg, y_h_seg, fs, nfft)






#  ii

# 16) Plot the magnitude responses in dB
plt.figure(figsize=(10, 5))
plt.plot(f, 20 * np.log10(np.abs(H1_low)), label='Low-pass Output (mag)', color='red')
plt.plot(f, 20 * np.log10(np.abs(H1_high)), label='High-pass Output (mag)', color='blue')
plt.axhline(-6, color='gray', linestyle='--', label='-6 dB')
plt.title('FRF Magnitude Response (H1)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 17) Plot the phase responses
plt.figure(figsize=(10, 5))
plt.plot(f, np.angle(H1_low, deg=True), label='Low-pass Output (phase)', color='red')
plt.plot(f, np.angle(H1_high, deg=True), label='High-pass Output (phase)', color='blue')
plt.title('FRF Phase Response (H1)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()




#  iii







#  iv

# 18) Convert the magnitude itno dB
mag_lp = 20 * np.log10(np.abs(H1_low))
mag_hp = 20 * np.log10(np.abs(H1_high))

# 19) Find how close the FRFs to -6 dB
diff_lp = np.abs(mag_lp + 6)
diff_hp = np.abs(mag_hp + 6)

# 20) add the differences that we have found
combined = diff_lp + diff_hp

# 21) Find where the %error is the smallest using argmin from the numpy library.
idx = np.argmin(combined)


#22) Find the cross over ffrequency. 
crossover_freq = f[idx]
print("Cross-over frequency (where both FRFs ≈ -6 dB):", crossover_freq, "Hz")



'''


'''





# part c

# i 


# The varibales just roll over from the initial decleration. 

# 23) Load input and output signals for sin3k 
fs, x_sin3k = wavfile.read('C:/Users/User/Documents/ASP Coursework/Question 3/x_sin3k.wav')
_, y_l_sin3k = wavfile.read('C:/Users/User/Documents/ASP Coursework/Question 3/y_l_sin3k.wav')

# 24) Truncate to the same length. This process is done earlier in part a 
min_len = min(len(x_sin3k), len(y_l_sin3k))
x_sin3k = x_sin3k[:min_len]
y_l_sin3k = y_l_sin3k[:min_len]

# 25) Create a function to estimate delay using cross-correlation
def estimate_delay(x_sin3k, y_l_sin3k, fs):
    corr = np.correlate(y_l_sin3k, x_sin3k, mode='full')  # Cross-correlation
    lags = np.arange(-len(x) + 1, len(x))   # Lag vector
    max_idx = np.argmax(corr)               # Index of max correlation
    lag_samples = lags[max_idx]             # Delay in samples
    delay_seconds = lag_samples / fs        # Convert to seconds
    return lag_samples, delay_seconds, corr, lags

# 26) Run the function to calculate a delay estimation
lag_samples, delay_seconds, corr, lags = estimate_delay(x_sin3k, y_l_sin3k, fs)



# ii


'''
 NOTE: I was unable to mix the variables and did not want to work with global 
       variables so I made new ones (where appropriate). The following code will have
       some different variable names but it is all made clear on the code raw file.
       I hope that makes sense. the next few comments are just the same job as what I 
       completed in the function.

'''


fs, x = wavfile.read('C:/Users/User/Documents/ASP Coursework/Question 3/x_sin3k.wav')
_, y = wavfile.read('C:/Users/User/Documents/ASP Coursework/Question 3/y_l_sin3k.wav')

#  Truncate both signals to the same length
min_len = min(len(x), len(y))
x = x[:min_len]
y = y[:min_len]

# Estimate delay using cross-correlation
corr = np.correlate(y, x, mode='full')
lags = np.arange(-len(x) + 1, len(x))
max_idx = np.argmax(corr)
lag_samples = lags[max_idx]
delay_seconds = lag_samples / fs


# 27) Plot the cross-correlation function with the delay marked

plt.figure(figsize=(10, 5))
plt.plot(lags, corr, color='red')
plt.axvline(lag_samples, color='red', linestyle='--', label=f'Delay = {lag_samples} samples')
plt.title('Cross-Correlation Between Input and Output')
plt.xlabel('Lag (samples)')
plt.ylabel('Correlation')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# 28)  Print the estimated delay
print("Estimated delay:", lag_samples, "samples")
print("Estimated delay time:", round(delay_seconds * 1000, 3), "ms")




'''

CONTINUED ON ANOTHER FILE 'Question 3 Part 2'

Issues began to compound because of the clutter of using just this one file. 

'''




