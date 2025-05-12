import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch
from scipy.signal import get_window

# part a 
# i

# 1) Loading the audio files
fs_y1, y1 = wavfile.read("C:/Users/User/Documents/ASP Coursework/Question 2/y1.wav")
fs_y2, y2 = wavfile.read("C:/Users/User/Documents/ASP Coursework/Question 2/y2.wav")

# 2) Checking if sampling frequencies match and setting the duration variables 
duration_y1 = len(y1) / fs_y1
duration_y2 = len(y2) / fs_y2

print(fs_y1, fs_y2, duration_y1, duration_y2)


# 3) Defining the Welch method parameters
window_duration = 0.0107  # in seconds
nperseg = int(fs_y1 * window_duration)  # number of samples per segment
noverlap = nperseg // 2
window = get_window("hann", nperseg)

#  4) Extract middle portion as signal (1s to 4.8s)
start_sample = int(1 * fs_y1)
end_sample = int(4.8 * fs_y1)
y1_signal = y1[start_sample:end_sample]
y2_signal = y2[start_sample:end_sample]

# 5)  Compute PSD using Welch's method
f1, Pxx_y1 = welch(y1_signal, fs=fs_y1, window=window, noverlap=noverlap, nperseg=nperseg)
f2, Pxx_y2 = welch(y2_signal, fs=fs_y2, window=window, noverlap=noverlap, nperseg=nperseg)



# 6) Convert PSDs to dB scale
Pxx_y1_dB = 10 * np.log10(Pxx_y1)
Pxx_y2_dB = 10 * np.log10(Pxx_y2)

# 7) Plotting the various graphs, changing parameters when required
fig, axs = plt.subplots(2, 2, figsize=(12,10))

#Linear frequency, linear scale
axs[0, 0].plot(f1, Pxx_y1, label='Mic 1', color='red')
axs[0, 0].plot(f2, Pxx_y2, label='Mic 2', color='blue')
axs[0, 0].set_title('PSD - Linear Frequency (Hz), Linear Scale')
axs[0, 0].set_xlabel('Frequency (Hz)')
axs[0, 0].set_ylabel('Power/Frequency (V^2/Hz)')
axs[0, 0].legend()

#Linear frequency, dB scale
axs[0, 1].plot(f1, Pxx_y1_dB, label='Mic 1', color='red')
axs[0, 1].plot(f2, Pxx_y2_dB, label='Mic 2', color='blue')
axs[0, 1].set_title('PSD - Linear Frequency (Hz), dB Scale')
axs[0, 1].set_xlabel('Frequency (Hz)')
axs[0, 1].set_ylabel('Power/Frequency (dB)')
axs[0, 1].legend()

#Log frequency, linear scale
axs[1, 0].semilogx(f1, Pxx_y1, label='Mic 1', color='red')
axs[1, 0].semilogx(f2, Pxx_y2, label='Mic 2', color='blue')
axs[1, 0].set_title('PSD - Log Frequency (Hz), Linear Scale')
axs[1, 0].set_xlabel('Frequency (Hz)')
axs[1, 0].set_ylabel('Power/Frequency (V^2/Hz)')
axs[1, 0].legend()

#Log frequency, dB scale
axs[1, 1].semilogx(f1, Pxx_y1_dB, label='Mic 1', color='red')
axs[1, 1].semilogx(f2, Pxx_y2_dB, label='Mic 2',color='blue')
axs[1, 1].set_title('PSD - Log Frequency (Hz), dB Scale')
axs[1, 1].set_xlabel('Frequency (Hz)')
axs[1, 1].set_ylabel('Power/Frequency (dB)')
axs[1, 1].legend()

plt.tight_layout()
plt.show()



# ii




# 8) import correlate and filtfilt

from scipy.signal import correlate, filtfilt

# 9) Extract the last second (nosie)

noise_y1 = y1[-fs_y1:]
noise_y2 = y2[-fs_y2:]

# 10) Compute PSD of the noise segments to estimate bandwidth

f_noise_y1, Pxx_noise_y1 = welch(noise_y1, fs=fs_y1, window=window, noverlap=noverlap, nperseg=nperseg)
f_noise_y2, Pxx_noise_y2 = welch(noise_y2, fs=fs_y2, window=window, noverlap=noverlap, nperseg=nperseg)

# 11) Plot PSD of the noise to identify its frequency band

plt.figure(figsize=(10, 5))
plt.semilogy(f_noise_y1, Pxx_noise_y1, label='Noise y1', color='red')
plt.semilogy(f_noise_y2, Pxx_noise_y2, label='Noise y2', color='blue')
plt.title('PSD of Noise Segments (Last 1s of y1 and y2)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (V^2/Hz)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

'''



'''


# part b 


from scipy.signal import butter

# 12) Define the Welch parameters for the PSD calculation
window_duration = 0.0107  # seconds
nperseg = int(fs_y1 * window_duration)
noverlap = nperseg // 2
window = get_window("hann", nperseg)

# 13) Extract last 1 second as noise-only segment
noise_y1 = y1[-fs_y1:]
noise_y2 = y2[-fs_y2:]

# 14( Compute PSD to estimate noise bandwidth
f_noise_y1, Pxx_noise_y1 = welch(noise_y1, fs=fs_y1, window=window, noverlap=noverlap, nperseg=nperseg)

# 15) Applying the Appendix 2 filter parameters
f1 = 1000   # Lower cutoff frequency in Hz
f2 = 2500   # Upper cutoff frequency in Hz
N = 6       # Filter order

# 16) Designing the  band-stop filter using scipy.signal.butter with the Appendix 2 parameters
b, a = butter(N, np.array([f1, f2]), btype='bandstop', fs=fs_y1)

# 17) Apply the filter to both y1 and y2 using scipy.signal.filtfilt
y1_filtered = filtfilt(b, a, y1)
y2_filtered = filtfilt(b, a, y2)

# 18) Extract middle portion (1s to 4.8s) from filtered signals
start_sample = int(1 * fs_y1)
end_sample = int(4.8 * fs_y1)
y1f_mid = y1_filtered[start_sample:end_sample]
y2f_mid = y2_filtered[start_sample:end_sample]

# 19) Compute PSD of filtered mid-segments
f1f, Pxx_y1f = welch(y1f_mid, fs=fs_y1, window=window, noverlap=noverlap, nperseg=nperseg)
f2f, Pxx_y2f = welch(y2f_mid, fs=fs_y2, window=window, noverlap=noverlap, nperseg=nperseg)

# 20) Plot the PSD with the filtered band.
plt.figure(figsize=(10, 6))
plt.semilogy(f1f, Pxx_y1f, label='Filtered y1', color='red')   # Red line for y1
plt.semilogy(f2f, Pxx_y2f, label='Filtered y2', color='blue')  # Blue line for y2
plt.axvspan(f1, f2, color='red', alpha=0.2, label='Filtered Band')
plt.title('PSD of Filtered Signals (1s–4.8s) Using Butterworth Band-Stop Filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (V²/Hz)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



'''



'''


# part c 


#21) make a function that takes in all the inputs

def estimate_delay(y1, y2, fs, d=2, c=344, f1=1000, f2=2500, N=6):
    

    # 22) Filter the signals using the Butterworth band-stop filter (made easy by the scipy.signal.butter module 
    b, a = butter(N, [f1, f2], btype='bandstop', fs=fs)
    y1_filtered = filtfilt(b, a, y1)
    y2_filtered = filtfilt(b, a, y2)

    # 23) Select the middle segment to avoid the effects of noise (1s to 4.8s)
    start = int(1 * fs)
    end = int(4.8 * fs)
    y1f = y1_filtered[start:end]
    y2f = y2_filtered[start:end]

    # 24) Use scipy.signal.correlate() to compute the cross correlation between the filtered signals and then np.arrange() generates the lag values (in samples)  
    R12 = correlate(y1f, y2f)
    lags = np.arange(-len(y1f) + 1, len(y1f))

    # 25) To find the sample lag we identify the peak of the cross-correlation, this then needs to be converted into seconds. This value is our Time Delay Estimate. 
    lag_samples = lags[np.argmax(R12)]
    delta_t = lag_samples / fs

    # 26)  Use the formula to find theta, which is our estimate for the Angle of Arrival
    arg = np.clip((c * delta_t) / d, -1.0, 1.0)
    theta_rad = np.arcsin(arg)
    theta_deg = np.degrees(theta_rad)
    
    
    print("Estimated delay time (s) =", delta_t )
    print("Lag (measured in samples)",lag_samples)
    print("Angle of Arrival (degrees)", theta_deg)
    
  

    
    return delta_t, lag_samples, theta_deg, lags, R12





'''
# I was unable to mix the variables and did not want to work with global variables so I made new ones (where appropriate). I hope that this is still clear. 
# I kept the numbers the same though. I hope that makes sense. 
'''

# 23) Extract the middle portion (1s to 4.8s) (no noise) )
start_sample = int(1 * fs_y1)
end_sample = int(4.8 * fs_y1)
y1f = y1_filtered[start_sample:end_sample]
y2f = y2_filtered[start_sample:end_sample]

# 24) Using the info from Appendix 3 to find the cross-correlation between the two same-length signals
R12 = correlate(y1f, y2f)  # cross-correlation
lags = np.arange(-len(y1f)+1, len(y1f))  # lag axis

# 25) Estimating  the delay between the signals 
lag_samples = lags[np.argmax(R12)]
delta_t = lag_samples / fs_y1

# 26) Solve for the angle
c = 344  # speed of sound in m/s
d = 2    # distance between mics in m
arg = np.clip((c * delta_t) / d, -1.0, 1.0)
theta_rad = np.arcsin(arg)
theta_deg = np.degrees(theta_rad)


# 27) Plotting the graph to dispplay cross-correlation
plt.figure(figsize=(10, 5)) 
plt.plot(lags, R12, color='red', label='Cross-correlation')
plt.axvline(lag_samples, color='blue', linestyle='--', label= 'Maximum Correlation')
plt.title('Cross-Correlation Between Filtered y1 and y2')
plt.xlabel('Lag (samples)')
plt.ylabel('Correlation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



estimate_delay(y1f, y2f, fs_y1 )




'''


'''


#part d








'''


'''


# part e 




# Extract mid segment (1s to 4.8s) from unfiltered signals
start_sample = int(1 * fs_y1)
end_sample = int(4.8 * fs_y1)
y1_unfiltered = y1[start_sample:end_sample]
y2_unfiltered = y2[start_sample:end_sample]

# Function to estimate delay and angle
def estimate_delay_and_angle(sig1, sig2, fs, c=344, d=2):
    corr = correlate(sig1, sig2, mode='full')
    lags = np.arange(-len(sig1) + 1, len(sig1))
    lag_samples = lags[np.argmax(corr)]
    time_delay = lag_samples / fs
    arg = np.clip((c * time_delay) / d, -1.0, 1.0)
    angle_rad = np.arcsin(arg)
    angle_deg = np.degrees(angle_rad)
    return time_delay, lag_samples, angle_deg

# Estimate delay and angle from unfiltered signals
delta_t_unfiltered, lag_unfiltered, theta_unfiltered_deg = estimate_delay_and_angle(y1_unfiltered, y2_unfiltered, fs_y1)

print(delta_t_unfiltered, lag_unfiltered, theta_unfiltered_deg)






















