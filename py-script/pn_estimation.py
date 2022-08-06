from numpy.random import default_rng
import numpy as np
import matplotlib.pyplot as plt

N  = 1024
Fs = 100e6

# mu, sigma = 0, 0.05 # mean and standard deviation
# s = np.random.normal(mu, sigma, N)
# s = np.random.rayleigh(0.4, N)
# s = np.random.uniform(0.0, 1.0, N)
# abs(mu - np.mean(s))
# abs(sigma - np.std(s, ddof=1))

dt     = 1 / Fs
t      = np.linspace(0, 1, N) * dt * N
f      = np.linspace(0, 1, N) * Fs



f1     = 5e6
f0     = 5e6
dF     = f1 - f0
Sr     = dF / (dt*N) / 2
y_0    = np.exp(1j*2*np.pi*(f0 + Sr*t)*t) # k*t + b

mu, sigma = 0, 0.05

noise  = np.random.normal(mu, sigma, N)

y_0   += noise

phi_t  = np.angle(y_0)
p_unwr = np.unwrap(phi_t)
freq_t = np.diff(p_unwr) / dt / (2*np.pi)

# FFT transform
yF     = np.fft.fft(y_0) / N
nF     = np.fft.fft(noise) / N

# plt.figure(1)
# count, bins, ignored = plt.hist(s, 30, density=True)
# # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
# plt.grid()

error = np.abs(5 - np.mean(freq_t)/1e6)
print(error)

plt.figure(num=1, figsize=(8,8))

plt.suptitle('Signal processing Time - Freq Analysis')

plt.subplot(221)
plt.plot(f/1e6, 10*np.log10(np.abs(yF)), ".-b")
plt.title("Signal + Noise Spectrum")
plt.xlabel('Freq, MHz')
plt.ylabel('Magnitude, dB')
plt.grid()

plt.subplot(222)
plt.plot(t/1e-6, p_unwr, ".-b")
plt.title("Instantaneous Phase")
plt.xlabel('Time, usec')
plt.grid()

plt.subplot(223)
plt.plot(t[1:]/1e-6, freq_t / 1e6 , ".-b")
plt.plot(t[1:]/1e-6, np.ones(N-1)*f0 / 1e6 , ".-r")
plt.title("Instantaneous Frequency")
plt.xlabel('Time, usec')
plt.ylabel('Freq, MHz')
plt.ylim([0, Fs/2/1e6])
plt.grid()

plt.subplot(224)
plt.plot(f/1e6, 10*np.log10(np.abs(nF)), ".-b")
plt.title("Noise Spectrum")
plt.xlabel('Freq, MHz')
plt.ylabel('Magnitude, dB')
plt.grid()
plt.tight_layout()




plt.figure(num=2, figsize=(8,8))

plt.suptitle('Signal Processing Statistic')

plt.subplot(211)
plt.hist(freq_t / 1e6, 30, density=True)
plt.title('Freq Histogramm')
plt.xlabel('Freq, MHz')
# plt.plot(np.real(y_0), ".-b")
# plt.plot(np.real(y_1), " .r")
plt.grid()

plt.subplot(212)
plt.hist(noise, 30, density=True)
plt.title('Noise Histogramm')
plt.grid()

plt.tight_layout()
plt.show()