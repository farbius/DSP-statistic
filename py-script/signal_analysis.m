clc
clear
close all


N           = 8192;
FFTn_dB     = 10*log10(N);
fprintf('FFT noise floor is %3.2f dB \n', 10*log10(N));
Fs          = 50e6;
ts          = 1 / Fs;
t_axis      = (0:N-1)*1/Fs;
f_axis      = (0:N-1)*Fs/N;
SNR_dB      = -20;
f0          = 12.5e6;
s           = exp(2*1i*pi*f0*t_axis);
n           = 10^(-SNR_dB/20)*(randn(1, N) + 1i.*randn(1, N));
u           = s + n;
uF          = fft(u)./N;


ph_raw      = unwrap(angle(u));
fr_raw      = 1 / 2 / pi * diff(ph_raw)   * Fs;

fr_raw_m    = fr_raw - mean(fr_raw);




figure
sgtitle('Signal Analysis: Time / Freq Domain') 

subplot(2, 2, 1)
plot(t_axis./1e-3, real(u), '.-b',t_axis./1e-3, imag(u), '.-r')
title('IQ components')
ylabel('Amplitude')
xlabel('t, ms')
grid on

subplot(2, 2, 3)
plot(f_axis./1e6, 20*log10(abs(uF)), '.-', f_axis./1e6, ones(1, N).*(-FFTn_dB-SNR_dB), '--r')
title('Signal Normalized Spectrum')
ylabel('Magnitude, dB')
xlabel('freq, MHz')
legend('Spectrum', ['Noise Floor, SNR is '  num2str(SNR_dB) ' dB'])
grid on

subplot(2, 2, 2)
plot(t_axis./1e-3, ph_raw, '.-')
title('Unwrapped Phase')
xlabel('t, ms')
ylabel('rad')
grid on

subplot(2, 2, 4)
plot(t_axis(1:end-1)./1e-3, fr_raw./1e6, '.-', t_axis(1:end-1)./1e-3, ones(1, N-1).*f0/1e6, '--r')
title('Instantenious Frequency')
ylabel('freq, MHz')
xlabel('t, ms')
legend('Inst freq', 'Carrier Freq')
grid on

scaling     =  0.05;
dphi_t      = (2*pi*ts).*cumsum(fr_raw_m, 2);
dphi_t      = unwrap(dphi_t, [], 2)*scaling;

% modulating phase_error on a 0Hz-carrier
xt          = exp(-1j.*dphi_t);
mywin       = hann(size(xt,2));

mywin_RMS_dB = -20.*log10(sqrt(mean(mywin.^2)));
% xt      = bsxfun(@times,mywin',xt);
N       = length(xt);
T_sequ  = N * ts;

% Voltage of single sample attenuated by window
Xf      = (1/N).*fft(xt,size(xt,2)*2,2);
Xf_dB   = 20.*log10(abs(Xf));
df      = 1/T_sequ/2;
f_bins  = 0:df:(df*(size(Xf_dB,2)-1));

PN_dBcHz= Xf_dB - 10.*log10(1/T_sequ); %  + mywin_RMS_dB;
Xf_lin  = 10.^(PN_dBcHz(:,:)./10);
% averaging the phasenoise of the single chirps
Xf      = 10.*log10(Xf_lin);


lin_Coef = polyfit(f_bins(1:end/2), Xf(1:end/2), 6);
Xfap = polyval(lin_Coef, f_bins(1:end/2));




figure
sgtitle('Signal Analysis: Statistic') 

subplot(2, 2, 1)
histogram(real(n),'Normalization','probability')
title(['Noise PDF, sigma is '  num2str(10^(-SNR_dB/20))])
grid on


subplot(2, 2, 2)
histogram(abs(n),'Normalization','probability')
title('Amplitude of Noise PDF')
grid on

subplot(2, 2, 3)
histogram(angle(n),'Normalization','probability')
title('Phase of Noise PDF')
grid on

subplot(2, 2, 4)
semilogx(f_bins(1:end/2),Xf(1:end/2), '.-r');
hold on
semilogx(f_bins(1:end/2),Xfap, '--b');
hold off
title('Phase noise estimation')
ylabel('Phasenoise [dBc/Hz]')
xlabel('df from carrier [Hz]')
grid on


