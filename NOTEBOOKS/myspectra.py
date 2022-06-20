import numpy as np
from scipy.interpolate import interp1d 
from scipy import signal

# See here: https://currents.soest.hawaii.edu/ocn_data_analysis/_static/Spectrum.html

def filt(ytmp, cutoff_dt, dt, btype='low', order=8, axis=-1):
    """
    Butterworth filter the time series

    Inputs:
        cutoff_dt - cuttoff period [seconds]
        btype - 'low' or 'high' or 'band'
    """
    if not btype == 'band':
        Wn = dt/cutoff_dt
    else:
        Wn = [dt/co for co in cutoff_dt]

    #(b, a) = signal.butter(order, Wn, btype=btype, analog=0, output='ba')
    #return signal.filtfilt(b, a, ytmp, axis=-1)

    sos = signal.butter(order, Wn, btype=btype, analog=0, output='sos', fs=1)    
    return signal.sosfiltfilt(sos, ytmp, axis=axis, padlen=0)


def filt_decompose(xraw, dt, low=34, high=3):
    
    x1 = filt(xraw, low*3600, dt, btype='low')
    x2 = filt(xraw, [low*3600, high*3600], dt, btype='band')
    x3 = filt(xraw, high*3600, dt, btype='high')
    
    xin = np.vstack([x1,x2,x3]).T
    
    return xin

def window_sinetaper(M,K=3):
    # Generate the time-domain taper for the FFT
    h_tk = np.zeros((K,M))
    t=np.arange(M,dtype=np.double)
    for k in range(K):
        h_tk[k,:] = np.sqrt(2./(M+1.))*np.sin( (k+1)*np.pi*t / (M+1.) ) 

    return h_tk
 
def power_spectra_old(tsec, u_r, power=2., axis=-1):
    """
    Calculates the power spectral density from a real valued quanity
    
    """
    
    M = tsec.shape[0]
    dt = tsec[1]-tsec[0]
    #dt *= 2*np.pi
    M_2 = int(np.floor(M/2))
    
    #h_tk = window_sinetaper(M,K=K)
   
    # Weight the time-series and perform the fft
    u_r_t = u_r#*h_tk
    S_k = np.fft.fft(u_r_t, axis=axis)
    S = dt*np.abs(S_k)**power 
    #S = np.mean(S_k,axis=-2)
        
    omega = np.fft.fftfreq(int(M),d=dt)
    
    #domega = 2*np.pi/(M*dt)
    #domega = 1/(M*dt)
    
    # Extract the positive and negative frequencies
    omega_ccw = omega[0:M_2]
    #omega_cw = omega[M_2::] # negative frequencies
    S_ccw = S[...,0:M_2]
    #S_cw = S[...,M_2::]
    
    #return omega, S
    return omega_ccw,S_ccw/M

def quadwin(n):
    """
    Quadratic (or "Welch") window
    """
    t = np.arange(n)
    win = 1 - ((t - 0.5 * n) / (0.5 * n)) ** 2
    return win

def power_spectra(h, dt=1, axis=-1):
    """
    First cut at spectral estimation: very crude.
    
    Returns frequencies, power spectrum, and
    power spectral density.
    Only positive frequencies between (and not including)
    zero and the Nyquist are output.
    """
    nt = h.shape[1]
    npositive = nt//2
    winweights = quadwin(nt)
    #winweights = np.ones((nt,))
    pslice = slice(1, npositive)
    freqs = np.fft.fftfreq(nt, d=dt)[pslice]
    ft = np.fft.fft(h, axis=axis)[...,pslice] 
    psraw = np.abs(ft) ** 2
    # Double to account for the energy in the negative frequencies.
    psraw *= 2
    
    # Normalization for Power Spectrum
    psraw /= nt**2
    
    # Convert PS to Power Spectral Density
    psdraw = psraw * dt * nt  # nt * dt is record length
    
    psdraw *= nt / (winweights**2).sum()
    
    return freqs,  psdraw

def fit_slope(highscale, lowscale, k, S):
    k_high = np.log10(2*np.pi/highscale)
    k_low = np.log10(2*np.pi/lowscale)
    k_i = np.linspace(k_high, k_low,100)

    Fi = interp1d(np.log10(k), np.log10(S))
    
    S_roms_i_log = Fi(k_i)
    p1 = np.polyfit(k_i, S_roms_i_log, 1)
    
    return p1, 10**k_i, 10**(p1[1]+p1[0]*k_i)
