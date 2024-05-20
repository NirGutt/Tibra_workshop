import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import bilby
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from gwpy.timeseries import TimeSeries
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from bilby.core.prior import PriorDict,Constraint
import tqdm
import gdown
import numpy as np
import scipy
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import h5py
import json
from scipy.special import logsumexp

def whiten(strain, psd_freq,psd, dt, phase_shift=0, time_shift=0):
    """Whitens strain data given the psd and sample rate, also applying a phase
    shift and time shift.

    Args:
        strain (ndarray): strain data
        interp_psd (interpolating function): function to take in freqs and output
            the average power at that freq
        dt (float): sample time interval of data
        phase_shift (float, optional): phase shift to apply to whitened data
        time_shift (float, optional): time shift to apply to whitened data (s)

    Returns:
        ndarray: array of whitened strain data
    """
    Nt = len(strain)
    # take the fourier transform of the data
    freqs = np.fft.rfftfreq(Nt, dt)

    # whitening: transform to freq domain, divide by square root of psd, then
    # transform back, taking care to get normalization right.
    hf = np.fft.rfft(strain)
    hf = hf*dt

    # apply time and phase shift
    hf = hf * np.exp(-1.j * 2 * np.pi * time_shift * freqs - 1.j * phase_shift)
    norm = np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(psd) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht,hf*norm,white_hf

def bandpass(strain, fband, fs):
    """Bandpasses strain data using a butterworth filter.

    Args:
        strain (ndarray): strain data to bandpass
        fband (ndarray): low and high-pass filter values to use
        fs (float): sample rate of data

    Returns:
        ndarray: array of bandpassed strain data
    """
    bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
    normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
    strain_bp = filtfilt(bb, ab, strain) / normalization
    return strain_bp


delta_t=0.25
maximum_frequency = 512
minimum_frequency = 20
duration=2
use_injection=False
outdir='outdir'
label='stocastic_sampler'
#sampling_frequency=1024

def time_domain_model(params):
  dt = (1/ifo.sampling_frequency)
  norm =  np.sqrt(1./(dt*2))
  return np.fft.irfft(model_GR(params)*norm)


def whiten_time_domain_model(params,psd):

  dt = (1/ifo.sampling_frequency)
  norm =  np.sqrt(1./(2*dt))
  hf= model_GR(params)
  Nt=len(hf)
  hf_norm = hf / np.sqrt(psd) * norm
  t_axis = np.arange(0,Nt*dt,dt)-delta_t
  return t_axis,np.fft.irfft(hf_norm, n=Nt)




def model_GR(params):
  # check for mass1 and mass2 params in the input and that mass1 > mass2
  params_to_eval= partial_params.copy()
  for k in partial_params.keys():
    if k in params:
      params_to_eval[k]=params[k]

  wf_pols = waveform_generator.frequency_domain_strain(params_to_eval)

  strain_waveform_freq = ifo.get_detector_response(wf_pols, params_to_eval)
  return strain_waveform_freq

bilby.core.utils.random.seed(88170235)




partial_params={'theta_jn': 3.097067,
 'luminosity_distance': 557.248554,
 'ra': 0.97,
 'dec': -1.271,
 #'azimuth': 2.8110605307230103,
 #'zenith': 2.3450004587195976,
 'mass_1': 37.973945,
 'mass_2': 32.408403,
 'a_1': 0.407249,
 'a_2': 0.834349,
 'tilt_1': 1.828981,
 'tilt_2': 1.652162,
 'psi': 2.851105,
 'phase': 2.138286,
 'geocent_time': 1126259462.414116,
 'time_jitter': 0.000033,
 'phi_12': 1.560456,
 'phi_jl': 0.997263,
'lambda_1' : 0,
'lambda_2' : 0}

ifo = bilby.gw.detector.get_empty_interferometer("H1")
trigger_time = 1126259462.4
duration = duration  # Analysis segment duration
post_trigger_duration = duration/2  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration
data_ifo = TimeSeries.fetch_open_data("H1", start_time, end_time)
ifo.strain_data.set_from_gwpy_timeseries(data_ifo)

psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

roll_off = 0.4
psd_data = TimeSeries.fetch_open_data("H1", psd_start_time, psd_end_time)
psd_alpha = 2 * roll_off / duration
psd = psd_data.psd(
        fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median"
)
ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value
    )
ifo.maximum_frequency = maximum_frequency
ifo.minimum_frequency = minimum_frequency






data= data_ifo.value

freq_array = ifo.frequency_array
time_array = ifo.time_array
psd_freq_array= ifo.power_spectral_density.frequency_array
psd = ifo.power_spectral_density.psd_array
psd= np.interp(freq_array,psd_freq_array,psd)


waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
        waveform_arguments={
            "waveform_approximant": "IMRPhenomPv2_NRTidal",
            "reference_frequency": minimum_frequency,
            "minimum_frequency" : minimum_frequency,
            "maximum_frequency" : maximum_frequency,
        },
        sampling_frequency=ifo.sampling_frequency,
    )

if use_injection:
  ifo.set_strain_data_from_power_spectral_density(
      sampling_frequency=ifo.sampling_frequency,
      duration=duration,
      start_time=start_time)

  ifo.inject_signal(
      waveform_generator=waveform_generator, parameters=partial_params
  )

data_time = ifo.time_domain_strain

data_freq =ifo.frequency_domain_strain
#ifo.  to_gwpy_frequencyseries()


def run_stocastic_sampler(data,model,psd,log_likelihood_func,prior_dict):
  # the prior_dict should contain teh min and max , and we assume evrything is a uniform distribution


  class SimpleGaussianLikelihood(bilby.Likelihood):
    def __init__(self, data,model,log_likelihood_func,psd):
        """
        A very simple Gaussian likelihood

        Parameters
        ----------
        data: array_like
            The data to analyse
        """
        super().__init__(parameters=partial_params.copy())
        self.data = data
        self.N = len(data)
        self.log_likelihood_func=log_likelihood_func
        self.psd=psd
        self.model=model


        self.ncores = multiprocessing.cpu_count()

    def log_likelihood(self):
        return self.log_likelihood_func(data,model,self.parameters,psd)


  def convert_m1_m2_to_dm(parameters):
    parameters['dm'] = parameters['mass_1'] - parameters['mass_2']
    return parameters

  priors = PriorDict(conversion_function=convert_m1_m2_to_dm)
  priors['dm'] = Constraint(minimum=0, maximum=30)
        
  likelihood = SimpleGaussianLikelihood(data,model,log_likelihood_func,psd)
  for k,v in partial_params.items(): 
      priors[k]=v
      
  for k,v in prior_dict.items():
    priors[k] = bilby.core.prior.Uniform(minimum=v[0],maximum=v[1],name=k)
      
  print(priors )
  # And run sampler
  result = bilby.run_sampler(
      likelihood=likelihood,
      priors=priors,
      #sampler="bilby_mcmc",
      #nsamples=300,
      sampler="dynesty",
      sample = "acceptance-walk",#"rwalk",
      naccept=1,
      resume=False,
      clean=True,
      nlive=50,
      npool=likelihood.ncores,
      outdir=outdir,
      label=label,
  )

  return result.posterior



def Generate_data(mass_1,mass_2):
  import time
  current_time = time.time()
  bilby.core.utils.random.seed(int(current_time))
  ifo.set_strain_data_from_power_spectral_density(
    sampling_frequency=ifo.sampling_frequency,
    duration=duration,
    start_time=start_time)

  p_copy = partial_params.copy()
  p_copy['mass_1']=mass_1
  p_copy['mass_2']=mass_2
  ifo.inject_signal(
    waveform_generator=waveform_generator, parameters=p_copy
  )

  return ifo.frequency_domain_strain

def corner_plot(samples):
  import os
  file_name=outdir+'/'+label+'_result.json'
  if not os.path.isfile(file_name):
    raise Exception('Did you run the sampler ? ')


  result=bilby.result.read_in_result(filename=file_name)
  result.posterior= samples
  result.plot_corner(save=False)
  plt.show()


# @title
def model(params):

 # check we have what we need
 # add a bunch of tests to the input
 if 'mass_1' not in params or 'mass_2' not in params:
  print(params)
  raise Exception('missing at least one parameter, input should contain mass_1,mass_2 in a dict form')


 return model_GR(params)


def extended_model(params):

 # check we have what we need
 if 'mass_1' not in params or 'mass_2' not in params or 'ra' not in params or 'a_1' not in params or 'a_2' not in params:
  print(params)
  raise Exception('missing at least one parameter, input should contain mass_1,mass_2, ra, a_1, a_2  in a dict form')

 return model_GR(params)
