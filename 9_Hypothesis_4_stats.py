"""
EEG Many Pipeline Project
Behzad Iravani, Neda Kaboodvand, Mehran Shabanpour
behzadiravani@gmail.com
n.kaboodvand@gmail.com
m.shbnpr@gmail.com
# -------------------------- HYPOTHESIS 4---------------------------
TThis script performs statistical tests for hypothesis 4
There are effects of successfully remembered vs. forgotten on a subsequent repetition ...
                a. ... on EEG voltage at any channels, at any time.
                b. ... on spectral power, at any frequencies, at any channels, at any time.
"""
import os
import numpy as np
import pandas as pd
import mne
import scipy as sp
from scipy import stats
# -----------Native libraries--------------
from modules import visualization as viz
from modules import preprocessing as prep
from modules import IO  # for writing notes

def get_evoked(fname):
    evoked = mne.read_evokeds(fname)
    evoked[0].apply_baseline(baseline=(None, 0))
    return evoked[0]

def get_tfrs(fname):
    power = mne.time_frequency.read_tfrs(fname)
    power[0].apply_baseline(baseline=[None, 0])

    tfr = power[0]
    return tfr

os.chdir("..")
# retrieve the dirs path
dirs = IO.read_dirs(os.getcwd())
#dirs["comp"]= dirs["comp"].replace("E:","D:") ##
subject_directories = os.listdir(dirs["comp"])

# subjects directories
subject_directories = os.listdir(dirs["comp"])
title = 'remembered > forgotten'
evoked_rememebered_dat = []
evoked_forgotten_dat   = []
# ----------------
tfr_rememebered_data   = []
tfr_forgotten_data     = []

for subs in subject_directories: #subject_directories
    s_num = ''.join([n for n in subs if n.isdigit()])
    dirs["comp_subject"] = {"root": dirs["comp"] + "/Subj" + s_num,
                            # path for computational work of the specific subject
                            "plot": dirs["comp"] + "/Subj" + s_num + "/" + "Plot",
                            "comp_subject_prepoc_time_series": dirs["comp"] + "/Subj" + s_num
                                                               + "/" + "Pre-processed time series data",
                            "comp_subject_ICA": dirs["comp"] + "/Subj" + s_num
                                                + "/" + "Removed ICA components (txt files)",
                            "comp_subject_BadTrials": dirs["comp"] + "/Subj" + s_num
                                                      + "/" + "Excluded trails (txt files)",
                            "comp_subject_BadSensors": dirs["comp"] + "/Subj" + s_num
                                                       + "/" + "Excluded sensors (txt files)"}

    print(f'selecting and averaging dat for {subs}')
    # ERP hypothesis 4a
    # read and baseline normalization
    print('read evoked and baseline normalization')
    evoked_rememebered  = get_evoked(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/" + "hypo_4a_remembered_time_domain_data_epo.fif")
    evoked_forgotten    = get_evoked(dirs["comp_subject"]["comp_subject_prepoc_time_series"]  + "/" + "hypo_4a_forgotten_time_domain_data_epo.fif")
    #  get the  post stimulus data
    evoked_rememebered_dat.append(evoked_rememebered.get_data(tmin=0, tmax= None))
    evoked_forgotten_dat.append(evoked_forgotten.get_data(tmin=0, tmax= None))
    print('done!')
    # TFR hypothesis 4b
    # reading and baseline correction
    print('loading tfr data')
    tfr_rememebered     = get_tfrs(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/" + "hypo_4b_remembered_freq_data_epo.fif")
    tfr_forgotten       = get_tfrs(dirs["comp_subject"]["comp_subject_prepoc_time_series"]  + "/" + "hypo_4b_forgotten_freq_data_epo.fif")
    print('done!')

    print('plotting tfrs')
    # creating contrast
    tfr_contrast        = mne.combine_evoked((tfr_rememebered, tfr_forgotten), (-.5, .5))
    tfr_rememebered_data.append(np.transpose(tfr_rememebered.data,(1,2,0)))
    tfr_forgotten_data.append(np.transpose(tfr_forgotten.data, (1, 2, 0)))
    # plotting contrast
    topo_plot_rememebered_forgotten = tfr_contrast.plot_topo(show=False, fig_facecolor='w', vmin = -5e-10, vmax = 5e-10)
    topo_plot_rememebered_forgotten.savefig(dirs["comp_subject"]['plot'] + "\\hypothesis4b_rememebered_forgotten_tfr.png")

# ERPs
erp1     = np.transpose(np.stack(evoked_rememebered_dat, axis = 0), (0,2,1))
erp2     = np.transpose(np.stack(evoked_forgotten_dat, axis = 0),(0,2,1))


ch_names = evoked_rememebered.ch_names
times    = evoked_rememebered.times
times    = times[times>=0]
# threshold = 2
sigma = 1e-3
X = erp1 - erp2
# Statistical hypothesis testing
# ttest with hat correction
ts = np.array(mne.stats.ttest_1samp_no_p(X, sigma=sigma))
print(ts.shape)
dummy =  evoked_rememebered.copy()
dummy.data = ts.T
dummy.times =  times
dummy.plot_joint(times = [0.190, 0.450, 0.700])

# calculate ps from ts
# create t distribution
df =  X.shape[0] - 1  # df for paired ttest
t_distribution = sp.stats.t(df)
ps = t_distribution.sf(np.abs(ts)) * 2 # p-values
print(ps.shape)
# fdr correction for multiple comparison
_, ps_fdr = mne.stats.fdr_correction(ps)
print(ps_fdr.shape)
# find significant clusters
prep.find_and_report_significantERP(dirs, ts, ps_fdr, times, ch_names, hypo= 'hypothesis_4')
viz.cluster_erp(dirs, erp1, erp2, ts, ps_fdr, times, ch_names, cond= 'rememebered > forgotten', hypo= 'hypothesis_4')
# --------------- TFRS ------------------
threshold = None
freqs = tfr_rememebered.freqs
times = tfr_rememebered.times
ch_names = tfr_rememebered.ch_names
tfr1 = np.stack(tfr_rememebered_data, axis = 0)
tfr2 = np.stack(tfr_forgotten_data, axis = 0)

X = tfr1 - tfr2
ts = np.array(mne.stats.ttest_1samp_no_p(X, sigma=sigma))
# calculate ps from ts
# create t distribution
df =  X.shape[0] - 1  # df for paired ttest
t_distribution = sp.stats.t(df)
ps = t_distribution.sf(np.abs(ts)) * 2 # p-values0
# fdr correction for multiple comparison
_, ps_fdr = mne.stats.fdr_correction(ps)
if np.any(ps_fdr<=0.05):
    # find significant clusters
    prep.find_and_report_significantTFR(dirs, ts, ps_fdr, times, freqs, ch_names,  hypo= 'hypothesis_4')
    viz.cluster_tfr(dirs, ts, ps_fdr, times, freqs, ch_names, cond= 'rememebered > forgotten', hypo= 'hypothesis_4')
else:
    prep.find_and_report_significantTFR(dirs, ts, ps, times, freqs, ch_names,  hypo= 'hypothesis_4')
    viz.cluster_tfr(dirs, ts, ps, times, freqs, ch_names, cond= 'rememebered > forgotten', hypo= 'hypothesis_4')
