"""
EEG Many Pipeline Project
Behzad Iravani, Neda Kaboodvand, Mehran Shabanpour
behzadiravani@gmail.com
n.kaboodvand@gmail.com
m.shbnpr@gmail.com
# -------------------------- HYPOTHESIS 1 ---------------------------
This script performs the grand averaging of the epochs and statistical testing of the hypothesis 1 where the aim is:
 -  There is an effect of scene category (i.e., a difference between images showing
man-made vs. natural environments) on the amplitude of the N1 component, i.e. the
first major negative EEG voltage deflection
"""

import os
from os.path import exists

import matplotlib.pyplot as plt
import mne
from mne.preprocessing import peak_finder
from itertools import compress
import pandas as pd
import numpy as np
import scipy as sp
import math as mth
# Import native modules'
from modules import visualization as viz
from modules import IO # for writing notes
from modules import preprocessing as prep
# Initializing

os.chdir("..")
# retrieve the dirs path
dirs = IO.read_dirs(os.getcwd())
#dirs["comp"]= dirs["comp"].replace("E:","D:") ## The drive letter where the data is
#dirs["root_folder"] = dirs["root_folder"].replace("E:","D:")
def find_N100(dat):
    dat.apply_baseline(baseline = (None, 0))
    dat = dat.crop(tmin = 0.070 , tmax = 0.140) # time in seconds
    time_ = dat.times
    x0 = np.squeeze(dat.get_data())
    th = (np.max(x0) - np.min(x0)) / 5
    N100_locs, N100_peaks = peak_finder(x0=x0,
                                        thresh= th,  extrema= -1.0)  # searching for minima -> extrema = -1
    print(N100_peaks)
    #select = N100_peaks<0
    #if any(select):
    return np.median(time_[N100_locs]), np.median(N100_peaks)
    #else:
    #return np.nan, np.nan
    #IO.write_text(dirs, "NG,No N100 found", [f'\n No N100 found for {subs} for {cond}'],'a')

subject_directories = os.listdir(dirs["comp"])

dat_man_made = []
dat_natural = []

# create the result folder
if not exists(dirs["root_folder"] + '/results/' ):
    os.mkdir(dirs["root_folder"] + '/results/')
# create the result folder
if not exists(dirs["root_folder"] + '/results/hypothesis_1' ):
    os.mkdir(dirs["root_folder"] + '/results/hypothesis_1')

for subs in subject_directories: # loop over subjects
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

    evoked = mne.read_evokeds(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/"  + "hypo_1_man_made_time_domain_data_epo.fif")
    chan_data_list = []
    print(f'--------------- {subs} -----------------')
    for ch in evoked[0].ch_names:
        # Man_Made
        chan_dat = evoked[0].copy()
        chan_dat.pick([ch])
        # Find N100
        print(f'--------------- Man Made {ch} -----------------')
        if not any([mth.isnan(items) for items in np.squeeze(chan_dat.get_data())]):
            N100_locs, N100_peaks = find_N100(chan_dat)
        else:
            N100_locs = np.nan
            N100_peaks = np.nan
            IO.write_text(dirs, "NG,NAN", [f'\n nan found for man made -- {subs}'],'a')

        chan_data_list.append((N100_locs, N100_peaks))
    dat_man_made.append(chan_data_list)
    # Natural
    evoked = mne.read_evokeds(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/" + "hypo_1_natural_time_domain_data_epo.fif")
    chan_data_list = []
    for ch in evoked[0].ch_names:
        chan_dat = evoked[0].copy()
        chan_dat.pick([ch])
        # Find N100
        print(f'--------------- Natural {ch} -----------------')
        if not any([mth.isnan(items) for items in np.squeeze(chan_dat.get_data())]):
            N100_locs, N100_peaks = find_N100(chan_dat)
        else:
            N100_locs = np.nan
            N100_peaks = np.nan
            IO.write_text(dirs["comp_subject"], "NG,NAN", [f'\n nan found for natural -- {subs}'], 'a','root')
        chan_data_list.append((N100_locs, N100_peaks))
    dat_natural.append(chan_data_list)

# [0] latency [1] amplitude, dimension 3
dat_man_made = np.squeeze(np.array(dat_man_made)[:,:,1])
dat_natural  = np.squeeze(np.array(dat_natural)[:,:,1])

# replacing outliers with median
dat_man_made, _ = prep.replace_outlier_with_median(dat_man_made)
dat_natural, _  = prep.replace_outlier_with_median(dat_natural)

# make note

IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis",
                      ["\n",
                       'replacing outliers (abs(z)>2.7 with median'],'a', 'root')


tval, pval = sp.stats.ttest_rel(dat_man_made, dat_natural, axis = 0, nan_policy= 'omit')

IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis",
                      ["\n",
                       '****FDR correction***'],'a', 'root')

(h, pval_FDR) = mne.stats.fdr_correction(pval, alpha=0.05, method='negcorr') #Benjamini/Yekutieli

mean_N100_man_made = np.squeeze(np.nanmean(np.array(dat_man_made), axis = 0))
mean_N100_natural = np.squeeze(np.nanmean(np.array(dat_natural), axis = 0))
SE_N100_man_made = np.squeeze(np.nanstd(np.array(dat_man_made), axis = 0)/np.sqrt(np.array(dat_man_made).shape[0]))
SE_N100_natural = np.squeeze(np.nanstd(np.array(dat_natural), axis = 0)/np.sqrt(np.array(dat_natural).shape[0]))


significants = pval_FDR<0.05
list_of_significant_channels = dict( chan = list(compress(evoked[0].ch_names, significants)),
                                     tval = tval[pval_FDR<0.05],
                                     pval = pval[pval_FDR<0.05],
                                     pval_FDR = pval_FDR[pval_FDR<0.05],
                                     N100_man_made_dat = dat_man_made[:,pval_FDR<0.05],
                                     N100_natural_dat = dat_natural[:,pval_FDR<0.05],
                                     N100_man_made = mean_N100_man_made[pval_FDR<0.05],
                                     N100_natural = mean_N100_natural[pval_FDR<0.05],
                                     N100_man_made_SE =   SE_N100_man_made[pval_FDR<0.05],
                                     N100_natural_SE  =   SE_N100_man_made[pval_FDR<0.05]
)

# saving the stats
lines = []
c= 0
for ch in list_of_significant_channels["chan"]:
    viz.N100_bar_graph(dirs, ch,
                       list_of_significant_channels,c
)
    lines.append(f'{ch} : t = {list_of_significant_channels["tval"][c]}, p < {list_of_significant_channels["pval"][c]}, P_corrected < {list_of_significant_channels["pval_FDR"][c]}')
    c += 1

with open(dirs['root_folder']+'/results/hypothesis_1/'+'hypothesis_1_ttest.txt', 'w') as f:
    f.write('\n'.join(lines))

evoked_man_made = []
evoked_natural = []
for subs in subject_directories: # loop over subjects
    tmp = mne.read_evokeds(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/" + "hypo_1_man_made_time_domain_data_epo.fif")
    evoked_man_made.append(tmp[0])
    tmp = mne.read_evokeds(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/" + "hypo_1_natural_time_domain_data_epo.fif")
    evoked_natural.append(tmp[0])

G_ManMade = mne.grand_average(evoked_man_made)
G_Natural = mne.grand_average(evoked_natural)
MnMd_minus_Nthrl = mne.combine_evoked([G_ManMade, G_Natural], weights=[1, -1])

ts_args = dict(xlim = (-.1, .5))
MnMd_minus_Nthrl.plot_joint(times=[.130], title = "Man-Made > Natural", ts_args= ts_args)
"""
# concatenate the data
group_man_made = np.stack(dat_man_made, axis = -1)
group_natural = np.stack(dat_natural, axis = -1)
# perform hypothesis testing
"""






