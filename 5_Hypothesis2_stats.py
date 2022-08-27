"""
EEG Many Pipeline Project
Behzad Iravani, Neda Kaboodvand, Mehran Shabanpour
behzadiravani@gmail.com
n.kaboodvand@gmail.com
m.shbnpr@gmail.com
# -------------------------- HYPOTHESIS 2 ---------------------------
This script performs statistical tests for the hypothesis 2
There are effects of image novelty (i.e., between images shown for the first time/new
vs. repeated/old images) within the time-range from 300–500 ms ...
a. ... on EEG voltage at fronto-central channels.
b. ... on theta power at fronto-central channels.
c. ... on alpha power at posterior channels.
"""
import os
import warnings
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
# -----------Native libraries--------------
from modules import IO
from modules import visualization as viz
from modules import preprocessing as prep

os.chdir("..")
# retrieve the dirs path
dirs = IO.read_dirs(os.getcwd())
# dirs["comp"]= dirs["comp"].replace("E:","D:") ##

# subjects directories
subject_directories = os.listdir(dirs["comp"])
title = 'new > old'
# define the frontocental electrodes
forntocentral_picks = ['FC5','FC3', 'FC1', 'FCz', 'FC2', 'FC4','FC6']
posterior_picks = ['PO7','PO3', 'POz', 'PO4', 'PO8',
                   'O1','Oz','O2']
def tfr_table_average(table, picks,subs):
    avg = []
    for items in picks:
        try:
            avg.append(table[items].mean()) # average over time and frequency
        except:
            warnings.warn(f'No {items} found in the data for {subs}')
            #IO.write_text(dirs, "_general_workflow_4_hyothesis", [f'\n \n *No {items} found in the data for {subs}* \n \n '], 'a')
    average = np.mean(avg) # average over channels
    return average


def get_forontocental_tfrs(fname, forntocentral_picks):
    power = mne.time_frequency.read_tfrs(fname)
    power[0].apply_baseline(baseline=[None, 0])

    tfr = power[0].pick(forntocentral_picks)
    return tfr

def get_forontocental_evoked(fname, forntocentral_picks):
    evoked = mne.read_evokeds(fname)
    evoked[0].apply_baseline(baseline=(None, 0))

    evoked = evoked[0].pick(forntocentral_picks)
    return evoked

def get_posterior_tfrs(fname, posterior_picks):
    power = mne.time_frequency.read_tfrs(fname)
    power[0].apply_baseline(baseline=[None, 0])

    tfr = power[0].pick(posterior_picks)
    return tfr

data = dict(sub = [], voltage_FC = [], theta_FC = [], alpha_O = [])

for subs in subject_directories: #
    # Extract subject number
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

    data['sub'].append(subs)
    print(f'selecting and averaging dat for {subs}')
    # -------- Frontocentral ---------------
    print('Frontocentral')
    # ------------ evoked ---------------
    print('loading evoked data')
    evoked_new = get_forontocental_evoked(dirs["comp_subject"]["comp_subject_prepoc_time_series"] +
                                           "/" + "hypo_2a_new_time_domain_data_epo.fif", forntocentral_picks)
    evoked_old = get_forontocental_evoked(dirs["comp_subject"]["comp_subject_prepoc_time_series"] +
                                          "/" + "hypo_2a_old_time_domain_data_epo.fif",
                                     forntocentral_picks)

    IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis",['\nHypothesis 2a: voltage of frontocentral electrodes',
                                                             ', '.join(forntocentral_picks)],'a','root')

    print('done!')
    # 300ms to 500ms
    tmin = .300   #s
    tmax = .500   #s
    print(f'averaging evoked from {tmin}ms to {tmax}ms ')

    IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis", ['\naveraging for time interval',
                                                        f't= [{tmin}, {tmax}] (s)'], 'a','root')
    evoked_new = np.mean(evoked_new.get_data(tmin = tmin, tmax = tmax))
    evoked_old = np.mean(evoked_old.get_data(tmin = tmin, tmax = tmax))
    data['voltage_FC'].append([evoked_new,evoked_old])
    print('done!')
    # ------------- tfrs ---------------

    IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis", ['\nHypothesis 2b: theta power of frontocentral electrodes',
                                                          ', '.join(forntocentral_picks)], 'a','root')

    print('loading tfr data')
    tfr_new = get_forontocental_tfrs(dirs["comp_subject"]["comp_subject_prepoc_time_series"]  + "/" + "hypo_2b,c_new_freq_data_epo.fif", forntocentral_picks)
    tfr_old = get_forontocental_tfrs(dirs["comp_subject"]["comp_subject_prepoc_time_series"]  + "/" + "hypo_2b,c_old_freq_data_epo.fif",
                                     forntocentral_picks)
    print('done!')
    print('plotting tfrs')
    plt.imshow(np.squeeze(np.mean(tfr_new.data, axis = 0)-
                          np.mean(tfr_old.data, axis = 0)),
               extent=[tfr_new.times[0], tfr_new.times[-1], tfr_new.freqs[0], tfr_new.freqs[-1]],
               aspect='auto', origin='lower', cmap='RdBu_r')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Power frontocentral (%s)' % title)
    plt.colorbar()
    plt.draw()
    plt.savefig(dirs["root_folder"]+"/results/hypothesis_2/hypothesis2b,c,frontocentral,tfr.png")
    plt.close()
    print('done!')
    # theta band
    fmin = 4
    fmax = 8
    # 300ms to 500ms
    tmin = .300 # in seconds
    tmax = .500
    print(f'averaging tfr from {tmin}s to {tmax}s for freqs =  [{fmin},{fmax}] ')
    data['theta_FC'].append([
                           tfr_table_average(tfr_new.crop(tmin = tmin, tmax = tmax, fmin = fmin, fmax = fmax).to_data_frame(),forntocentral_picks,subs),
                           tfr_table_average(tfr_old.crop(tmin = tmin, tmax = tmax, fmin = fmin, fmax = fmax).to_data_frame(),forntocentral_picks,subs)])

    IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis", ['\naveraging for time interval',
                                                          f't= [{tmin}, {tmax}] (s)'
                                                           f'freqs = [{fmin}, {fmax}]'], 'a', 'root')


    #-------------------- Posterior -----------------

    IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis", ['\nHypothesis 2c: alpha power of posterior electrodes',
                                                          ', '.join(posterior_picks)], 'a', 'root')

    tfr_new = get_posterior_tfrs(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/" + "hypo_2b,c_new_freq_data_epo.fif",
                                     posterior_picks)
    tfr_old = get_posterior_tfrs(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/" + "hypo_2b,c_old_freq_data_epo.fif",
                                     posterior_picks)
    print('plotting tfrs')
    plt.imshow(np.squeeze(np.mean(tfr_new.data, axis=0) -
                          np.mean(tfr_old.data, axis=0)),
               extent=[tfr_new.times[0], tfr_new.times[-1], tfr_new.freqs[0], tfr_new.freqs[-1]],
               aspect='auto', origin='lower', cmap='RdBu_r')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Power posterior (%s)' % title)
    plt.colorbar()
    plt.draw()
    plt.savefig(dirs['root_folder']+'/results/hypothesis_2/hypothesis2b,c,Posterior,tfr.png')
    plt.close()
    print('done!')
    # alpha band
    fmin = 8
    fmax = 12
    # 300ms to 500ms
    tmin = .300 #in seconds
    tmax = .500
    print(f'averaging tfr from {tmin} s to {tmax} s for freqs =  [{fmin},{fmax}] ')
    data['alpha_O'].append([
                           tfr_table_average(tfr_new.crop(tmin = tmin, tmax = tmax, fmin = fmin, fmax = fmax).to_data_frame(),posterior_picks,subs),
                           tfr_table_average(tfr_old.crop(tmin = tmin, tmax = tmax, fmin = fmin, fmax = fmax).to_data_frame(),posterior_picks,subs)])

    IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis", ['\naveraging for time interval',
                                                          f't= [{tmin}, {tmax}] (s)'
                                                          f'freqs = [{fmin}, {fmax}]'], 'a','root')

# Hypothesis testing

pd.DataFrame.from_dict(data).to_csv(path_or_buf=dirs['root_folder'] + '\\' + 'results' + '\\' + 'hypothesis_2\\data.txt', sep=',')
# Voltage frontocentral
tmp = np.stack(data['voltage_FC'],axis=0)
# replacing outlier with median
tmp[:,0],_ = prep.replace_outlier_with_median(tmp[:,0])
tmp[:,1],_ = prep.replace_outlier_with_median(tmp[:,1])

tval, pval = sp.stats.ttest_rel(np.array(tmp[:,0]), np.array(tmp[:,1]), axis = 0, nan_policy= 'omit')
viz.bar_graph(dirs,tmp[:,0],tmp[:,1],
              'Voltage frontocentral, [300-500 ms]',
              'Amplitude (s.e.m)')

with open(dirs['root_folder']+'/results/hypothesis_2/'+'hypothesis_2_ttest.txt', 'w') as f:
    f.write(f'new>old frontocentral volatge : t({.5*(len(tmp[:,0])+len(tmp[:,1]))-1}) = {tval}, p =  {pval}\n')
# Power theta frontocentral
tmp = np.stack(data['theta_FC'],axis=0)
# replacing outlier with median
tmp[:,0],_ = prep.replace_outlier_with_median(tmp[:,0])
tmp[:,1],_ = prep.replace_outlier_with_median(tmp[:,1])

tval, pval = sp.stats.ttest_rel(np.array(tmp[:,0]), np.array(tmp[:,1]), axis = 0, nan_policy= 'omit')
viz.bar_graph(dirs,tmp[:,0],tmp[:,1],
              'Theta power frontocentral, [300-500 ms]',
              'Magnitude (s.e.m)')

with open(dirs['root_folder']+'/results/hypothesis_2/'+'hypothesis_2_ttest.txt', 'a') as f:
    f.write(f'new>old frontocentral theta power : t({.5*(len(tmp[:,0])+len(tmp[:,1]))-1}) = {tval}, p =  {pval}\n')
# Power alpha occipital
tmp = np.stack(data['alpha_O'],axis=0)
# replacing outlier with median
tmp[:,0],_ = prep.replace_outlier_with_median(tmp[:,0])
tmp[:,1],_ = prep.replace_outlier_with_median(tmp[:,1])
tval, pval = sp.stats.ttest_rel(np.array(tmp[:,0]), np.array(tmp[:,1]), axis = 0, nan_policy= 'omit')
viz.bar_graph(dirs,tmp[:,0],tmp[:,1],
              'Alpha power occipital, [300-500 ms]',
              'Magnitude (s.e.m)')

with open(dirs['root_folder']+'/results/hypothesis_2/'+'hypothesis_2_ttest.txt', 'a') as f:
    f.write(f'new>old posterior alpha power : t({.5*(len(tmp[:,0])+len(tmp[:,1]))-1}) = {tval}, p =  {pval}\n')

IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis", ['\n ++ replaced outliers with median (abs(Z)>2.7) ++\n'], 'a', 'root')

