"""
EEG Many Pipeline Project
Behzad Iravani, Neda Kaboodvand, Mehran Shabanpour
behzadiravani@gmail.com
n.kaboodvand@gmail.com
m.shbnpr@gmail.com
# -------------------------- HYPOTHESIS 3 ---------------------------
There are effects of [hits] vs. [misses] ...
                a. ... on EEG voltage at any channels, at any time.
                b. ... on spectral power, at any frequencies, at any channels, at any time.
"""
import os
import mne
import pandas as pd
import numpy as np
import math
from modules import IO  # for writing notes

# Initializing

os.chdir("..")
# retrieve the dirs path
dirs = IO.read_dirs(os.getcwd())
#dirs["comp"]= dirs["comp"].replace("E:","D:") ##

subject_directories = os.listdir(dirs["comp"])

for subs in subject_directories:
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
    IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis",
                      ["\n", '------------------------------------',
                       '              Hypothesis 3             ',
                       '------------------------------------',
                       '              -------------               ',
                       'filtering data to [1 40] for ERP analysis of Hypo2',
                       '                 -------------           ',
                       '-------------- REFERENCE TO AVERAGE-------------'
                       ], 'a', 'root')

    print(f'loading {subs} epoched data ...')
    events = pd.read_csv(dirs["comp_subject"]['root'] + "/" + "event_trigger_values.txt",
                         delimiter="\t", header=6)
    conditions = dict(
        hit = np.unique(events.event[events.id3 == 1].astype(int)),
        miss=np.unique(events.event[events.id3 == 2].astype(int)))
    print(conditions)
    epoch = mne.read_epochs(
        dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/" + "time_domain_data_epo.fif")  # loading subjcet epoched data
    # combing relevant events
    epoch = mne.epochs.combine_event_ids(epoch, [str(items) for items in conditions["hit"]], {'hit': 1})
    epoch = mne.epochs.combine_event_ids(epoch, [str(items) for items in conditions["miss"]], {'miss': 2})
    print(epoch['miss'])
    print('Equalizing the number of trials')
    # replace combined epochs with the original
    epoch = mne.concatenate_epochs([epoch["hit"], epoch["miss"]])
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # make note

    IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis", ['\nEqualizing the number of trials',
                                                          "before: ",
                                                          "hit = " + str(len(epoch["hit"])),
                                                          "miss = " + str(len(epoch["miss"]))], 'a', 'root')

    epoch.equalize_event_counts()
    print(epoch)

    IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis", ['\nafter:',
                                                          "hit = " + str(len(epoch["hit"])),
                                                          "miss = " + str(len(epoch["miss"]))], 'a', 'root')

    # filtering
    epoch.filter(1, 40, method='iir')
    # average (ERP)
    evoked_hit  = epoch['hit'].average()
    evoked_miss = epoch['miss'].average()
    print('done!')
    print('saving the evoked responses')
    evoked_hit.save(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/" + "hypo_3a_hit_time_domain_data_epo.fif", overwrite=True)
    evoked_miss.save(dirs["comp_subject"]["comp_subject_prepoc_time_series"]+ "/" + "hypo_3a_miss_time_domain_data_epo.fif", overwrite=True)

    print('performing time frequency analysis using Morlet wavelet')

    IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis", ['\ntfrs using Morlet wavelet',
                                                           'frequency bins = ' +
                                                           '[' + ''.join([' {:1.1f},'.format(elem) for elem in
                                                                          2 ** np.linspace(math.log2(4), math.log2(40),
                                                                                           10)]) + ']',
                                                           'n_cycles = 3'], 'a', 'root')

    power_hit = mne.time_frequency.tfr_morlet(epoch["hit"], freqs=2 ** np.linspace(math.log2(4), math.log2(40), 10),
                                              n_cycles=3,
                                              average=True, return_itc=False)
    power_miss = mne.time_frequency.tfr_morlet(epoch["miss"], freqs=2 ** np.linspace(math.log2(4), math.log2(40), 10),
                                              n_cycles=3,
                                              average=True, return_itc=False)
    print('saving tfr data')
    mne.time_frequency.write_tfrs(dirs["comp_subject"]["comp_subject_prepoc_time_series"]+ "/" + "hypo_3b_hit_freq_data_epo.fif", power_hit,
                                  overwrite=True)
    mne.time_frequency.write_tfrs(dirs["comp_subject"]["comp_subject_prepoc_time_series"]+ "/" + "hypo_3b_miss_freq_data_epo.fif", power_miss,
                                  overwrite=True)
    print('done!')


