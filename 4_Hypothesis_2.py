
"""
EEG Many Pipeline Project
Behzad Iravani, Neda Kaboodvand, Mehran Shabanpour
behzadiravani@gmail.com
n.kaboodvand@gmail.com
m.shbnpr@gmail.com
# -------------------------- HYPOTHESIS 2 ---------------------------
There are effects of image novelty (i.e., between images shown for the first time/new
vs. repeated/old images) within the time-range from 300–500 ms ...
a. ... on EEG voltage at fronto-central channels.
b. ... on theta power at fronto-central channels.
c. ... on alpha power at posterior channels.

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
# dirs["comp"]= dirs["comp"].replace("E:","D:") ##

subject_directories = os.listdir(dirs["comp"])
for subs in  subject_directories:  # loop over
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
                       '              Hypothesis 2             ',
                       '------------------------------------',
                       '              -------------               ',
                       'filtering data to [1 40] for ERP analysis of Hypo2',
                       '                 -------------           ',
                       '-------------- REFERENCE TO AVERAGE-------------'
                       ], 'a', 'root')


    print(f'loading {subs} epoched data ...')
    events = pd.read_csv(dirs["comp"] + "/" + subs + "/" + "event_trigger_values.txt",
                             delimiter="\t", header=6)
    conditions = dict(
            new=np.unique(events.event[events.id2 == 0].astype(int)),
            old=np.unique(events.event[events.id2 == 1].astype(int)))
    # Check if the conditions are balanced
    # isImbalance = prep.report_number_of_trais(dirs, conditions)
    epoch = mne.read_epochs(
            dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/" + "time_domain_data_epo.fif")  # loading subjcet epoched data
    # combing relevant events
    epoch = mne.epochs.combine_event_ids(epoch, [str(items) for items in conditions["new"]], {'new': 0})
    epoch = mne.epochs.combine_event_ids(epoch, [str(items) for items in conditions["old"]], {'old': 1})

    # replace the combined epochs with the original evoked
    epoch = mne.concatenate_epochs([epoch["new"], epoch["old"]])
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    print(epoch.info)
    print('Equalizing the number of trials')
    # make note

    IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis",['\nEqualizing the number of trials',
                                                             "before: ",
                                                             "new = " + str(len(epoch["new"])),
                                                             "old = " + str(len(epoch["old"]))],'a', "root")

    epoch.equalize_event_counts()
    print(epoch)

    IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis",['\nafter:',
                                                             "new = " + str(len(epoch["new"])),
                                                             "old = " + str(len(epoch["old"]))],'a', "root")

    print('filtering data to [1 40] for ERP analysis of Hypo1')
    epoch.filter(1, 40, method='iir')
    print('done!')
    print('computing evoked response')
    evoked_new = epoch["new"].average()
    evoked_old = epoch["old"].average()
    print('done!')
    print('saving the evoked responses')
    evoked_new.save(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/" + "hypo_2a_new_time_domain_data_epo.fif", overwrite=True)
    evoked_old.save(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/" + "hypo_2a_old_time_domain_data_epo.fif", overwrite=True)

    print('performing time frequency analysis using Morlet wavelet')
    IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis", ['\ntfrs using Morlet wavelet',
                                                          'frequency bins = '+
                                                          '[' +''.join([' {:1.1f},'.format(elem) for elem in 2 ** np.linspace(math.log2(4), math.log2(40),10)])+']',
                                                          'n_cycles = 3'], 'a', "root")
    power_new = mne.time_frequency.tfr_morlet(epoch["new"], freqs=2 ** np.linspace(math.log2(4), math.log2(40),10), n_cycles=3,
                                                  average=True, return_itc=False)
    power_old = mne.time_frequency.tfr_morlet(epoch["old"], freqs=2 ** np.linspace(math.log2(4), math.log2(40), 10),
                                              n_cycles=3,
                                              average=True, return_itc=False)
    print('saving tfr data')
    mne.time_frequency.write_tfrs(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/" + "hypo_2b,c_new_freq_data_epo.fif", power_new,overwrite=True)
    mne.time_frequency.write_tfrs(dirs["comp_subject"]["comp_subject_prepoc_time_series"]+ "/" + "hypo_2b,c_old_freq_data_epo.fif", power_old,overwrite=True)
    print('done!')

