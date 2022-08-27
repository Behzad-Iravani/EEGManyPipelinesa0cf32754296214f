
"""
EEG Many Pipeline Project
Behzad Iravani, Neda Kaboodvand, Mehran Shabanpour
behzadiravani@gmail.com
n.kaboodvand@gmail.com
m.shbnpr@gmail.com
# -------------------------- HYPOTHESIS 1 ---------------------------
This script test the hypothesis 1 where the aim is:
 -  There is an effect of scene category (i.e., a difference between images showing
man-made vs. natural environments) on the amplitude of the N1 component, i.e. the
first major negative EEG voltage deflection
"""
import math
import os
import reprlib

import mne
import pandas as pd
import numpy as np
from modules import IO # for writing notes

# Initializing

os.chdir("..")
# retrieve the dirs path
dirs = IO.read_dirs(os.getcwd())
# dirs["comp"]= dirs["comp"].replace("E:","D:") ## The drive letter where the data is, given that the data was stroed on an external HDD
subject_directories = os.listdir(dirs['comp'])

for subs in subject_directories: # loop over subjects
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
                   '              Hypothesis 1               ',
                   '------------------------------------',
                   '              -------------               ',
                   'filtering data to [1 40] for ERP analysis of Hypo1',
                   '                 -------------           ',
                   '-------------- REFERENCE TO AVERAGE-------------'
                   ], 'a', "root")


    print(f'loading {subs} epoched data ...')
    events = pd.read_csv(dirs["comp"] + "/" + subs + "/" + "\event_trigger_values.txt",
                         delimiter="\t", header=6)
    conditions = dict(
        Man_made=np.unique(events.event[events.id1 == 1].astype(int)),
        Natural=np.unique(events.event[events.id1 == 2].astype(int)))
    # Check if the conditions are balanced
    #isImbalance = prep.report_number_of_trais(dirs, conditions)
    epoch = mne.read_epochs(
        dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/" + "time_domain_data_epo.fif")  # loading subjcet epoched data
    # combing relevant events
    epoch = mne.epochs.combine_event_ids(epoch,[str(items) for items in conditions["Man_made"]], {'Man_made' : 1})
    epoch = mne.epochs.combine_event_ids(epoch, [str(items) for items in conditions["Natural"]], {'Natural': 2})

    # replace the combined epochs with the original evoked
    epoch = mne.concatenate_epochs([epoch["Man_made"], epoch["Natural"]])
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    print(epoch.info)
    print('Equalizing the number of trials')
    # make note

    IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis",['\nEqualizing the number of trials',
                                                         "before: ",
                                                         "Man made = " + str(len(epoch["Man_made"])),
                                                         "Natural = " + str(len(epoch["Natural"]))],'a', "root")
    epoch.equalize_event_counts()
    print(epoch)
    IO.write_text(dirs["comp_subject"], "_general_workflow_4_hypothesis",['\nafter:',
                                                         "Man made = " + str(len(epoch["Man_made"])),
                                                         "Natural = " + str(len(epoch["Natural"]))],'a', "root")
    print('filtering data to [1 40] for ERP analysis of Hypo1')
    epoch.filter(1,40, method = 'iir')
    print('done!')
    print('computing evoked response')
    evoked_man_made = epoch["Man_made"].average()
    evoked_natural = epoch["Natural"].average()

    print('done!')
    print('saving the evoked responses')
    evoked_man_made.save(dirs["comp_subject"]["comp_subject_prepoc_time_series"]+ "/" + "hypo_1_man_made_time_domain_data_epo.fif",overwrite=True)
    evoked_natural.save(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + "/" + "hypo_1_natural_time_domain_data_epo.fif",overwrite=True)


