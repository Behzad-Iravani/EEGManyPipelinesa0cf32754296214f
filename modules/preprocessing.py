'''This is a native module for preprocessing of EEG many pipline project using mne python
it consists of several functions for loading EEG file and administrative tasks

# BEHZAD IRAVANI, Neda Kaboodvand, Mehran Shabanpour
behzadiravani@gmail.com
n.kaboodvand@gmail.com
m.shbnpr@gmail.com
# MARCH 2022
'''
# ---- import
import os
import numpy as np
import mne as mne
import pandas as pd
from mne_bids import BIDSPath, read, print_dir_tree
import matplotlib.pyplot as plt
from modules import IO
from scipy import stats
import re
# end import ----

def replace_outlier_with_median(x):
    mx = np.nanmean(x, axis=0)
    sx = np.nanstd(x,axis=0)
    mdn = np.nanmedian(x,axis=0)
    if not np.isscalar(mdn):
        zscore_x = (x-np.tile(mx, (x.shape[0],1)))/np.tile(sx, (x.shape[0],1))
        outliers = np.abs(zscore_x)>2.7
        for cols in range(outliers.shape[1]):
            x[outliers[:, cols], cols] = mdn[cols]
            print(f'{np.sum(outliers[:, cols])} outliers were replaced with median')
    else:
        zscore_x = (x - mx) / sx
        outliers = np.abs(zscore_x) > 2.7
        x[outliers] = mdn
        print(f'{np.sum(outliers)} outliers were replaced with median')

    return x, zscore_x

def get_subject_folders(bids_root = "C:\\", verbose = False):
    # This function retrieves the subject directories from the BIDS-path
    # Input:
    #       bids_root     = path to BIDS directory where the subjects' data are located
    #       verbose       = if True then the directory tree is printed out for user.
    # Output:
    #       subject_folder= a string list contains the name of subjects' directories
    #       num_subjects  = the total number of subjects

    if verbose:
        print_dir_tree(bids_root, max_depth=4)
    subject_folders = os.listdir(bids_root)
    isdir = [os.path.isdir(bids_root +'/'+items) for items in subject_folders]
    # remove non-directory items
    subject_folders = [d for (d, keep) in zip(subject_folders, isdir) if keep]
     # ---
    num_subjects = np.shape(subject_folders)
    print(f'{num_subjects[0]} found in this directory.')
    return subject_folders, num_subjects
def create_montage_from_CED(dirs):
    # This function creates EEG montage from the CED.txt file
    # Input:
    #       dirs         =  a dictionary contains the essential paths
    # Output:
    #       montage_subject= subject montage that is created using mne toolbox and the CED.txt file

    df_ = pd.read_csv(dirs["Montage"]+'/'+'chanlocs_ced.txt',sep= '\t') # reading the csv file using pandas package
    ch_names_ = df_.labels.to_list() # converting the dataframe to list
    # converting the strings to numeric values
    df_['X'] = pd.to_numeric(df_['X'], errors='coerce')
    df_['Y']= pd.to_numeric(df_['Y'],errors='coerce')
    df_['Z'] = pd.to_numeric(df_['Z'], errors='coerce')
    pos = df_[['X', 'Y', 'Z']].values
    montage_subject =mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names_,pos.astype(float))))
    #montage.plot()
    return montage_subject

def load_data(subject, bids_root):
    # This function loads the subject data into workspace
    # Input:
    #       subject   = a string contains the subject identifier
    #       bids_root = a string contains the path to BIDS directory
    # Output:
    #        raw      = a raw object contains the subject EEG raw data
    #        bids_path= a string contains the BIDS path of the subject

    m = re.search('\d+', subject) # extracting the subject numeric identifier
    task = 'xxxx'
    # suffix = 'eeg'
    datatype = 'eeg'
    bids_path = BIDSPath(subject=m.group(0), task=task,
                         root=bids_root, datatype=datatype) # suffix=suffix
    print(bids_path.match())
    raw = read.read_raw_bids(bids_path=bids_path, verbose=False)

    return raw, bids_path

def read_events(events_path):
    # This function reads the events from tsv/text file and extracts the relevant information
    # Input:
    #       events_path = a string contains the path to event text file
    # Output:
    #        events     = a list contains the events' onset, sample and value

    event = open(events_path) # opening the event tsv file
    line_count = 0 # counter for lines in the text file
    l=0 # counter for adding line number to the event list
    events = [] # initializing the event list
    while True:
        if line_count ==0: # HEADER
            HEADER = event.readline().split('\t')
            counter_column = 0
            column = []
            for items in HEADER:
                if "onset" in items:
                    column.append(counter_column)
                elif "sample"in items:
                    column.append(counter_column)
                elif "value" in items:
                    column.append(counter_column)
                counter_column +=1
            line_count += 1
            continue

        line = event.readline()[:-1].split('\t') # last character is a line breaker
        if len(line) < 5: # if the line contains less than 5 characters the files has come to the end
            print('End of the event file')
            break
        else:
            if np.all([line[x].replace('.','',1).isdigit()for x in column]):
                events.append(np.array([float(line[x]) for x in column]))
                print(f'trial {l+1}: ' + ','.join([line[x] for x in column]))
                l+=1
        line_count += 1

    return events

def load_events(bids_path):
    # This function reading events form .tsv to the workspace
    # Input:
    #        bids_path = a string contains the BIDS path
    # Output:
    #        events    = a list contains the events' onset, sample and value
    events_path = bids_path.update(suffix='events',
                                   extension='.tsv')
    events = read_events(events_path)
    return events


def remove_eog_with_ICA(raw, dirs):
    # This function removes the EOG artifact using ICA and mne functions
    # Input:
    #       raw        = a raw mne object contains raw EEG data
    #       dirs       = a dictionary contains the essential paths
    # Output:
    #       ica_removed= a raw mne object that contains the corrected EEG data
    # ---- load EEG data ----
    print('loading data to perform ICA... ')
    raw.load_data()
    print('loading is done! ')
    # ------------------------
    # ---- running ica -------
    print('starting ICA... ')
    ica = mne.preprocessing.ICA(n_components=20, random_state=0)
    ica.fit(raw.copy().filter(8, 35)) # filter data before performing ICA
    # identify the components to remove with threshold of 2
    blinks_bad_idx, _ = ica.find_bads_eog(raw, ['VEOG'], threshold=2)
    eyemovement_bad_idx, _ = ica.find_bads_eog(raw, ['HEOG'], threshold=2)
    ica.exclude = np.concatenate([blinks_bad_idx, eyemovement_bad_idx]) # components to be removed
    plt.close()
    ica_plot = ica.plot_components(outlines="skirt")
    ica_plot[0].savefig(dirs["comp_subject"]["plot"]+ '/' + "ICA_maps.png")
    # -------------------------
    # ----- plot a minute of data before ICA for comparison
    raw.plot(duration=60.0, start=0.0, n_channels=16,scalings = dict(eeg=100e-6, eog=200e-6))
    plt.savefig(dirs["comp_subject"]["plot"] + '/' + "beforeICA.png")
    # ------- applying ICA ------
    ica_removed = ica.apply(raw.copy(), exclude=ica.exclude) # ICA components were removed from unfiltered data
    ica_removed.plot(duration=60.0, start=0.0, n_channels=16,scalings = dict(eeg=100e-6, eog=200e-6))
    # ----- plot data after removing bad ICA components
    plt.savefig(dirs["comp_subject"]["plot"] + '/' + "afterICA.png")
    ica_removed.save(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + '/' + "time_domain_preproc_continuous.fif")
    # ------ write the report ------
    IO.write_text(dirs["comp_subject"], 'Removed ICA',[f'ICA was used to remove eog artifacts including blink and eye-movement: ',
                              'Total number of components = 20.',
                              f'The overall number of components to remove was = {len(ica.exclude)}.',
                              f'blink bad components = {blinks_bad_idx}.'
                              f'eye-movement components =  {eyemovement_bad_idx}.'],'a',"comp_subject_ICA")
    ica.unmixing_matrix_.tofile(dirs["comp_subject"]["comp_subject_ICA"] + '/Unmixing_matrix.txt',sep=',', format='%f')
    ica.mixing_matrix_.tofile(dirs["comp_subject"]["comp_subject_ICA"] + '/mixing_matrix.txt',sep=',', format='%f')
    try:
        ica.save(dirs["comp_subject"]["comp_subject_ICA"]+'\ICA.fif')
    except:
        print('Cannot safely write data with dtype float64 as int')

    print(ica.mixing_matrix_)
    return ica_removed
def detect_bad_chennels(raw,th):
    # This function detects bad channels based on absolute z-scored variance
    # Input:
    #       raw  = raw mne object contains EEG data
    #       th   = a scalar determines the threshold
    # Output:
    #       ch_bad_idx = a ndarray contains indicating the bad channels
    dat = raw.get_data() # retrieving the data matrix
    v = np.abs(stats.zscore(np.var(dat, axis=1), ddof = 1)) # absolute z-scored variance
    ch_bad_idx = np.where(v>th) # indicating where the absolute z-score variance is above threshold
    return ch_bad_idx

def epoch_data(mne_raw, events):
    # This function epochs EEG data from -500ms to 1000ms
    # Input:
    #       mne_raw  = raw mne object contains continuous EEG data
    #       events   = a dictionary of events contains onsets, samples and values
    # Output:
    #        raw_epoched = epoched mne object

    num_tr = len(events['onset'][0])
    print(f'total {num_tr} trials detected.')
    mneEVENTS = np.array([events['onset'][0],np.zeros(num_tr), np.squeeze(events["event"])])

    # reject based on the amplitude
    print('remove trials with large amplitude')
    reject_criteria = dict(eeg=500e-6,  # 500 µV
                           eog=800e-6)  # 800 µV
    #flat_criteria = dict(eeg=.1e-6)  # .1 µV
    tmin = -.5  # start of each epoch (500ms before the trigger)
    tmax = 1  # end of each epoch (1000ms after the trigger)
    # Epoching EEG data using mne.epoch method
    raw_epoched = mne.Epochs(raw = mne_raw, events= np.transpose(mneEVENTS.astype(int)),
                             event_id =None, tmin=tmin, tmax= tmax,
                             event_repeated = 'error', reject=reject_criteria)#, flat=flat_criteria)

    raw_epoched.drop_bad() # drop the events with artifacts

    return raw_epoched


def organized_events(Hypo1, Hypo2, Hypo3, Hypo4):
    # This function sorts events and create specific event value for each hypothesis
    # Input:
    #       Hypo1, Hypo2, Hypo3, Hypo4 = hypothesis object defined by hypoclass object
    # Output:
    #        organized_events = a dictionary contains a specific event values for each hypothesis
    # --- Hypothesis 1 ----
    trnum = len(Hypo1.events["onset"][0])
    # --- End Hypothesis 1 ---
    # --- Hypothesis 2 ----
    index2 = []
    for items in Hypo2.events["onset"][0]:
        mapindex = np.where(Hypo1.events["onset"][0] == items)
        if mapindex:
            index2.append(mapindex)
    new_event2 = np.ones(trnum)*9 # unrelated conditions are determined by value of 9
    new_event2[np.squeeze(index2)] = Hypo2.events["Hypo_related_events"][np.where([items < 9 for items in  list(Hypo2.events["Hypo_related_events"])])]
    # --- End Hypothesis 2 ---
    # --- Hypothesis 3 ----
    index3 = []
    for items in Hypo3.events["onset"][0]:
        mapindex = np.where(Hypo1.events["onset"][0] == items)
        if mapindex:
            index3.append(mapindex)
    new_event3 = np.ones(trnum) * 9 # unrelated conditions are determined by value of 9
    new_event3[np.squeeze(index3)] = Hypo3.events["Hypo_related_events"][np.where([items < 9 for items in  list(Hypo3.events["Hypo_related_events"])])]
    # --- End Hypothesis 3 ---
    # --- Hypothesis 4 ----
    index4 = []
    for items in Hypo4.events["onset"][0]:
        mapindex = np.where(Hypo1.events["onset"][0] == items)
        if mapindex:
            index4.append(mapindex)
    new_event4 = np.ones(trnum)*9 # unrelated conditions are determined by value of 9
    new_event4[np.squeeze(index4)] =  Hypo4.events["Hypo_related_events"][np.where([items < 9 for items in  list(Hypo4.events["Hypo_related_events"])])]
    # --- End Hypothesis 4 ---
    organized_events = dict(trial_no = np.array(range(trnum))+1,
                        onset = Hypo1.events["onset"],
                        event = Hypo1.events['event'],
                        id1 = Hypo1.events['Hypo_related_events'],
                        id2 = new_event2,
                        id3 = new_event3,
                        id4 = new_event4)
    return organized_events


def find_and_report_significantERP(dirs, t_obs, ps_fdr, times, ch_names, hypo):
    # This function produces an automatic report for significant ERP instances
    # Input:
    #       dirs   =  a dictionary contains the essential paths
    #       t_obs  =  a ndarray contains the observed t-values
    #       ps_fdr =  a ndarray contains either corrected or not corrected p-values
    #       times  =  a ndarray contains time bins
    #       ch_names = a list contains the channels' name
    #       hypo     = a string contains the hypothesis number

    lines = [] # a list that contains the lines of report
    # ch_names
    dm_chan = len(ch_names)
    dm_times = times.shape[0]
    ch_names = np.tile(np.array(ch_names), (dm_times,1)) # expanding the channels dimension
    times    = np.tile(times, (dm_chan,1))

    # vectorize
    t_obs    = np.reshape(t_obs ,-1)
    ch_names = np.reshape(ch_names,-1)
    times    = np.reshape(times.T,-1)
    # -----
    c = 0
    lines.append( 'ch_names, t, p, times')
    for p in np.reshape(ps_fdr,-1):
        if p <= 0.05:
            print(f'{ch_names[c]}, {t_obs[c]}, {p}, {times[c]}')
            lines.append(f'{ch_names[c]}, {t_obs[c]}, {p}, {times[c]}')
        c += 1
    with open(dirs['root_folder']+'\\results\\'+ hypo + '\\_erp_cluster_permutation.csv','w') as f:
        f.write('\n'.join(lines))

def find_and_report_significantTFR(dirs, t_obs, p_values, times, freqs, ch_names, hypo):
    # This function produces an automatic report for significant TFR instances
    # Input:
    #       dirs   =  a dictionary contains the essential paths
    #       t_obs  =  a ndarray contains the observed t-values
    #       ps_values =  a ndarray contains either corrected or not corrected p-values
    #       times  =  a ndarray contains time bins
    #       freqs    = a ndarray contains frequency bins
    #       ch_names = a list contains the channels' name
    #       hypo     = a string contains the hypothesis number
    lines = []
    lines.append('ch_names, t, p, freqs, times')
    cc = 0
    for ch in range(t_obs.shape[-1]):
        ct = 0
        for t in range(t_obs.shape[1]):
            cf = 0
            for f in range (t_obs.shape[0]):
                if p_values[cf,ct,cc]<= 0.002:
                    lines.append(f' {ch_names[cc]}, {t_obs[cf,ct,cc]}, {p_values[cf,ct,cc]}, {freqs[cf]}, {times[ct]}')
                cf += 1
            ct += 1
        cc += 1
    with open(dirs['root_folder'] + '\\results\\' + hypo + '\\_tfr_cluster_permutation.csv', 'w') as f:
        f.write('\n'.join(lines))
    #for items in cluster_p_values if items < 0.05


"""
def report_number_of_trais(dirs, Conditions):
    lines = []
    num_trials = []
    imbalance = False
    for keys, values in Conditions.items():
        num_trials.append(len(values[0]))
        lines.append("\n number of trials for " + keys + " : " + str(len(values[0])))
    percentage = num_trials/np.sum(num_trials)
    IO.write_text(dirs, "_general_workflow_4_hyothesis",lines, 'a')

    if np.std(percentage)>.5:
        warnings.warn("The trails are imbalanced")
        imbalance = True

    return imbalance
"""


