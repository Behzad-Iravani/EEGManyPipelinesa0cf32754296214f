"""
EEG Many Pipeline Project
Behzad Iravani, Neda Kaboodvand, Mehran Shabanpour
behzadiravani@gmail.com
n.kaboodvand@gmail.com
m.shbnpr@gmail.com

This is the pre-processing script that handles administrating the files, epoching and
performing ICA to remove blinks and eye movements

- No filtering has been applied
Data has been filtered later according to the research question of each hypothesis
in separate script.

"""
# ----
# # Initializing
# including core libraries:
import os
import matplotlib.pyplot as plt
from os.path import exists
import numpy as np
import mne
# ----
# Including native libraries:
from modules import preprocessing as prep
from modules import Hypothesis as Hypo
from modules import IO

# ----Initializing
os.chdir("..") # one level up in the directory list from the Scripts subfolder
# creating a dictionary of important paths
dirs = {'root_folder' : os.getcwd(),
        "BIDS_subfolder": os.getcwd() + '/raw_data/eeg_BIDS',
        "Montage": os.getcwd() + '/raw_data/channel_locations',
        'comp': os.getcwd() + '/Data' }
# -------------
# Printing the current working directory
print("The root directory is: {0}".format(dirs["root_folder"])) # This is the root folder wherein Scripts, comp, data
# are located
print("The BIDS directory is: {0}".format(dirs["BIDS_subfolder"])) # This is path to BIDS folder  ~/data/eeg-BIDS
print('Initializing done!')
# ---- End of initializing
sub = [name for name in os.listdir(dirs["BIDS_subfolder"]) if os.path.isdir(os.path.join(dirs["BIDS_subfolder"], name))]
print(f'data for {len(sub)} subjects was found.')
preprocessing = True # Change this to True to run preprocessing
if preprocessing:
    for s in sub:  # subject to process
        intro = ["This file is generated automatically for EEG many pipelines project",
         "Behzad Iravani, Neda Kaboodvand, Mehran Shabanpour",
         "behzadiravani@gmail.com",
         "n.kaboodvand@gmail.com",
         "m.shbnpr@gmail.com",
         f"------      {s}     ------\n"]
        print(f'----------- PREPROCESSING of {s} -------------')
        # Extract subject number
        s_num = ''.join([n for n in s if n.isdigit()])
        # define paths within subject directory
        dirs["comp_subject"] = { "root" : dirs["comp"] + "/Subj" + s_num[1:], # path for computational work of the specific subject
                                  "plot" :  dirs["comp"] + "/Subj" + s_num[1:] + "/" + "Plot",
                                  "comp_subject_prepoc_time_series" : dirs["comp"] + "/Subj" + s_num[1:] + "/" + "Pre-processed time series data",
                                  "comp_subject_ICA" :  dirs["comp"] + "/Subj" + s_num[1:] + "/" + "Removed ICA components (txt files)",
                                  "comp_subject_BadTrials" : dirs["comp"] + "/Subj" + s_num[1:] + "/" + "Excluded trails (txt files)",
                                  "comp_subject_BadSensors" : dirs["comp"] + "/Subj" + s_num[1:] + "/" + "Excluded sensors (txt files)"}

        IO.save_dict(dirs) # save dictionary contains paths as text file
        if not exists(dirs["comp_subject"]["root"] ):      # check if the directory already exits
            print(f' Create subject {s} folder for computation at:{dirs["comp_subject"]["root"]}')
            os.mkdir(dirs["comp_subject"]["root"] ) # creating subject directory if it does not exist
            for keys, values in dirs["comp_subject"].items():
                if keys is not "root":
                    os.mkdir(values)        # creating the subject sub-directories
        if exists(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + '/' + 'time_domain_data_epo.fif'): # checking if the data is already pre-processed
            print(f'{s} is already preprocessed, moving to next subject')
            continue
        print(f'subject {s} is initializing for processing ')
        # creating the bad trials file for the subject
        IO.write_text(dirs["comp_subject"], 'Excluded trials', intro, 'w', "comp_subject_BadTrials")
        # creating the bad sensors file for the subject
        IO.write_text(dirs["comp_subject"], 'Excluded sensors', intro, 'w', "comp_subject_BadSensors")
        # creating the bad trials file for the subject
        IO.write_text(dirs["comp_subject"], 'Removed ICA', intro, 'w', "comp_subject_ICA")
        # creating the hypothesis file for the subject
        IO.write_text(dirs["comp_subject"], '_general_workflow_4_hypothesis', intro, 'w',"root")
        # creating the event file for the subject
        IO.write_text(dirs["comp_subject"],'event_trigger_values', intro,'w',"root")

        # ------ CREATE MONTAGE ------
        print(f'creating montage for {s} from BESA text file...')
        montage = prep.create_montage_from_CED(dirs)
        # ----------------------------
        # getting the subjects' EEG folder's name
        prep.get_subject_folders(dirs["BIDS_subfolder"],verbose= False)
        # load the EEG in BIDS format for a give subjects
        print(f'reading raw for sub:{s}')
        raw, bids_path = prep.load_data(s, dirs["BIDS_subfolder"])
        # reading the events timing from the text file
        events = prep.load_events(bids_path=bids_path)
        # load raw to memory
        raw.load_data()
        print('done!')
        print('setting the  Montage...')
        print('done!')
        # add channel types, 70 EEG, 2 EOG
        raw.set_channel_types(dict(zip(raw.info.ch_names, np.concatenate([['eeg'] * 70, ['eog']*2]) )))
        raw.set_montage(montage)
        # --------------------------------------
        # Perform ICA to remove ocular artifacts
        if not exists(dirs["comp_subject"] ["comp_subject_prepoc_time_series"]+ '/' +  'time_domain_preproc_continuous.fif'):
            data_ica = prep.remove_eog_with_ICA(raw, dirs)
            plt.close('all')  # close the saved ICA figures
            print('ICA is done!')
        else:
            print('ICA file exists, skipping performing ICA!')
            data_ica = mne.io.Raw(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + '/' +  'time_domain_preproc_continuous.fif', preload= True)
            print('done!')
        # look for bad channels based on variance
        print('detecting bad channels based on their variance....')
        th = 3 # threshold for detecting bad channels
        ch_bad_idx = prep.detect_bad_chennels(raw=data_ica, th=th)
        # create slices
        slc = np.concatenate([list(items) for items in ch_bad_idx])
        #
        data_ica.info["bads"] = [data_ica.info.ch_names[x] for x in slc] # add the bad channel indices to data object
        print(f'{data_ica.info["bads"]} was(were) interpolated with spline.')
        IO.write_text(dirs["comp_subject"], 'Excluded sensors', [f'\nBad channels based on the z-scored variance |Z(Var(ch))|>{th}: {data_ica.info["bads"]}'], 'a', "comp_subject_BadSensors")
        # ----------- Interpolating the bad channels -----------------------
        data_ica.interpolate_bads(method= dict(eeg = 'spline'))
        IO.write_text(dirs["comp_subject"], 'Excluded sensors', ['\nInterpolating channels with the spline method.'], 'a',"comp_subject_BadSensors") # make a note that spline was used for interpolating the bad channels
       #  ---------- Detecting bad trials based on amplitude ---------------
        IO.write_text(dirs["comp_subject"], 'Excluded trials', ['\nTrial indices rejected at step 1 (before ICA): N/A'], 'a',"comp_subject_BadTrials") # make a note that no trials were removed before ICA
        # Load events data
        events = prep.load_events(bids_path=bids_path)
        events = np.stack(events,axis = 0)
        # Creating the hypothesis object
        # HYPOTHESIS 1
        Hypo1 = Hypo.HypoClass.find_relevant_events(dirs= dirs,
                                             hypothesis_num= 1,
                                             description = "scene category and N1 amplitude",
                                             events = events)
        # HYPOTHESIS 2
        Hypo2 = Hypo.HypoClass.find_relevant_events(dirs= dirs,
                                             hypothesis_num= 2,
                                             description= "effects of image novelty and fronto-central from 300–500 ms",
                                             events= events)
        # HYPOTHESIS 3
        Hypo3 = Hypo.HypoClass.find_relevant_events(dirs= dirs,
                                             hypothesis_num= 3,
                                             description= "There are effects of [hits] vs. [misses] anywhere",
                                             events= events)
        # HYPOTHESIS 4
        Hypo4 = Hypo.HypoClass.find_relevant_events(dirs=dirs,
                                             hypothesis_num= 4,
                                             description= "There are effects of successfully remembered vs. forgotten on a subsequent",
                                             events= events)
        org_events = prep.organized_events(Hypo1, Hypo2, Hypo3, Hypo4) # organizing events


        print(f'Epoching {s} EEG data and detecting bad epochs... ')
        epoched = prep.epoch_data(mne_raw=data_ica, events=org_events)
        droped_trials = np.zeros(len(np.squeeze(org_events['event'])))
        droped_trials[np.squeeze(np.where(epoched.drop_log))]=1
        # make note of the dropped epoch indices after ICA
        IO.write_text(dirs["comp_subject"], 'Excluded trials', [f'\nTrial indices rejected at step 2 (after ICA): {np.squeeze(np.where(droped_trials))}'],
                      'a',"comp_subject_BadTrials")

        org_events['dropped'] =droped_trials
        # converting the events dictionary to text
        event_text = IO.write_organized_events(org_events)
        # write the events as text file
        IO.write_text(dirs["comp_subject"], 'event_trigger_values',event_text, 'a',"root")
        # save epoched and preprocessed data
        epoched.save(dirs["comp_subject"]["comp_subject_prepoc_time_series"] + '/' + 'time_domain_data_epo.fif', overwrite=True)

        print(f'----------- PREPROCESSING DONE for {s}! ----------' )



