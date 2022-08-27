"""
The hypothesis class encapsulate the information and methods for each specific hypothesis
In total there ara four hypothesis sought to assess:
    Hypothesis 1) There is an effect of scene category for N1 amplitude
    Hypothesis 2) There are effects of image novelty within the time-range from 300–500 ms ...
                a. ... on EEG voltage at fronto-central channels.
                b. ... on theta power at fronto-central channels.
                c. ... on alpha power at posterior channels.
    Hypothesis 3) There are effects of [hits] vs. [misses] ...
                a. ... on EEG voltage at any channels, at any time.
                b. ... on spectral power, at any frequencies, at any channels, at any time.
    Hypothesis 4) There are effects of successfully remembered vs. forgotten on a subsequent repetition ...
                a. ... on EEG voltage at any channels, at any time.
                b. ... on spectral power, at any frequencies, at any channels, at any time.
"""
import numpy as np
from modules import IO

class HypoClass:
    def __init__(self, dirs, hypothesis_num, description, events = None, levels_out = None, info= None):
        self.dirs = dirs # a dictionary contains the path to sub-folders
        self.hypothesis_num = hypothesis_num # an integer indexing the hypothesis number (1 to 4)
        self.description = description # a string summarizes the hypothesis.
        self.events = events # a numpy array with 3 columns [onset(time), samples, event_id] read from events.tsv
        self.levels_out = levels_out # a dictionary contains the ids and labels for the events
        self.info = info # info to report

    def generate_event_codes(hypothesis_num):
        """
        This function generates any possible events codes for a given hypothesis
        Input:
            hypothesis_num: integer, indexing the hypothesis to be examined
        Output:
            event_codes: a list contains the possible events id for a given hypothesis
            levels: a list contains dictionaries encompassing the events' name and ids

        """
        levels = [{'man-made': 1, 'natural': 2},
                  {'new': 0, 'old': 1},
                  {'hit': 1, 'miss_forgotten': 2, 'false_alarm': 3, 'correct_rejection': 4, 'NA': 9},
                  {'subsqnt_forgotten': 0, 'subsqnt_remembered': 1, 'NA': 9}
                  ]

        event_codes = [[], []]
        if hypothesis_num == 1:
            for items1 in levels[1]:
                for items2 in levels[2]:
                    for items3 in levels[3]:
                        event_codes[0].append(
                            levels[0]["man-made"] * 1e3 + levels[1][items1] * 1e2 + levels[2][items2] * 1e1 + levels[3][
                                items3])
                        event_codes[1].append(
                            levels[0]["natural"] * 1e3 + levels[1][items1] * 1e2 + levels[2][items2] * 1e1 + levels[3][
                                items3])
        elif hypothesis_num == 2:
            for items0 in levels[0]:
                for items2 in levels[2]:
                    for items3 in levels[3]:
                        event_codes[0].append(
                            levels[0][items0] * 1e3 + levels[1]["new"] * 1e2 + levels[2][items2] * 1e1 + levels[3][
                                items3])
                        event_codes[1].append(
                            levels[0][items0] * 1e3 + levels[1]["old"] * 1e2 + levels[2][items2] * 1e1 + levels[3][
                                items3])
        elif hypothesis_num == 3:
            for items0 in levels[0]:
                for items1 in levels[1]:
                    for items3 in levels[3]:
                        event_codes[0].append(
                            levels[0][items0] * 1e3 + levels[1][items1] * 1e2 + levels[2]['hit'] * 1e1 + levels[3][
                                items3])
                        event_codes[1].append(
                            levels[0][items0] * 1e3 + levels[1][items1] * 1e2 + levels[2]['miss_forgotten'] * 1e1 + levels[3][
                                items3])
        elif hypothesis_num == 4:
            for items0 in levels[0]:
                for items1 in levels[1]:
                    for items2 in levels[2]:
                        event_codes[0].append(
                            levels[0][items0] * 1e3 + levels[1][items1] * 1e2 + levels[2][items2] * 1e1 + levels[3][
                                "subsqnt_forgotten"])
                        event_codes[1].append(
                            levels[0][items0] * 1e3 + levels[1][items1] * 1e2 + levels[2][items2] * 1e1 + levels[3][
                                "subsqnt_remembered"])


        return event_codes, levels

    def obtain_sort_indexs_of_items(levels, events_code, events):
        """
        Input:
            levels : a dictionary containing the labels and ids for each condition, e.g., 'Man-made': 1
            events_code: a list contains the possible id (1010 etc.) for each condition in the levels
            events: a numpy array with 3 columns [onset(time), samples, event_id] read from events.tsv
        Output:
            index_w: trialx0 numpy array containing the available events in the file
            sorted_index: trialx0 numpy array, sorted index_w based on the time of onset
            num_trials4items: numpy array containing the number of trials per each condition

        """
        events_relate2hypo = np.ones(len(events))*9
        index = []
        num_trials4items = np.zeros(len(levels))  # initializing the number of trails for a given item across conditions
        value_ = list(levels.values()) # get the values for the events of this given hypothesis

        for con in range(len(levels)):
            for items in events_code[con]:
                if len(np.where(events[:, -1] == items)[0]) > 0:  # if such items exist in the event list
                    index.append(np.array(np.where(events[:, -1] == items)).reshape(
                        -1))  # search for the index of events relevant to Hypo1
                    events_relate2hypo[np.array(np.where(events[:, -1] == items))] = value_[con]

                    num_trials4items[con] += np.sum(
                        events[:, -1] == items)  # obtaining the number of trials for such item
            # ------

        index_w = np.concatenate(index)  # concatenating the conditions
        sorted_index = np.argsort(events[index_w, 0], axis=-1) # sort the events based on the time of onset
        return index_w, sorted_index, num_trials4items,events_relate2hypo

    @classmethod
    def find_relevant_events(cls, dirs, hypothesis_num, description, events = None):
        """
        A class method to obtain the relevant events for a given hypothesis from the list of all possible combinations.
        Input:
            dirs: a dictionary contains the paths to different sub-folders
            hypothesis_num: an integer indexing the hypothesis to be tested
            description:  a string describing the hypothesis
            events: a numpy array with 3 columns [onset(time), samples, event_id] read from events.tsv

        """
        def event_codes2decsriptive(event_code, levels):
            """
            this function creates a descriptive events id for the record.
            input:
                event_code: a list contains the possible id (1010 etc.) for each condition in the levels
                levels: a dictionary containing the labels and ids for each condition, e.g., 'Man-made': 1
            output:
                event_descriptive: a list contains the descriptive ids of events
            """
            global report_lines # golobal variable containing the lines to be written in report text

            def reverse_dictionary(D, l, single_event):
                # d: dictionary to be reversed
                # l: level of the list where dictionary is
                pp = 3 - l
                inv_map = {v: k for k, v in D[l].items()}
                label = inv_map[single_event // 10 ** pp % 10]
                return label

            event_descriptive = []
            for items in event_code:
                event_descriptive.append(reverse_dictionary(levels, 0, items) + ';' +
                                         reverse_dictionary(levels, 1, items) + ';' +
                                         reverse_dictionary(levels, 2, items) + ';' +
                                         reverse_dictionary(levels, 3, items))
            return event_descriptive

        relevant_events = dict(onset=[], event=[],
                               Hypo_related_events=[])
        # assess the hypothesis number
        if hypothesis_num == 1:  # Hypothesis 1, scene category and N1 amplitude

            info = ['---------------------------',
                                       f'Hypothesis {hypothesis_num}: {description}',
                                       '---------------------------']
            events_code, levels = cls.generate_event_codes(
                hypothesis_num)  # Events for 2 conditions [0]: Man-Made, [1]: Natural
            info.append(f'Condition 1 {list(levels[0])[0]} possible events_code:{events_code[0]}')
            info.append(f'Condition 1 {list(levels[0])[1]} possible events_code:{events_code[1]}')
            info.append('\n'.join(event_codes2decsriptive(events_code[0], levels)))
            # Extract the event indices relevant to Conditions, Hypothesis 1
            index_w, sorted_index, num_trials4items, events_relate2hypo = cls.obtain_sort_indexs_of_items(levels[0], events_code, events)
            # creating a dictionary that contains the onset
            relevant_events["onset"].append(np.squeeze(events[index_w[sorted_index], 1]))
            relevant_events["event"].append(np.squeeze(events[index_w[sorted_index], -1]))
            relevant_events["Hypo_related_events"] = events_relate2hypo
            levels_out = levels[0]
        elif hypothesis_num == 2:  # Hypothesis 2, effects of image novelty and fronto-central from 300–500 ms
            info = ['---------------------------',
                                       f'Hypothesis {hypothesis_num}: {description}',
                                       '---------------------------']
            events_code, levels = cls.generate_event_codes(
                hypothesis_num)  # Events for 2 conditions [0]: new, [1]: old
            info.append(f'Condition 1 {list(levels[1])[0]} possible events_code:{events_code[0]}')
            info.append(f'Condition 1 {list(levels[1])[1]} possible events_code:{events_code[1]}')
            info.append('\n'.join(event_codes2decsriptive(events_code[0], levels)))
            # Extract the event indices relevant to Conditions, Hypothesis 1
            index_w, sorted_index, num_trials4items, events_relate2hypo = cls.obtain_sort_indexs_of_items(levels[1], events_code, events)
            # creating a dictionary that contains the onset
            relevant_events["onset"].append(np.squeeze(events[index_w[sorted_index], 1]))
            relevant_events["event"].append(np.squeeze(events[index_w[sorted_index], -1]))
            relevant_events["Hypo_related_events"] = events_relate2hypo
            levels_out = levels[1]
        elif hypothesis_num == 3:  # Hypothesis 3, There are effects of [hits] vs. [misses] anywhere
            info = ['---------------------------',
                                       f'Hypothesis {hypothesis_num}: {description}',
                                       '---------------------------']
            events_code, levels = cls.generate_event_codes(
                hypothesis_num)  # Events for 2 conditions [1]: Hit, [2]: Miss
            info.append(f'Condition 1 {list(levels[2])[0]} possible events_code:{events_code[0]}')
            info.append(f'Condition 1 {list(levels[2])[1]} possible events_code:{events_code[1]}')
            info.append('\n'.join(event_codes2decsriptive(events_code[0], levels)))
            # Extract the event indices relevant to Conditions, Hypothesis 1
            index_w, sorted_index, num_trials4items, events_relate2hypo = cls.obtain_sort_indexs_of_items(
                {key: value for key, value in levels[2].items() if value == 1 or value ==2}, # select subset of ids ->[hits] and [miss]
                events_code, events)
            # creating a dictionary that contains the onset
            relevant_events["onset"].append(np.squeeze(events[index_w[sorted_index], 1]))
            relevant_events["event"].append(np.squeeze(events[index_w[sorted_index], -1]))
            relevant_events["Hypo_related_events"] = events_relate2hypo
            levels_out = {key: value for key, value in levels[2].items() if value == 1 or value ==2}

        elif hypothesis_num == 4:  # There are effects of successfully remembered vs. forgotten on a subsequent
            info = ['---------------------------',
                    f'Hypothesis {hypothesis_num}: {description}',
                    '---------------------------']
            events_code, levels = cls.generate_event_codes(
                hypothesis_num)  # Events for 2 conditions [1]: Hit, [2]: Miss
            info.append(f'Condition 1 {list(levels[3])[0]} possible events_code:{events_code[0]}')
            info.append(f'Condition 1 {list(levels[3])[1]} possible events_code:{events_code[1]}')
            info.append('\n'.join(event_codes2decsriptive(events_code[0], levels)))
            # Extract the event indices relevant to Conditions, Hypothesis 1
            index_w, sorted_index, num_trials4items, events_relate2hypo = cls.obtain_sort_indexs_of_items(
                {key: value for key, value in levels[3].items() if value<=1}, # select subset of ids ->[subseqnt. forgotten] and [subseqnt. remembered]
                events_code,events)
            # creating a dictionary that contains the onset
            relevant_events["onset"].append(np.squeeze(events[index_w[sorted_index], 1]))
            relevant_events["event"].append(np.squeeze(events[index_w[sorted_index], -1]))
            relevant_events["Hypo_related_events"] = events_relate2hypo
            levels_out = {key: value for key, value in levels[3].items() if value<=1}
        else:
            raise 'This hypothesis has not been defined!'



        return cls(dirs, hypothesis_num, description, relevant_events, levels_out, info)


