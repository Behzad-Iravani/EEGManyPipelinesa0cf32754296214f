"""
EEG Many Pipeline Project
Behzad Iravani, Neda Kaboodvand, Mehran Shabanpour
behzadiravani@gmail.com
n.kaboodvand@gmail.com
m.shbnpr@gmail.com

This script provides sub-function for visualizing results
"""
# --- Import modules
import matplotlib.pyplot as plt
import numpy as np
# --- End import

def set_sizes(fig_size=(9, 6), font_size= 10):
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["font.size"] = font_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["axes.labelsize"] = font_size
    plt.rcParams["axes.titlesize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size

def N100_bar_graph(dirs,ch,list,c):
    ax = plt.subplot(111)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.bar(np.array([0,  1]),
            np.array([list["N100_man_made"][c],
                     list["N100_natural"][c]]),
    # add error bars
           yerr= np.array(list["N100_man_made_SE"][c],
                     list["N100_natural_SE"][c]), align='center', alpha=0.5, ecolor = [0,0,0], capsize=10
                )
    n_sub = list["N100_man_made_dat"][:,c].shape[0]

    r = np.random.randn(n_sub, 2)
    xx1 = .1 * r[:, 0]
    xx2 = .1 * r[:, 1]
    ax.scatter(0 + xx1, list["N100_man_made_dat"][:,c], s=8, facecolors='none', edgecolors='k')
    ax.scatter(1 + xx2, list["N100_natural_dat"][:,c], s=8, facecolors='none', edgecolors='k')
    for i in range(len(xx1)):
        ax.plot([0 + xx1[i], 1 + xx2[i]], [list["N100_man_made_dat"][i,c], list["N100_natural_dat"][i,c]], color='k', ls=':', linewidth=.85)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    bottom, top = plt.ylim()
    ax.set_xticks([0,1])
    ax.set_yticks([bottom, top])
    ax.set_ylabel('Amplitude (s.e.m)')
    ax.set_xticklabels(['Man-made', 'Natural'])
    plt.title(f'N100 for {ch}')
    plt.draw()
    plt.savefig(dirs["root_folder"]+'/results/'+'hypothesis_1/'+ f'hypothesis_1_bar{ch},N100.png')
    plt.close()

def bar_graph(dirs,new, old, title, ylable_):
    ax = plt.subplot(121)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    mean_new = np.mean(new)
    mean_old = np.mean(old)
    ax.bar(np.array([0,  1]),
                  np.array([mean_new,
                            mean_old]),
    # add error bars
           yerr= np.array([np.std(new)/np.sqrt(len(new)),
                        np.std(old)/np.sqrt(len(old))]), align='center', alpha=0.5, ecolor = [0,0,0], capsize=10
                )
    r = np.random.randn(len(new), 2)
    xx1 = .1 * r[:, 0]
    xx2 = .1 * r[:, 1]

    ax.scatter(0 + xx1, new,s=8, facecolors='none', edgecolors='k')
    ax.scatter(1 + xx2, old,s=8, facecolors='none', edgecolors='k')
    for i in range(len(xx1)):
        ax.plot([0 + xx1[i], 1 + xx2[i]], [new[i], old[i]], color='k', ls=':', linewidth=.85)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    bottom, top = plt.ylim()
    ax.set_xticks([0,1])
    ax.set_yticks([bottom, top])
    ax.set_xticklabels(['New', 'Old'])
    ax.set_ylabel(ylable_)

    ax = plt.subplot(122)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.bar(np.array([1]),
           np.array(np.mean([new-old])),
           # add error bars
           yerr=np.array(np.std([new-old])/np.sqrt(.5*(len(new)+len(old)))), align='center', alpha=0.5, ecolor=[0, 0, 0], capsize=10
           )
    ax.scatter(1 + xx1, new-old,s=8, facecolors='none', edgecolors='k')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    bottom, top = plt.ylim()
    ax.set_xticks([1])
    ax.set_yticks([bottom, top])
    ax.set_xticklabels(['New > Old'])
    ax.set_ylabel('delta' + ylable_)
    #ax.set_aspect('equal')
    plt.title(title)
    plt.draw()
    plt.savefig(dirs["root_folder"]+'/results/'+'hypothesis_2/'+ 'hypothesis_2a,b,c_bar,' + title +'.png')
    plt.savefig(dirs["root_folder"] + '/results/' + 'hypothesis_2/' + 'hypothesis_2a,b,c_bar,' + title + '.svg')
    plt.close()

def cluster_tfr(dirs, t_obs, p_values, times, freqs, ch_names, cond, hypo):

    #fig.subplots_adjust(0.12, 0.08, 0.96, 0.94, 0.2, 0.43)
    # Create new stats image with only significant clusters
    for ch in range(t_obs.shape[-1]):
        # Compute the difference in tfr to determine which was greater since

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        t_obs_plot = np.nan * np.ones_like(t_obs[:,:,ch])
        tmp = t_obs[:,:,ch]
        t_obs_plot[p_values[:,:, ch] <= 0.002] = tmp[p_values[:,:, ch] <= 0.002]
        a = ax.imshow(t_obs[:,:,ch],
                extent=[times[0], times[-1], freqs[0], freqs[-1]],
                aspect='auto', origin='lower', cmap='gray',interpolation = 'none', resample= False)
        max_F = np.nanmax(abs(t_obs_plot))
        ax.imshow(t_obs_plot,
                  extent=[times[0], times[-1], freqs[0], freqs[-1]],
                  aspect='auto', origin='lower', cmap='RdBu_r',
                  vmin=-max_F, vmax=max_F, interpolation = 'none', resample= False)
        ax.set_yticks(np.linspace(4,40,10))
        ax.set_yticklabels(np.round_(freqs, 1))
        fig.colorbar(a)
        plt.title(ch_names[ch] + ' ' + cond)
        plt.draw()
        plt.savefig(dirs['root_folder']+'/results/' + hypo +'/tfr_'+ch_names[ch]+'.png')

def cluster_erp(dirs,erp1, erp2, t_obs, p_values, times, ch_names, cond, hypo):

    #fig.subplots_adjust(0.12, 0.08, 0.96, 0.94, 0.2, 0.43)
    for ch in range(t_obs.shape[-1]):
        # Compute the difference in tfr to determine which was greater since
        evoked_power_1 = np.mean(erp1[:,:,ch], axis=0)
        evoked_power_2 = np.mean(erp2[:,:,ch], axis=0)
        evoked_power_contrast = evoked_power_1 - evoked_power_2

    # Create new stats image with only significant clusters
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        t_obs_plot = np.nan * np.ones_like(evoked_power_contrast)
        t_obs_plot[p_values[:,ch]<0.05] = evoked_power_contrast[p_values[:,ch]<0.05]
        ax.plot(times,evoked_power_contrast,
                color='k')
        ax.plot(times, t_obs_plot,
                   color='r')
        plt.title(ch_names[ch] + ' ' + cond)
        plt.draw()
        plt.savefig(dirs['root_folder']+'/results/' + hypo + '/erp_'+ch_names[ch]+'.png')
