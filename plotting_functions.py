import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def plot_raster(EED, group_name=None, recording_name=None, cellid_name=None):
    """
    Plot raster plots for pre and post epochs at different stimulation levels.

    Parameters:
    EED (object): The object containing the electrophysiology data.
    group_name (str, optional): The name of the group to plot. Defaults to None, in which case all groups are plotted.
    recording_name (str, optional): The name of the recording to plot. Defaults to None, in which case all recordings are plotted.
    cellid_name (str, optional): The name of the cell ID to plot. Defaults to None, in which case all cell IDs are plotted.

    Returns:
    None: The function saves the plots to the 'rasterplots' directory.
    """
    
    # Ensure 'rasterplots' directory exists
    if not os.path.exists('rasterplots'):
        os.makedirs('rasterplots')
    
    # Define stimulation levels
    stim_levels = ['Zero', 'Low', 'Mid', 'Max']
    
    # Get list of group names
    group_names = [group_name] if group_name else EED.group_names
    
    # Loop through each group, recording, and cell ID
    for group in group_names:
        recording_names = [recording_name] if recording_name else EED.get_recording_names(group)
        for recording in recording_names:
            cellid_names = [cellid_name] if cellid_name else EED.get_cellid_names(group, recording)
            for cell_id in cellid_names:
                # Get the pre and post stim data
                data = EED.get_pre_post_data(group, recording, cell_id)
                
                # Create a new figure
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                
                # Loop through each stimulation level and plot raster plots for pre and post epochs
                for stim_index, stim_level in enumerate(stim_levels):
                    ax = axes[stim_index // 2, stim_index % 2]
                    
                    # Get spike trains for the current stimulation level
                    pre_spiketrains = data['Pre']['SpikeTrains_for_PSTHs'][stim_index]
                    post_spiketrains = data['Post']['SpikeTrains_for_PSTHs'][stim_index]
                    
                    # Plot raster plot for pre epoch in grey
                    for trial_index, train in enumerate(pre_spiketrains):
                        ax.eventplot(np.where(train)[0] - 500, lineoffsets=trial_index, linelengths=0.8, color='grey')
                    
                    # Plot raster plot for post epoch in blue
                    for trial_index, train in enumerate(post_spiketrains):
                        ax.eventplot(np.where(train)[0] - 500, lineoffsets=trial_index + len(pre_spiketrains), linelengths=0.8, color='blue')
                    
                    # Add a faint red line at 0 ms to indicate the stimulus onset
                    ax.axvline(0, color='red', linestyle='--', linewidth=0.5)
                    
                    # Set subplot title and labels
                    ax.set_title(stim_level)
                    ax.set_xlabel('Time (ms)')
                    ax.set_ylabel('Trial')
                
                # Adjust the vertical spacing between subplots to prevent overlap
                plt.subplots_adjust(hspace=0.4)
                
                # Set main plot title
                fig.suptitle(f'Group: {group}, Recording: {recording}, Cell ID: {cell_id}')
                
                # Save the plot to the 'rasterplots' directory
                plt.savefig(f'rasterplots/{group}_{recording}_{cell_id}.png')
                
                # Close the plot to prevent it from displaying inline in the notebook
                plt.close(fig)


def plot_psth(EED, group_name=None, recording_name=None, cellid_name=None, window_size=5):
    """
    Plot PSTH (Peri-Stimulus Time Histogram) for pre and post epochs at different stimulation levels.

    Parameters:
    EED (object): The object containing the electrophysiology data.
    group_name (str, optional): The name of the group to plot. Defaults to None, in which case all groups are plotted.
    recording_name (str, optional): The name of the recording to plot. Defaults to None, in which case all recordings are plotted.
    cellid_name (str, optional): The name of the cell ID to plot. Defaults to None, in which case all cell IDs are plotted.
    window_size (int, optional): The size of the smoothing window. Defaults to 5.

    Returns:
    None: The function saves the plots to the 'psthplots' directory.
    """
    
    # Ensure 'psthplots' directory exists
    if not os.path.exists('psthplots'):
        os.makedirs('psthplots')
    
    # Define stimulation levels
    stim_levels = ['Zero', 'Low', 'Mid', 'Max']
    
    # Create a smoothing window
    window = np.ones(window_size) / window_size
    
    # Get list of group names
    group_names = [group_name] if group_name else EED.group_names
    
    # Loop through each group, recording, and cell ID
    for group in group_names:
        recording_names = [recording_name] if recording_name else EED.get_recording_names(group)
        for recording in recording_names:
            cellid_names = [cellid_name] if cellid_name else EED.get_cellid_names(group, recording)
            for cell_id in cellid_names:
                # Get the pre and post stim data
                data = EED.get_pre_post_data(group, recording, cell_id)
                
                # Create a new figure
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                
                # Loop through each stimulation level and plot PSTH for pre and post epochs
                for stim_index, stim_level in enumerate(stim_levels):
                    ax = axes[stim_index // 2, stim_index % 2]
                    
                    # Get spike trains for the current stimulation level
                    pre_spiketrains = data['Pre']['SpikeTrains_for_PSTHs'][stim_index]
                    post_spiketrains = data['Post']['SpikeTrains_for_PSTHs'][stim_index]
                    
                    # Convert spike trains to xarray DataArray
                    pre_da = xr.DataArray(pre_spiketrains, dims=('trial', 'time'))
                    post_da = xr.DataArray(post_spiketrains, dims=('trial', 'time'))
                    
                    # Calculate the average firing rate across trials and smooth the data
                    pre_psth = np.convolve(pre_da.mean(dim='trial'), window, mode='same')
                    post_psth = np.convolve(post_da.mean(dim='trial'), window, mode='same')
                    
                    # Plot PSTH for pre and post epochs
                    ax.plot(pre_psth, color='grey')
                    ax.plot(post_psth, color='blue')
                    
                    # Add a faint red line at 500 ms to indicate the stimulus onset
                    ax.axvline(500, color='red', linestyle='--', linewidth=0.5)
                    
                    # Set subplot title and labels
                    ax.set_title(stim_level)
                    ax.set_xlabel('Time (ms)')
                    ax.set_ylabel('Average firing rate')
                
                # Adjust the vertical spacing between subplots to prevent overlap
                plt.subplots_adjust(hspace=0.4)
                
                # Set main plot title
                fig.suptitle(f'Group: {group}, Recording: {recording}, Cell ID: {cell_id}')
                
                # Save the plot to the 'psthplots' directory
                plt.savefig(f'psthplots/{group}_{recording}_{cell_id}.png')
                
                # Close the plot to prevent it from displaying inline in the notebook
                plt.close(fig)


def plot_spike_distribution(EED, group_name=None, pool_stimulations=False):
    """
    Plot the distribution of spikes during the early phase (0-50 ms post-stimulus) for each group.

    Parameters:
    EED (object): The object containing the electrophysiology data.
    group_name (str, optional): The name of the group to plot. Defaults to None, in which case all groups are plotted.
    pool_stimulations (bool, optional): Whether to pool low, mid, and high stimulations into a single group called "Stimulation". Defaults to False.

    Returns:
    None: The function saves the plots to the 'spike_distribution' directory.
    """
    
    # Ensure 'spike_distribution' directory exists
    if not os.path.exists('spike_distribution'):
        os.makedirs('spike_distribution')
    
    # Get list of group names
    group_names = [group_name] if group_name else EED.group_names
    
    # Loop through each group
    for group in group_names:
        recording_names = EED.get_recording_names(group)
        
        # Loop through each recording
        for recording in recording_names:
            cellid_names = EED.get_cellid_names(group, recording)
            
            # Loop through each cell ID
            for cell_id in cellid_names:
                # Get the pre and post stim data
                data = EED.get_pre_post_data(group, recording, cell_id)
                
                # Define stimulation levels based on whether stimulations should be pooled
                if pool_stimulations:
                    stim_levels = ['Zero', 'Stimulation']
                else:
                    stim_levels = ['Zero', 'Low', 'Mid', 'Max']
                
                # Loop through each stimulation level
                for stim_index, stim_level in enumerate(stim_levels):
                    # Get spike trains for the current stimulation level
                    if stim_level == 'Stimulation':
                        pre_spiketrains = np.concatenate([data['Pre']['SpikeTrains_for_PSTHs'][i] for i in range(1, 4)], axis=0)
                        post_spiketrains = np.concatenate([data['Post']['SpikeTrains_for_PSTHs'][i] for i in range(1, 4)], axis=0)
                    else:
                        pre_spiketrains = data['Pre']['SpikeTrains_for_PSTHs'][stim_index]
                        post_spiketrains = data['Post']['SpikeTrains_for_PSTHs'][stim_index]
                    
                    # Extract spike data for the early phase (0-50 ms post-stimulus)
                    early_phase_pre = pre_spiketrains[:, 500:550]  # Adjust indices as necessary
                    early_phase_post = post_spiketrains[:, 500:550]  # Adjust indices as necessary
                    
                    # Calculate the total number of spikes in each trial during the early phase
                    spike_counts_pre = early_phase_pre.sum(axis=1) 
                    spike_counts_post = early_phase_post.sum(axis=1)
                    
                    # Create a new figure
                    plt.figure()
                    
                    # Plot the distribution of spike counts for the pre and post epochs
                    plt.hist(spike_counts_pre, bins=np.arange(spike_counts_pre.max() + 2) - 0.5, 
                        alpha=0.5, label='Pre', color='grey', edgecolor='grey', facecolor='white')

                    plt.hist(spike_counts_post, bins=np.arange(spike_counts_post.max() + 2) - 0.5, 
                        alpha=0.5, label='Post', color='blue', edgecolor='blue', facecolor='white')
                    
                    
                    # Set plot title and labels with a two-line title
                    plt.title(f'Group: {group}, Recording: {recording}, Cell ID: {cell_id}\nStim Level: {stim_level}', fontsize=10)
                    plt.xlabel('Spike Count')
                    plt.ylabel('Frequency')

                    # Add a legend
                    plt.legend()

                    # Adjust layout to prevent titles and labels from being cut off
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.85)  # Adjust top margin to provide more space for the title

                    # Save the plot to the 'spike_distribution' directory with higher resolution
                    plt.savefig(f'spike_distribution/{group}_{recording}_{cell_id}_{stim_level}.png', dpi=300)
                    
                    # Close the plot to prevent it from displaying inline in the notebook
                    plt.close()
                    
def plot_spike_distribution_boxplots(EED, group_name=None, pool_stimulations=False):
    """
    Plot the distribution of spikes during the early phase (0-50 ms post-stimulus) for each group using boxplots.

    Parameters:
    EED (object): The object containing the electrophysiology data.
    group_name (str, optional): The name of the group to plot. Defaults to None, in which case all groups are plotted.
    pool_stimulations (bool, optional): Whether to pool low, mid, and high stimulations into a single group called "Stimulation". Defaults to False.

    Returns:
    None: The function saves the plots to the 'spike_distributions_boxplots' directory.
    """
    
    # Ensure 'spike_distributions_boxplots' directory exists
    if not os.path.exists('spike_distributions_boxplots'):
        os.makedirs('spike_distributions_boxplots')
    
    # Get list of group names
    group_names = [group_name] if group_name else EED.group_names
    
    # Loop through each group
    for group in group_names:
        recording_names = EED.get_recording_names(group)
        
        # Loop through each recording
        for recording in recording_names:
            cellid_names = EED.get_cellid_names(group, recording)
            
            # Loop through each cell ID
            for cell_id in cellid_names:
                # Get the pre and post stim data
                data = EED.get_pre_post_data(group, recording, cell_id)
                
                # Define stimulation levels based on whether stimulations should be pooled
                if pool_stimulations:
                    stim_levels = ['Zero', 'Stimulation']
                else:
                    stim_levels = ['Zero', 'Low', 'Mid', 'Max']
                
                # Loop through each stimulation level
                for stim_index, stim_level in enumerate(stim_levels):
                    # Get spike trains for the current stimulation level
                    if stim_level == 'Stimulation':
                        pre_spiketrains = np.concatenate([data['Pre']['SpikeTrains_for_PSTHs'][i] for i in range(1, 4)], axis=0)
                        post_spiketrains = np.concatenate([data['Post']['SpikeTrains_for_PSTHs'][i] for i in range(1, 4)], axis=0)
                    else:
                        pre_spiketrains = data['Pre']['SpikeTrains_for_PSTHs'][stim_index]
                        post_spiketrains = data['Post']['SpikeTrains_for_PSTHs'][stim_index]
                    
                    # Extract spike data for the early phase (0-50 ms post-stimulus)
                    early_phase_pre = pre_spiketrains[:, 500:550]  # Adjust indices as necessary
                    early_phase_post = post_spiketrains[:, 500:550]  # Adjust indices as necessary
                    
                    # Calculate the total number of spikes in each trial during the early phase
                    spike_counts_pre = early_phase_pre.sum(axis=1)
                    spike_counts_post = early_phase_post.sum(axis=1)
                    
                    # Create a new figure
                    plt.figure()
                    
                    # Plot the boxplots for the pre and post epochs
                    plt.boxplot([spike_counts_pre, spike_counts_post], labels=['Pre', 'Post'])
                    
                    # Set plot title and labels
                    plt.title(f'Group: {group}, Recording: {recording}, Cell ID: {cell_id}\nStim Level: {stim_level}', fontsize=10)
                    plt.xlabel('Condition')
                    plt.ylabel('Spike Count')
                    
                    # Adjust layout to prevent titles and labels from being cut off
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.85)  # Adjust top margin to provide more space for the title
                    
                    # Save the plot to the 'spike_distributions_boxplots' directory with higher resolution
                    plt.savefig(f'spike_distributions_boxplots/{group}_{recording}_{cell_id}_{stim_level}.png', dpi=300)
                    
                    # Close the plot to prevent it from displaying inline in the notebook
                    plt.close()
