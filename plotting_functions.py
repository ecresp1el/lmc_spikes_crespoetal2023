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
