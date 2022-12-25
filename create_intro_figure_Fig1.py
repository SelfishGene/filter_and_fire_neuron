import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from scipy import signal

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

#%% script params

num_axons = 3
num_dendrites = 9

axon_colors = ['magenta', 'teal', 'purple']

num_spikes_per_axon = 3
experiment_time_ms  = 300

tau_rise_range  = [1,9]
tau_decay_range = [5,30]

v_reset     = -80
v_threshold = -55
current_to_voltage_mult_factor = 4
refreactory_time_constant = 25

save_figures = False
all_file_endings_to_use = ['.png', '.pdf', '.svg']

figure_folder = '/filter_and_fire_neuron/saved_figures/'

#%% helper functions


def create_single_PSP_profile(tau_rise, tau_decay, temporal_filter_length=50):

    if tau_rise >= tau_decay:
        tau_decay = tau_rise + 5

    exp_r = signal.exponential(M=temporal_filter_length, center=0, tau=tau_rise , sym=False)
    exp_d = signal.exponential(M=temporal_filter_length, center=0, tau=tau_decay, sym=False)

    post_syn_potential = exp_d - exp_r
    post_syn_potential /= post_syn_potential.max()

    return post_syn_potential


def construct_normlized_synaptic_filter(tau_rise_vec, tau_decay_vec):

    num_synapses = tau_rise_vec.shape[0]
    temporal_filter_length = int(7 * tau_decay_vec.max()) + 1

    syn_filter = np.zeros((num_synapses, temporal_filter_length))

    for k, (tau_r, tau_d) in enumerate(zip(tau_rise_vec, tau_decay_vec)):
        syn_filter[k,:] = create_single_PSP_profile(tau_r, tau_d, temporal_filter_length=temporal_filter_length)

    return syn_filter


def simulate_filter_and_fire_cell_training(presynaptic_input_spikes, synaptic_weights, tau_rise_vec, tau_decay_vec,
                                           refreactory_time_constant=20, v_reset=-75, v_threshold=-55, current_to_voltage_mult_factor=2):

    temporal_filter_length = int(5 * refreactory_time_constant) + 1
    refreactory_filter = signal.exponential(M=temporal_filter_length,center=0,tau=refreactory_time_constant,sym=False)[np.newaxis,:]

    # padd input and get all synaptic filters
    normlized_syn_filter = np.flipud(construct_normlized_synaptic_filter(tau_rise_vec, tau_decay_vec))
    padded_input = np.hstack((np.zeros(normlized_syn_filter.shape), presynaptic_input_spikes))

    # calc local currents
    local_normlized_currents = np.zeros(presynaptic_input_spikes.shape, dtype=np.float16)
    for k in range(normlized_syn_filter.shape[0]):
        local_normlized_currents[k] = signal.convolve(padded_input[k], normlized_syn_filter[k], mode='valid')[1:]

    # multiply by weights to get the somatic current
    soma_current = signal.convolve(local_normlized_currents, synaptic_weights, mode='valid')

    # simulate the cell
    soma_voltage = v_reset + current_to_voltage_mult_factor * soma_current.ravel()
    output_spike_times_in_ms = []
    # after a spike inject current that is exactly required to bring the cell back to v_reset (this current slowly decays)
    for t in range(len(soma_voltage)):
        # after a spike inject current that is exactly required to bring the cell back to v_reset (this current slowly decays)
        if (soma_voltage[t] > v_threshold) and ((t + 1) < len(soma_voltage)):
            t_start = t + 1
            t_end = min(len(soma_voltage), t_start + temporal_filter_length)
            soma_voltage[t_start:t_end] -= (soma_voltage[t + 1] - v_reset) * refreactory_filter.ravel()[:(t_end - t_start)]
            output_spike_times_in_ms.append(t)

    return local_normlized_currents, soma_voltage, output_spike_times_in_ms


def add_offset_for_plotting(traces_matrix, offset_size=1.1):

    traces_matrix_with_offset = offset_size * np.kron(np.arange(traces_matrix.shape[0])[:,np.newaxis], np.ones((1,traces_matrix.shape[1])))
    traces_matrix_with_offset = traces_matrix_with_offset + traces_matrix

    return traces_matrix_with_offset


#%% simulate the cell

connections_per_axon = int(num_dendrites / num_axons)
num_synapses = num_dendrites

tau_rise_vec  = np.linspace(tau_rise_range[0] , tau_rise_range[1] , num_synapses)[:,np.newaxis]
tau_decay_vec = np.linspace(tau_decay_range[0], tau_decay_range[1], num_synapses)[:,np.newaxis]

# synapse learnable parameters
synaptic_weights_vec = 1.0 + 0.1 * np.random.uniform(size=(num_synapses, 1))

synaptic_filters = construct_normlized_synaptic_filter(tau_rise_vec, tau_decay_vec)

#%% run once

# generate sample input
stimulus_duration_ms = experiment_time_ms

axon_input_spike_train = np.zeros((num_axons, stimulus_duration_ms))
for k in range(num_axons):
    curr_axon_spike_times = 20 + np.random.randint(stimulus_duration_ms -80, size=num_spikes_per_axon)
    axon_input_spike_train[k,curr_axon_spike_times] = 1.0

presynaptic_input_spikes = np.kron(np.ones((connections_per_axon,1), dtype=bool), axon_input_spike_train).astype(bool)

# simulate F&F cell with normlized currents
local_normlized_currents, soma_voltage, output_spike_times_in_ms = simulate_filter_and_fire_cell_training(presynaptic_input_spikes,
                                                                                                          synaptic_weights_vec, tau_rise_vec, tau_decay_vec,
                                                                                                          refreactory_time_constant=refreactory_time_constant,
                                                                                                          v_reset=v_reset, v_threshold=v_threshold,
                                                                                                          current_to_voltage_mult_factor=current_to_voltage_mult_factor)

local_normlized_currents = np.flipud(local_normlized_currents)

soma_voltage_with_spikes = soma_voltage
soma_voltage_with_spikes[output_spike_times_in_ms] = -25


max_local_added_voltage = add_offset_for_plotting(local_normlized_currents).T.max()

print('running once')


#%% run again until we have at least 1 spike

while len(output_spike_times_in_ms) != 1 or max_local_added_voltage > 10.15:
    axon_input_spike_train = np.zeros((num_axons, stimulus_duration_ms))
    for k in range(num_axons):
        curr_axon_spike_times = 30 + np.random.randint(stimulus_duration_ms -60, size=num_spikes_per_axon)
        axon_input_spike_train[k,curr_axon_spike_times] = 1.0

    presynaptic_input_spikes = np.kron(np.ones((connections_per_axon,1), dtype=bool), axon_input_spike_train).astype(bool)

    # simulate F&F cell with normlized currents
    local_normlized_currents, soma_voltage, output_spike_times_in_ms = simulate_filter_and_fire_cell_training(presynaptic_input_spikes,
                                                                                                              synaptic_weights_vec, tau_rise_vec, tau_decay_vec,
                                                                                                              refreactory_time_constant=refreactory_time_constant,
                                                                                                              v_reset=v_reset, v_threshold=v_threshold,
                                                                                                              current_to_voltage_mult_factor=current_to_voltage_mult_factor)

    local_normlized_currents = np.flipud(local_normlized_currents)

    soma_voltage_with_spikes = soma_voltage
    soma_voltage_with_spikes[output_spike_times_in_ms] = -15

    max_local_added_voltage = add_offset_for_plotting(local_normlized_currents).T.max()

print('there is at least 1 spike')

#%% run untill we have a changed enough spike location

min_spike_time_diff = 35
max_local_added_voltage = 0

while True:

    output_spike_times_in_ms =[]

    # generate input axons with a single spike
    while len(output_spike_times_in_ms) != 1 or max_local_added_voltage > 10.15:
        axon_input_spike_train = np.zeros((num_axons, stimulus_duration_ms))
        for k in range(num_axons):
            curr_axon_spike_times = 30 + np.random.randint(stimulus_duration_ms -60, size=num_spikes_per_axon)
            axon_input_spike_train[k,curr_axon_spike_times] = 1.0

        presynaptic_input_spikes = np.kron(np.ones((connections_per_axon,1), dtype=bool), axon_input_spike_train).astype(bool)

        # simulate F&F cell with normlized currents
        local_normlized_currents, soma_voltage, output_spike_times_in_ms = simulate_filter_and_fire_cell_training(presynaptic_input_spikes,
                                                                                                                  synaptic_weights_vec, tau_rise_vec, tau_decay_vec,
                                                                                                                  refreactory_time_constant=refreactory_time_constant,
                                                                                                                  v_reset=v_reset, v_threshold=v_threshold,
                                                                                                                  current_to_voltage_mult_factor=current_to_voltage_mult_factor)

        local_normlized_currents = np.flipud(local_normlized_currents)

        soma_voltage_with_spikes = soma_voltage
        soma_voltage_with_spikes[output_spike_times_in_ms] = -15

        max_local_added_voltage = add_offset_for_plotting(local_normlized_currents).T.max()

        if len(output_spike_times_in_ms) == 1 and (output_spike_times_in_ms[0] > 200 or output_spike_times_in_ms[0] < 120):
            output_spike_times_in_ms = []

    print('there is at least 1 spike, spike times: ', output_spike_times_in_ms)

    # generate a weights change that will move this spikes by a minimum amount
    mult_vector = np.random.permutation([0.25,0.5,0.5,0.75,1.25,1.5,1.75,1.75,2.0])[:,np.newaxis]
    synaptic_weights_vec_2 = mult_vector * synaptic_weights_vec

    # make sure only the a the middle axon weights are changed (for non cluttered visualization)
    synaptic_weights_vec_2[0::3] = synaptic_weights_vec[0::3]
    synaptic_weights_vec_2[2::3] = synaptic_weights_vec[2::3]
    mult_vector[0::3] = 1
    mult_vector[2::3] = 1

    # simulate F&F cell with normlized currents
    local_normlized_currents_2, soma_voltage_2, output_spike_times_in_ms_2 = simulate_filter_and_fire_cell_training(presynaptic_input_spikes,
                                                                                                                    synaptic_weights_vec_2, tau_rise_vec, tau_decay_vec,
                                                                                                                    refreactory_time_constant=refreactory_time_constant,
                                                                                                                    v_reset=v_reset, v_threshold=v_threshold,
                                                                                                                    current_to_voltage_mult_factor=current_to_voltage_mult_factor)

    local_normlized_currents_2 = np.flipud(local_normlized_currents_2)

    soma_voltage_with_spikes_2 = soma_voltage_2
    soma_voltage_with_spikes_2[output_spike_times_in_ms_2] = -15

    max_local_added_voltage_2 = add_offset_for_plotting(local_normlized_currents_2).T.max()

    local_normlized_currents_2 = mult_vector * local_normlized_currents_2

    if len(output_spike_times_in_ms_2) == 1 and ((output_spike_times_in_ms[0] - output_spike_times_in_ms_2[0]) <= -min_spike_time_diff):
        break

print('changed weights such that the spike changed location for more than %d ms' %(min_spike_time_diff))


#%% Display Input Axons, Synaptic Filters and Local Voltage Traces "Slopily"

plt.close('all')
fig = plt.figure(figsize=(20,8))
plt.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05, wspace=0.25, hspace=0.35)

plt.subplot(1,3,1);
for k, axon_color in enumerate(axon_colors):
    plt.plot(add_offset_for_plotting(np.flipud(axon_input_spike_train)).T[:,k], color=axon_color); plt.title('Input Axons', fontsize=24);

plt.subplot(1,3,2);
for k, axon_color in enumerate(axon_colors):
    plt.plot(add_offset_for_plotting(synaptic_filters).T[:100,k::num_axons], color=axon_color); plt.title('Synaptic Filters', fontsize=24);

plt.subplot(1,3,3);
for k, axon_color in enumerate(axon_colors):
    plt.plot(add_offset_for_plotting(local_normlized_currents).T[:,k::num_axons], color=axon_color); plt.title('Synaptic Contact Voltage Contribution', fontsize=24);
    plt.plot(add_offset_for_plotting(local_normlized_currents_2).T[:,k::num_axons], color=axon_color, linestyle='dashed');

#%% plot one below each other

plt.close('all')
plt.figure(figsize=(10,25))
gs_figure = gridspec.GridSpec(nrows=11,ncols=1)
gs_figure.update(left=0.04, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.8)

ax_axons          = plt.subplot(gs_figure[:3,:])
ax_local_voltages = plt.subplot(gs_figure[3:9,:])
ax_soma_voltage   = plt.subplot(gs_figure[9:,:])

ax_axons.plot(add_offset_for_plotting(axon_input_spike_train).T); ax_axons.set_title('Input Axons', fontsize=24);
ax_local_voltages.plot(add_offset_for_plotting(local_normlized_currents).T); ax_local_voltages.set_title('Synaptic Contact Voltage Contribution', fontsize=24);
ax_local_voltages.plot(add_offset_for_plotting(local_normlized_currents_2).T, linestyle='dashed');
ax_soma_voltage.plot(soma_voltage_with_spikes); ax_soma_voltage.set_title('Somatic Voltage', fontsize=24);
ax_soma_voltage.plot(soma_voltage_with_spikes_2, linestyle='dashed');

#%% Display The Full Figure

plt.close('all')
fig = plt.figure(figsize=(25,18))
gs_figure = gridspec.GridSpec(nrows=11,ncols=30)
gs_figure.update(left=0.04, right=0.95, bottom=0.05, top=0.92, wspace=0.2, hspace=0.7)

ax_axons         = plt.subplot(gs_figure[2:6,:9])
ax_syn_filters   = plt.subplot(gs_figure[:8,13:17])
ax_local_voltges = plt.subplot(gs_figure[:8,21:])
ax_soma_voltage  = plt.subplot(gs_figure[8:,21:])


for k, axon_color in enumerate(axon_colors):
    ax_axons.plot(add_offset_for_plotting(np.flipud(axon_input_spike_train)).T[:,k], color=axon_color, lw=3);
ax_axons.set_yticks([])
ax_axons.set_xticks([])
ax_axons.set_title('Input Axons', fontsize=20);
ax_axons.spines['top'].set_visible(False)
ax_axons.spines['bottom'].set_visible(False)
ax_axons.spines['left'].set_visible(False)
ax_axons.spines['right'].set_visible(False)

for k, axon_color in enumerate(axon_colors):
    ax_syn_filters.plot(add_offset_for_plotting(synaptic_filters).T[:100,k::num_axons], color=axon_color, lw=3);
ax_syn_filters.set_yticks([])
ax_syn_filters.set_xticks([])
ax_syn_filters.set_title('Synaptic Filters', fontsize=20);
ax_syn_filters.spines['top'].set_visible(False)
ax_syn_filters.spines['bottom'].set_visible(False)
ax_syn_filters.spines['left'].set_visible(False)
ax_syn_filters.spines['right'].set_visible(False)

for k, axon_color in enumerate(axon_colors):
    ax_local_voltges.plot(add_offset_for_plotting(local_normlized_currents).T[:,k::num_axons], color=axon_color, lw=3);
    ax_local_voltges.plot(add_offset_for_plotting(local_normlized_currents_2).T[:,k::num_axons], color=axon_color, ls='dashed', lw=2.5);
ax_local_voltges.set_xticks([])
ax_local_voltges.set_yticks([])
ax_local_voltges.set_title('Synaptic Contact Voltage Contribution', fontsize=20);
ax_local_voltges.spines['top'].set_visible(False)
ax_local_voltges.spines['bottom'].set_visible(False)
ax_local_voltges.spines['left'].set_visible(False)
ax_local_voltges.spines['right'].set_visible(False)

ax_soma_voltage.plot(soma_voltage_with_spikes, color='0.1', lw=3, alpha=0.85);
ax_soma_voltage.plot(soma_voltage_with_spikes_2, color='dodgerblue', lw=3);
ax_soma_voltage.set_yticks([])
ax_soma_voltage.set_xticks([])
ax_soma_voltage.set_title('Somatic Voltage', fontsize=20);
ax_soma_voltage.spines['top'].set_visible(False)
ax_soma_voltage.spines['bottom'].set_visible(False)
ax_soma_voltage.spines['left'].set_visible(False)
ax_soma_voltage.spines['right'].set_visible(False)

# save figure
if save_figures:
    figure_name = 'F&F_A_Introduction_Figure_1_%d' %(np.random.randint(2000))
    for file_ending in all_file_endings_to_use:
        if file_ending == '.png':
            fig.savefig(figure_folder + figure_name + file_ending, bbox_inches='tight')
        else:
            fig.savefig(figure_folder + figure_name + file_ending, bbox_inches='tight')


#%%



