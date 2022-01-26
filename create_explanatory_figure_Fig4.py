import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import linear_model
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.decomposition import TruncatedSVD, NMF
import matplotlib.gridspec as gridspec

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

#%% script params

save_figures = True
save_figures = False
all_file_endings_to_use = ['.png', '.pdf', '.svg']

data_folder   = '/filter_and_fire_neuron/results_data_capacity/'
figure_folder = '/filter_and_fire_neuron/saved_figures/'

#%% helper functions

def create_single_PSP_profile(tau_rise, tau_decay, temporal_filter_length=50):

    safety_factor = 1.5
    if tau_rise >= (tau_decay / safety_factor):
        tau_decay = safety_factor * tau_rise

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
    local_normlized_currents = np.zeros(presynaptic_input_spikes.shape)
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


#%% script params

# input parameters
num_values_per_param = 12

# neuron model parameters
connections_per_axon = 5
num_synapses = num_values_per_param * num_values_per_param

v_reset     = -80
v_threshold = -55
current_to_voltage_mult_factor = 3
refreactory_time_constant = 20

model_type = 'F&F'
#model_type = 'I&F'

time_limit_ms = 120

# synapse non-learnable parameters
if model_type == 'F&F':
    tau_rise_range  = [1,18]
    tau_decay_range = [7,25]
elif model_type == 'I&F':
    tau_rise_range  = [3,3]
    tau_decay_range = [25,25]

tau_rise_vec = np.linspace(tau_rise_range[0], tau_rise_range[1] , num_values_per_param)[:,np.newaxis]
tau_rise_vec = np.kron(np.ones((num_values_per_param,1)), tau_rise_vec)

tau_decay_vec = np.linspace(tau_decay_range[0], tau_decay_range[1] , num_values_per_param)[:,np.newaxis]
tau_decay_vec = np.kron(tau_decay_vec, np.ones((num_values_per_param,1)))

normlized_syn_filter_small = construct_normlized_synaptic_filter(tau_rise_vec, tau_decay_vec)

offset_size = 0.15

plt.close('all')
plt.figure(figsize=(25,20));
plt.subplot(1,2,1); plt.imshow(normlized_syn_filter_small);
plt.title('normlized synaptic filters as heatmaps', fontsize=22); plt.xlabel('time [ms]', fontsize=22); plt.ylabel('synaptic filter index', fontsize=22); plt.xlim(0,time_limit_ms);
plt.subplot(1,2,2);

use_colors = False
if use_colors:
    colors = 'rgbymcrgbymc'

    end_ind = 0
    for k in range(num_values_per_param):
        start_ind = end_ind
        end_ind = start_ind + num_values_per_param
        print(start_ind, end_ind, colors[k])
        plt.plot(offset_size * k * num_values_per_param + add_offset_for_plotting(normlized_syn_filter_small[start_ind:end_ind], offset_size=offset_size).T, c=colors[k]);
else:
    plt.plot(add_offset_for_plotting(normlized_syn_filter_small, offset_size=offset_size).T, c='k');

plt.title('normlized synaptic filters as PSPs', fontsize=22); plt.xlabel('time [ms]', fontsize=22); plt.ylabel('normalized PSP (A.U)', fontsize=22); plt.xlim(-1,time_limit_ms);

#%% Create all possible combinations

# input parameters
num_values_per_param = 50

tau_rise_vec = np.linspace(tau_rise_range[0], tau_rise_range[1] , num_values_per_param)[:,np.newaxis]
tau_rise_vec = np.kron(np.ones((num_values_per_param,1)), tau_rise_vec)

tau_decay_vec = np.linspace(tau_decay_range[0], tau_decay_range[1] , num_values_per_param)[:,np.newaxis]
tau_decay_vec = np.kron(tau_decay_vec, np.ones((num_values_per_param,1)))

normlized_syn_filter_large = construct_normlized_synaptic_filter(tau_rise_vec, tau_decay_vec)

plt.close('all')
plt.figure(figsize=(25,20));
plt.subplot(1,2,1); plt.imshow(normlized_syn_filter_large);
plt.title('normlized synaptic filters as heatmaps', fontsize=22); plt.xlabel('time [ms]', fontsize=22); plt.ylabel('synaptic filter index', fontsize=22)
plt.xlim(0,time_limit_ms);
plt.subplot(1,2,2); plt.plot(normlized_syn_filter_large.T, alpha=0.15);
plt.title('normlized synaptic filters as PSPs', fontsize=22); plt.xlabel('time [ms]', fontsize=22); plt.ylabel('normalized PSP (A.U)', fontsize=22);
plt.xlim(0,time_limit_ms);

#%% apply SVD and display

X = normlized_syn_filter_large
PSP_SVD_model = TruncatedSVD(n_components=100)
PSP_SVD_model.fit(X)

SVD_cutoff_ind = 3
max_SVD_basis_to_present = 18

plt.close('all')
plt.figure(figsize=(25,20));
plt.subplot(3,1,1); plt.imshow(PSP_SVD_model.components_[:max_SVD_basis_to_present]);
plt.title('normlized synaptic filters as heatmaps', fontsize=22); plt.xlabel('time [ms]', fontsize=22); plt.ylabel('synaptic filter index', fontsize=22); plt.xlim(0,time_limit_ms);
plt.subplot(3,1,2); plt.plot(PSP_SVD_model.components_[:SVD_cutoff_ind].T);
plt.title('first 3 basis functions', fontsize=22); plt.xlabel('time [ms]', fontsize=22); plt.ylabel('normalized PSP (A.U)', fontsize=22); plt.xlim(0,time_limit_ms);
plt.subplot(3,1,3); plt.plot(PSP_SVD_model.components_[SVD_cutoff_ind:max_SVD_basis_to_present].T);
plt.title('rest of the basis functions', fontsize=22); plt.xlabel('time [ms]', fontsize=22); plt.ylabel('normalized PSP (A.U)', fontsize=22); plt.xlim(0,time_limit_ms);

#%% show variance explained

num_basis_functions = PSP_SVD_model.explained_variance_ratio_.shape[0]
explained_var_percent = 100 * PSP_SVD_model.explained_variance_ratio_
cumsum_explained_var_percent = np.concatenate((np.array([0]), np.cumsum(explained_var_percent)))
dot_selected_ind = 3

plt.close('all')
plt.figure(figsize=(10,7));
plt.plot(np.arange(num_basis_functions + 1), cumsum_explained_var_percent, c='k')
plt.scatter(dot_selected_ind, cumsum_explained_var_percent[dot_selected_ind+1], c='r', s=200)
plt.xlabel('num basis functions', fontsize=16); plt.ylabel('explained %s' %('%'), fontsize=16);
plt.title('SVD cumulative explained percent \ntotal variance explained = %.2f%s' %(cumsum_explained_var_percent[dot_selected_ind+1],'%'), fontsize=18);
plt.ylim(-1,105); plt.xlim(-1,num_basis_functions+1);
plt.xlim(-0.3,12)

#%% Apply NMF

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# to avoid numberic instability, replicate the data and add some noise
noisy_data_for_NMF = np.tile(X,[3,1])
noisy_data_for_NMF = noisy_data_for_NMF + 0.0 * np.random.rand(noisy_data_for_NMF.shape[0], noisy_data_for_NMF.shape[1])

PSP_NMF_model = NMF(n_components=20)
PSP_NMF_model.fit(noisy_data_for_NMF)

NMF_cutoff_ind = 3
max_basis_to_present = 10

# normalize each basis vector to it's maximum (for presentation)
NMF_basis = PSP_NMF_model.components_
NMF_basis_norm = NMF_basis / np.tile(NMF_basis.max(axis=1, keepdims=True), [1, NMF_basis.shape[1]])

plt.close('all')
plt.figure(figsize=(25,20));
plt.subplot(3,1,1); plt.imshow(NMF_basis_norm[:max_basis_to_present]);
plt.title('normlized synaptic filters as heatmaps', fontsize=22); plt.xlabel('time [ms]', fontsize=22); plt.ylabel('synaptic filter index', fontsize=22); plt.xlim(0,time_limit_ms);
plt.subplot(3,1,2); plt.plot(NMF_basis_norm[:NMF_cutoff_ind].T);
plt.title('first 4 basis functions', fontsize=22); plt.xlabel('time [ms]', fontsize=22); plt.ylabel('normalized PSP (A.U)', fontsize=22); plt.xlim(0,time_limit_ms);
plt.subplot(3,1,3); plt.plot(NMF_basis_norm[NMF_cutoff_ind:max_basis_to_present].T);
plt.title('rest of the basis functions', fontsize=22); plt.xlabel('time [ms]', fontsize=22); plt.ylabel('normalized PSP (A.U)', fontsize=22); plt.xlim(0,time_limit_ms);

#%% load file for capacity plot

results_filename = data_folder + 'FF_vs_IF_capacity_comparision__num_axons_200__sim_duration_sec_120__num_mult_conn_6__rand_rep_18.pickle'

loaded_script_results_dict = pickle.load(open(results_filename, "rb" ))

processed_res_curves = loaded_script_results_dict['processed_res_curves']
all_results_curves   = loaded_script_results_dict['all_results_curves']

num_axons = loaded_script_results_dict['script_main_params']['num_axons']
stimulus_duration_sec = loaded_script_results_dict['script_main_params']['stimulus_duration_sec']

filename_str = 'FF_vs_IF_capacity_comparision__num_axons_%d__sim_duration_sec_120__num_mult_conn_6__rand_rep_18.pickle'
num_axons_list = sorted([100, 112, 125, 137, 150, 162, 175, 187, 200, 212, 225, 237])
all_filenames_str = [filename_str %(x) for x in num_axons_list]

model_keys = list(loaded_script_results_dict['processed_res_curves'].keys())
connections_per_axon_2C = loaded_script_results_dict['processed_res_curves'][model_keys[0]]['connections_per_axon']

precisely_timed_spikes_per_axon_2C = {}
precisely_timed_spikes_per_axon_error_2C = {}
for key in model_keys:
    precisely_timed_spikes_per_axon_2C[key] = np.zeros((len(all_filenames_str), len(connections_per_axon_2C)))
    precisely_timed_spikes_per_axon_error_2C[key] = np.zeros((len(all_filenames_str), len(connections_per_axon_2C)))

for k, (curr_num_axons, curr_filename) in enumerate(zip(num_axons_list, all_filenames_str)):
    curr_results_filename = data_folder + curr_filename
    curr_loaded_results_dict = pickle.load(open(curr_results_filename, "rb" ))

    for key in model_keys:
        precisely_timed_spikes_per_axon_2C[key][k,:] = curr_loaded_results_dict['processed_res_curves'][key]['num_almost_perfectly_placed_spikes'] / curr_num_axons

        for j, (num_M_conn, num_spikes) in enumerate(zip(curr_loaded_results_dict['processed_res_curves'][key]['connections_per_axon'],
                                                         curr_loaded_results_dict['processed_res_curves'][key]['num_almost_perfectly_placed_spikes'])):

            model_connections_str = '%s, %d connections' %(key, num_M_conn)
            error_index = list(curr_loaded_results_dict['all_results_curves'][model_connections_str]['num_spikes']).index(num_spikes)
            error_scale = (curr_loaded_results_dict['all_results_curves'][model_connections_str]['num_spikes'][error_index + 1] -
                           curr_loaded_results_dict['all_results_curves'][model_connections_str]['num_spikes'][max(0, error_index - 1)])

            if error_index > 1:
                error_scale /= 2

            precisely_timed_spikes_per_axon_error_2C[key][k,j] = error_scale / curr_num_axons


color_map = {}
color_map['I&F'] = '0.05'
color_map['F&F'] = 'orange'

#%% Calculate Capacity with optimal PSP profiles (m = 3)

optimal_basis_PSPs = NMF_basis_norm[:NMF_cutoff_ind]
print(optimal_basis_PSPs.shape)

num_axons = 3
normlized_syn_filter = np.kron(optimal_basis_PSPs, np.ones((num_axons,1)))

#%% Define several new helper functions (including two simulation functions)

def simulate_filter_and_fire_cell_training_PSPs(presynaptic_input_spikes, synaptic_weights, normlized_syn_filter,
                                                refreactory_time_constant=20, v_reset=-75, v_threshold=-55, current_to_voltage_mult_factor=2):

    temporal_filter_length = int(5 * refreactory_time_constant) + 1
    refreactory_filter = signal.exponential(M=temporal_filter_length,center=0,tau=refreactory_time_constant,sym=False)[np.newaxis,:]

    # padd input and get all synaptic filters
    padded_input = np.hstack((np.zeros(normlized_syn_filter.shape), presynaptic_input_spikes))

    # calc local currents
    local_normlized_currents = np.zeros(presynaptic_input_spikes.shape)
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


# use local currents as "features" and fit a linear model to the data
def prepare_training_dataset(local_normlized_currents, desired_output_spikes, spike_safety_range_ms=10, negative_subsampling_fraction=0.1):

    # remove all "negative" time points that are too close to spikes
    desired_output_spikes_LPF = signal.convolve(desired_output_spikes, np.ones((spike_safety_range_ms,)), mode='same') > 0.1
    desired_timepoints = ~desired_output_spikes_LPF

    # massivly subsample the remaining timepoints
    desired_timepoints[np.random.rand(desired_timepoints.shape[0]) > negative_subsampling_fraction] = 0
    desired_timepoints[desired_output_spikes > 0.1] = 1

    X = local_normlized_currents.T[desired_timepoints,:]
    y = desired_output_spikes[desired_timepoints]

    return X, y


#%% check if desired number of spikes is better than desired AUC score

requested_number_of_output_spikes = 93

optimal_basis_PSPs = NMF_basis_norm[:NMF_cutoff_ind]

# input parameters
num_axons = 200

# neuron model parameters
connections_per_axon = NMF_cutoff_ind
num_synapses = connections_per_axon * num_axons

v_reset     = -80
v_threshold = -55
current_to_voltage_mult_factor = 3
refreactory_time_constant = 15

model_type = 'F&F optimal'

# synapse learnable parameters
synaptic_weights_vec = np.random.normal(size=(num_synapses, 1))

# generate sample input
stimulus_duration_ms = 90000
instantanious_input_spike_probability = 0.004

axons_input_spikes = np.random.rand(num_axons, stimulus_duration_ms) < instantanious_input_spike_probability
presynaptic_input_spikes = np.kron(np.ones((connections_per_axon,1)), axons_input_spikes)
normlized_syn_filter = np.kron(optimal_basis_PSPs, np.ones((num_axons,1)))

assert presynaptic_input_spikes.shape[0] == num_synapses, 'number of synapses doesnt match the number of presynaptic inputs'

# generate desired pattern of output spikes
min_time_between_spikes_ms = 90

desired_output_spike_times = min_time_between_spikes_ms * np.random.randint(int(stimulus_duration_ms / min_time_between_spikes_ms), size=requested_number_of_output_spikes)
desired_output_spike_times = np.sort(np.unique(desired_output_spike_times))

desired_output_spikes = np.zeros((stimulus_duration_ms,))
desired_output_spikes[desired_output_spike_times] = 1.0

print('number of requested output spikes = %d' %(requested_number_of_output_spikes))

# simulate cell with normlized currents
local_normlized_currents, soma_voltage, output_spike_times_in_ms = simulate_filter_and_fire_cell_training_PSPs(presynaptic_input_spikes,
                                                                                                               synaptic_weights_vec, normlized_syn_filter,
                                                                                                               refreactory_time_constant=refreactory_time_constant,
                                                                                                               v_reset=v_reset, v_threshold=v_threshold,
                                                                                                               current_to_voltage_mult_factor=current_to_voltage_mult_factor)

output_spikes = np.zeros((stimulus_duration_ms,))
try:
    output_spikes[np.array(output_spike_times_in_ms)] = 1.0
except:
    print('no output spikes created')

#%% fit linear model to local currents

logistic_reg_model = linear_model.LogisticRegression(C=100000, fit_intercept=True, penalty='l2', max_iter=3000)

spike_safety_range_ms = 5
negative_subsampling_fraction = 0.5

X, y = prepare_training_dataset(local_normlized_currents, desired_output_spikes,
                                spike_safety_range_ms=spike_safety_range_ms,
                                negative_subsampling_fraction=negative_subsampling_fraction)

# fit model
logistic_reg_model.fit(X,y)

print('number of data points = %d (%.2f%s positive class)' %(X.shape[0], 100 * y.mean(),'%'))

y_hat = logistic_reg_model.predict_proba(X)[:,1]

# calculate AUC
train_AUC = roc_auc_score(y, y_hat)

fitted_output_spike_prob = logistic_reg_model.predict_proba(local_normlized_currents.T)[:,1]
full_AUC = roc_auc_score(desired_output_spikes, fitted_output_spike_prob)

# get desired FP threshold
desired_false_positive_rate = 0.004

fpr, tpr, thresholds = roc_curve(desired_output_spikes, fitted_output_spike_prob)

desired_fp_ind = np.argmin(abs(fpr-desired_false_positive_rate))
if desired_fp_ind == 0:
    desired_fp_ind = 1

actual_false_positive_rate = fpr[desired_fp_ind]
true_positive_rate         = tpr[desired_fp_ind]
desired_fp_threshold       = thresholds[desired_fp_ind]

AUC_score = auc(fpr, tpr)

print('AUC = %.4f' %(AUC_score))
print('at %.4f FP rate, TP = %.4f' %(actual_false_positive_rate, true_positive_rate))

output_spikes_after_learning = fitted_output_spike_prob > desired_fp_threshold

#%% Build the final figure

xy_label_fontsize = 16
title_fontsize = 21

plt.close('all')
fig = plt.figure(figsize=(20,18.5))
gs_figure = gridspec.GridSpec(nrows=8,ncols=5)
gs_figure.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.6, hspace=0.9)

ax_PSP_heatmap     = plt.subplot(gs_figure[ :6, :3])
ax_SVD_heatmap     = plt.subplot(gs_figure[6: , :3])
ax_PSP_traces      = plt.subplot(gs_figure[ :2,3: ])
ax_NMF_trance      = plt.subplot(gs_figure[2:4,3: ])
ax_explained_var   = plt.subplot(gs_figure[4:6,3: ])
ax_n_spikes_m_cons = plt.subplot(gs_figure[6: ,3: ])

interp_method_PSP = 'spline16'
interp_method_SVD = 'bilinear'
colormap = 'jet'

ax_PSP_heatmap.imshow(normlized_syn_filter_small, cmap=colormap, interpolation=interp_method_PSP);
ax_PSP_heatmap.set_xlim(0,time_limit_ms);
ax_PSP_heatmap.set_title('All PSPs as heatmap', fontsize=title_fontsize)
ax_PSP_heatmap.set_xlabel('Time (ms)', fontsize=xy_label_fontsize)
ax_PSP_heatmap.set_xticks([0,30,60,90,120])
ax_PSP_heatmap.set_xticklabels([0,30,60,90,120], fontsize=xy_label_fontsize)
ax_PSP_heatmap.set_ylabel('PSP index', fontsize=xy_label_fontsize)
ax_PSP_heatmap.set_yticks([0,24,48,72,96,120])
ax_PSP_heatmap.set_yticklabels([1,25,49,73,97,121], fontsize=xy_label_fontsize)

ax_SVD_heatmap.imshow(np.kron(PSP_SVD_model.components_[:max_SVD_basis_to_present], np.ones((2,1))), cmap=colormap, interpolation=interp_method_SVD);
ax_SVD_heatmap.set_xlim(0,time_limit_ms);
ax_SVD_heatmap.set_title('SVD basis functions as heatmap', fontsize=title_fontsize)
ax_SVD_heatmap.set_xlabel('Time (ms)', fontsize=xy_label_fontsize)
ax_SVD_heatmap.set_xticks([0,30,60,90,120])
ax_SVD_heatmap.set_xticklabels([0,30,60,90,120], fontsize=xy_label_fontsize)
ax_SVD_heatmap.set_ylabel('Basis function index', fontsize=xy_label_fontsize)
ax_SVD_heatmap.set_yticks([0,9,19,29])
ax_SVD_heatmap.set_yticklabels([1,10,20,30], fontsize=xy_label_fontsize)

ax_PSP_traces.plot(normlized_syn_filter_large.T, alpha=0.15);
ax_PSP_traces.set_xlim(-1,time_limit_ms);
ax_PSP_traces.set_title('All PSPs as traces', fontsize=title_fontsize)
ax_PSP_traces.set_ylabel('Magnitude (A.U.)', fontsize=xy_label_fontsize);
ax_PSP_traces.set_xlabel('Time (ms)', fontsize=xy_label_fontsize)
ax_PSP_traces.set_yticks([0.0,0.25,0.50,0.75,1.00])
ax_PSP_traces.set_yticklabels([0.0,0.25,0.50,0.75,1.00], fontsize=xy_label_fontsize)
ax_PSP_traces.set_xticks([0,30,60,90,120])
ax_PSP_traces.set_xticklabels([0,30,60,90,120], fontsize=xy_label_fontsize)

ax_NMF_trance.plot(NMF_basis_norm[:NMF_cutoff_ind].T);
ax_NMF_trance.set_xlim(-1,time_limit_ms);
ax_NMF_trance.set_title('NMF first %d basis functions' %(NMF_cutoff_ind), fontsize=title_fontsize)
ax_NMF_trance.set_ylabel('Magnitude  (A.U.)', fontsize=xy_label_fontsize);
ax_NMF_trance.set_xlabel('Time (ms)', fontsize=xy_label_fontsize)
ax_NMF_trance.set_yticks([0.0,0.25,0.50,0.75,1.00])
ax_NMF_trance.set_yticklabels([0.0,0.25,0.50,0.75,1.00], fontsize=xy_label_fontsize)
ax_NMF_trance.set_xticks([0,30,60,90,120])
ax_NMF_trance.set_xticklabels([0,30,60,90,120], fontsize=xy_label_fontsize)

ax_explained_var.plot(np.arange(num_basis_functions + 1), cumsum_explained_var_percent, c='k')

ax_explained_var.scatter(dot_selected_ind, cumsum_explained_var_percent[NMF_cutoff_ind + 1], c='r', s=200)
ax_explained_var.set_title('Variance explained = %.2f%s' %(cumsum_explained_var_percent[NMF_cutoff_ind + 1],'%'), fontsize=title_fontsize);
ax_explained_var.set_xlabel('Num basis functions', fontsize=xy_label_fontsize);
ax_explained_var.set_ylabel('Explained Percent (%s)' %('%'), fontsize=xy_label_fontsize);
ax_explained_var.set_ylim(-1,115);
ax_explained_var.set_yticks([0,25,50,75,100])
ax_explained_var.set_yticklabels([0,25,50,75,100], fontsize=xy_label_fontsize)
ax_explained_var.set_xlim(-0.3,12);
ax_explained_var.set_xticks([0,3,6,9,12])
ax_explained_var.set_xticklabels([0,3,6,9,12], fontsize=xy_label_fontsize)

for key in processed_res_curves.keys():
    y_error = precisely_timed_spikes_per_axon_2C[key].std(axis=0)
    ax_n_spikes_m_cons.errorbar(connections_per_axon_2C, precisely_timed_spikes_per_axon_2C[key].mean(axis=0), yerr=y_error, label=key, lw=4, color=color_map[key])

ax_n_spikes_m_cons.legend(loc='upper left', fontsize=22)
ax_n_spikes_m_cons.set_title('Placing Precisely Timed output Spikes', fontsize=title_fontsize)
ax_n_spikes_m_cons.set_xlabel('Number of Multiple Contacts - M', fontsize=xy_label_fontsize)
ax_n_spikes_m_cons.set_ylabel('Precisely Timed Spikes / Axon', fontsize=xy_label_fontsize);
ax_n_spikes_m_cons.spines['top'].set_visible(False)
ax_n_spikes_m_cons.spines['right'].set_visible(False)
ax_n_spikes_m_cons.set_yticks([0.15,0.3,0.45])
ax_n_spikes_m_cons.set_yticklabels([0.15,0.3,0.45], fontsize=xy_label_fontsize)
ax_n_spikes_m_cons.set_xticks([1,2,3,5,10,15])
ax_n_spikes_m_cons.set_xticklabels([1,2,3,5,10,15], fontsize=xy_label_fontsize)

# add the asimptote line
if AUC_score > 0.99:
    optimal_const_value = np.ones(connections_per_axon_2C.shape) * requested_number_of_output_spikes / num_axons
    ax_n_spikes_m_cons.plot(connections_per_axon_2C, optimal_const_value, label='Optimal 3 PSPs', ls='dashed', lw=2, color='red')
    ax_n_spikes_m_cons.scatter(3, optimal_const_value[0], label='Optimal 3 PSPs', s=200, color='red')
    ax_n_spikes_m_cons.set_ylim(0.09,1.06 * optimal_const_value[0])
    ax_n_spikes_m_cons.legend(loc='center right', fontsize=17)

# save figure
if save_figures:
    figure_name = 'F&F_explanatory_Figure_4_%d' %(np.random.randint(200))
    for file_ending in all_file_endings_to_use:
        if file_ending == '.png':
            fig.savefig(figure_folder + figure_name + file_ending, bbox_inches='tight')
        else:
            fig.savefig(figure_folder + figure_name + file_ending, bbox_inches='tight')


#%%
