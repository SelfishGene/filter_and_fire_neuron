import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import linear_model
from sklearn.metrics import roc_curve, roc_auc_score, auc
import pickle
import matplotlib
import matplotlib.gridspec as gridspec

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

#%% main script params

# input parameters
num_axons = 100

# neuron model parameters
connections_per_axon = 5
num_synapses = connections_per_axon * num_axons

v_reset     = -80
v_threshold = -55
current_to_voltage_mult_factor = 3
refreactory_time_constant = 15

model_type = 'F&F'
#model_type = 'I&F'

# synapse non-learnable parameters
if model_type == 'F&F':
    tau_rise_range  = [1,16]
    tau_decay_range = [8,24]
elif model_type == 'I&F':
    tau_rise_range  = [1,1]
    tau_decay_range = [24,24]

tau_rise_vec  = np.random.uniform(low=tau_rise_range[0] , high=tau_rise_range[1] , size=(num_synapses, 1))
tau_decay_vec = np.random.uniform(low=tau_decay_range[0], high=tau_decay_range[1], size=(num_synapses, 1))

# synapse learnable parameters
synaptic_weights_vec = np.random.normal(size=(num_synapses, 1))

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
    temporal_filter_length = int(4 * tau_decay_vec.max()) + 1

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


def simulate_filter_and_fire_cell_inference(presynaptic_input_spikes, synaptic_weights, tau_rise_vec, tau_decay_vec,
                                            refreactory_time_constant=20, v_reset=-75, v_threshold=-55, current_to_voltage_mult_factor=2):

    temporal_filter_length = int(5 * refreactory_time_constant) + 1
    refreactory_filter = signal.exponential(M=temporal_filter_length,center=0,tau=refreactory_time_constant,sym=False)[np.newaxis,:]

    # padd input and get all synaptic filters
    normlized_syn_filter = construct_normlized_synaptic_filter(tau_rise_vec, tau_decay_vec)
    padded_input = np.hstack((np.zeros(normlized_syn_filter.shape), presynaptic_input_spikes))

    # calc somatic current
    weighted_syn_filter  = synaptic_weights * normlized_syn_filter
    soma_current = signal.convolve(padded_input, weighted_syn_filter, mode='valid')[:,1:]

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

    return soma_voltage, output_spike_times_in_ms


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


#%% generate sample input

# generate sample input
stimulus_duration_ms = 60000

axons_input_spikes = np.random.rand(num_axons, stimulus_duration_ms) < 0.001

presynaptic_input_spikes = np.kron(np.ones((connections_per_axon,1)), axons_input_spikes)

assert presynaptic_input_spikes.shape[0] == num_synapses, 'number of synapses doesnt match the number of presynaptic inputs'

# generate desired pattern of output spikes
requested_number_of_output_spikes = 40
min_time_between_spikes_ms = 125

desired_output_spike_times = min_time_between_spikes_ms * np.random.randint(int(stimulus_duration_ms / min_time_between_spikes_ms), size=requested_number_of_output_spikes)
desired_output_spike_times = np.sort(np.unique(desired_output_spike_times))

desired_output_spikes = np.zeros((stimulus_duration_ms,))
desired_output_spikes[desired_output_spike_times] = 1.0

print('number of requested output spikes = %d' %(requested_number_of_output_spikes))

# simulate cell with normlized currents
local_normlized_currents, soma_voltage, output_spike_times_in_ms = simulate_filter_and_fire_cell_training(presynaptic_input_spikes,
                                                                                                          synaptic_weights_vec, tau_rise_vec, tau_decay_vec,
                                                                                                          refreactory_time_constant=refreactory_time_constant,
                                                                                                          v_reset=v_reset, v_threshold=v_threshold,
                                                                                                          current_to_voltage_mult_factor=current_to_voltage_mult_factor)

output_spikes = np.zeros((stimulus_duration_ms,))
try:
    output_spikes[np.array(output_spike_times_in_ms)] = 1.0
except:
    print('no output spikes created')

#%% fit linear model to local currents and display GT vs prediction

# fit linear model to local currents
logistic_reg_model = linear_model.LogisticRegression(C=30000, fit_intercept=True, penalty='l2', max_iter=3000)

spike_safety_range_ms = 1
negative_subsampling_fraction = 0.99

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

plt.close('all')
plt.figure(figsize=(25,12))
plt.subplot(2,1,1);
plt.plot(1.05 * y - 0.025); plt.title('train AUC = %.5f' %(train_AUC))
plt.plot(y_hat); plt.xlabel('training samples'); plt.legend(['GT', 'prediction'])

plt.subplot(2,1,2);
plt.plot(1.05 * desired_output_spikes - 0.025); plt.title('full trace AUC = %.5f' %(full_AUC))
plt.plot(fitted_output_spike_prob); plt.xlabel('time [ms]'); plt.legend(['GT', 'prediction'])

#%% Display Input Raster, input and output before and after learning

plt.close('all')
fig = plt.figure(figsize=(20,16))
gs_figure = gridspec.GridSpec(nrows=9,ncols=1)
gs_figure.update(left=0.04, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.4)

ax_axons           = plt.subplot(gs_figure[:6,:])
ax_before_learning = plt.subplot(gs_figure[6,:])
ax_after_learning  = plt.subplot(gs_figure[7,:])
ax_desired_output  = plt.subplot(gs_figure[8,:])

syn_activation_time, syn_activation_index = np.nonzero(axons_input_spikes.T)
syn_activation_time = syn_activation_time / 1000

min_time_sec = 0
max_time_sec = stimulus_duration_ms / 1000

time_sec = np.linspace(min_time_sec, max_time_sec, output_spikes.shape[0])

ax_axons.scatter(syn_activation_time, syn_activation_index, s=2, c='k'); ax_axons.set_title('input axons raster', fontsize=15)
ax_axons.set_xlim(min_time_sec, max_time_sec);
ax_axons.set_xticks([])
ax_axons.spines['top'].set_visible(False)
ax_axons.spines['bottom'].set_visible(False)
ax_axons.spines['left'].set_visible(False)
ax_axons.spines['right'].set_visible(False)

ax_before_learning.plot(time_sec, output_spikes, c='k'); ax_before_learning.set_title('before learning', fontsize=15)
ax_before_learning.set_xlim(min_time_sec, max_time_sec);
ax_before_learning.set_xticks([])
ax_before_learning.set_yticks([])
ax_before_learning.spines['top'].set_visible(False)
ax_before_learning.spines['bottom'].set_visible(False)
ax_before_learning.spines['left'].set_visible(False)
ax_before_learning.spines['right'].set_visible(False)

ax_after_learning.plot(time_sec, output_spikes_after_learning, c='k'); ax_after_learning.set_title('after learning', fontsize=15)
ax_after_learning.set_xlim(min_time_sec, max_time_sec);
ax_after_learning.set_xticks([])
ax_after_learning.set_yticks([])
ax_after_learning.spines['top'].set_visible(False)
ax_after_learning.spines['bottom'].set_visible(False)
ax_after_learning.spines['left'].set_visible(False)
ax_after_learning.spines['right'].set_visible(False)

ax_desired_output.plot(time_sec, desired_output_spikes, c='k'); ax_desired_output.set_title('desired output (num spikes = %d)' %(requested_number_of_output_spikes), fontsize=15);
ax_desired_output.set_xlim(min_time_sec, max_time_sec);
ax_desired_output.set_yticks([]);
ax_desired_output.spines['top'].set_visible(False)
ax_desired_output.spines['bottom'].set_visible(False)
ax_desired_output.spines['left'].set_visible(False)
ax_desired_output.spines['right'].set_visible(False)

#%% Load pickle file with previously stored results and check that it's OK

results_filename = data_folder + 'FF_vs_IF_capacity_comparision__num_axons_200__sim_duration_sec_120__num_mult_conn_6__rand_rep_18.pickle'

loaded_script_results_dict = pickle.load(open(results_filename, "rb" ))

print('-----------------------------------------------------------------------------------------------------------')
print('loaded_script_results_dict.keys():')
print('----------')
print(loaded_script_results_dict.keys())
print('-----------------------------------------------------------------------------------------------------------')
print('loaded_script_results_dict["script_main_params"].keys():')
print('----------')
[print(x) for x in loaded_script_results_dict['script_main_params'].keys()]
print('-----------------------------------------------------------------------------------------------------------')
print('num_axons =', loaded_script_results_dict['script_main_params']['num_axons'])
print('stimulus_duration_sec =', loaded_script_results_dict['script_main_params']['stimulus_duration_sec'])
print('min_time_between_spikes_ms =', loaded_script_results_dict['script_main_params']['min_time_between_spikes_ms'])
print('refreactory_time_constant =', loaded_script_results_dict['script_main_params']['refreactory_time_constant'])
print('num_random_iter =', loaded_script_results_dict['script_main_params']['num_random_iter'])
print('spike_safety_range_ms =', loaded_script_results_dict['script_main_params']['spike_safety_range_ms'])
print('negative_subsampling_fraction =', loaded_script_results_dict['script_main_params']['negative_subsampling_fraction'])
print('-----------------------------------------------------------------------------------------------------------')

#%% extract params

processed_res_curves = loaded_script_results_dict['processed_res_curves']
all_results_curves   = loaded_script_results_dict['all_results_curves']

num_axons = loaded_script_results_dict['script_main_params']['num_axons']
stimulus_duration_sec = loaded_script_results_dict['script_main_params']['stimulus_duration_sec']

#%% Fig 2B

num_plots = len(all_results_curves.keys())

num_random_iterations = 5

fig = plt.figure(figsize=(30,15));
for key, value in all_results_curves.items():
    plt.errorbar(value['num_spikes'], value['mean_AUC'], yerr=value['std_AUC'] / np.sqrt(num_random_iterations), label=key, lw=4)
plt.legend(loc='lower left', fontsize=23, ncol=2)
plt.title('learning to place precisly timed output spikes (random input, %d sec window, %d axons)' %(stimulus_duration_sec, num_axons), fontsize=24)
plt.ylabel('accuracy at 1ms precision (AUC)', fontsize=24)
plt.xlabel('num requested spikes to place', fontsize=30);

#%% Fig 2C

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

num_plots = len(processed_res_curves.keys())

color_map = {}
color_map['I&F'] = '0.05'
color_map['F&F'] = 'orange'

fig = plt.figure(figsize=(30,12));
for key in processed_res_curves.keys():
    y_error = precisely_timed_spikes_per_axon_2C[key].std(axis=0)
    plt.errorbar(connections_per_axon_2C, precisely_timed_spikes_per_axon_2C[key].mean(axis=0), yerr=y_error, label=key, lw=4, color=color_map[key])
plt.title('learning to place precisely timed output spikes (random input, %d sec window, %d axons)' %(stimulus_duration_sec, num_axons), fontsize=24)
plt.xlabel('Number of Multiple Contacts - M', fontsize=24)
plt.ylabel('Number of Accuractly Timed Spikes\n per Input Axon', fontsize=24);
plt.legend(loc='upper left', fontsize=40)
plt.yticks([0.15,0.3,0.45])

#%% Fig 2D

filename_str = 'FF_vs_IF_capacity_comparision__num_axons_%d__sim_duration_sec_120__num_mult_conn_5__rand_rep_12.pickle'

num_axons_list = sorted([50,100,200,300,400])
num_multiple_conn_list = [1,3,5,10]

all_filenames_str = [filename_str %(x) for x in num_axons_list]

FF_num_placed_spikes = {}
IF_num_placed_spikes = {}

for num_M_conn in num_multiple_conn_list:

    FF_num_placed_spikes[num_M_conn] = {}
    FF_num_placed_spikes[num_M_conn]['num_accurately_placed_spikes'] = []
    FF_num_placed_spikes[num_M_conn]['num_accurately_placed_spikes_error'] = []

    IF_num_placed_spikes[num_M_conn] = {}
    IF_num_placed_spikes[num_M_conn]['num_accurately_placed_spikes'] = []
    IF_num_placed_spikes[num_M_conn]['num_accurately_placed_spikes_error'] = []

    for curr_num_axons, curr_filename in zip(num_axons_list, all_filenames_str):

        curr_results_filename = data_folder + curr_filename
        curr_loaded_results_dict = pickle.load(open(curr_results_filename, "rb" ))

        FF_ind = list(curr_loaded_results_dict['processed_res_curves']['F&F']['connections_per_axon']).index(num_M_conn)
        FF_num_spikes = curr_loaded_results_dict['processed_res_curves']['F&F']['num_almost_perfectly_placed_spikes'][FF_ind]

        FF_error_index = list(curr_loaded_results_dict['all_results_curves']['F&F, %d connections' %(num_M_conn)]['num_spikes']).index(FF_num_spikes)
        FF_error_scale = (curr_loaded_results_dict['all_results_curves']['F&F, %d connections' %(num_M_conn)]['num_spikes'][FF_error_index + 1] -
                          curr_loaded_results_dict['all_results_curves']['F&F, %d connections' %(num_M_conn)]['num_spikes'][max(0, FF_error_index - 1)])

        if FF_error_index > 1:
            FF_error_scale /= 2

        IF_ind = list(curr_loaded_results_dict['processed_res_curves']['I&F']['connections_per_axon']).index(num_M_conn)
        IF_num_spikes = curr_loaded_results_dict['processed_res_curves']['I&F']['num_almost_perfectly_placed_spikes'][IF_ind]

        IF_error_index = list(curr_loaded_results_dict['all_results_curves']['I&F, %d connections' %(num_M_conn)]['num_spikes']).index(IF_num_spikes)
        IF_error_scale = (curr_loaded_results_dict['all_results_curves']['F&F, %d connections' %(num_M_conn)]['num_spikes'][IF_error_index + 1] -
                          curr_loaded_results_dict['all_results_curves']['F&F, %d connections' %(num_M_conn)]['num_spikes'][max(0, IF_error_index - 1)])

        if IF_error_index > 1:
            IF_error_scale /= 2

        FF_num_placed_spikes[num_M_conn]['num_accurately_placed_spikes'].append(FF_num_spikes)
        FF_num_placed_spikes[num_M_conn]['num_accurately_placed_spikes_error'].append(FF_error_scale)
        IF_num_placed_spikes[num_M_conn]['num_accurately_placed_spikes'].append(IF_num_spikes)
        IF_num_placed_spikes[num_M_conn]['num_accurately_placed_spikes_error'].append(IF_error_scale)


fig = plt.figure(figsize=(25,16))

for M in [10,3]:
    plt.errorbar(num_axons_list, FF_num_placed_spikes[M]['num_accurately_placed_spikes'], yerr=FF_num_placed_spikes[M]['num_accurately_placed_spikes_error'], label='F&F (M = %d)' %(M), lw=5)
plt.errorbar(num_axons_list, IF_num_placed_spikes[1]['num_accurately_placed_spikes'], yerr=FF_num_placed_spikes[1]['num_accurately_placed_spikes_error'], label='I&F', lw=5)

plt.legend(loc='upper left', fontsize=30)
plt.title('Capacity Linearly scales with Number of Axons', fontsize=30)
plt.ylabel('Number of Accuratley Timed Spikes', fontsize=25)
plt.xlabel('Number of Input Axons', fontsize=25);

#%% Full Figure 2

plt.close('all')
fig = plt.figure(figsize=(25,16))
gs_figure = gridspec.GridSpec(nrows=13,ncols=6)
gs_figure.update(left=0.04, right=0.95, bottom=0.05, top=0.95, wspace=0.45, hspace=0.4)

ax_axons            = plt.subplot(gs_figure[:4,:4])
ax_before_learning  = plt.subplot(gs_figure[4,:4])
ax_after_learning   = plt.subplot(gs_figure[5,:4])
ax_desired_output   = plt.subplot(gs_figure[6,:4])
ax_acc_per_n_spikes = plt.subplot(gs_figure[8:,:4])

ax_n_spikes_m_cons  = plt.subplot(gs_figure[:6,4:])
ax_n_spikes_n_axons = plt.subplot(gs_figure[7:,4:])

# 2 A
before_color = '0.15'
after_color  = 'blue'
target_color = 'red'

syn_activation_time, syn_activation_index = np.nonzero(axons_input_spikes.T)
syn_activation_time = syn_activation_time / 1000

min_time_sec = 0
max_time_sec = stimulus_duration_ms / 1000

time_sec = np.linspace(min_time_sec, max_time_sec, output_spikes.shape[0])

ax_axons.scatter(syn_activation_time, syn_activation_index, s=3, c='k'); ax_axons.set_title('Input Axons Raster', fontsize=18)
ax_axons.set_xlim(min_time_sec, max_time_sec);
ax_axons.set_xticks([])
ax_axons.set_yticks([0,25,50,75,100])
ax_axons.set_yticklabels([0,25,50,75,100],fontsize=15)
ax_axons.spines['top'].set_visible(False)
ax_axons.spines['bottom'].set_visible(False)
ax_axons.spines['left'].set_visible(False)
ax_axons.spines['right'].set_visible(False)

ax_before_learning.plot(time_sec, output_spikes, c=before_color, lw=2.5);
ax_before_learning.set_title('Before Learning', fontsize=17, color=before_color)
ax_before_learning.set_xlim(min_time_sec, max_time_sec);
ax_before_learning.set_xticks([])
ax_before_learning.set_yticks([])
ax_before_learning.spines['top'].set_visible(False)
ax_before_learning.spines['bottom'].set_visible(False)
ax_before_learning.spines['left'].set_visible(False)
ax_before_learning.spines['right'].set_visible(False)

ax_after_learning.plot(time_sec, output_spikes_after_learning, c=after_color, lw=2.5);
ax_after_learning.set_title('After Learning', fontsize=17, color=after_color)
ax_after_learning.set_xlim(min_time_sec, max_time_sec);
ax_after_learning.set_xticks([])
ax_after_learning.set_yticks([])
ax_after_learning.spines['top'].set_visible(False)
ax_after_learning.spines['bottom'].set_visible(False)
ax_after_learning.spines['left'].set_visible(False)
ax_after_learning.spines['right'].set_visible(False)

ax_desired_output.plot(time_sec, desired_output_spikes, c=target_color, lw=2.5);
ax_desired_output.set_title('Desired Output (num spikes = %d)' %(requested_number_of_output_spikes), fontsize=17, color=target_color);
ax_desired_output.set_xlim(min_time_sec, max_time_sec);
ax_desired_output.set_yticks([]);
ax_desired_output.set_xticks([0,15,30,45,60])
ax_desired_output.set_xticklabels([0,15,30,45,60],fontsize=15)
ax_desired_output.spines['top'].set_visible(False)
ax_desired_output.spines['bottom'].set_visible(False)
ax_desired_output.spines['left'].set_visible(False)
ax_desired_output.spines['right'].set_visible(False)
ax_desired_output.set_xlabel('time (sec)', fontsize=17);


# 2 B
num_plots = len(all_results_curves.keys())

key_to_label_map = {
    'F&F, 1 connections' : 'F&F (M =  1)',
    'F&F, 2 connections' : 'F&F (M =  2)',
    'F&F, 3 connections' : 'F&F (M =  3)',
    'F&F, 5 connections' : 'F&F (M =  5)',
    'F&F, 10 connections': 'F&F (M = 10)',
    'F&F, 15 connections': 'F&F (M = 15)',
    'I&F, 1 connections' : 'I&F (M =  1)',
    'I&F, 2 connections' : 'I&F (M =  2)',
    'I&F, 3 connections' : 'I&F (M =  3)',
    'I&F, 5 connections' : 'I&F (M =  5)',
    'I&F, 10 connections': 'I&F (M = 10)',
    'I&F, 15 connections': 'I&F (M = 15)'}

key_to_color_map = {
    'F&F, 1 connections' : 'blue',
    'F&F, 2 connections' : 'orange',
    'F&F, 3 connections' : 'green',
    'F&F, 5 connections' : 'crimson',
    'F&F, 10 connections': 'brown',
    'F&F, 15 connections': 'purple',
    'I&F, 1 connections' : 'gray',
    'I&F, 2 connections' : 'gray',
    'I&F, 3 connections' : 'gray',
    'I&F, 5 connections' : 'gray',
    'I&F, 10 connections': 'gray',
    'I&F, 15 connections': 'gray'}

keys_ordering = ['I&F, 1 connections', 'I&F, 2 connections', 'I&F, 3 connections', 'I&F, 5 connections', 'I&F, 10 connections', 'I&F, 15 connections',
                 'F&F, 1 connections', 'F&F, 2 connections', 'F&F, 3 connections', 'F&F, 5 connections', 'F&F, 10 connections', 'F&F, 15 connections']

# for key, value in all_results_curves.items():
for key in keys_ordering:
    value = all_results_curves[key]
    curr_color = key_to_color_map[key]
    curr_label = key_to_label_map[key]
    if curr_color == 'gray':
        ax_acc_per_n_spikes.errorbar(value['num_spikes'], value['mean_AUC'], yerr=value['std_AUC'] / np.sqrt(num_random_iterations),
                                     label=curr_label, color=curr_color, lw=3, alpha=0.6)
    else:
        ax_acc_per_n_spikes.errorbar(value['num_spikes'], value['mean_AUC'], yerr=value['std_AUC'] / np.sqrt(num_random_iterations),
                                     label=curr_label, color=curr_color, lw=3)

ax_acc_per_n_spikes.legend(loc='lower left', fontsize=18, ncol=2)
ax_acc_per_n_spikes.set_title('Placing Precisely Timed output Spikes', fontsize=18)
ax_acc_per_n_spikes.set_ylabel('Accuracy at 1ms Precision (AUC)', fontsize=17)
ax_acc_per_n_spikes.set_xlabel('Number of Requried Precisely Timed Spikes', fontsize=17);
ax_acc_per_n_spikes.spines['top'].set_visible(False)
ax_acc_per_n_spikes.spines['right'].set_visible(False)
ax_acc_per_n_spikes.set_yticks([0.7,0.8,0.9,1.0])
ax_acc_per_n_spikes.set_yticklabels([0.7,0.8,0.9,1.0],fontsize=15)
ax_acc_per_n_spikes.set_xticks([0,50,100,150,200])
ax_acc_per_n_spikes.set_xticklabels([0,50,100,150,200],fontsize=15)


# 2 C
for key in processed_res_curves.keys():
    y_error = precisely_timed_spikes_per_axon_2C[key].std(axis=0)
    ax_n_spikes_m_cons.errorbar(connections_per_axon_2C, precisely_timed_spikes_per_axon_2C[key].mean(axis=0), yerr=y_error, label=key, lw=4, color=color_map[key])

ax_n_spikes_m_cons.legend(loc='upper left', fontsize=22)
ax_n_spikes_m_cons.set_title('Placing Precisely Timed output Spikes', fontsize=18)
ax_n_spikes_m_cons.set_xlabel('Number of Multiple Contacts - M', fontsize=17)
ax_n_spikes_m_cons.set_ylabel('Number of Precisely Timed Spikes / Input Axon', fontsize=17);
ax_n_spikes_m_cons.spines['top'].set_visible(False)
ax_n_spikes_m_cons.spines['right'].set_visible(False)
ax_n_spikes_m_cons.set_yticks([0.15,0.3,0.45])
ax_n_spikes_m_cons.set_ylim([0.08,0.53])
ax_n_spikes_m_cons.set_yticklabels([0.15,0.3,0.45], fontsize=15)
ax_n_spikes_m_cons.set_xticks([1,2,3,5,10,15])
ax_n_spikes_m_cons.set_xticklabels([1,2,3,5,10,15], fontsize=15)


# 2 D
for M in [10,3]:
    ax_n_spikes_n_axons.errorbar(num_axons_list, FF_num_placed_spikes[M]['num_accurately_placed_spikes'], yerr=FF_num_placed_spikes[M]['num_accurately_placed_spikes_error'], label='F&F (M = %d)' %(M), lw=5)
ax_n_spikes_n_axons.errorbar(num_axons_list, IF_num_placed_spikes[1]['num_accurately_placed_spikes'], yerr=FF_num_placed_spikes[1]['num_accurately_placed_spikes_error'], label='I&F', lw=5)

ax_n_spikes_n_axons.legend(loc='upper left', fontsize=22)
ax_n_spikes_n_axons.set_title('Capacity Linearly Scales with Number of Axons', fontsize=18)
ax_n_spikes_n_axons.set_ylabel('Number of Precisely Timed Spikes', fontsize=17)
ax_n_spikes_n_axons.set_xlabel('Number of Input Axons', fontsize=17);
ax_n_spikes_n_axons.set_yticks([0,100,200])
ax_n_spikes_n_axons.set_yticklabels([0,100,200], fontsize=15)
ax_n_spikes_n_axons.set_xticks(num_axons_list)
ax_n_spikes_n_axons.set_xticklabels(num_axons_list, fontsize=15)
ax_n_spikes_n_axons.spines['top'].set_visible(False)
ax_n_spikes_n_axons.spines['right'].set_visible(False)


# save figure
if save_figures:
    figure_name = 'F&F_capacity_Figure_2_%d' %(np.random.randint(200))
    for file_ending in all_file_endings_to_use:
        if file_ending == '.png':
            fig.savefig(figure_folder + figure_name + file_ending, bbox_inches='tight')
        else:
            fig.savefig(figure_folder + figure_name + file_ending, bbox_inches='tight')
