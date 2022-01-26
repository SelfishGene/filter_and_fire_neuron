import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import linear_model
from sklearn.metrics import roc_curve, roc_auc_score, auc
from tensorflow import keras
import matplotlib.gridspec as gridspec

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

#%% script params

# input parameters
num_axons_FF_cap = 100
time_delays_list_IF_cap = [250, 500]
num_time_delays_IF = len(time_delays_list_IF_cap) + 1
num_axons_IF = num_time_delays_IF * num_axons_FF_cap

stimulus_duration_ms = 10000
requested_number_of_output_spikes = 40
min_time_between_spikes_ms = 135

# neuron model parameters
v_reset     = -80
v_threshold = -55
current_to_voltage_mult_factor = 3
refreactory_time_constant = 15

# F&F neuron model parameters
connections_per_axon_FF = 5
num_synapses_FF = connections_per_axon_FF * num_axons_FF_cap

# synapse non-learnable parameters
tau_rise_range_FF  = [1,16]
tau_decay_range_FF = [8,24]

tau_rise_vec_FF  = np.random.uniform(low=tau_rise_range_FF[0] , high=tau_rise_range_FF[1] , size=(num_synapses_FF, 1))
tau_decay_vec_FF = np.random.uniform(low=tau_decay_range_FF[0], high=tau_decay_range_FF[1], size=(num_synapses_FF, 1))

# synapse learnable parameters
synaptic_weights_vec_FF = np.random.normal(size=(num_synapses_FF, 1))

# I&F neuron model parameters
connections_per_axon_IF = 1
num_synapses_IF = connections_per_axon_IF * num_axons_IF

# synapse non-learnable parameters
tau_rise_range_IF  = [1,1]
tau_decay_range_IF = [24,24]

tau_rise_vec_IF  = np.random.uniform(low=tau_rise_range_IF[0] , high=tau_rise_range_IF[1] , size=(num_synapses_IF, 1))
tau_decay_vec_IF = np.random.uniform(low=tau_decay_range_IF[0], high=tau_decay_range_IF[1], size=(num_synapses_IF, 1))

# synapse learnable parameters
synaptic_weights_vec_IF = np.random.normal(size=(num_synapses_IF, 1))

# book-keeping
save_figures = True
save_figures = False
all_file_endings_to_use = ['.png', '.pdf', '.svg']

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


def simulate_filter_and_fire_cell_training_long(presynaptic_input_spikes, synaptic_weights, tau_rise_vec, tau_decay_vec,
                                                refreactory_time_constant=20, v_reset=-75, v_threshold=-55, current_to_voltage_mult_factor=2):


    total_duration_ms = presynaptic_input_spikes.shape[1]
    max_duration_per_call_ms = 50000
    overlap_time_ms = 500

    if max_duration_per_call_ms >= total_duration_ms:
        local_normlized_currents, soma_voltage, output_spike_times_in_ms = simulate_filter_and_fire_cell_training(presynaptic_input_spikes, synaptic_weights, tau_rise_vec, tau_decay_vec,
                                                                                                                  refreactory_time_constant=refreactory_time_constant, v_reset=v_reset, v_threshold=v_threshold,
                                                                                                                  current_to_voltage_mult_factor=current_to_voltage_mult_factor)
        return local_normlized_currents, soma_voltage, output_spike_times_in_ms


    local_normlized_currents = np.zeros(presynaptic_input_spikes.shape, dtype=np.float16)
    soma_voltage = np.zeros((total_duration_ms,))
    output_spike_times_in_ms = []

    num_sub_calls = int(np.ceil(total_duration_ms / (max_duration_per_call_ms - overlap_time_ms)))
    end_ind = overlap_time_ms
    for k in range(num_sub_calls):
        start_ind = end_ind - overlap_time_ms
        end_ind = min(start_ind + max_duration_per_call_ms, total_duration_ms)

        curr_loc_norm_c, curr_soma_v, curr_out_sp_t = simulate_filter_and_fire_cell_training(presynaptic_input_spikes[:,start_ind:end_ind], synaptic_weights, tau_rise_vec, tau_decay_vec,
                                                                                             refreactory_time_constant=refreactory_time_constant, v_reset=v_reset, v_threshold=v_threshold,
                                                                                             current_to_voltage_mult_factor=current_to_voltage_mult_factor)

        # update fields
        if k == 0:
            local_normlized_currents[:,start_ind:end_ind] = curr_loc_norm_c
            soma_voltage[start_ind:end_ind] = curr_soma_v
            output_spike_times_in_ms += curr_out_sp_t
        else:
            local_normlized_currents[:,(start_ind+overlap_time_ms):end_ind] = curr_loc_norm_c[:,overlap_time_ms:end_ind]
            soma_voltage[(start_ind+overlap_time_ms):end_ind] = curr_soma_v[overlap_time_ms:]
            curr_out_sp_t = [x for x in curr_out_sp_t if x >= (overlap_time_ms-1)]
            output_spike_times_in_ms += [(start_ind + x) for x in curr_out_sp_t]

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


def simulate_filter_and_fire_cell_inference_long(presynaptic_input_spikes, synaptic_weights, tau_rise_vec, tau_decay_vec,
                                                refreactory_time_constant=20, v_reset=-75, v_threshold=-55, current_to_voltage_mult_factor=2):

    total_duration_ms = presynaptic_input_spikes.shape[1]
    max_duration_per_call_ms = 50000
    overlap_time_ms = 500

    if max_duration_per_call_ms >= total_duration_ms:
        soma_voltage, output_spike_times_in_ms = simulate_filter_and_fire_cell_inference(presynaptic_input_spikes, synaptic_weights, tau_rise_vec, tau_decay_vec,
                                                                                         refreactory_time_constant=refreactory_time_constant, v_reset=v_reset, v_threshold=v_threshold,
                                                                                         current_to_voltage_mult_factor=current_to_voltage_mult_factor)
        return soma_voltage, output_spike_times_in_ms


    soma_voltage = np.zeros((total_duration_ms,))
    output_spike_times_in_ms = []

    num_sub_calls = int(np.ceil(total_duration_ms / (max_duration_per_call_ms - overlap_time_ms)))
    end_ind = overlap_time_ms
    for k in range(num_sub_calls):
        start_ind = end_ind - overlap_time_ms
        end_ind = min(start_ind + max_duration_per_call_ms, total_duration_ms)

        curr_soma_v, curr_out_sp_t = simulate_filter_and_fire_cell_inference(presynaptic_input_spikes[:,start_ind:end_ind], synaptic_weights, tau_rise_vec, tau_decay_vec,
                                                                             refreactory_time_constant=refreactory_time_constant, v_reset=v_reset, v_threshold=v_threshold,
                                                                             current_to_voltage_mult_factor=current_to_voltage_mult_factor)

        # update fields
        if k == 0:
            soma_voltage[start_ind:end_ind] = curr_soma_v
            output_spike_times_in_ms += curr_out_sp_t
        else:
            soma_voltage[(start_ind+overlap_time_ms):end_ind] = curr_soma_v[overlap_time_ms:]
            curr_out_sp_t = [x for x in curr_out_sp_t if x >= (overlap_time_ms-1)]
            output_spike_times_in_ms += [(start_ind + x) for x in curr_out_sp_t]

    return soma_voltage, output_spike_times_in_ms


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


#%% define random input for capacity plot

# generate sample input
axons_input_spikes_capacity = np.random.rand(num_axons_FF_cap, stimulus_duration_ms) < 0.0016

# F&F
presynaptic_input_spikes_FF = np.kron(np.ones((connections_per_axon_FF,1)), axons_input_spikes_capacity)
assert presynaptic_input_spikes_FF.shape[0] == num_synapses_FF, 'number of synapses doesnt match the number of presynaptic inputs'

# I&F
presynaptic_input_spikes_IF = axons_input_spikes_capacity.copy()
for delay_ms in time_delays_list_IF_cap:
    curr_delay_axons_input_spikes_IF = np.zeros(axons_input_spikes_capacity.shape)
    curr_delay_axons_input_spikes_IF[:,delay_ms:] = axons_input_spikes_capacity[:,:-delay_ms]
    presynaptic_input_spikes_IF = np.vstack((presynaptic_input_spikes_IF, curr_delay_axons_input_spikes_IF))
assert presynaptic_input_spikes_IF.shape[0] == num_synapses_IF, 'number of synapses doesnt match the number of presynaptic inputs'


# generate desired pattern of output spikes
desired_output_spike_times = min_time_between_spikes_ms * np.random.randint(int(stimulus_duration_ms / min_time_between_spikes_ms), size=requested_number_of_output_spikes)
desired_output_spike_times = np.sort(np.unique(desired_output_spike_times))

desired_output_spikes = np.zeros((stimulus_duration_ms,))
desired_output_spikes[desired_output_spike_times] = 1.0

print('number of requested output spikes = %d' %(requested_number_of_output_spikes))


#%% fit F&F model to the input

# simulate cell with normlized currents
local_normlized_currents_FF, _, _ = simulate_filter_and_fire_cell_training(presynaptic_input_spikes_FF,
                                                                           synaptic_weights_vec_FF, tau_rise_vec_FF, tau_decay_vec_FF,
                                                                           refreactory_time_constant=refreactory_time_constant,
                                                                           v_reset=v_reset, v_threshold=v_threshold,
                                                                           current_to_voltage_mult_factor=current_to_voltage_mult_factor)


# fit linear model to local currents
filter_and_fire_model = linear_model.LogisticRegression(C=30000, fit_intercept=True, penalty='l2', max_iter=3000)

spike_safety_range_ms = 1
negative_subsampling_fraction = 0.99

X, y = prepare_training_dataset(local_normlized_currents_FF, desired_output_spikes,
                                spike_safety_range_ms=spike_safety_range_ms,
                                negative_subsampling_fraction=negative_subsampling_fraction)

# fit model
filter_and_fire_model.fit(X,y)

print('number of data points = %d (%.2f%s positive class)' %(X.shape[0], 100 * y.mean(),'%'))

y_hat = filter_and_fire_model.predict_proba(X)[:,1]

# calculate AUC
train_AUC = roc_auc_score(y, y_hat)

fitted_output_spike_prob_FF = filter_and_fire_model.predict_proba(local_normlized_currents_FF.T)[:,1]
full_AUC = roc_auc_score(desired_output_spikes, fitted_output_spike_prob_FF)

# get desired FP threshold
desired_false_positive_rate = 0.004

fpr, tpr, thresholds = roc_curve(desired_output_spikes, fitted_output_spike_prob_FF)

desired_fp_ind = np.argmin(abs(fpr-desired_false_positive_rate))
if desired_fp_ind == 0:
    desired_fp_ind = 1

actual_false_positive_rate = fpr[desired_fp_ind]
true_positive_rate         = tpr[desired_fp_ind]
desired_fp_threshold       = thresholds[desired_fp_ind]

AUC_score = auc(fpr, tpr)

if AUC_score > 0.9995:
    desired_fp_threshold = 0.15

print('F&F fitting AUC = %.4f' %(AUC_score))
print('at %.4f FP rate, TP = %.4f' %(actual_false_positive_rate, true_positive_rate))

output_spikes_after_learning_FF = fitted_output_spike_prob_FF > desired_fp_threshold

#%% fit I&F to the input with 2 delayed versions of the same input

# simulate cell with normlized currents
local_normlized_currents_IF, _, _ = simulate_filter_and_fire_cell_training(presynaptic_input_spikes_IF,
                                                                           synaptic_weights_vec_IF, tau_rise_vec_IF, tau_decay_vec_IF,
                                                                           refreactory_time_constant=refreactory_time_constant,
                                                                           v_reset=v_reset, v_threshold=v_threshold,
                                                                           current_to_voltage_mult_factor=current_to_voltage_mult_factor)


# fit linear model to local currents
integrate_and_fire_model = linear_model.LogisticRegression(C=30000, fit_intercept=True, penalty='l2', max_iter=3000)

spike_safety_range_ms = 1
negative_subsampling_fraction = 0.99

X, y = prepare_training_dataset(local_normlized_currents_IF, desired_output_spikes,
                                spike_safety_range_ms=spike_safety_range_ms,
                                negative_subsampling_fraction=negative_subsampling_fraction)

# fit model
integrate_and_fire_model.fit(X,y)

print('number of data points = %d (%.2f%s positive class)' %(X.shape[0], 100 * y.mean(),'%'))

y_hat = integrate_and_fire_model.predict_proba(X)[:,1]

# calculate AUC
train_AUC = roc_auc_score(y, y_hat)

fitted_output_spike_prob_IF = integrate_and_fire_model.predict_proba(local_normlized_currents_IF.T)[:,1]
full_AUC = roc_auc_score(desired_output_spikes, fitted_output_spike_prob_IF)

# get desired FP threshold
desired_false_positive_rate = 0.004

fpr, tpr, thresholds = roc_curve(desired_output_spikes, fitted_output_spike_prob_IF)

desired_fp_ind = np.argmin(abs(fpr - desired_false_positive_rate))
if desired_fp_ind == 0:
    desired_fp_ind = 1

actual_false_positive_rate = fpr[desired_fp_ind]
true_positive_rate         = tpr[desired_fp_ind]
desired_fp_threshold       = thresholds[desired_fp_ind]

AUC_score = auc(fpr, tpr)

if AUC_score > 0.9995:
    desired_fp_threshold = 0.15

print('I&F fitting AUC = %.4f' %(AUC_score))
print('at %.4f FP rate, TP = %.4f' %(actual_false_positive_rate, true_positive_rate))

output_spikes_after_learning_IF = fitted_output_spike_prob_IF > desired_fp_threshold

#%% MNIST params

spatial_extent_factor = 5
temporal_extent_factor_numerator = 2
temporal_extent_factor_denumerator = 1

num_const_firing_channels = 20
temporal_silence_ms = 70

positive_digit = 7
num_train_positive_patterns = 2000

release_probability = 1.0
apply_release_prob_during_train = False
apply_releash_prob_during_test = False

output_spike_tolorance_window_duration = 20
output_spike_tolorance_window_offset   = 5

time_delays_list_IF = [12,24]
num_axons_FF = spatial_extent_factor * 20 + num_const_firing_channels
num_time_delays_IF = len(time_delays_list_IF) + 1
num_axons_IF = num_time_delays_IF * num_axons_FF

num_synapses_FF = connections_per_axon_FF * num_axons_FF
connections_per_axon_IF = 1
num_synapses_IF = connections_per_axon_IF * num_axons_IF

# synapse non-learnable parameters
tau_rise_range_FF  = [1,16]
tau_decay_range_FF = [8,24]

tau_rise_vec_FF  = np.random.uniform(low=tau_rise_range_FF[0] , high=tau_rise_range_FF[1] , size=(num_synapses_FF, 1))
tau_decay_vec_FF = np.random.uniform(low=tau_decay_range_FF[0], high=tau_decay_range_FF[1], size=(num_synapses_FF, 1))

# synapse learnable parameters
synaptic_weights_vec_FF = np.random.normal(size=(num_synapses_FF, 1))

# I&F neuron model parameters
connections_per_axon_IF = 1
num_synapses_IF = connections_per_axon_IF * num_axons_IF

# synapse non-learnable parameters
tau_rise_range_IF  = [1,1]
tau_decay_range_IF = [24,24]

tau_rise_vec_IF  = np.random.uniform(low=tau_rise_range_IF[0] , high=tau_rise_range_IF[1] , size=(num_synapses_IF, 1))
tau_decay_vec_IF = np.random.uniform(low=tau_decay_range_IF[0], high=tau_decay_range_IF[1], size=(num_synapses_IF, 1))

# synapse learnable parameters
synaptic_weights_vec_IF = np.random.normal(size=(num_synapses_IF, 1))

#%% load MNIST dataset and and transform into spikes

(x_train_original, y_train), (x_test_original, y_test) = keras.datasets.mnist.load_data()

# crop the data and binarize it
h_crop_range = [4,24]
w_crop_range = [4,24]

positive_threshold = 150

x_train_original = x_train_original[:,h_crop_range[0]:h_crop_range[1],w_crop_range[0]:w_crop_range[1]] > positive_threshold
x_test_original  = x_test_original[: ,h_crop_range[0]:h_crop_range[1],w_crop_range[0]:w_crop_range[1]] > positive_threshold

#%% Transform Xs to spatio-temporal spike trains

# extend according to "temporal_extent_factor"
kernel = np.ones((1, spatial_extent_factor, temporal_extent_factor_numerator), dtype=bool)

x_train = x_train_original.copy()
x_test  = x_test_original.copy()

# reshape X according to what is needed
x_train = np.kron(x_train, kernel)
x_test = np.kron(x_test, kernel)

# subsample according to "temporal_extent_factor_denumerator"
x_train = x_train[:,:,::temporal_extent_factor_denumerator]
x_test = x_test[:,:,::temporal_extent_factor_denumerator]

# padd with ones on top (for "bias" learning)
top_pad_train = np.ones((1, num_const_firing_channels, x_train.shape[2]), dtype=bool)
top_pad_test  = np.ones((1, num_const_firing_channels, x_test.shape[2] ), dtype=bool)

# add a few zero rows for clear seperation for visualization purpuses
top_pad_train[:,-5:,:] = 0
top_pad_test[:,-5:,:] = 0

x_train = np.concatenate((np.tile(top_pad_train, [x_train.shape[0],1,1]), x_train), axis=1)
x_test  = np.concatenate((np.tile(top_pad_test , [x_test.shape[0],1,1] ), x_test ), axis=1)

# pad with "temporal_silence_ms" zeros in the begining of each pattern (for silence between patterns)
left_pad_train = np.zeros((1, x_train.shape[1], temporal_silence_ms), dtype=bool)
left_pad_test  = np.zeros((1, x_test.shape[1] , temporal_silence_ms), dtype=bool)

x_train = np.concatenate((np.tile(left_pad_train, [x_train.shape[0],1,1]), x_train), axis=2)
x_test  = np.concatenate((np.tile(left_pad_test , [x_test.shape[0],1,1] ), x_test ), axis=2)

# add background activity
desired_background_activity_firing_rate_Hz = 10
background_activity_fraction = desired_background_activity_firing_rate_Hz / 1000

x_train[np.random.rand(x_train.shape[0], x_train.shape[1], x_train.shape[2]) < background_activity_fraction] = 1
x_test[ np.random.rand(x_test.shape[0] , x_test.shape[1] , x_test.shape[2] ) < background_activity_fraction] = 1

# subsample the input spikes
desired_average_input_firing_rate_Hz = 20
actual_mean_firing_rate_Hz = 1000 * x_train.mean()

fraction_of_spikes_to_eliminate = desired_average_input_firing_rate_Hz / actual_mean_firing_rate_Hz

x_train = x_train * (np.random.rand(x_train.shape[0], x_train.shape[1], x_train.shape[2]) < fraction_of_spikes_to_eliminate)
x_test  = x_test  * (np.random.rand(x_test.shape[0], x_test.shape[1], x_test.shape[2]) < fraction_of_spikes_to_eliminate)

final_mean_firing_rate_Hz = 1000 * x_train.mean()

#%% Create "one-vs-all" dataset

y_train_binary = y_train == positive_digit
y_test_binary  = y_test  == positive_digit

num_train_positive_patterns = min(int(y_train_binary.sum()), num_train_positive_patterns)

num_train_negative_patterns = int(2.0 * num_train_positive_patterns)

positive_inds = np.where(y_train_binary)[0]
negative_inds = np.where(~y_train_binary)[0]

selected_train_positives = np.random.choice(positive_inds, size=num_train_positive_patterns)
selected_train_negatives = np.random.choice(negative_inds, size=num_train_negative_patterns)

all_selected = np.random.permutation(np.concatenate((selected_train_positives, selected_train_negatives)))

X_train_spikes = x_train[all_selected]
Y_train_spikes = y_train_binary[all_selected]

X_test_spikes = x_test.copy()
Y_test_spikes = y_test_binary.copy()

zero_pred_baseline_accuracy = 100 * (1 - Y_test_spikes.mean())


#%% prepare input spikes for F&F and I&F training

axons_input_spikes = np.concatenate([X_train_spikes[k] for k in range(X_train_spikes.shape[0])],axis=1)

# prepare output spikes
pattern_duration_ms = X_train_spikes[0].shape[1]
output_spike_offset = 1
output_kernel = np.zeros((pattern_duration_ms,))
output_kernel[-output_spike_offset] = 1

desired_output_spikes_mnist = np.kron(Y_train_spikes, output_kernel)

# F&F
presynaptic_input_spikes_FF = np.kron(np.ones((connections_per_axon_FF,1)), axons_input_spikes)
assert presynaptic_input_spikes_FF.shape[0] == num_synapses_FF, 'number of synapses doesnt match the number of presynaptic inputs'

# I&F
presynaptic_input_spikes_IF = axons_input_spikes.copy()
for delay_ms in time_delays_list_IF:
    curr_delay_axons_input_spikes_IF = np.zeros(axons_input_spikes.shape)
    curr_delay_axons_input_spikes_IF[:,delay_ms:] = axons_input_spikes[:,:-delay_ms]
    presynaptic_input_spikes_IF = np.vstack((presynaptic_input_spikes_IF, curr_delay_axons_input_spikes_IF))
assert presynaptic_input_spikes_IF.shape[0] == num_synapses_IF, 'number of synapses doesnt match the number of presynaptic inputs'


#%% prepare input spikes for F&F and I&F testing

num_test_patterns = X_test_spikes.shape[0]

# prepare test outputs
output_kernel_test = np.zeros((X_test_spikes[0].shape[1],))
output_kernel_test[-output_spike_tolorance_window_duration:] = 1

desired_output_spikes_test = np.kron(Y_test_spikes[:num_test_patterns], output_kernel_test)
desired_output_spikes_test = np.concatenate((np.zeros((output_spike_tolorance_window_offset,)), desired_output_spikes_test[:-output_spike_tolorance_window_offset]))

# prepare test inputs
axons_input_spikes_test = np.concatenate([X_test_spikes[k] for k in range(num_test_patterns)],axis=1)

# F&F
presynaptic_input_spikes_test_FF = np.kron(np.ones((connections_per_axon_FF,1)), axons_input_spikes_test)
assert presynaptic_input_spikes_test_FF.shape[0] == num_synapses_FF, 'number of synapses doesnt match the number of presynaptic inputs'

# I&F
presynaptic_input_spikes_test_IF = axons_input_spikes_test.copy()
for delay_ms in time_delays_list_IF:
    curr_delay_axons_input_spikes_IF = np.zeros(axons_input_spikes_test.shape)
    curr_delay_axons_input_spikes_IF[:,delay_ms:] = axons_input_spikes_test[:,:-delay_ms]
    presynaptic_input_spikes_test_IF = np.vstack((presynaptic_input_spikes_test_IF, curr_delay_axons_input_spikes_IF))
assert presynaptic_input_spikes_test_IF.shape[0] == num_synapses_IF, 'number of synapses doesnt match the number of presynaptic inputs'


#%% simulate F&F cell with normlized currents on train

local_normlized_currents_FF, _, _ = simulate_filter_and_fire_cell_training_long(presynaptic_input_spikes_FF,
                                                                                synaptic_weights_vec_FF, tau_rise_vec_FF, tau_decay_vec_FF,
                                                                                refreactory_time_constant=refreactory_time_constant,
                                                                                v_reset=v_reset, v_threshold=v_threshold,
                                                                                current_to_voltage_mult_factor=current_to_voltage_mult_factor)

#%% fit linear model to local currents

filter_and_fire_model = linear_model.LogisticRegression(C=10000, fit_intercept=False, penalty='l2')

spike_safety_range_ms = 20
negative_subsampling_fraction = 0.5

X, y = prepare_training_dataset(local_normlized_currents_FF, desired_output_spikes_mnist,
                                spike_safety_range_ms=spike_safety_range_ms,
                                negative_subsampling_fraction=negative_subsampling_fraction)

# fit model
filter_and_fire_model.fit(X, y)

print('number of data points = %d (%.2f%s positive class)' %(X.shape[0], 100 * y.mean(),'%'))

# calculate train AUC
y_hat = filter_and_fire_model.predict_proba(X)[:,1]
train_AUC = roc_auc_score(y, y_hat)

FF_learned_synaptic_weights = np.fliplr(filter_and_fire_model.coef_).T

#%% find the best multiplicative factor for test prediction F&F

# FF_weight_mult_factors_list = [1,2,3,4,5,6,9,12,20,50,120,250]
FF_weight_mult_factors_list = [1,4,7,11,15]
FF_accuracy_list = []
FF_true_positive_list = []
FF_false_positive_list = []
for weight_mult_factor in FF_weight_mult_factors_list:

    # collect learned synaptic weights
    synaptic_weights_post_learning = weight_mult_factor * FF_learned_synaptic_weights

    soma_voltage_test, output_spike_times_in_ms_test = simulate_filter_and_fire_cell_inference_long(presynaptic_input_spikes_test_FF,
                                                                                                    synaptic_weights_post_learning, tau_rise_vec_FF, tau_decay_vec_FF,
                                                                                                    refreactory_time_constant=refreactory_time_constant,
                                                                                                    v_reset=v_reset, v_threshold=v_threshold,
                                                                                                    current_to_voltage_mult_factor=current_to_voltage_mult_factor)


    output_spikes_test = np.zeros(soma_voltage_test.shape)
    try:
        output_spikes_test[np.array(output_spike_times_in_ms_test)] = 1.0
    except:
        print('no output spikes created')


    # calculate test accuracy
    compact_desired_output_test = Y_test_spikes[:num_test_patterns]

    compact_desired_output_test2 = np.zeros(compact_desired_output_test.shape, dtype=bool)
    compact_predicted_output_test = np.zeros(compact_desired_output_test.shape, dtype=bool)

    # go over all patterns and extract the prediction (depends if the spikes are in the desired window)
    for pattern_ind in range(num_test_patterns):
        start_ind = pattern_duration_ms * pattern_ind + output_spike_tolorance_window_offset
        end_ind = start_ind + pattern_duration_ms

        # extract prediction
        predicted_spike_train_for_pattern = output_spikes_test[start_ind:end_ind]
        desired_spike_train_for_pattern = desired_output_spikes_test[start_ind:end_ind]

        compact_desired_output_test2[pattern_ind]  = desired_spike_train_for_pattern.sum() > 0.1

        if Y_test_spikes[pattern_ind] == 1:
            # check if there is a spike in the desired window only
            compact_predicted_output_test[pattern_ind] = (desired_spike_train_for_pattern * predicted_spike_train_for_pattern).sum() > 0.1
        else:
            # check if there is any spike in the full pattern duration
            compact_predicted_output_test[pattern_ind] = predicted_spike_train_for_pattern.sum() > 0.1

    # small verificaiton
    assert((compact_desired_output_test == compact_desired_output_test2).sum() == num_test_patterns)

    # display accuracy
    percent_accuracy = 100 * (compact_desired_output_test == compact_predicted_output_test).mean()
    true_positive    = 100 * (np.logical_and(compact_desired_output_test == True , compact_predicted_output_test == True).sum() / (compact_desired_output_test == True).sum())
    false_positive   = 100 * (np.logical_and(compact_desired_output_test == False, compact_predicted_output_test == True).sum() / (compact_desired_output_test == False).sum())

    print('weights mult factor = %.1f: Accuracy = %.3f%s. (TP, FP) = (%.3f%s, %.3f%s)' %(weight_mult_factor, percent_accuracy,'%',true_positive,'%',false_positive,'%'))

    FF_accuracy_list.append(percent_accuracy)
    FF_true_positive_list.append(true_positive)
    FF_false_positive_list.append(false_positive)

#%% make a final prediction on the test set F&F

# get the max accuracy weight matrix
max_accuracy_weight_mult_factor = FF_weight_mult_factors_list[np.argsort(np.array(FF_accuracy_list))[-1]]

synaptic_weights_vec_after_learning_FF = max_accuracy_weight_mult_factor * FF_learned_synaptic_weights

# simulate the max accuracy output after learning
soma_voltage_test, output_spike_times_in_ms_test_FF = simulate_filter_and_fire_cell_inference_long(presynaptic_input_spikes_test_FF,
                                                                                                   synaptic_weights_vec_after_learning_FF, tau_rise_vec_FF, tau_decay_vec_FF,
                                                                                                   refreactory_time_constant=refreactory_time_constant,
                                                                                                   v_reset=v_reset, v_threshold=v_threshold,
                                                                                                   current_to_voltage_mult_factor=current_to_voltage_mult_factor)


output_spikes_test_after_learning_full_FF = np.zeros(soma_voltage_test.shape)
try:
    output_spikes_test_after_learning_full_FF[np.array(output_spike_times_in_ms_test)] = 1.0
except:
    print('no output spikes created')


#%% simulate I&F cell with normlized currents on train

local_normlized_currents_IF, _, _ = simulate_filter_and_fire_cell_training_long(presynaptic_input_spikes_IF,
                                                                                synaptic_weights_vec_IF, tau_rise_vec_IF, tau_decay_vec_IF,
                                                                                refreactory_time_constant=refreactory_time_constant,
                                                                                v_reset=v_reset, v_threshold=v_threshold,
                                                                                current_to_voltage_mult_factor=current_to_voltage_mult_factor)

#%% fit linear model to local currents

integrate_and_fire_model = linear_model.LogisticRegression(C=10000, fit_intercept=False, penalty='l2')

spike_safety_range_ms = 20
negative_subsampling_fraction = 0.5

X, y = prepare_training_dataset(local_normlized_currents_IF, desired_output_spikes_mnist,
                                spike_safety_range_ms=spike_safety_range_ms,
                                negative_subsampling_fraction=negative_subsampling_fraction)

# fit model
integrate_and_fire_model.fit(X, y)

print('number of data points = %d (%.2f%s positive class)' %(X.shape[0], 100 * y.mean(),'%'))

# calculate train AUC
y_hat = integrate_and_fire_model.predict_proba(X)[:,1]
train_AUC = roc_auc_score(y, y_hat)

IF_learned_synaptic_weights = np.fliplr(integrate_and_fire_model.coef_).T

#%% find the best multiplicative factor for test prediction F&F

# FF_weight_mult_factors_list = [1,2,3,4,5,6,9,12,20,50,120,250]
IF_weight_mult_factors_list = [1,4,7,11,15]
IF_accuracy_list = []
IF_true_positive_list = []
IF_false_positive_list = []
for weight_mult_factor in IF_weight_mult_factors_list:

    # collect learned synaptic weights
    synaptic_weights_post_learning = weight_mult_factor * IF_learned_synaptic_weights

    soma_voltage_test, output_spike_times_in_ms_test = simulate_filter_and_fire_cell_inference_long(presynaptic_input_spikes_test_IF,
                                                                                                    synaptic_weights_post_learning, tau_rise_vec_IF, tau_decay_vec_IF,
                                                                                                    refreactory_time_constant=refreactory_time_constant,
                                                                                                    v_reset=v_reset, v_threshold=v_threshold,
                                                                                                    current_to_voltage_mult_factor=current_to_voltage_mult_factor)


    output_spikes_test = np.zeros(soma_voltage_test.shape)
    try:
        output_spikes_test[np.array(output_spike_times_in_ms_test)] = 1.0
    except:
        print('no output spikes created')


    # calculate test accuracy
    compact_desired_output_test = Y_test_spikes[:num_test_patterns]

    compact_desired_output_test2 = np.zeros(compact_desired_output_test.shape, dtype=bool)
    compact_predicted_output_test = np.zeros(compact_desired_output_test.shape, dtype=bool)

    # go over all patterns and extract the prediction (depends if the spikes are in the desired window)
    for pattern_ind in range(num_test_patterns):
        start_ind = pattern_duration_ms * pattern_ind + output_spike_tolorance_window_offset
        end_ind = start_ind + pattern_duration_ms

        # extract prediction
        predicted_spike_train_for_pattern = output_spikes_test[start_ind:end_ind]
        desired_spike_train_for_pattern = desired_output_spikes_test[start_ind:end_ind]

        compact_desired_output_test2[pattern_ind]  = desired_spike_train_for_pattern.sum() > 0.1

        if Y_test_spikes[pattern_ind] == 1:
            # check if there is a spike in the desired window only
            compact_predicted_output_test[pattern_ind] = (desired_spike_train_for_pattern * predicted_spike_train_for_pattern).sum() > 0.1
        else:
            # check if there is any spike in the full pattern duration
            compact_predicted_output_test[pattern_ind] = predicted_spike_train_for_pattern.sum() > 0.1

    # small verificaiton
    assert((compact_desired_output_test == compact_desired_output_test2).sum() == num_test_patterns)

    # display accuracy
    percent_accuracy = 100 * (compact_desired_output_test == compact_predicted_output_test).mean()
    true_positive    = 100 * (np.logical_and(compact_desired_output_test == True , compact_predicted_output_test == True).sum() / (compact_desired_output_test == True).sum())
    false_positive   = 100 * (np.logical_and(compact_desired_output_test == False, compact_predicted_output_test == True).sum() / (compact_desired_output_test == False).sum())

    print('weights mult factor = %.1f: Accuracy = %.3f%s. (TP, FP) = (%.3f%s, %.3f%s)' %(weight_mult_factor, percent_accuracy,'%',true_positive,'%',false_positive,'%'))

    IF_accuracy_list.append(percent_accuracy)
    IF_true_positive_list.append(true_positive)
    IF_false_positive_list.append(false_positive)

#%% make a final prediction on the test set F&F

# get the max accuracy weight matrix
max_accuracy_weight_mult_factor = IF_weight_mult_factors_list[np.argsort(np.array(IF_accuracy_list))[-1]]

synaptic_weights_vec_after_learning_IF = max_accuracy_weight_mult_factor * IF_learned_synaptic_weights

# simulate the max accuracy output after learning
soma_voltage_test, output_spike_times_in_ms_test_IF = simulate_filter_and_fire_cell_inference_long(presynaptic_input_spikes_test_IF,
                                                                                                   synaptic_weights_vec_after_learning_IF, tau_rise_vec_IF, tau_decay_vec_IF,
                                                                                                   refreactory_time_constant=refreactory_time_constant,
                                                                                                   v_reset=v_reset, v_threshold=v_threshold,
                                                                                                   current_to_voltage_mult_factor=current_to_voltage_mult_factor)


output_spikes_test_after_learning_full_IF = np.zeros(soma_voltage_test.shape)
try:
    output_spikes_test_after_learning_full_IF[np.array(output_spike_times_in_ms_test)] = 1.0
except:
    print('no output spikes created')


#%% Build the figure


plt.close('all')
fig = plt.figure(figsize=(12,22))
gs_figure = gridspec.GridSpec(nrows=17,ncols=1)
gs_figure.update(left=0.04, right=0.95, bottom=0.02, top=0.98, wspace=0.45, hspace=0.4)

ax_axons          = plt.subplot(gs_figure[:5,:])
ax_FF_spikes      = plt.subplot(gs_figure[5,:])
ax_IF_spikes      = plt.subplot(gs_figure[6,:])
ax_desired_output = plt.subplot(gs_figure[7,:])

ax_digits            = plt.subplot(gs_figure[9:11,:])
ax_axons_mnist       = plt.subplot(gs_figure[11:15,:])
ax_learning_outcomes = plt.subplot(gs_figure[15:,:])

FF_color = 'orange'
IF_color = '0.05'
target_color = 'red'

syn_activation_time, syn_activation_index = np.nonzero(presynaptic_input_spikes_IF.T)

syn_activation_time_1_cap, syn_activation_index_1_cap = np.nonzero(axons_input_spikes_capacity.T)
syn_activation_time_1_cap = syn_activation_time_1_cap / 1000

syn_activation_time_2_cap = syn_activation_time_1_cap + (time_delays_list_IF_cap[0] / 1000)
syn_activation_time_3_cap = syn_activation_time_1_cap + (time_delays_list_IF_cap[1] / 1000)
syn_activation_index_2_cap = syn_activation_index_1_cap + num_axons_FF_cap
syn_activation_index_3_cap = syn_activation_index_2_cap + num_axons_FF_cap

min_time_sec = -0.1
max_time_sec = stimulus_duration_ms / 1000
time_sec = np.linspace(0, max_time_sec, output_spikes_after_learning_FF.shape[0])

ax_axons.scatter(syn_activation_time_1_cap, syn_activation_index_1_cap, s=8, c='black');
ax_axons.scatter(syn_activation_time_2_cap, syn_activation_index_2_cap, s=8, c='chocolate');
ax_axons.scatter(syn_activation_time_3_cap, syn_activation_index_3_cap, s=8, c='purple');
ax_axons.set_xlim(min_time_sec, max_time_sec);
ax_axons.set_xticks([])
ax_axons.set_yticks([])
ax_axons.spines['top'].set_visible(False)
ax_axons.spines['bottom'].set_visible(False)
ax_axons.spines['left'].set_visible(False)
ax_axons.spines['right'].set_visible(False)

ax_FF_spikes.plot(time_sec, output_spikes_after_learning_FF, c=FF_color, lw=2.5);
ax_FF_spikes.set_title('F&F (M = 5)', fontsize=17, color=FF_color)
ax_FF_spikes.set_xlim(min_time_sec, max_time_sec);
ax_FF_spikes.set_xticks([])
ax_FF_spikes.set_yticks([])
ax_FF_spikes.spines['top'].set_visible(False)
ax_FF_spikes.spines['bottom'].set_visible(False)
ax_FF_spikes.spines['left'].set_visible(False)
ax_FF_spikes.spines['right'].set_visible(False)

ax_IF_spikes.plot(time_sec, output_spikes_after_learning_IF, c=IF_color, lw=2.5);
ax_IF_spikes.set_title('I&F (Orig Axons + 2 delayed)', fontsize=17, color=IF_color)
ax_IF_spikes.set_xlim(min_time_sec, max_time_sec);
ax_IF_spikes.set_xticks([])
ax_IF_spikes.set_yticks([])
ax_IF_spikes.spines['top'].set_visible(False)
ax_IF_spikes.spines['bottom'].set_visible(False)
ax_IF_spikes.spines['left'].set_visible(False)
ax_IF_spikes.spines['right'].set_visible(False)

ax_desired_output.plot(time_sec, desired_output_spikes, c=target_color, lw=2.5);
ax_desired_output.set_title('Desired Output (num spikes = %d)' %(requested_number_of_output_spikes), fontsize=17, color=target_color);
ax_desired_output.set_xlim(min_time_sec, max_time_sec);
ax_desired_output.set_xticks([]);
ax_desired_output.set_yticks([]);
ax_desired_output.spines['top'].set_visible(False)
ax_desired_output.spines['bottom'].set_visible(False)
ax_desired_output.spines['left'].set_visible(False)
ax_desired_output.spines['right'].set_visible(False)





##%% organize MNIST part into a figure

# test digit images
extention_kernel = np.ones((1, spatial_extent_factor, temporal_extent_factor_numerator), dtype=bool)
x_test_original_extended = np.kron(x_test_original, extention_kernel)
left_pad_test  = np.zeros((1, x_test_original_extended.shape[1] , temporal_silence_ms), dtype=bool)
x_test_original_extended  = np.concatenate((np.tile(left_pad_test , [x_test_original_extended.shape[0],1,1] ), x_test_original_extended ), axis=2)
x_test_axons_input_spikes = np.concatenate([x_test_original_extended[k] for k in range(x_test_original_extended.shape[0])],axis=1)

test_set_full_duration_ms = x_test_axons_input_spikes.shape[1]

# select a subset of time to display
num_digits_to_display = 9
start_time = x_test_original_extended.shape[2] * np.random.randint(int(test_set_full_duration_ms / x_test_original_extended.shape[2] - temporal_silence_ms))
end_time = start_time + 1 + x_test_original_extended.shape[2] * num_digits_to_display + temporal_silence_ms

# make sure we have something decent to show (randomise start and end times untill we do)
output_spikes_test_after_learning = output_spikes_test_after_learning_full_FF[start_time:end_time]
while output_spikes_test_after_learning.sum() < 2:
    start_time = x_test_original_extended.shape[2] * np.random.randint(int(test_set_full_duration_ms / x_test_original_extended.shape[2] - temporal_silence_ms))
    end_time = start_time + 1 + x_test_original_extended.shape[2] * num_digits_to_display + temporal_silence_ms
    output_spikes_test_after_learning = output_spikes_test_after_learning_full_FF[start_time:end_time]

min_time_ms = 0
max_time_ms = end_time - start_time

time_sec_mnist = np.arange(min_time_ms, max_time_ms) / 1000
min_time_sec_mnist = min_time_ms / 1000
max_time_sec_mnist = max_time_ms / 1000

# input digits
x_test_input_digits = x_test_axons_input_spikes[:,start_time:end_time]

syn_activation_time_1, syn_activation_index_1 = np.nonzero(axons_input_spikes_test[-x_test_axons_input_spikes.shape[0]:,start_time:end_time].T)
syn_activation_time_1 = syn_activation_time_1 / 1000
syn_activation_index_1 = x_test_axons_input_spikes.shape[0] - syn_activation_index_1

syn_activation_time_2 = syn_activation_time_1 + (time_delays_list_IF[0] / 1000)
syn_activation_time_3 = syn_activation_time_1 + (time_delays_list_IF[1] / 1000)
syn_activation_index_2 = syn_activation_index_1 + num_axons_FF
syn_activation_index_3 = syn_activation_index_2 + num_axons_FF


# output after learning
output_spikes_test_after_learning_FF = output_spikes_test_after_learning_full_FF[start_time:end_time]
output_spikes_test_after_learning_IF = output_spikes_test_after_learning_full_IF[start_time:end_time]

# desired output
output_kernel_test = np.zeros((X_test_spikes[0].shape[1],))
output_kernel_test[-1] = 1
desired_output_spikes_test_full = np.kron(Y_test_spikes[:num_test_patterns], output_kernel_test)
desired_output_spikes_test_mnist = desired_output_spikes_test_full[start_time:end_time]


# build figure
ax_digits.imshow(x_test_input_digits, cmap='gray');
ax_digits.set_xticks([])
ax_digits.set_yticks([])
ax_digits.spines['top'].set_visible(False)
ax_digits.spines['bottom'].set_visible(False)
ax_digits.spines['left'].set_visible(False)
ax_digits.spines['right'].set_visible(False)

ax_axons_mnist.scatter(syn_activation_time_1, syn_activation_index_1, s=8, c='black');
ax_axons_mnist.scatter(syn_activation_time_2, syn_activation_index_2, s=8, c='chocolate');
ax_axons_mnist.scatter(syn_activation_time_3, syn_activation_index_3, s=8, c='purple');
ax_axons_mnist.set_xlim(min_time_sec_mnist, max_time_sec_mnist);
ax_axons_mnist.set_xticks([])
ax_axons_mnist.set_yticks([])
ax_axons_mnist.spines['top'].set_visible(False)
ax_axons_mnist.spines['bottom'].set_visible(False)
ax_axons_mnist.spines['left'].set_visible(False)
ax_axons_mnist.spines['right'].set_visible(False)


ax_learning_outcomes.plot(time_sec_mnist, 2.2 + output_spikes_test_after_learning_FF, c=FF_color, lw=2.5);
ax_learning_outcomes.plot(time_sec_mnist, 1.1 + output_spikes_test_after_learning_IF, c=IF_color, lw=2.5);
ax_learning_outcomes.plot(time_sec_mnist, 0.0 + desired_output_spikes_test_mnist, c=target_color, lw=2.5);

ax_learning_outcomes.set_xlim(min_time_sec_mnist, max_time_sec_mnist);
ax_learning_outcomes.set_xticks([])
ax_learning_outcomes.set_yticks([])
ax_learning_outcomes.spines['top'].set_visible(False)
ax_learning_outcomes.spines['bottom'].set_visible(False)
ax_learning_outcomes.spines['left'].set_visible(False)
ax_learning_outcomes.spines['right'].set_visible(False)

ax_learning_outcomes.text(0.02,2.5, 'F&F (M=5)', color=FF_color, fontsize=20)
ax_learning_outcomes.text(0.02,1.4, 'I&F (Orig Axons + 2 delayed)', color=IF_color, fontsize=20)
ax_learning_outcomes.text(0.02,0.3, 'Desired Output', color=target_color, fontsize=20)


# save figure
if save_figures:
    figure_name = 'F&F_hardware_saving_Figure_5_%d' %(np.random.randint(200))
    for file_ending in all_file_endings_to_use:
        if file_ending == '.png':
            fig.savefig(figure_folder + figure_name + file_ending, bbox_inches='tight')
        else:
            fig.savefig(figure_folder + figure_name + file_ending, bbox_inches='tight')

