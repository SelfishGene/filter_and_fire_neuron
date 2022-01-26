import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import linear_model
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow import keras
import glob
import matplotlib
import matplotlib.gridspec as gridspec

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'

#%% script params

save_figures = True
save_figures = False
all_file_endings_to_use = ['.png', '.pdf', '.svg']

data_folder   = '/filter_and_fire_neuron/results_data_mnist/'
figure_folder = '/filter_and_fire_neuron/saved_figures/'

build_dataframe_from_scratch = False

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


#%% load MNIST dataset and show the data

(x_train_original, y_train), (x_test_original, y_test) = keras.datasets.mnist.load_data()

num_rows = 5
num_cols = 7

plt.close('all')
plt.figure(figsize=(20,15))
for k in range(num_rows * num_cols):
    rand_sample_ind = np.random.randint(x_train_original.shape[0])
    plt.subplot(num_rows, num_cols, k + 1);
    plt.imshow(x_train_original[k]); plt.title('digit "%s"' %(y_train[k]))

#%% Crop the data and binarize it

h_crop_range = [4,24]
w_crop_range = [4,24]

positive_threshold = 150

x_train_original = x_train_original[:,h_crop_range[0]:h_crop_range[1],w_crop_range[0]:w_crop_range[1]] > positive_threshold
x_test_original  = x_test_original[: ,h_crop_range[0]:h_crop_range[1],w_crop_range[0]:w_crop_range[1]] > positive_threshold

num_rows = 5
num_cols = 7

plt.close('all')
plt.figure(figsize=(20,15))
for k in range(num_rows * num_cols):
    rand_sample_ind = np.random.randint(x_train_original.shape[0])
    plt.subplot(num_rows, num_cols, k + 1);
    plt.imshow(x_train_original[k]); plt.title('digit "%s"' %(y_train[k]))

#%% Transform Xs to spatio-temporal spike trains

spatial_extent_factor = 5
temporal_extent_factor_numerator = 2
temporal_extent_factor_denumerator = 1

num_const_firing_channels = 20
temporal_silence_ms = 70

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

# display the patterns
num_rows = 5
num_cols = 7

plt.close('all')
plt.figure(figsize=(20,15))
for k in range(num_rows * num_cols):
    rand_sample_ind = np.random.randint(x_train.shape[0])
    plt.subplot(num_rows, num_cols, k + 1);
    plt.imshow(x_train[k], cmap='gray'); plt.title('digit "%s"' %(y_train[k]))

#%% Create "one-vs-all" dataset

positive_digit = 3
num_train_positive_patterns = 7000

release_probability = 1.0
apply_release_prob_during_train = False
apply_releash_prob_during_test = False

y_train_binary = y_train == positive_digit
y_test_binary = y_test == positive_digit

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

#%% Create a regularized logistic regression baseline

logistic_reg_model = linear_model.LogisticRegression(C=0.1, fit_intercept=False, penalty='l2',verbose=False)

# fit model
logistic_reg_model.fit(X_train_spikes.reshape([X_train_spikes.shape[0],-1]), Y_train_spikes)

# predict and calculate AUC on train data
Y_train_spikes_hat = logistic_reg_model.predict_proba(X_train_spikes.reshape([X_train_spikes.shape[0],-1]))[:,1]
Y_test_spikes_hat = logistic_reg_model.predict_proba(X_test_spikes.reshape([X_test_spikes.shape[0],-1]))[:,1]

train_AUC = roc_auc_score(Y_train_spikes, Y_train_spikes_hat)
test_AUC = roc_auc_score(Y_test_spikes, Y_test_spikes_hat)

print('for (# pos = %d, # neg = %d): (train AUC, test AUC) = (%.5f, %.5f)' %(num_train_positive_patterns, num_train_negative_patterns, train_AUC, test_AUC))

plt.close('all')
plt.figure(figsize=(8,8))
plt.imshow(logistic_reg_model.coef_.reshape([X_train_spikes.shape[1], X_train_spikes.shape[2]]));
plt.title('Learned Weights \n (spatio-temporal ("image") logistic regression)');

#%% Calculate and Display LogReg Accuracy

LL_false_positive_list, LL_true_positive_list, LL_thresholds_list = roc_curve(Y_test_spikes, Y_test_spikes_hat)

num_pos_class = int((Y_test_spikes == True).sum())
num_neg_class = int((Y_test_spikes == False).sum())

tp = LL_true_positive_list * num_pos_class
tn = (1 - LL_false_positive_list) * num_neg_class
LL_accuracy_list = (tp + tn) / (num_pos_class + num_neg_class)

LL_false_positive_list = LL_false_positive_list[LL_false_positive_list < 0.05]
LL_true_positive_list = LL_true_positive_list[:len(LL_false_positive_list)]
LL_thresholds_list = LL_thresholds_list[:len(LL_false_positive_list)]
LL_accuracy_list = LL_accuracy_list[:len(LL_false_positive_list)]

LL_false_positive_list = 100 * LL_false_positive_list
LL_true_positive_list = 100 * LL_true_positive_list
LL_accuracy_list = 100 * LL_accuracy_list

LL_accuracy_max = LL_accuracy_list.max()

LL_accuracy_subsampled = LL_accuracy_list[30::30]
LL_thresholds_subsampled = LL_thresholds_list[30::30]

acc_bar_x_axis = range(LL_accuracy_subsampled.shape[0])

plt.close('all')
plt.figure(figsize=(15,8));
plt.subplot(1,2,1); plt.bar(x=acc_bar_x_axis,height=LL_accuracy_subsampled);
plt.xticks(acc_bar_x_axis, LL_thresholds_subsampled, rotation='vertical');
plt.title('max accuracy = %.2f%s' %(LL_accuracy_max,'%'), fontsize=24)
plt.ylim(87.8,100); plt.xlabel('threshold', fontsize=20); plt.ylabel('Accuracy (%)', fontsize=20);
plt.plot([acc_bar_x_axis[0]-1, acc_bar_x_axis[-1]+1], [zero_pred_baseline_accuracy, zero_pred_baseline_accuracy], color='r')
plt.subplot(1,2,2); plt.plot(LL_false_positive_list, LL_true_positive_list);
plt.ylabel('True Positive (%)', fontsize=20); plt.xlabel('False Positive (%)', fontsize=20);


#%% Fit a F&F model

# main parameters
connections_per_axon = 5
model_type = 'F&F'
#model_type = 'I&F'

# neuron model parameters
num_axons = X_train_spikes[0].shape[0]
num_synapses = connections_per_axon * num_axons

v_reset     = -80
v_threshold = -55
current_to_voltage_mult_factor = 3
refreactory_time_constant = 15

# synapse non-learnable parameters
if model_type == 'F&F':
    tau_rise_range  = [1,18]
    tau_decay_range = [8,27]
elif model_type == 'I&F':
    tau_rise_range  = [1,1]
    tau_decay_range = [27,27]

tau_rise_vec  = np.random.uniform(low=tau_rise_range[0] , high=tau_rise_range[1] , size=(num_synapses, 1))
tau_decay_vec = np.random.uniform(low=tau_decay_range[0], high=tau_decay_range[1], size=(num_synapses, 1))

# synapse learnable parameters
synaptic_weights_vec = np.random.normal(size=(num_synapses, 1))

# prepare input spikes
axons_input_spikes = np.concatenate([X_train_spikes[k] for k in range(X_train_spikes.shape[0])],axis=1)

# prepare output spikes
pattern_duration_ms = X_train_spikes[0].shape[1]
output_spike_offset = 1
output_kernel = np.zeros((pattern_duration_ms,))
output_kernel[-output_spike_offset] = 1

desired_output_spikes = np.kron(Y_train_spikes, output_kernel)

plt.close('all')
plt.figure(figsize=(30,15));
plt.imshow(axons_input_spikes[:,:1101], cmap='gray')
plt.title('input axons raster', fontsize=22)
plt.ylabel('axon index', fontsize=22);
plt.xlabel('time [ms]', fontsize=22);

plt.figure(figsize=(30,1));
plt.plot(desired_output_spikes[:1101]); plt.xlim(0,1101)
plt.ylabel('output spike', fontsize=22)
plt.xlabel('time [ms]', fontsize=22);

#%% simulate cell with normlized currents

presynaptic_input_spikes = np.kron(np.ones((connections_per_axon,1), dtype=bool), axons_input_spikes).astype(bool)

local_normlized_currents, soma_voltage, output_spike_times_in_ms = simulate_filter_and_fire_cell_training_long(presynaptic_input_spikes,
                                                                                                               synaptic_weights_vec, tau_rise_vec, tau_decay_vec,
                                                                                                               refreactory_time_constant=refreactory_time_constant,
                                                                                                               v_reset=v_reset, v_threshold=v_threshold,
                                                                                                               current_to_voltage_mult_factor=current_to_voltage_mult_factor)

#%% fit linear model to local currents

filter_and_fire_model = linear_model.LogisticRegression(C=10000, fit_intercept=False, penalty='l2')

spike_safety_range_ms = 20
negative_subsampling_fraction = 0.5

X, y = prepare_training_dataset(local_normlized_currents, desired_output_spikes,
                                spike_safety_range_ms=spike_safety_range_ms,
                                negative_subsampling_fraction=negative_subsampling_fraction)

# fit model
filter_and_fire_model.fit(X, y)

print('number of data points = %d (%.2f%s positive class)' %(X.shape[0], 100 * y.mean(),'%'))

# calculate train AUC
y_hat = filter_and_fire_model.predict_proba(X)[:,1]
train_AUC = roc_auc_score(y, y_hat)

# display some training data predictions
num_timepoints_to_show = 10000
fitted_output_spike_prob = filter_and_fire_model.predict_proba(local_normlized_currents[:,:num_timepoints_to_show].T)[:,1]

plt.close('all')
plt.figure(figsize=(30,10))
plt.plot(1.05 * desired_output_spikes[:num_timepoints_to_show] - 0.025); plt.title('train AUC = %.5f' %(train_AUC), fontsize=22)
plt.plot(fitted_output_spike_prob[:num_timepoints_to_show]); plt.xlabel('time [ms]'); plt.legend(['GT', 'prediction'], fontsize=22);

#%% display learned weights

normlized_syn_filter = construct_normlized_synaptic_filter(tau_rise_vec, tau_decay_vec)

# collect learned synaptic weights
FF_learned_synaptic_weights = np.fliplr(filter_and_fire_model.coef_).T
weighted_syn_filter = FF_learned_synaptic_weights * normlized_syn_filter

axon_spatio_temporal_pattern = np.zeros((num_axons, weighted_syn_filter.shape[1]))
for k in range(num_axons):
    axon_spatio_temporal_pattern[k] = weighted_syn_filter[k::num_axons].sum(axis=0)

axon_spatio_temporal_pattern_short = axon_spatio_temporal_pattern[:,:X_train_spikes.shape[2]]

plt.close('all')
plt.figure(figsize=(18,8))
plt.subplot(1,2,1); plt.imshow(logistic_reg_model.coef_.reshape([X_train_spikes.shape[1], X_train_spikes.shape[2]])); plt.title('logistic regression', fontsize=20)
plt.subplot(1,2,2); plt.imshow(np.flip(axon_spatio_temporal_pattern_short)); plt.title('filter and fire neuron', fontsize=20);

#%% Make a prediction on the entire test trace

num_test_patterns = X_test_spikes.shape[0]

# prepare test outputs
output_spike_tolorance_window_duration = 20
output_spike_tolorance_window_offset   = 5
output_kernel_test = np.zeros((X_test_spikes[0].shape[1],))
output_kernel_test[-output_spike_tolorance_window_duration:] = 1

desired_output_spikes_test = np.kron(Y_test_spikes[:num_test_patterns], output_kernel_test)
desired_output_spikes_test = np.concatenate((np.zeros((output_spike_tolorance_window_offset,)), desired_output_spikes_test[:-output_spike_tolorance_window_offset]))

# prepare test inputs
axons_input_spikes_test = np.concatenate([X_test_spikes[k] for k in range(num_test_patterns)],axis=1)
presynaptic_input_spikes_test = np.kron(np.ones((connections_per_axon,1), dtype=bool), axons_input_spikes_test).astype(bool)

# add synaptic unrelability ("release probability" that is not 100%)
if apply_releash_prob_during_test:
    presynaptic_input_spikes_test = presynaptic_input_spikes_test * (np.random.rand(presynaptic_input_spikes_test.shape[0], presynaptic_input_spikes_test.shape[1]) < release_probability)

FF_weight_mult_factors_list = [x for x in [1,2,3,4,5,6,9,12,20,50,120,250]]
FF_accuracy_list = []
FF_true_positive_list = []
FF_false_positive_list = []
for weight_mult_factor in FF_weight_mult_factors_list:

    # collect learned synaptic weights
    synaptic_weights_post_learning = weight_mult_factor * FF_learned_synaptic_weights

    soma_voltage_test, output_spike_times_in_ms_test = simulate_filter_and_fire_cell_inference_long(presynaptic_input_spikes_test,
                                                                                                    synaptic_weights_post_learning, tau_rise_vec, tau_decay_vec,
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

#%% "after learning" Build the nice looking figure of before and after learning

# get the max accuracy weight matrix
max_accuracy_weight_mult_factor = FF_weight_mult_factors_list[np.argsort(np.array(FF_accuracy_list))[-1]]

synaptic_weights_vec_after_learning = max_accuracy_weight_mult_factor * FF_learned_synaptic_weights

# simulate the max accuracy output after learning
soma_voltage_test, output_spike_times_in_ms_test = simulate_filter_and_fire_cell_inference_long(presynaptic_input_spikes_test,
                                                                                                synaptic_weights_vec_after_learning, tau_rise_vec, tau_decay_vec,
                                                                                                refreactory_time_constant=refreactory_time_constant,
                                                                                                v_reset=v_reset, v_threshold=v_threshold,
                                                                                                current_to_voltage_mult_factor=current_to_voltage_mult_factor)


output_spikes_test_after_learning_full = np.zeros(soma_voltage_test.shape)
try:
    output_spikes_test_after_learning_full[np.array(output_spike_times_in_ms_test)] = 1.0
except:
    print('no output spikes created')

#%% "before learning" Simulate response to test set before learning

synaptic_weights_vec_before_learning = 0.01 + 0.3 * np.random.normal(size=(num_synapses, 1))

# simulate response to test set before learning (randomly permuted learned weights vector)
soma_voltage_test, output_spike_times_in_ms_test = simulate_filter_and_fire_cell_inference_long(presynaptic_input_spikes_test,
                                                                                                synaptic_weights_vec_before_learning, tau_rise_vec, tau_decay_vec,
                                                                                                refreactory_time_constant=refreactory_time_constant,
                                                                                                v_reset=v_reset, v_threshold=v_threshold,
                                                                                                current_to_voltage_mult_factor=current_to_voltage_mult_factor)


output_spikes_test_before_learning_full = np.zeros(soma_voltage_test.shape)
try:
    output_spikes_test_before_learning_full[np.array(output_spike_times_in_ms_test)] = 1.0
except:
    print('no output spikes created')


#%% organize everything into a figure

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

num_spikes_in_window = 0
while num_spikes_in_window != 3:
    start_time = x_test_original_extended.shape[2] * np.random.randint(int(test_set_full_duration_ms / x_test_original_extended.shape[2] - temporal_silence_ms))
    end_time = start_time + 1 + x_test_original_extended.shape[2] * num_digits_to_display + temporal_silence_ms

    output_spike_tolorance_window_duration = 20
    output_spike_tolorance_window_offset   = 5

    output_kernel_test = np.zeros((X_test_spikes[0].shape[1],))
    output_kernel_test[-1] = 1
    desired_output_spikes_test_full = np.kron(Y_test_spikes[:num_test_patterns], output_kernel_test)
    desired_output_spikes_test = desired_output_spikes_test_full[start_time:end_time]

    num_spikes_in_window = desired_output_spikes_test.sum()

# make sure we have something decent to show (randomise start and end times untill we do)
output_spikes_test_after_learning = output_spikes_test_after_learning_full[start_time:end_time]
while output_spikes_test_after_learning.sum() < 2:
    start_time = x_test_original_extended.shape[2] * np.random.randint(int(test_set_full_duration_ms / x_test_original_extended.shape[2] - temporal_silence_ms))
    end_time = start_time + 1 + x_test_original_extended.shape[2] * num_digits_to_display + temporal_silence_ms
    output_spikes_test_after_learning = output_spikes_test_after_learning_full[start_time:end_time]

min_time_ms = 0
max_time_ms = end_time - start_time

time_sec = np.arange(min_time_ms, max_time_ms) / 1000
min_time_sec = min_time_ms / 1000
max_time_sec = max_time_ms / 1000


before_color = '0.15'
after_color = 'blue'
target_color = 'red'

# input digits
x_test_input_digits = x_test_axons_input_spikes[:,start_time:end_time]

# axon input raster
syn_activation_time, syn_activation_index = np.nonzero(axons_input_spikes_test[-x_test_axons_input_spikes.shape[0]:,start_time:end_time].T)
syn_activation_time = syn_activation_time / 1000
syn_activation_index = x_test_axons_input_spikes.shape[0] - syn_activation_index

# output before learning
output_spikes_test_before_learning = output_spikes_test_before_learning_full[start_time:end_time]

# output after learning
output_spikes_test_after_learning = output_spikes_test_after_learning_full[start_time:end_time]

# desired output
output_spike_tolorance_window_duration = 20
output_spike_tolorance_window_offset   = 5
output_kernel_test = np.zeros((X_test_spikes[0].shape[1],))
output_kernel_test[-1] = 1
desired_output_spikes_test_full = np.kron(Y_test_spikes[:num_test_patterns], output_kernel_test)
desired_output_spikes_test = desired_output_spikes_test_full[start_time:end_time]


# build full figure
plt.close('all')
fig = plt.figure(figsize=(19,22))
gs_figure = gridspec.GridSpec(nrows=4,ncols=3)
gs_figure.update(left=0.03, right=0.97, bottom=0.57, top=0.985, wspace=0.1, hspace=0.11)

ax_digits            = plt.subplot(gs_figure[0,:])
ax_axons             = plt.subplot(gs_figure[1:3,:])
ax_learning_outcomes = plt.subplot(gs_figure[3,:])

ax_digits.imshow(x_test_input_digits, cmap='gray'); ax_digits.set_title('Input Digits', fontsize=18)
ax_digits.set_xticks([])
ax_digits.set_yticks([])
ax_digits.spines['top'].set_visible(False)
ax_digits.spines['bottom'].set_visible(False)
ax_digits.spines['left'].set_visible(False)
ax_digits.spines['right'].set_visible(False)

ax_axons.scatter(syn_activation_time, syn_activation_index, s=6, c='k');
ax_axons.set_ylabel('Input Axons Raster', fontsize=18)
ax_axons.set_xlim(min_time_sec, max_time_sec);
ax_axons.set_xticks([])
ax_axons.set_yticks([])
ax_axons.spines['top'].set_visible(False)
ax_axons.spines['bottom'].set_visible(False)
ax_axons.spines['left'].set_visible(False)
ax_axons.spines['right'].set_visible(False)


ax_learning_outcomes.plot(time_sec, 2.2 + output_spikes_test_before_learning, c=before_color, lw=2.5);
ax_learning_outcomes.plot(time_sec, 1.1 + output_spikes_test_after_learning, c=after_color, lw=2.5);
ax_learning_outcomes.plot(time_sec, 0.0 + desired_output_spikes_test, c=target_color, lw=2.5);

ax_learning_outcomes.set_xlim(min_time_sec, max_time_sec);
ax_learning_outcomes.set_xticks([])
ax_learning_outcomes.set_yticks([])
ax_learning_outcomes.spines['top'].set_visible(False)
ax_learning_outcomes.spines['bottom'].set_visible(False)
ax_learning_outcomes.spines['left'].set_visible(False)
ax_learning_outcomes.spines['right'].set_visible(False)

ax_learning_outcomes.text(0.025,2.5, 'Before Learning', color=before_color, fontsize=20)
ax_learning_outcomes.text(0.025,1.4, 'After Learning', color=after_color, fontsize=20)
ax_learning_outcomes.text(0.025,0.3, 'Desired Output', color=target_color, fontsize=20)

# load data into dataframe

list_of_files = glob.glob(data_folder + 'MNIST_*.pickle')
print(len(list_of_files))

if build_dataframe_from_scratch:

    # Load one saved pickle
    filename_to_load = list_of_files[np.random.randint(len(list_of_files))]
    loaded_script_results_dict = pickle.load(open(filename_to_load, "rb" ))

    # display basic fields in the saved pickle file
    print('-----------------------------------------------------------------------------------------------------------')
    print('loaded_script_results_dict.keys():')
    print('----------')
    print(list(loaded_script_results_dict.keys()))
    print('-----------------------------------------------------------------------------------------------------------')
    print('loaded_script_results_dict["script_main_params"].keys():')
    print('----------')
    print(list(loaded_script_results_dict["script_main_params"].keys()))
    print('-----------------------------------------------------------------------------------------------------------')
    print('positive_digit =', loaded_script_results_dict['script_main_params']['positive_digit'])
    print('connections_per_axon =', loaded_script_results_dict['script_main_params']['connections_per_axon'])
    print('digit_sample_image_shape_cropped =', loaded_script_results_dict['script_main_params']['digit_sample_image_shape_cropped'])
    print('digit_sample_image_shape_expanded =', loaded_script_results_dict['script_main_params']['digit_sample_image_shape_expanded'])
    print('num_train_positive_patterns =', loaded_script_results_dict['script_main_params']['num_train_positive_patterns'])
    print('temporal_silence_ms =', loaded_script_results_dict['script_main_params']['temporal_silence_ms'])
    print('-----------------------------------------------------------------------------------------------------------')
    print('model_accuracy_LR =', loaded_script_results_dict['model_accuracy_LR'])
    print('model_accuracy_FF =', loaded_script_results_dict['model_accuracy_FF'])
    print('model_accuracy_IF =', loaded_script_results_dict['model_accuracy_IF'])
    print('model_accuracy_baseline =', loaded_script_results_dict['model_accuracy_baseline'])
    print('-----------------------------------------------------------------------------------------------------------')

    try:
        print('-----------------------------------------------------------------------------------------------------------')
        print('positive_digit =', loaded_script_results_dict['script_main_params']['positive_digit'])
        print('num_train_positive_patterns =', loaded_script_results_dict['script_main_params']['num_train_positive_patterns'])
        print('release_probability =', loaded_script_results_dict['script_main_params']['release_probability'])
        print('train_epochs =', loaded_script_results_dict['script_main_params']['train_epochs'])
        print('test_epochs =', loaded_script_results_dict['script_main_params']['test_epochs'])
        print('create_output_burst =', loaded_script_results_dict['script_main_params']['create_output_burst'])
        print('-----------------------------------------------------------------------------------------------------------')
    except:
        print('no prob release fields')

    # display the learned weights
    # plt.close('all')
    # plt.figure(figsize=(24,10))
    # plt.subplot(1,3,1); plt.imshow(loaded_script_results_dict['learned_weights_LR']); plt.title('logistic regression', fontsize=24)
    # plt.subplot(1,3,2); plt.imshow(loaded_script_results_dict['learned_weights_FF']); plt.title('filter and fire neuron', fontsize=24)
    # plt.subplot(1,3,3); plt.imshow(loaded_script_results_dict['learned_weights_IF']); plt.title('integrate and fire neuron', fontsize=24)

    # go over all files and insert into a large dataframe
    columns_to_use = ['digit','M_connections','N_axons', 'T', 'N_positive_samples',
                      'Accuracy LR', 'Accuracy FF', 'Accuracy IF', 'Accuracy baseline',
                      'release probability', 'train_epochs', 'test_epochs']
    results_df = pd.DataFrame(index=range(len(list_of_files)), columns=columns_to_use)

    for k, filename_to_load in enumerate(list_of_files):
        loaded_script_results_dict = pickle.load(open(filename_to_load, "rb" ))

        results_df.loc[k, 'digit']              = loaded_script_results_dict['script_main_params']['positive_digit']
        results_df.loc[k, 'M_connections']      = loaded_script_results_dict['script_main_params']['connections_per_axon']
        results_df.loc[k, 'N_axons']            = loaded_script_results_dict['script_main_params']['digit_sample_image_shape_expanded'][0]
        results_df.loc[k, 'T']                  = loaded_script_results_dict['script_main_params']['digit_sample_image_shape_expanded'][1]
        results_df.loc[k, 'N_positive_samples'] = loaded_script_results_dict['script_main_params']['num_train_positive_patterns']
        results_df.loc[k, 'Accuracy LR']        = loaded_script_results_dict['model_accuracy_LR']
        results_df.loc[k, 'Accuracy FF']        = loaded_script_results_dict['model_accuracy_FF']
        results_df.loc[k, 'Accuracy IF']        = loaded_script_results_dict['model_accuracy_IF']
        results_df.loc[k, 'Accuracy baseline']  = loaded_script_results_dict['model_accuracy_baseline']

        try:
            results_df.loc[k, 'release probability'] = loaded_script_results_dict['script_main_params']['release_probability']
            results_df.loc[k, 'train_epochs']        = loaded_script_results_dict['script_main_params']['train_epochs']
            results_df.loc[k, 'test_epochs']         = loaded_script_results_dict['script_main_params']['test_epochs']
        except:
            results_df.loc[k, 'release probability'] = 1.0
            results_df.loc[k, 'train_epochs']        = 1
            results_df.loc[k, 'test_epochs']         = 1

    print(results_df.shape)

    # save the dataframe
    filename = 'MNIST_classification_LR_FF_IF_%d_rows_%d_cols.csv' %(results_df.shape[0], results_df.shape[1])
    results_df.to_csv(data_folder + filename, index=False)
else:
    filename = 'MNIST_classification_LR_FF_IF_5162_rows_12_cols.csv'

# open previously saved file
results_df = pd.read_csv(data_folder + filename)
print(results_df.shape)

# display accuracy per digit at specific condition for (I&F, F&F, LR)
selected_T = 40
selected_M = 5
selected_N_axons = 100
selected_N_samples = 4000
selected_release_prob = 1.0

condition_rows = results_df.loc[:, 'T'] == selected_T
condition_rows = np.logical_and(condition_rows, results_df.loc[:, 'M_connections']       == selected_M)
condition_rows = np.logical_and(condition_rows, results_df.loc[:, 'N_axons']             == selected_N_axons)
condition_rows = np.logical_and(condition_rows, results_df.loc[:, 'N_positive_samples']  >= selected_N_samples)
condition_rows = np.logical_and(condition_rows, results_df.loc[:, 'release probability'] == selected_release_prob)

average_for_condition_per_digit_df = results_df.loc[condition_rows,:].groupby('digit').mean()
stddev_for_condition_per_digit_df  = results_df.loc[condition_rows,:].groupby('digit').std()
counts_for_condition_per_digit_df  = results_df.loc[condition_rows,:].groupby('digit').count().iloc[:,0]

digits_list = average_for_condition_per_digit_df.index.tolist()
LR_accuracy       = average_for_condition_per_digit_df['Accuracy LR'].tolist()
FF_accuracy       = average_for_condition_per_digit_df['Accuracy FF'].tolist()
IF_accuracy       = average_for_condition_per_digit_df['Accuracy IF'].tolist()
baseline_accuracy = average_for_condition_per_digit_df['Accuracy baseline'].tolist()

LR_stderr = stddev_for_condition_per_digit_df['Accuracy LR'].tolist()
FF_stderr = stddev_for_condition_per_digit_df['Accuracy FF'].tolist()
IF_stderr = stddev_for_condition_per_digit_df['Accuracy IF'].tolist()

# display plot
bar_plot_x_axis = 1.0 * np.arange(len(digits_list))
bar_widths = 0.7 / 3
x_tick_names = ['"%s"' %(str(x)) for x in digits_list]

gs_accuracy_per_digit = gridspec.GridSpec(nrows=2,ncols=1)
gs_accuracy_per_digit.update(left=0.045, right=0.675, bottom=0.32, top=0.52, wspace=0.1, hspace=0.1)
ax_accuracy_per_digit = plt.subplot(gs_accuracy_per_digit[:,:])

ax_accuracy_per_digit.bar(bar_plot_x_axis + 0 * bar_widths, IF_accuracy, bar_widths, yerr=IF_stderr, color='0.05'  , label='I&F')
ax_accuracy_per_digit.bar(bar_plot_x_axis + 1 * bar_widths, FF_accuracy, bar_widths, yerr=FF_stderr, color='orange', alpha=0.95, label='F&F')
ax_accuracy_per_digit.bar(bar_plot_x_axis + 2 * bar_widths, LR_accuracy, bar_widths, yerr=LR_stderr, color='0.45'  , label='Spatio\nTemporal LR')

for k in range(len(digits_list)):
    if k == 0:
        ax_accuracy_per_digit.plot([bar_plot_x_axis[k] - 0.7 * bar_widths, bar_plot_x_axis[k] + 2.7 * bar_widths], [baseline_accuracy[k], baseline_accuracy[k]], color='r', label='Baseline')
    else:
        ax_accuracy_per_digit.plot([bar_plot_x_axis[k] - 0.7 * bar_widths, bar_plot_x_axis[k] + 2.7 * bar_widths], [baseline_accuracy[k], baseline_accuracy[k]], color='r')

ax_accuracy_per_digit.set_title('Accuracy Comparison per digit (M = %d, T = %d (ms))' %(selected_M, selected_T), fontsize=22)
ax_accuracy_per_digit.set_yticks([90,92,94,96,98]);
ax_accuracy_per_digit.set_yticklabels([90,92,94,96,98], fontsize=16);
ax_accuracy_per_digit.set_ylim(87.9,98.9);
ax_accuracy_per_digit.set_xticks(bar_plot_x_axis + bar_widths);
ax_accuracy_per_digit.set_xticklabels(x_tick_names, rotation=0, fontsize=26);
ax_accuracy_per_digit.set_ylabel('Test accuracy (%)', fontsize=18)
ax_accuracy_per_digit.legend(fontsize=18, ncol=1);
ax_accuracy_per_digit.set_xlim(-0.5,10);
ax_accuracy_per_digit.spines['top'].set_visible(False)
ax_accuracy_per_digit.spines['right'].set_visible(False)

# display accuracy across all digits as function of pattern presentation duration for (I&F, F&F, LR)
selected_M = 5
selected_N_axons = 100
selected_N_samples = 4000
selected_release_prob = 1.0

condition_rows = results_df.loc[:, 'M_connections'] == selected_M
condition_rows = np.logical_and(condition_rows, results_df.loc[:, 'N_axons']             == selected_N_axons)
condition_rows = np.logical_and(condition_rows, results_df.loc[:, 'N_positive_samples']  >= selected_N_samples)
condition_rows = np.logical_and(condition_rows, results_df.loc[:, 'release probability'] == selected_release_prob)

all_T_values = sorted(results_df['T'].unique().tolist())
digits_list  = sorted(results_df['digit'].unique().tolist())
num_digits   = len(digits_list)

LR_acc_mean = []
FF_acc_mean = []
IF_acc_mean = []
baseline_acc_mean = []

LR_acc_std = []
FF_acc_std = []
IF_acc_std = []
baseline_acc_std = []

for selected_T in all_T_values:
    curr_rows = np.logical_and(condition_rows, results_df.loc[:, 'T'] == selected_T)

    average_acc_per_T_per_digit_df = results_df.loc[curr_rows,:].groupby('digit').mean()
    stddev_acc_per_T_per_digit_df  = results_df.loc[curr_rows,:].groupby('digit').std()
    # print(selected_T, ':\n', results_df.loc[curr_rows,:].groupby('digit').count().iloc[:,0])

    LR_acc_mean.append(average_acc_per_T_per_digit_df['Accuracy LR'].mean())
    FF_acc_mean.append(average_acc_per_T_per_digit_df['Accuracy FF'].mean())
    IF_acc_mean.append(average_acc_per_T_per_digit_df['Accuracy IF'].mean())
    baseline_acc_mean.append(average_acc_per_T_per_digit_df['Accuracy baseline'].mean())

    LR_acc_std.append(average_acc_per_T_per_digit_df['Accuracy LR'].std() / np.sqrt(num_digits))
    FF_acc_std.append(average_acc_per_T_per_digit_df['Accuracy FF'].std() / np.sqrt(num_digits))
    IF_acc_std.append(average_acc_per_T_per_digit_df['Accuracy IF'].std() / np.sqrt(num_digits))
    baseline_acc_std.append(average_acc_per_T_per_digit_df['Accuracy baseline'].std() / np.sqrt(num_digits))

gs_accuracy_vs_T = gridspec.GridSpec(nrows=2,ncols=1)
gs_accuracy_vs_T.update(left=0.045, right=0.3375, bottom=0.04, top=0.28, wspace=0.1, hspace=0.1)
ax_accuracy_vs_T = plt.subplot(gs_accuracy_vs_T[:,:])

ax_accuracy_vs_T.errorbar(all_T_values, LR_acc_mean, yerr=LR_acc_std, linewidth=4, color='0.45')
ax_accuracy_vs_T.errorbar(all_T_values, FF_acc_mean, yerr=FF_acc_std, linewidth=4, color='orange')
ax_accuracy_vs_T.errorbar(all_T_values, IF_acc_mean, yerr=IF_acc_std, linewidth=4, color='0.05')
ax_accuracy_vs_T.errorbar(all_T_values, baseline_acc_mean, yerr=baseline_acc_std, linewidth=1, color='red')
ax_accuracy_vs_T.set_xlabel('Pattern presentation duration - T (ms)', fontsize=18)
ax_accuracy_vs_T.set_ylabel('Test Accuracy (%)', fontsize=18)
ax_accuracy_vs_T.set_ylim(89.5,96.5)
ax_accuracy_vs_T.set_xticks(all_T_values)
ax_accuracy_vs_T.set_xticklabels(all_T_values, fontsize=15)
ax_accuracy_vs_T.set_yticks([90,92,94,96]);
ax_accuracy_vs_T.set_yticklabels([90,92,94,96], fontsize=15);
ax_accuracy_vs_T.spines['top'].set_visible(False)
ax_accuracy_vs_T.spines['right'].set_visible(False)


# display accuracy as function of M, for a specific digit and multiple release probabilities
selected_digit = 2
selected_N_axons = 100
selected_N_samples = 2048

condition_rows = results_df.loc[:, 'digit'] == selected_digit
condition_rows = np.logical_and(condition_rows, results_df.loc[:, 'N_axons']             == selected_N_axons)
condition_rows = np.logical_and(condition_rows, results_df.loc[:, 'N_positive_samples']  == selected_N_samples)

all_M_values = sorted(results_df.loc[condition_rows, 'M_connections'].unique().tolist())
all_P_values = sorted(results_df.loc[condition_rows, 'release probability'].unique().tolist())

all_M_values = [1,2,3,5,8]

results_dict_Acc_vs_M = {}
for selected_release_P in all_P_values:
    results_dict_Acc_vs_M[selected_release_P] = {}
    results_dict_Acc_vs_M[selected_release_P]['LR_acc_mean'] = []
    results_dict_Acc_vs_M[selected_release_P]['FF_acc_mean'] = []
    results_dict_Acc_vs_M[selected_release_P]['IF_acc_mean'] = []
    results_dict_Acc_vs_M[selected_release_P]['baseline_acc_mean'] = []

    results_dict_Acc_vs_M[selected_release_P]['LR_acc_std'] = []
    results_dict_Acc_vs_M[selected_release_P]['FF_acc_std'] = []
    results_dict_Acc_vs_M[selected_release_P]['IF_acc_std'] = []
    results_dict_Acc_vs_M[selected_release_P]['baseline_acc_std'] = []


for selected_M in all_M_values:
    curr_condition_rows = np.logical_and(condition_rows, results_df.loc[:, 'M_connections'] == selected_M)

    for selected_release_P in all_P_values:
        curr_rows = np.logical_and(curr_condition_rows, results_df.loc[:, 'release probability'] == selected_release_P)

        num_rows = curr_rows.sum()
        num_rows = 1
        results_dict_Acc_vs_M[selected_release_P]['LR_acc_mean'].append(results_df.loc[curr_rows,'Accuracy LR'].mean())
        results_dict_Acc_vs_M[selected_release_P]['FF_acc_mean'].append(results_df.loc[curr_rows,'Accuracy FF'].mean())
        results_dict_Acc_vs_M[selected_release_P]['IF_acc_mean'].append(results_df.loc[curr_rows,'Accuracy IF'].mean())
        results_dict_Acc_vs_M[selected_release_P]['baseline_acc_mean'].append(results_df.loc[curr_rows,'Accuracy baseline'].mean())

        results_dict_Acc_vs_M[selected_release_P]['LR_acc_std'].append(results_df.loc[curr_rows,'Accuracy LR'].std() / np.sqrt(num_rows))
        results_dict_Acc_vs_M[selected_release_P]['FF_acc_std'].append(results_df.loc[curr_rows,'Accuracy FF'].std() / np.sqrt(num_rows))
        results_dict_Acc_vs_M[selected_release_P]['IF_acc_std'].append(results_df.loc[curr_rows,'Accuracy IF'].std() / np.sqrt(num_rows))
        results_dict_Acc_vs_M[selected_release_P]['baseline_acc_std'].append(results_df.loc[curr_rows,'Accuracy baseline'].std() / np.sqrt(num_rows))

gs_accuracy_vs_M = gridspec.GridSpec(nrows=2,ncols=1)
gs_accuracy_vs_M.update(left=0.37, right=0.66, bottom=0.04, top=0.28, wspace=0.1, hspace=0.1)
ax_accuracy_vs_M = plt.subplot(gs_accuracy_vs_M[:,:])

selected_release_P = 1.0
ax_accuracy_vs_M.errorbar(all_M_values, results_dict_Acc_vs_M[selected_release_P]['LR_acc_mean'], yerr=results_dict_Acc_vs_M[selected_release_P]['LR_acc_std'], lw=4, color='0.45')
ax_accuracy_vs_M.errorbar(all_M_values, results_dict_Acc_vs_M[selected_release_P]['FF_acc_mean'], yerr=results_dict_Acc_vs_M[selected_release_P]['FF_acc_std'], lw=4, color='orange')
ax_accuracy_vs_M.errorbar(all_M_values, results_dict_Acc_vs_M[selected_release_P]['IF_acc_mean'], yerr=results_dict_Acc_vs_M[selected_release_P]['IF_acc_std'], lw=4, color='0.05')

selected_release_P = 0.5
ax_accuracy_vs_M.errorbar(all_M_values, results_dict_Acc_vs_M[selected_release_P]['LR_acc_mean'], yerr=results_dict_Acc_vs_M[selected_release_P]['LR_acc_std'], lw=4, ls=':', color='0.45')
ax_accuracy_vs_M.errorbar(all_M_values, results_dict_Acc_vs_M[selected_release_P]['FF_acc_mean'], yerr=results_dict_Acc_vs_M[selected_release_P]['FF_acc_std'], lw=4, ls=':', color='orange')
ax_accuracy_vs_M.errorbar(all_M_values, results_dict_Acc_vs_M[selected_release_P]['IF_acc_mean'], yerr=results_dict_Acc_vs_M[selected_release_P]['IF_acc_std'], lw=4, ls=':', color='0.05')

legend_list = ['release Prob = 1.0', 'release Prob = 1.0', 'release Prob = 1.0',
               'P = 0.5', 'P = 0.5', 'P = 0.5']
ax_accuracy_vs_M.legend(legend_list, ncol=2, mode='expand', fontsize=18)

ax_accuracy_vs_M.set_xlabel('Number of Multiple Contacts - M', fontsize=18)
ax_accuracy_vs_M.set_xticks(all_M_values)
ax_accuracy_vs_M.set_xticklabels(all_M_values, fontsize=15)
ax_accuracy_vs_M.set_yticks([91,93,95,97]);
ax_accuracy_vs_M.set_yticklabels([91,93,95,97], fontsize=15);
ax_accuracy_vs_M.set_ylim(90.5,98.2)
ax_accuracy_vs_M.spines['top'].set_visible(False)
ax_accuracy_vs_M.spines['right'].set_visible(False)
ax_accuracy_vs_M.set_xlim(0.8,8.2)


# display accuracy as function of number of positive training samples
selected_digit = 7
selected_T = 30
selected_M = 5
selected_N_axons = 100
max_N_samples = 1100
num_train_epochs = 15

condition_rows = results_df.loc[:, 'digit'] == selected_digit
condition_rows = np.logical_and(condition_rows, results_df.loc[:, 'T']                  == selected_T)
condition_rows = np.logical_and(condition_rows, results_df.loc[:, 'M_connections']      == selected_M)
condition_rows = np.logical_and(condition_rows, results_df.loc[:, 'N_axons']            == selected_N_axons)
condition_rows = np.logical_and(condition_rows, results_df.loc[:, 'N_positive_samples'] <= max_N_samples)

good_epoch_rows = np.logical_or(results_df.loc[:, 'release probability'] == 1.0, results_df.loc[:, 'train_epochs'] == num_train_epochs)
condition_rows  = np.logical_and(condition_rows, good_epoch_rows)

all_N_samples_values = sorted(results_df.loc[condition_rows, 'N_positive_samples'].unique().tolist())
all_P_values         = sorted(results_df.loc[condition_rows, 'release probability'].unique().tolist())

all_N_samples_values = [16, 32, 64, 128, 256, 512, 1024]

results_dict_Acc_vs_N_samples = {}
for selected_release_P in all_P_values:
    results_dict_Acc_vs_N_samples[selected_release_P] = {}
    results_dict_Acc_vs_N_samples[selected_release_P]['LR_acc_mean'] = []
    results_dict_Acc_vs_N_samples[selected_release_P]['FF_acc_mean'] = []
    results_dict_Acc_vs_N_samples[selected_release_P]['IF_acc_mean'] = []
    results_dict_Acc_vs_N_samples[selected_release_P]['baseline_acc_mean'] = []

    results_dict_Acc_vs_N_samples[selected_release_P]['LR_acc_std'] = []
    results_dict_Acc_vs_N_samples[selected_release_P]['FF_acc_std'] = []
    results_dict_Acc_vs_N_samples[selected_release_P]['IF_acc_std'] = []
    results_dict_Acc_vs_N_samples[selected_release_P]['baseline_acc_std'] = []


for selected_N_samples in all_N_samples_values:
    curr_condition_rows = np.logical_and(condition_rows, results_df.loc[:, 'N_positive_samples'] == selected_N_samples)

    for selected_release_P in all_P_values:
        curr_rows = np.logical_and(curr_condition_rows, results_df.loc[:, 'release probability'] == selected_release_P)
        num_rows = curr_rows.sum()

        results_dict_Acc_vs_N_samples[selected_release_P]['LR_acc_mean'].append(results_df.loc[curr_rows,'Accuracy LR'].mean())
        results_dict_Acc_vs_N_samples[selected_release_P]['FF_acc_mean'].append(results_df.loc[curr_rows,'Accuracy FF'].mean())
        results_dict_Acc_vs_N_samples[selected_release_P]['IF_acc_mean'].append(results_df.loc[curr_rows,'Accuracy IF'].mean())
        results_dict_Acc_vs_N_samples[selected_release_P]['baseline_acc_mean'].append(results_df.loc[curr_rows,'Accuracy baseline'].mean())

        results_dict_Acc_vs_N_samples[selected_release_P]['LR_acc_std'].append(results_df.loc[curr_rows,'Accuracy LR'].std() / np.sqrt(num_rows))
        results_dict_Acc_vs_N_samples[selected_release_P]['FF_acc_std'].append(results_df.loc[curr_rows,'Accuracy FF'].std() / np.sqrt(num_rows))
        results_dict_Acc_vs_N_samples[selected_release_P]['IF_acc_std'].append(results_df.loc[curr_rows,'Accuracy IF'].std() / np.sqrt(num_rows))
        results_dict_Acc_vs_N_samples[selected_release_P]['baseline_acc_std'].append(results_df.loc[curr_rows,'Accuracy baseline'].std() / np.sqrt(num_rows))


gs_accuracy_vs_N_samples = gridspec.GridSpec(nrows=2,ncols=1)
gs_accuracy_vs_N_samples.update(left=0.6925, right=0.97, bottom=0.04, top=0.28, wspace=0.1, hspace=0.1)
ax_accuracy_vs_N_samples = plt.subplot(gs_accuracy_vs_N_samples[:,:])

selected_release_P = 1.0
ax_accuracy_vs_N_samples.errorbar(all_N_samples_values, results_dict_Acc_vs_N_samples[selected_release_P]['LR_acc_mean'], yerr=results_dict_Acc_vs_N_samples[selected_release_P]['LR_acc_std'], lw=4, color='0.45')
ax_accuracy_vs_N_samples.errorbar(all_N_samples_values, results_dict_Acc_vs_N_samples[selected_release_P]['FF_acc_mean'], yerr=results_dict_Acc_vs_N_samples[selected_release_P]['FF_acc_std'], lw=4, color='orange')
ax_accuracy_vs_N_samples.errorbar(all_N_samples_values, results_dict_Acc_vs_N_samples[selected_release_P]['IF_acc_mean'], yerr=results_dict_Acc_vs_N_samples[selected_release_P]['IF_acc_std'], lw=4, color='0.05')

selected_release_P = 0.5
ax_accuracy_vs_N_samples.errorbar(all_N_samples_values, results_dict_Acc_vs_N_samples[selected_release_P]['LR_acc_mean'], yerr=results_dict_Acc_vs_N_samples[selected_release_P]['LR_acc_std'], lw=4, ls=':', color='0.45')
ax_accuracy_vs_N_samples.errorbar(all_N_samples_values, results_dict_Acc_vs_N_samples[selected_release_P]['FF_acc_mean'], yerr=results_dict_Acc_vs_N_samples[selected_release_P]['FF_acc_std'], lw=4, ls=':', color='orange')
ax_accuracy_vs_N_samples.errorbar(all_N_samples_values, results_dict_Acc_vs_N_samples[selected_release_P]['IF_acc_mean'], yerr=results_dict_Acc_vs_N_samples[selected_release_P]['IF_acc_std'], lw=4, ls=':', color='0.05')

ax_accuracy_vs_N_samples.set_xlabel('Number of Positive Training Patterns - N', fontsize=18)
ax_accuracy_vs_N_samples.set_xticks(all_N_samples_values)
ax_accuracy_vs_N_samples.set_xticklabels(all_N_samples_values, fontsize=15)
ax_accuracy_vs_N_samples.set_ylim(89.5,97.2)
ax_accuracy_vs_N_samples.set_yticks([90,92,94,96]);
ax_accuracy_vs_N_samples.set_yticklabels([90,92,94,96], fontsize=15);
ax_accuracy_vs_N_samples.spines['top'].set_visible(False)
ax_accuracy_vs_N_samples.spines['right'].set_visible(False)
ax_accuracy_vs_N_samples.set_xticks([16,128,256,512,1024]);
ax_accuracy_vs_N_samples.set_xticklabels([16,128,256,512,1024], fontsize=15);


# open digit 3 and display the learned weights of the 3 models for it
digit = 3
T = 50
N_axons = 100
M_connections = 5
temporal_silence_ms = 70

min_pos_samples = 5000

all_LR_weights = []
all_FF_weights = []
all_IF_weights = []

for k, filename_to_load in enumerate(list_of_files):
    loaded_script_results_dict = pickle.load(open(filename_to_load, "rb" ))

    digit_OK   = loaded_script_results_dict['script_main_params']['positive_digit'] == digit
    M_OK       = loaded_script_results_dict['script_main_params']['connections_per_axon'] == M_connections
    N_axons_OK = loaded_script_results_dict['script_main_params']['digit_sample_image_shape_expanded'][0] == N_axons
    T_OK       = loaded_script_results_dict['script_main_params']['digit_sample_image_shape_expanded'][1] == T
    N_pos_OK   = loaded_script_results_dict['script_main_params']['num_train_positive_patterns'] >= min_pos_samples
    ISS_OK     = loaded_script_results_dict['script_main_params']['temporal_silence_ms'] == temporal_silence_ms

    if digit_OK and M_OK and N_axons_OK and T_OK and N_pos_OK and ISS_OK:
        all_LR_weights.append(loaded_script_results_dict['learned_weights_LR'])
        all_FF_weights.append(loaded_script_results_dict['learned_weights_FF'])
        all_IF_weights.append(loaded_script_results_dict['learned_weights_IF'])

all_LR_weights = np.array(all_LR_weights)
all_FF_weights = np.array(all_FF_weights)
all_IF_weights = np.array(all_IF_weights)

# the first "num_const_firing_channels" axons are a "bias" term, so don't show them
h_start = loaded_script_results_dict['script_main_params']['num_const_firing_channels']
# the first "temporal_silence_ms" time points are silence, so they are boring
w_start = 39

rand_index = np.random.randint(all_LR_weights.shape[0])

single_trial_weights_LR = all_LR_weights[rand_index][h_start:,w_start:]
single_trial_weights_FF = all_FF_weights[rand_index][h_start:,w_start:]
single_trial_weights_IF = all_IF_weights[rand_index][h_start:,w_start:]

mean_weights_LR = all_LR_weights.mean(axis=0)[h_start:,w_start:]
mean_weights_FF = all_FF_weights.mean(axis=0)[h_start:,w_start:]
mean_weights_IF = all_IF_weights.mean(axis=0)[h_start:,w_start:]


def get_weight_symmetric_range(weights_matrix):
    top_value = np.percentile(weights_matrix, 99)
    bottom_value = np.percentile(weights_matrix, 1)
    symmetric_range = np.array([-1,1]) * max(np.abs(top_value), np.abs(bottom_value))

    return symmetric_range


symmetric_weight_range = get_weight_symmetric_range(single_trial_weights_LR)
symmetric_weight_range = get_weight_symmetric_range(mean_weights_LR)
colormap = 'viridis'
vmin = symmetric_weight_range[0]
vmax = symmetric_weight_range[1]

gs_learned_weights = gridspec.GridSpec(nrows=2,ncols=3)
gs_learned_weights.update(left=0.6925, right=0.97, bottom=0.32, top=0.529, wspace=0.06, hspace=0.06)

ax_learned_weights_00 = plt.subplot(gs_learned_weights[0,0])
ax_learned_weights_01 = plt.subplot(gs_learned_weights[0,1])
ax_learned_weights_02 = plt.subplot(gs_learned_weights[0,2])
ax_learned_weights_10 = plt.subplot(gs_learned_weights[1,0])
ax_learned_weights_11 = plt.subplot(gs_learned_weights[1,1])
ax_learned_weights_12 = plt.subplot(gs_learned_weights[1,2])

ax_learned_weights_00.set_title('Spatio-Temporal\n Logistic Regression', fontsize=13);
ax_learned_weights_00.set_ylabel('single trial', fontsize=14)
ax_learned_weights_00.imshow(single_trial_weights_LR, vmin=vmin, vmax=vmax, cmap=colormap);
ax_learned_weights_01.imshow(single_trial_weights_FF, vmin=vmin, vmax=vmax, cmap=colormap);
ax_learned_weights_01.set_title('Filter & Fire\n neuron', fontsize=13)
ax_learned_weights_02.imshow(single_trial_weights_IF, vmin=vmin, vmax=vmax, cmap=colormap);
ax_learned_weights_02.set_title('Integrate & Fire\n neuron', fontsize=13)

ax_learned_weights_10.set_ylabel('mean of %d trials' %(all_LR_weights.shape[0]), fontsize=14)
ax_learned_weights_10.imshow(mean_weights_LR, vmin=vmin, vmax=vmax, cmap=colormap);
ax_learned_weights_11.imshow(mean_weights_FF, vmin=vmin, vmax=vmax, cmap=colormap);
ax_learned_weights_12.imshow(mean_weights_IF, vmin=vmin, vmax=vmax, cmap=colormap);

def set_xy_ticks_ticklabels_to_None(ax_input):
    ax_input.set_xticks([])
    ax_input.set_xticklabels([])
    ax_input.set_yticks([])
    ax_input.set_yticklabels([])

set_xy_ticks_ticklabels_to_None(ax_learned_weights_00)
set_xy_ticks_ticklabels_to_None(ax_learned_weights_01)
set_xy_ticks_ticklabels_to_None(ax_learned_weights_02)
set_xy_ticks_ticklabels_to_None(ax_learned_weights_10)
set_xy_ticks_ticklabels_to_None(ax_learned_weights_11)
set_xy_ticks_ticklabels_to_None(ax_learned_weights_12)


# save figure
if save_figures:
    figure_name = 'F&F_MNIST_Figure_3_%d' %(np.random.randint(200))
    for file_ending in all_file_endings_to_use:
        if file_ending == '.png':
            fig.savefig(figure_folder + figure_name + file_ending, bbox_inches='tight')
        else:
            fig.savefig(figure_folder + figure_name + file_ending, bbox_inches='tight')

#%%

