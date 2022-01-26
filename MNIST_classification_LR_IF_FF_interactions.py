import os
import sys
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import linear_model
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow import keras

#%% script params

start_time = time.time()

try:
    print('----------------------------')
    print('----------------------------')
    random_seed = int(sys.argv[1])
    positive_digit = int(sys.argv[2])
    connections_per_axon = int(sys.argv[3])
    temporal_extent_factor_numerator = int(sys.argv[4])
    temporal_extent_factor_denumerator = int(sys.argv[5])
    release_probability = int(sys.argv[6])
    num_train_positive_patterns = int(sys.argv[7])
    print('"random_seed" selected by user - %d' %(random_seed))
    print('"positive_digit" selected by user - %d' %(positive_digit))
    print('"connections_per_axon" selected by user - %d' %(connections_per_axon))
    print('"temporal_extent_factor_numerator" selected by user - %d' %(temporal_extent_factor_numerator))
    print('"temporal_extent_factor_denumerator" selected by user - %d' %(temporal_extent_factor_denumerator))
    print('"release_probability" selected by user - %d' %(release_probability))
    print('"num_train_positive_patterns" selected by user - %d' %(num_train_positive_patterns))

    determine_internally = False
except:
    determine_internally = True
    try:
        random_seed = int(sys.argv[1])
        print('random seed selected by user - %d' %(random_seed))
    except:
        random_seed = np.random.randint(100000)
        print('randomly choose seed - %d' %(random_seed))

np.random.seed(random_seed)
print('----------------------------')
print('----------------------------')


if determine_internally:
    positive_digit = np.random.randint(10)
    connections_per_axon = np.random.choice([1,2,3,5,10], size=1)[0]
    temporal_extent_factor_numerator = np.random.choice([1,2,3,4,5], size=1)[0]
    temporal_extent_factor_denumerator = np.random.choice([1,2], size=1)[0]
    release_probability = np.random.choice([0.25, 0.5,0.5,0.5, 0.75, 1.0,1.0,1.0], size=1)[0]
    num_train_positive_patterns = np.random.choice([16,32,64,128,256,512,1024,2048,4096,5000], size=1)[0]

# interactions set to False
use_interaction_terms = False
interactions_degree  = 2

spatial_extent_factor = 5
num_const_firing_channels = 20
temporal_silence_ms = 70
#num_train_positive_patterns = 7000
num_train_negative_patterns_mult_factor = 5
spike_safety_range_ms = 20
negative_subsampling_fraction = 0.2

# release probability related params
# release_probability = 1.0
# release_probability = 0.5
train_epochs = 15
test_epochs  = 3

# what to consider as good prediction
output_spike_tolorance_window_duration = 30
output_spike_tolorance_window_offset   = 10

FF_weight_mult_factors_list = [0.01,0.03,0.07,0.1,0.3,0.5,0.8,1,1.3,2,3,4,5,7,10,25,50,120,250]
IF_weight_mult_factors_list = [0.01,0.03,0.07,0.1,0.3,0.5,0.8,1,1.3,2,3,4,5,7,10,25,50,120,250,1000,10000]

if use_interaction_terms is False:
    non_interaction_fraction_FF = 1.0
    non_interaction_fraction_IF = 1.0

# setting to create a 1 spike out or a 3 spike burst as supervising signal
create_output_burst = False
# create_output_burst = True

# setting for quick learning
#quick_test = True
quick_test = False

if quick_test:
    num_train_positive_patterns = 500
    num_train_negative_patterns_mult_factor = 2
    negative_subsampling_fraction = 0.1

    FF_weight_mult_factors_list = [0.25, 1, 2, 4, 8, 16, 64]
    IF_weight_mult_factors_list = [0.25, 1, 2, 4, 8, 16, 64]


show_plots = True
show_plots = False

data_folder = '/filter_and_fire_neuron/results_data_mnist/'

experiment_results_dict = {}
experiment_results_dict['script_main_params'] = {}
experiment_results_dict['script_main_params']['positive_digit'] = positive_digit
experiment_results_dict['script_main_params']['connections_per_axon'] = connections_per_axon
experiment_results_dict['script_main_params']['random_seed'] = random_seed
experiment_results_dict['script_main_params']['interactions_degree'] = interactions_degree

experiment_results_dict['script_main_params']['temporal_extent_factor_numerator'] = temporal_extent_factor_numerator
experiment_results_dict['script_main_params']['temporal_extent_factor_denumerator'] = temporal_extent_factor_denumerator
experiment_results_dict['script_main_params']['spatial_extent_factor'] = spatial_extent_factor
experiment_results_dict['script_main_params']['num_const_firing_channels'] = num_const_firing_channels
experiment_results_dict['script_main_params']['temporal_silence_ms'] = temporal_silence_ms

experiment_results_dict['script_main_params']['num_train_positive_patterns'] = num_train_positive_patterns
experiment_results_dict['script_main_params']['num_train_negative_patterns_mult_factor'] = num_train_negative_patterns_mult_factor
experiment_results_dict['script_main_params']['spike_safety_range_ms'] = spike_safety_range_ms
experiment_results_dict['script_main_params']['negative_subsampling_fraction'] = negative_subsampling_fraction

experiment_results_dict['script_main_params']['release_probability'] = release_probability
experiment_results_dict['script_main_params']['train_epochs'] = train_epochs
experiment_results_dict['script_main_params']['test_epochs'] = test_epochs
experiment_results_dict['script_main_params']['create_output_burst'] = create_output_burst

experiment_results_dict['script_main_params']['output_spike_tolorance_window_duration'] = output_spike_tolorance_window_duration
experiment_results_dict['script_main_params']['output_spike_tolorance_window_offset']   = output_spike_tolorance_window_offset
experiment_results_dict['script_main_params']['FF_weight_mult_factors_list'] = FF_weight_mult_factors_list
experiment_results_dict['script_main_params']['IF_weight_mult_factors_list'] = IF_weight_mult_factors_list


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
    temporal_filter_length = int(4 * tau_decay_vec.max()) + 1

    syn_filter = np.zeros((num_synapses, temporal_filter_length))

    for k, (tau_r, tau_d) in enumerate(zip(tau_rise_vec, tau_decay_vec)):
        syn_filter[k,:] = create_single_PSP_profile(tau_r, tau_d, temporal_filter_length=temporal_filter_length)

    return syn_filter


def simulate_filter_and_fire_cell_with_interactions(presynaptic_input_spikes, interactions_map, synaptic_weights, tau_rise_vec, tau_decay_vec,
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

    # apply interactions
    local_normlized_currents = apply_dendritic_interactions(local_normlized_currents, interactions_map)

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


def simulate_filter_and_fire_cell_with_interactions_long(presynaptic_input_spikes, interactions_map, synaptic_weights, tau_rise_vec, tau_decay_vec,
                                                         refreactory_time_constant=20, v_reset=-75, v_threshold=-55, current_to_voltage_mult_factor=2):

    total_duration_ms = presynaptic_input_spikes.shape[1]
    max_duration_per_call_ms = 50000
    overlap_time_ms = 500

    if max_duration_per_call_ms >= total_duration_ms:
        local_normlized_currents, soma_voltage, output_spike_times_in_ms = simulate_filter_and_fire_cell_with_interactions(presynaptic_input_spikes, interactions_map, synaptic_weights, tau_rise_vec, tau_decay_vec,
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

        curr_loc_norm_c, curr_soma_v, curr_out_sp_t = simulate_filter_and_fire_cell_with_interactions(presynaptic_input_spikes[:,start_ind:end_ind], interactions_map, synaptic_weights, tau_rise_vec, tau_decay_vec,
                                                                                                      refreactory_time_constant=refreactory_time_constant, v_reset=v_reset, v_threshold=v_threshold,
                                                                                                      current_to_voltage_mult_factor=current_to_voltage_mult_factor)

        # update fields
        if k == 0:
            local_normlized_currents[:,start_ind:end_ind] = curr_loc_norm_c
            soma_voltage[start_ind:end_ind] = curr_soma_v
            output_spike_times_in_ms += curr_out_sp_t
        else:
            local_normlized_currents[:,(start_ind + overlap_time_ms):end_ind] = curr_loc_norm_c[:,overlap_time_ms:end_ind]
            soma_voltage[(start_ind + overlap_time_ms):end_ind] = curr_soma_v[overlap_time_ms:]
            curr_out_sp_t = [x for x in curr_out_sp_t if x >= (overlap_time_ms - 1)]
            output_spike_times_in_ms += [(start_ind + x) for x in curr_out_sp_t]

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


def generate_dendritic_interactions_map(num_synapses, interactions_degree=2, non_interaction_fraction=0.2):

    interactions_map = {}
    interactions_map['degree_permutations'] = {}

    for degree in range(interactions_degree - 1):
        interactions_map['degree_permutations'][degree] = np.random.permutation(num_synapses)
    interactions_map['non_interacting_indices'] = np.random.permutation(num_synapses)[:int(num_synapses * non_interaction_fraction)]

    return interactions_map


def apply_dendritic_interactions(normlized_synaptic_currents, interactions_map):
    output_normlized_synaptic_currents = normlized_synaptic_currents.copy()

    # apply d times random interactions
    for degree in range(interactions_degree - 1):
        output_normlized_synaptic_currents = output_normlized_synaptic_currents * normlized_synaptic_currents[interactions_map['degree_permutations'][degree]]

    # keep some fraction of only individual interactions
    output_normlized_synaptic_currents[interactions_map['non_interacting_indices']] = normlized_synaptic_currents[interactions_map['non_interacting_indices']]

    return output_normlized_synaptic_currents


#%% Load MNIST dataset and show the data

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

if show_plots:
    num_rows = 5
    num_cols = 7

    plt.figure(figsize=(20,15))
    for k in range(num_rows * num_cols):
        rand_sample_ind = np.random.randint(x_train.shape[0])
        plt.subplot(num_rows, num_cols, k + 1)
        plt.imshow(x_train[k]); plt.title('digit "%s"' %(y_train[k]))

#%% display mean and std images, as well as histograms

mean_image = x_train.mean(axis=0)
std_image  = x_train.std(axis=0)

if show_plots:
    plt.figure(figsize=(21,14))
    plt.subplot(2,3,1); plt.imshow(mean_image); plt.title('mean image')
    plt.subplot(2,3,2); plt.bar(np.arange(mean_image.shape[0]), mean_image.sum(axis=0)); plt.title('"temporal" (columns) histogram (mean image)')
    plt.subplot(2,3,3); plt.bar(np.arange(mean_image.shape[0]), mean_image.sum(axis=1)); plt.title('"spatial" (rows) histogram (mean image)')

    plt.subplot(2,3,4); plt.imshow(std_image); plt.title('std image')
    plt.subplot(2,3,5); plt.bar(np.arange(std_image.shape[0]), std_image.sum(axis=0)); plt.title('"temporal" (columns) histogram (std image)')
    plt.subplot(2,3,6); plt.bar(np.arange(std_image.shape[0]), std_image.sum(axis=1)); plt.title('"spatial" (rows) histogram (std image)')


#%% Crop the data and binarize it

h_crop_range = [4,24]
w_crop_range = [4,24]

positive_threshold = 150

x_train = x_train[:,h_crop_range[0]:h_crop_range[1],w_crop_range[0]:w_crop_range[1]] > positive_threshold
x_test  = x_test[: ,h_crop_range[0]:h_crop_range[1],w_crop_range[0]:w_crop_range[1]] > positive_threshold

if show_plots:
    num_rows = 5
    num_cols = 7

    plt.figure(figsize=(20,15))
    for k in range(num_rows * num_cols):
        rand_sample_ind = np.random.randint(x_train.shape[0])
        plt.subplot(num_rows, num_cols, k + 1)
        plt.imshow(x_train[k]); plt.title('digit "%s"' %(y_train[k]))

experiment_results_dict['script_main_params']['digit_sample_image_shape_cropped'] = x_train[0].shape

#%% Transform Xs to spatio-temporal spike trains

# extend according to "spatial_extent_factor" and "temporal_extent_factor"
kernel = np.ones((1, spatial_extent_factor, temporal_extent_factor_numerator), dtype=bool)

# reshape X according to what is needed
x_train = np.kron(x_train, kernel)
x_test  = np.kron(x_test , kernel)

# subsample according to "temporal_extent_factor_denumerator"
x_train = x_train[:,:,::temporal_extent_factor_denumerator]
x_test  = x_test[:,:, ::temporal_extent_factor_denumerator]

experiment_results_dict['script_main_params']['digit_sample_image_shape_expanded'] = x_train[0].shape

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
x_test  = x_test  * (np.random.rand(x_test.shape[0] , x_test.shape[1] , x_test.shape[2])  < fraction_of_spikes_to_eliminate)

final_mean_firing_rate_Hz = 1000 * x_train.mean()

# display the patterns
if show_plots:
    num_rows = 5
    num_cols = 7

    plt.figure(figsize=(20,15))
    for k in range(num_rows * num_cols):
        rand_sample_ind = np.random.randint(x_train.shape[0])
        plt.subplot(num_rows, num_cols, k + 1)
        plt.imshow(x_train[k], cmap='gray'); plt.title('digit "%s"' %(y_train[k]))


#%% display distribution of number of spikes per pattern

if show_plots:
    plt.close('all')
    plt.figure(figsize=(12,8))
    plt.hist(x_train.sum(axis=2).sum(axis=1), bins=40); plt.title('distribution of number of spikes per pattern')
    plt.ylabel('number of patterns'); plt.xlabel('number of incoming spikes per pattern')

#%% Create "one-vs-all" dataset

y_train_binary = y_train == positive_digit
y_test_binary  = y_test  == positive_digit

num_train_positive_patterns = min(int(y_train_binary.sum()), num_train_positive_patterns)
num_train_negative_patterns = int(num_train_negative_patterns_mult_factor * num_train_positive_patterns)

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

if release_probability < 1.0:
    # replicate train and test by corresponding factors (epochs)
    X_train_spikes = np.tile(X_train_spikes, (train_epochs, 1, 1))
    Y_train_spikes = np.tile(Y_train_spikes, (train_epochs, ))

    X_test_spikes = np.tile(X_test_spikes, (test_epochs, 1, 1))
    Y_test_spikes = np.tile(Y_test_spikes, (test_epochs, ))

    # add synaptic unrelability to all patterns after replication
    rand_matrix = np.random.rand(X_train_spikes.shape[0], X_train_spikes.shape[1], X_train_spikes.shape[2])
    X_train_spikes = X_train_spikes * (rand_matrix < release_probability)

    rand_matrix = np.random.rand(X_test_spikes.shape[0], X_test_spikes.shape[1], X_test_spikes.shape[2])
    X_test_spikes = X_test_spikes * (rand_matrix < release_probability)

experiment_results_dict['script_main_params']['num_train_positive_patterns'] = num_train_positive_patterns
experiment_results_dict['script_main_params']['num_train_negative_patterns_mult_factor'] = num_train_negative_patterns_mult_factor
experiment_results_dict['script_main_params']['release_probability'] = release_probability
experiment_results_dict['script_main_params']['train_epochs'] = train_epochs
experiment_results_dict['script_main_params']['test_epochs']  = test_epochs

#%% Create a regularized logistic regression baseline

logistic_reg_model = linear_model.LogisticRegression(C=0.1, fit_intercept=False, penalty='l2',verbose=False)

# fit model
logistic_reg_model.fit(X_train_spikes.reshape([X_train_spikes.shape[0],-1]), Y_train_spikes)

# predict and calculate AUC on train data
Y_train_spikes_hat = logistic_reg_model.predict_proba(X_train_spikes.reshape([X_train_spikes.shape[0],-1]))[:,1]
Y_test_spikes_hat = logistic_reg_model.predict_proba(X_test_spikes.reshape([X_test_spikes.shape[0],-1]))[:,1]

train_AUC = roc_auc_score(Y_train_spikes, Y_train_spikes_hat)
test_AUC = roc_auc_score(Y_test_spikes, Y_test_spikes_hat)

print('----------------------------')
print('----------------------------')
print('for (# pos = %d, # neg = %d): (train AUC, test AUC) = (%.5f, %.5f)' %(num_train_positive_patterns, num_train_negative_patterns, train_AUC, test_AUC))
print('----------------------------')

logistic_regression_learned_weights = logistic_reg_model.coef_.reshape([X_train_spikes.shape[1], X_train_spikes.shape[2]])

if show_plots:
    plt.figure(figsize=(8,8))
    plt.imshow(logistic_regression_learned_weights)
    plt.title('Learned Weights \n (spatio-temporal ("image") logistic regression)')

experiment_results_dict['learned_weights_LR'] = logistic_regression_learned_weights

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

if show_plots:
    plt.figure(figsize=(15,8))
    plt.subplot(1,2,1); plt.bar(x=acc_bar_x_axis,height=LL_accuracy_subsampled)
    plt.xticks(acc_bar_x_axis, LL_thresholds_subsampled, rotation='vertical')
    plt.title('max accuracy = %.2f%s' %(LL_accuracy_max,'%'), fontsize=24)
    plt.ylim(87.8,100); plt.xlabel('threshold', fontsize=20); plt.ylabel('Accuracy (%)', fontsize=20)
    plt.plot([acc_bar_x_axis[0] - 1, acc_bar_x_axis[-1] + 1], [zero_pred_baseline_accuracy, zero_pred_baseline_accuracy], color='r')

    plt.subplot(1,2,2); plt.plot(LL_false_positive_list, LL_true_positive_list)
    plt.ylabel('True Positive (%)', fontsize=20); plt.xlabel('False Positive (%)', fontsize=20)

experiment_results_dict['model_accuracy_LR'] = LL_accuracy_max

#%% Fit a F&F model

# main parameters
# connections_per_axon = 5
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
    tau_rise_range  = [1, 18]
    tau_decay_range = [8, 30]
    # tau_decay_range = [8,48]
elif model_type == 'I&F':
    tau_rise_range  = [ 1, 1]
    tau_decay_range = [30,30]

tau_rise_vec  = np.random.uniform(low=tau_rise_range[0] , high=tau_rise_range[1] , size=(num_synapses, 1))
tau_decay_vec = np.random.uniform(low=tau_decay_range[0], high=tau_decay_range[1], size=(num_synapses, 1))

experiment_results_dict['tau_rise_vec_FF']  = tau_rise_vec
experiment_results_dict['tau_decay_vec_FF'] = tau_decay_vec

# synapse learnable parameters
synaptic_weights_vec = np.random.normal(size=(num_synapses, 1))

# prepare input spikes
axons_input_spikes = np.concatenate([X_train_spikes[k] for k in range(X_train_spikes.shape[0])], axis=1)

# prepare output spikes
pattern_duration_ms = X_train_spikes[0].shape[1]
output_kernel = np.zeros((pattern_duration_ms,))
output_spike_offset = 1
output_kernel[-output_spike_offset] = 1

if create_output_burst:
    output_spike_offset = 6
    output_kernel[-output_spike_offset] = 1
    output_spike_offset = 11
    output_kernel[-output_spike_offset] = 1

desired_output_spikes = np.kron(Y_train_spikes, output_kernel)

if show_plots:
    plt.figure(figsize=(30,15))
    plt.imshow(axons_input_spikes[:,:1101], cmap='gray')
    plt.title('input axons raster', fontsize=22)
    plt.ylabel('axon index', fontsize=22)
    plt.xlabel('time [ms]', fontsize=22)

    plt.figure(figsize=(30,1))
    plt.plot(desired_output_spikes[:1101]); plt.xlim(0,1101)
    plt.ylabel('output spike', fontsize=22)
    plt.xlabel('time [ms]', fontsize=22)

#%%

presynaptic_input_spikes = np.kron(np.ones((connections_per_axon, 1), dtype=bool), axons_input_spikes).astype(bool)

if use_interaction_terms:
    non_interaction_fraction_FF = min(1.0, 2.5 * num_axons / num_synapses)
    non_interaction_fraction_FF = 0.40
interactions_map_FF = generate_dendritic_interactions_map(num_synapses, interactions_degree=interactions_degree, non_interaction_fraction=non_interaction_fraction_FF)

experiment_results_dict['script_main_params']['non_interaction_fraction_FF'] = non_interaction_fraction_FF
experiment_results_dict['interactions_map_FF'] = interactions_map_FF

# simulate cell with normlized currents
local_normlized_currents, soma_voltage, output_spike_times_in_ms = simulate_filter_and_fire_cell_with_interactions_long(presynaptic_input_spikes, interactions_map_FF,
                                                                                                                        synaptic_weights_vec, tau_rise_vec, tau_decay_vec,
                                                                                                                        refreactory_time_constant=refreactory_time_constant,
                                                                                                                        v_reset=v_reset, v_threshold=v_threshold,
                                                                                                                        current_to_voltage_mult_factor=current_to_voltage_mult_factor)


# fit linear model to local currents
filter_and_fire_model = linear_model.LogisticRegression(C=100000, fit_intercept=False, penalty='l2')

# spike_safety_range_ms = 20
# negative_subsampling_fraction = 0.25

X, y = prepare_training_dataset(local_normlized_currents, desired_output_spikes,
                                spike_safety_range_ms=spike_safety_range_ms,
                                negative_subsampling_fraction=negative_subsampling_fraction)

print('----------------------------')
print('number of data points = %d (%.2f%s positive class)' %(X.shape[0], 100 * y.mean(),'%'))
print('----------------------------')

# fit model
filter_and_fire_model.fit(X, y)

# calculate train AUC
y_hat = filter_and_fire_model.predict_proba(X)[:,1]
train_AUC = roc_auc_score(y, y_hat)

print('F&F train AUC = %.5f' %(train_AUC))

if show_plots:
    # display some training data predictions
    num_timepoints_to_show = 10000
    fitted_output_spike_prob = filter_and_fire_model.predict_proba(local_normlized_currents[:,:num_timepoints_to_show].T)[:,1]

    plt.figure(figsize=(30,10))
    plt.plot(1.05 * desired_output_spikes[:num_timepoints_to_show] - 0.025); plt.title('train AUC = %.5f' %(train_AUC), fontsize=22)
    plt.plot(fitted_output_spike_prob[:num_timepoints_to_show]); plt.xlabel('time [ms]'); plt.legend(['GT', 'prediction'], fontsize=22)

#%% display learned weights
normlized_syn_filter = construct_normlized_synaptic_filter(tau_rise_vec, tau_decay_vec)

# collect learned synaptic weights
FF_learned_synaptic_weights = np.fliplr(filter_and_fire_model.coef_).T
weighted_syn_filter = FF_learned_synaptic_weights * normlized_syn_filter

axon_spatio_temporal_pattern = np.zeros((num_axons, weighted_syn_filter.shape[1]))
for k in range(num_axons):
    axon_spatio_temporal_pattern[k] = weighted_syn_filter[k::num_axons].sum(axis=0)

axon_spatio_temporal_pattern_short = axon_spatio_temporal_pattern[:,:X_train_spikes.shape[2]]

if show_plots:
    plt.figure(figsize=(18,8))
    plt.subplot(1,2,1); plt.imshow(logistic_reg_model.coef_.reshape([X_train_spikes.shape[1], X_train_spikes.shape[2]])); plt.title('logistic regression', fontsize=20)
    plt.subplot(1,2,2); plt.imshow(np.flip(axon_spatio_temporal_pattern_short)); plt.title('filter and fire neuron', fontsize=20)

experiment_results_dict['learned_weights_FF'] = np.flip(axon_spatio_temporal_pattern_short)

#%% Make a prediction on the entire test trace

num_test_patterns = X_test_spikes.shape[0]

# prepare test outputs
# output_spike_tolorance_window_duration = 20
# output_spike_tolorance_window_offset   = 5
output_kernel_test = np.zeros((X_test_spikes[0].shape[1],))
output_kernel_test[-output_spike_tolorance_window_duration:] = 1

desired_output_spikes_test = np.kron(Y_test_spikes[:num_test_patterns], output_kernel_test)
desired_output_spikes_test = np.concatenate((np.zeros((output_spike_tolorance_window_offset,)), desired_output_spikes_test[:-output_spike_tolorance_window_offset]))

# prepare test inputs
axons_input_spikes_test = np.concatenate([X_test_spikes[k] for k in range(num_test_patterns)],axis=1)
presynaptic_input_spikes_test = np.kron(np.ones((connections_per_axon, 1), dtype=bool), axons_input_spikes_test).astype(bool)

# FF_weight_mult_factors_list = [x for x in [2,3,4,5,6,9,20,50,120,250]]
FF_accuracy_list = []
FF_true_positive_list = []
FF_false_positive_list = []
for weight_mult_factor in FF_weight_mult_factors_list:

    # collect learned synaptic weights
    synaptic_weights_post_learning = weight_mult_factor * FF_learned_synaptic_weights

    _, soma_voltage_test, output_spike_times_in_ms_test = simulate_filter_and_fire_cell_with_interactions_long(presynaptic_input_spikes_test, interactions_map_FF,
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

    print('F&F: weights mult factor = %.1f: Accuracy = %.3f%s. (TP, FP) = (%.3f%s, %.3f%s)' %(weight_mult_factor, percent_accuracy,'%',true_positive,'%',false_positive,'%'))

    FF_accuracy_list.append(percent_accuracy)
    FF_true_positive_list.append(true_positive)
    FF_false_positive_list.append(false_positive)

experiment_results_dict['model_accuracy_FF'] = np.array(FF_accuracy_list).max()
experiment_results_dict['model_accuracy_baseline'] = zero_pred_baseline_accuracy

#%% Display accuracy results for F&F model

if show_plots:
    plt.figure(figsize=(15,8))
    plt.subplot(1,2,1); plt.bar(x=range(len(FF_accuracy_list)),height=FF_accuracy_list)
    plt.xticks(range(len(FF_accuracy_list)), FF_weight_mult_factors_list); plt.title('max accuracy = %.2f%s' %(np.array(FF_accuracy_list).max(),'%'), fontsize=24)
    plt.ylim(87.8,100); plt.xlabel('weight mult factor ("gain")', fontsize=20); plt.ylabel('Accuracy (%)', fontsize=20)
    plt.plot([-1, len(FF_accuracy_list)], [zero_pred_baseline_accuracy, zero_pred_baseline_accuracy], color='r')

    plt.subplot(1,2,2); plt.plot(FF_false_positive_list, FF_true_positive_list)
    plt.ylabel('True Positive (%)', fontsize=20); plt.xlabel('False Positive (%)', fontsize=20)

    plt.figure(figsize=(30,6))
    plt.plot(1.05 * desired_output_spikes_test[:25000] - 0.025)
    plt.plot(output_spikes_test[:25000])
    plt.legend(['desired outputs', 'actual outputs'])

    plt.figure(figsize=(30,6))
    plt.plot(1.05 * desired_output_spikes_test[2900:4600] - 0.025)
    plt.plot(output_spikes_test[2900:4600])
    plt.legend(['desired outputs', 'actual outputs'])

    plt.figure(figsize=(30,6))
    plt.plot(1.05 * desired_output_spikes_test[8500:9500] - 0.025)
    plt.plot(output_spikes_test[8500:9500])
    plt.legend(['desired outputs', 'actual outputs'])

#%% F&F model predictions

if show_plots:
    start_time = 100 * np.random.randint(int(axons_input_spikes_test.shape[1] / 100 - 20))
    end_time = start_time + 1 + 100 * 18

    plt.figure(figsize=(30,15))
    plt.subplot(3,1,1); plt.imshow(axons_input_spikes_test[:,start_time:end_time], cmap='gray'); plt.title('input axons raster (test set)', fontsize=22)
    plt.subplot(3,1,2); plt.plot(output_spikes_test[start_time:end_time]); plt.xlim(0,1 + 100 * 18); plt.ylabel('output spike', fontsize=22)
    plt.subplot(3,1,3); plt.plot(soma_voltage_test[start_time:end_time]); plt.xlim(0,1 + 100 * 18); plt.ylabel('soma voltage [mV]', fontsize=22)
    plt.xlabel('time [ms]', fontsize=22)

#%% Fit an I&F model

# main parameters
# connections_per_axon = 5
#model_type = 'F&F'
model_type = 'I&F'

# neuron model parameters
num_axons = X_train_spikes[0].shape[0]
num_synapses = connections_per_axon * num_axons

v_reset     = -80
v_threshold = -55
current_to_voltage_mult_factor = 3
refreactory_time_constant = 15

# synapse non-learnable parameters
if model_type == 'F&F':
    tau_rise_range  = [1, 18]
    tau_decay_range = [8, 30]
elif model_type == 'I&F':
    tau_rise_range  = [ 1, 1]
    tau_decay_range = [30,30]

tau_rise_vec_IF  = np.random.uniform(low=tau_rise_range[0] , high=tau_rise_range[1] , size=(num_synapses, 1))
tau_decay_vec_IF = np.random.uniform(low=tau_decay_range[0], high=tau_decay_range[1], size=(num_synapses, 1))

# synapse learnable parameters
synaptic_weights_vec_IF = np.random.normal(size=(num_synapses, 1))

#%%

if use_interaction_terms:
    non_interaction_fraction_IF = 1.0 * num_axons / num_synapses
    non_interaction_fraction_IF = 0.4
interactions_map_IF = generate_dendritic_interactions_map(num_synapses, interactions_degree=interactions_degree, non_interaction_fraction=non_interaction_fraction_IF)

experiment_results_dict['script_main_params']['non_interaction_fraction_IF'] = non_interaction_fraction_IF
experiment_results_dict['interactions_map_IF'] = interactions_map_IF

# simulate cell with normlized currents
local_normlized_currents_IF, soma_voltage_IF, output_spike_times_in_ms_IF = simulate_filter_and_fire_cell_with_interactions_long(presynaptic_input_spikes, interactions_map_IF,
                                                                                                                                 synaptic_weights_vec_IF, tau_rise_vec_IF, tau_decay_vec_IF,
                                                                                                                                 refreactory_time_constant=refreactory_time_constant,
                                                                                                                                 v_reset=v_reset, v_threshold=v_threshold,
                                                                                                                                 current_to_voltage_mult_factor=current_to_voltage_mult_factor)

# fit linear model to local currents
integrate_and_fire_model = linear_model.LogisticRegression(C=100000, fit_intercept=False, penalty='l2')

# spike_safety_range_ms = 20
# negative_subsampling_fraction = 0.25

X, y = prepare_training_dataset(local_normlized_currents_IF, desired_output_spikes,
                                spike_safety_range_ms=spike_safety_range_ms,
                                negative_subsampling_fraction=negative_subsampling_fraction)

print('----------------------------')
print('number of data points = %d (%.2f%s positive class)' %(X.shape[0], 100 * y.mean(),'%'))
print('----------------------------')

# fit model
integrate_and_fire_model.fit(X, y)

# calculate train AUC
y_hat = integrate_and_fire_model.predict_proba(X)[:,1]
train_AUC = roc_auc_score(y, y_hat)

print('I&F train AUC = %.5f' %(train_AUC))

if show_plots:
    # display some training data predictions
    num_timepoints_to_show = 10000
    fitted_output_spike_prob = integrate_and_fire_model.predict_proba(local_normlized_currents_IF[:,:num_timepoints_to_show].T)[:,1]

    plt.figure(figsize=(30,10))
    plt.plot(1.05 * desired_output_spikes[:num_timepoints_to_show] - 0.025); plt.title('train AUC = %.5f' %(train_AUC), fontsize=22)
    plt.plot(fitted_output_spike_prob[:num_timepoints_to_show]); plt.xlabel('time [ms]'); plt.legend(['GT', 'prediction'], fontsize=22)

#%% display learned weights

# display learned weights
normlized_syn_filter_IF = construct_normlized_synaptic_filter(tau_rise_vec_IF, tau_decay_vec_IF)

# collect learned synaptic weights
IF_learned_synaptic_weights = np.fliplr(integrate_and_fire_model.coef_).T
weighted_syn_filter_IF = IF_learned_synaptic_weights * normlized_syn_filter_IF

axon_spatio_temporal_pattern_IF = np.zeros((num_axons, weighted_syn_filter_IF.shape[1]))
for k in range(num_axons):
    axon_spatio_temporal_pattern_IF[k] = weighted_syn_filter_IF[k::num_axons].sum(axis=0)

axon_spatio_temporal_pattern_short_IF = axon_spatio_temporal_pattern_IF[:,:X_train_spikes.shape[2]]

if show_plots:
    plt.figure(figsize=(24,10))
    plt.subplot(1,3,1); plt.imshow(logistic_reg_model.coef_.reshape([X_train_spikes.shape[1], X_train_spikes.shape[2]])); plt.title('logistic regression', fontsize=24)
    plt.subplot(1,3,2); plt.imshow(np.flip(axon_spatio_temporal_pattern_short)); plt.title('filter and fire neuron', fontsize=24)
    plt.subplot(1,3,3); plt.imshow(np.flip(axon_spatio_temporal_pattern_short_IF)); plt.title('integrate and fire neuron', fontsize=24)

experiment_results_dict['learned_weights_IF'] = np.flip(axon_spatio_temporal_pattern_short_IF)

#%% Display I&F accuracy

# IF_weight_mult_factors_list = [x for x in [10,50,100,1000,10000]]
IF_accuracy_list = []
IF_true_positive_list = []
IF_false_positive_list = []

for weight_mult_factor in IF_weight_mult_factors_list:

    # collect learned synaptic weights
    synaptic_weights_post_learning = weight_mult_factor * IF_learned_synaptic_weights

    _, soma_voltage_test, output_spike_times_in_ms_test = simulate_filter_and_fire_cell_with_interactions_long(presynaptic_input_spikes_test, interactions_map_IF,
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

    print('I&F: weights mult factor = %.1f: Accuracy = %.3f%s. (TP, FP) = (%.3f%s, %.3f%s)' %(weight_mult_factor, percent_accuracy,'%',true_positive,'%',false_positive,'%'))

    IF_accuracy_list.append(percent_accuracy)
    IF_true_positive_list.append(true_positive)
    IF_false_positive_list.append(false_positive)

if show_plots:
    plt.figure(figsize=(15,8))
    plt.subplot(1,2,1); plt.bar(x=range(len(IF_accuracy_list)),height=IF_accuracy_list)
    plt.xticks(range(len(IF_accuracy_list)), IF_weight_mult_factors_list); plt.title('max accuracy = %.2f%s' %(np.array(IF_accuracy_list).max(),'%'), fontsize=24)
    plt.ylim(87.8,100); plt.xlabel('weight mult factor ("gain")', fontsize=20); plt.ylabel('Accuracy (%)', fontsize=20)
    plt.plot([-1, len(IF_accuracy_list)], [zero_pred_baseline_accuracy, zero_pred_baseline_accuracy], color='r')

    plt.subplot(1,2,2); plt.plot(IF_false_positive_list, IF_true_positive_list)
    plt.ylabel('True Positive (%)', fontsize=20); plt.xlabel('False Positive (%)', fontsize=20)


experiment_results_dict['model_accuracy_IF'] = np.array(IF_accuracy_list).max()


if show_plots:
    plt.figure(figsize=(30,6))
    plt.plot(1.05 * desired_output_spikes_test[:25000] - 0.025)
    plt.plot(output_spikes_test[:25000])
    plt.legend(['desired outputs', 'actual outputs'])

    plt.figure(figsize=(30,6))
    plt.plot(1.05 * desired_output_spikes_test[2900:4600] - 0.025)
    plt.plot(output_spikes_test[2900:4600])
    plt.legend(['desired outputs', 'actual outputs'])

    plt.figure(figsize=(30,6))
    plt.plot(1.05 * desired_output_spikes_test[8500:9500] - 0.025)
    plt.plot(output_spikes_test[8500:9500])
    plt.legend(['desired outputs', 'actual outputs'])

#%% I&F model predictions

if show_plots:
    start_time = 100 * np.random.randint(int(axons_input_spikes_test.shape[1] / 100 - 20))
    end_time = start_time + 1 + 100 * 18

    plt.figure(figsize=(30,15))
    plt.subplot(3,1,1); plt.imshow(axons_input_spikes_test[:,start_time:end_time], cmap='gray'); plt.title('input axons raster (test set)', fontsize=22)
    plt.subplot(3,1,2); plt.plot(output_spikes_test[start_time:end_time]); plt.xlim(0,1 + 100 * 18); plt.ylabel('output spike', fontsize=22)
    plt.subplot(3,1,3); plt.plot(soma_voltage_test[start_time:end_time]); plt.xlim(0,1 + 100 * 18); plt.ylabel('soma voltage [mV]', fontsize=22)
    plt.xlabel('time [ms]', fontsize=22)

#%% save results

if use_interaction_terms:
    results_filename = 'MNIST__interactions_%d__digit_%d__N_axons_%d__T_%d__M_%d__p_%0.3d__N_pos_samples_%d__randseed_%d.pickle' %(interactions_degree, positive_digit,
                                                                                                                                  experiment_results_dict['script_main_params']['digit_sample_image_shape_expanded'][0],
                                                                                                                                  experiment_results_dict['script_main_params']['digit_sample_image_shape_expanded'][1],
                                                                                                                                  experiment_results_dict['script_main_params']['connections_per_axon'],
                                                                                                                                  100 * experiment_results_dict['script_main_params']['release_probability'],
                                                                                                                                  num_train_positive_patterns, random_seed)
else:
    results_filename = 'MNIST__digit_%d__N_axons_%d__T_%d__M_%d__p_%0.3d__N_pos_samples_%d__randseed_%d.pickle' %(positive_digit,
                                                                                                                 experiment_results_dict['script_main_params']['digit_sample_image_shape_expanded'][0],
                                                                                                                 experiment_results_dict['script_main_params']['digit_sample_image_shape_expanded'][1],
                                                                                                                 experiment_results_dict['script_main_params']['connections_per_axon'],
                                                                                                                 100 * experiment_results_dict['script_main_params']['release_probability'],
                                                                                                                 num_train_positive_patterns, random_seed)

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# pickle everythin
pickle.dump(experiment_results_dict, open(data_folder + results_filename, "wb"))

#%% Load the saved pickle just to check it's OK

loaded_script_results_dict = pickle.load(open(data_folder + results_filename, "rb" ))

print('-----------------------------------------------------------------------------------------------------------')
print('loaded_script_results_dict.keys():')
print('----------')
print(list(loaded_script_results_dict.keys()))
print('-----------------------------------------------------------------------------------------------------------')
print('loaded_script_results_dict["script_main_params"].keys():')
print('----------')
print(list(loaded_script_results_dict["script_main_params"].keys()))
print('-----------------------------------------------------------------------------------------------------------')
print('interactions_degree =', loaded_script_results_dict['script_main_params']['interactions_degree'])
print('positive_digit =', loaded_script_results_dict['script_main_params']['positive_digit'])
print('connections_per_axon =', loaded_script_results_dict['script_main_params']['connections_per_axon'])
print('digit_sample_image_shape_cropped =', loaded_script_results_dict['script_main_params']['digit_sample_image_shape_cropped'])
print('digit_sample_image_shape_expanded =', loaded_script_results_dict['script_main_params']['digit_sample_image_shape_expanded'])
print('num_train_positive_patterns =', loaded_script_results_dict['script_main_params']['num_train_positive_patterns'])
print('temporal_silence_ms =', loaded_script_results_dict['script_main_params']['temporal_silence_ms'])
print('release_probability =', loaded_script_results_dict['script_main_params']['release_probability'])
print('train_epochs =', loaded_script_results_dict['script_main_params']['train_epochs'])
print('test_epochs =', loaded_script_results_dict['script_main_params']['test_epochs'])
print('-----------------------------------------------------------------------------------------------------------')
print('model_accuracy_LR =', loaded_script_results_dict['model_accuracy_LR'])
print('model_accuracy_FF =', loaded_script_results_dict['model_accuracy_FF'])
print('model_accuracy_IF =', loaded_script_results_dict['model_accuracy_IF'])
print('model_accuracy_baseline =', loaded_script_results_dict['model_accuracy_baseline'])
print('-----------------------------------------------------------------------------------------------------------')

if show_plots:
    plt.figure(figsize=(24,10))
    plt.subplot(1,3,1); plt.imshow(loaded_script_results_dict['learned_weights_LR']); plt.title('logistic regression', fontsize=24)
    plt.subplot(1,3,2); plt.imshow(loaded_script_results_dict['learned_weights_FF']); plt.title('filter and fire neuron', fontsize=24)
    plt.subplot(1,3,3); plt.imshow(loaded_script_results_dict['learned_weights_IF']); plt.title('integrate and fire neuron', fontsize=24)

script_duration_min = (time.time() - start_time) / 60
print('-----------------------------------')
print('finished script! took %.1f minutes' %(script_duration_min))
print('-----------------------------------')
