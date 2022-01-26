import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
import pickle

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

#%% script params

start_time = time.time()

try:
    print('----------------------------')
    random_seed = int(sys.argv[1])
    num_axons = int(sys.argv[2])
    use_interaction_terms = (sys.argv[3] == 'True')
    interactions_degree = int(sys.argv[4])
    print('"random_seed" selected by user - %d' %(random_seed))
    print('"num_axons" selected by user - %d' %(num_axons))
    print('"use_interaction_terms" selected by user - %s' %(use_interaction_terms))
    print('"interactions_degree" selected by user - %d' %(interactions_degree))

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


# input parameters
if determine_internally:
    num_axons = 100
    use_interaction_terms = True
    interactions_degree  = 2


# experiment parameters
stimulus_duration_sec = 90
min_time_between_spikes_ms = 110

instantanious_input_spike_probability = 0.004
almost_prefect_accuracy_AUC_threshold = 0.99

# full
model_type_list = ['F&F', 'I&F']
connections_per_axon_list = [1,3,5,10,15]
req_num_output_spikes_list = np.linspace(5,200,40).astype(int)
req_num_output_spikes_list = np.logspace(np.log10(11),np.log10(200),40).astype(int)
random_iterations_list = np.linspace(1,10,10).astype(int)

# neuron model parameters
v_reset     = -80
v_threshold = -55
current_to_voltage_mult_factor = 3
refreactory_time_constant = 15

tau_rise_range_FF  = [ 1,16]
tau_decay_range_FF = [ 8,30]
tau_rise_range_IF  = [ 1, 1]
tau_decay_range_IF = [30,30]

show_plots = True
show_plots = False

data_folder = '/filter_and_fire_neuron/results_data_capacity/'

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


def apply_dendritic_interactions(normlized_synaptic_currents, interactions_degree=2, non_interaction_fraction=0.2):
    output_normlized_synaptic_currents = normlized_synaptic_currents.copy()

    # apply d times random interactions
    for degree in range(interactions_degree - 1):
        random_permutation = np.random.permutation(normlized_synaptic_currents.shape[0])
        output_normlized_synaptic_currents = output_normlized_synaptic_currents * normlized_synaptic_currents[random_permutation]

    # keep some fraction of only individual interactions
    random_permutation = np.random.permutation(normlized_synaptic_currents.shape[0])[:int(normlized_synaptic_currents.shape[0] * non_interaction_fraction)]
    output_normlized_synaptic_currents[random_permutation] = normlized_synaptic_currents[random_permutation]

    return output_normlized_synaptic_currents


#%% Fit I&F and F&F several times with variable multiple connections and different requested number of spikes

stimulus_duration_ms = 1000 * stimulus_duration_sec

all_results_dicts = []
print('------------------------------------------------------------------------------------------')
for model_type in model_type_list:
    for connections_per_axon in connections_per_axon_list:
        for requested_number_of_output_spikes in req_num_output_spikes_list:

            #print('------------------')
            #print('model type = "%s"' %(model_type))
            #print('number of connections per axon = %d' %(connections_per_axon))
            #print('number of requested output spikes = %d' %(requested_number_of_output_spikes))
            #print('------------------')

            full_AUC_list = []
            result_entry_dict = {}

            result_entry_dict['model type'] = model_type
            result_entry_dict['connections per axon'] = connections_per_axon
            result_entry_dict['num output spikes'] = requested_number_of_output_spikes

            for random_iter in random_iterations_list:

                # neuron model parameters
                num_synapses = connections_per_axon * num_axons

                # synapse non-learnable parameters
                if model_type == 'F&F':
                    tau_rise_range  = tau_rise_range_FF
                    tau_decay_range = tau_decay_range_FF
                elif model_type == 'I&F':
                    tau_rise_range  = tau_rise_range_IF
                    tau_decay_range = tau_decay_range_IF

                tau_rise_vec  = np.random.uniform(low=tau_rise_range[0] , high=tau_rise_range[1] , size=(num_synapses, 1))
                tau_decay_vec = np.random.uniform(low=tau_decay_range[0], high=tau_decay_range[1], size=(num_synapses, 1))

                # synapse learnable parameters
                synaptic_weights_vec = np.random.normal(size=(num_synapses, 1))

                # generate sample input
                axons_input_spikes = np.random.rand(num_axons, stimulus_duration_ms) < instantanious_input_spike_probability
                presynaptic_input_spikes = np.kron(np.ones((connections_per_axon,1)), axons_input_spikes)
                assert presynaptic_input_spikes.shape[0] == num_synapses, 'number of synapses doesnt match the number of presynaptic inputs'

                # generate desired pattern of output spikes

                desired_output_spike_times = min_time_between_spikes_ms * np.random.randint(int(stimulus_duration_ms / min_time_between_spikes_ms), size=requested_number_of_output_spikes)
                desired_output_spike_times = np.sort(np.unique(desired_output_spike_times))

                desired_output_spikes = np.zeros((stimulus_duration_ms,))
                desired_output_spikes[desired_output_spike_times] = 1.0

                # simulate cell with normlized currents
                local_normlized_currents, soma_voltage, output_spike_times_in_ms = simulate_filter_and_fire_cell_training(presynaptic_input_spikes,
                                                                                                                          synaptic_weights_vec, tau_rise_vec, tau_decay_vec,
                                                                                                                          refreactory_time_constant=refreactory_time_constant,
                                                                                                                          v_reset=v_reset, v_threshold=v_threshold,
                                                                                                                          current_to_voltage_mult_factor=current_to_voltage_mult_factor)


                # normlized_synaptic_currents = local_normlized_currents
                # plt.close('all')
                # plt.figure(figsize=(16,10))
                # plt.subplot(2,1,1); plt.imshow(normlized_synaptic_currents[:,:1000])
                # plt.subplot(2,1,2); plt.imshow(apply_dendritic_interactions(normlized_synaptic_currents, interactions_degree=2, non_interaction_fraction=0.001)[:,:1000])

                if use_interaction_terms:
                    non_interaction_fraction = num_axons / local_normlized_currents.shape[0]
                    local_normlized_currents = apply_dendritic_interactions(local_normlized_currents, interactions_degree=interactions_degree, non_interaction_fraction=non_interaction_fraction)

                # fit linear model to local currents
                logistic_reg_model = linear_model.LogisticRegression(C=100000, fit_intercept=False, penalty='l2', solver='lbfgs', verbose=False)

                spike_safety_range_ms = 5
                negative_subsampling_fraction = 0.5

                # use local currents as "features" and fit a linear model to the data
                X, y = prepare_training_dataset(local_normlized_currents, desired_output_spikes,
                                                spike_safety_range_ms=spike_safety_range_ms,
                                                negative_subsampling_fraction=negative_subsampling_fraction)
                #print('number of extracted data points for training = %d (%.2f%s positive class)' %(X.shape[0], 100 * y.mean(),'%'))

                # fit model
                logistic_reg_model.fit(X,y)

                # predict and calculate AUC on train data
                y_hat = logistic_reg_model.predict_proba(X)[:,1]
                train_AUC = roc_auc_score(y, y_hat)

                # predict and calculate AUC on full trace
                fitted_output_spike_prob = logistic_reg_model.predict_proba(local_normlized_currents.T)[:,1]
                full_AUC = roc_auc_score(desired_output_spikes, fitted_output_spike_prob)

                # print progress and save result
                #print('random iteration %d: (train AUC, full trace AUC) = (%.5f, %.5f)' %(random_iter, train_AUC, full_AUC))
                full_AUC_list.append(full_AUC)

            # convert to array:
            full_AUC_vec = np.array(full_AUC_list)
            mean_AUC = full_AUC_vec.mean()
            std_AUC = full_AUC_vec.std()

            result_entry_dict['full_AUC_vec'] = full_AUC_vec
            result_entry_dict['mean_AUC'] = mean_AUC
            result_entry_dict['std_AUC'] = std_AUC

            # append
            all_results_dicts.append(result_entry_dict)
            print(result_entry_dict)
            print('------------------------------------------------------------------------------------------')


#%% process results dict for showing capacity results

all_results_curves = {}


for model_type in model_type_list:
    for connections_per_axon in connections_per_axon_list:

        num_spikes_list = []
        mean_AUC_list = []
        std_AUC_list = []

        for res_dict in all_results_dicts:
            if model_type == res_dict['model type'] and connections_per_axon == res_dict['connections per axon']:
                num_spikes_list.append(res_dict['num output spikes'])
                mean_AUC_list.append(res_dict['mean_AUC'])
                std_AUC_list.append(res_dict['std_AUC'])

        num_spikes = np.array(num_spikes_list)
        mean_AUC = np.array(mean_AUC_list)
        std_AUC = np.array(std_AUC_list)

        sorted_inds = np.argsort(num_spikes)

        num_spikes = num_spikes[sorted_inds]
        mean_AUC = mean_AUC[sorted_inds]
        std_AUC = std_AUC[sorted_inds]

        try:
            almost_perfect_accuracy_inds = np.where(mean_AUC > almost_prefect_accuracy_AUC_threshold)
            num_almost_perfectly_placed_spikes = num_spikes[almost_perfect_accuracy_inds[0][-1]]

            dict_key = '%s, %d connections' %(model_type, connections_per_axon)
            all_results_curves[dict_key] = {}
            all_results_curves[dict_key]['num_spikes'] = num_spikes
            all_results_curves[dict_key]['mean_AUC'] = mean_AUC
            all_results_curves[dict_key]['std_AUC'] = std_AUC
            all_results_curves[dict_key]['model_type'] = model_type
            all_results_curves[dict_key]['connections_per_axon'] = connections_per_axon
            all_results_curves[dict_key]['num_almost_perfectly_placed_spikes'] = num_almost_perfectly_placed_spikes
        except:
            print('something wrong skipping')


#%% Accuracy as function of spikes for various conditions

if show_plots:
    plt.figure(figsize=(30,15))
    for key, value in all_results_curves.items():
        plt.errorbar(value['num_spikes'], value['mean_AUC'], yerr=value['std_AUC'], label=key)
    plt.legend(loc='upper right', fontsize=23, ncol=2)
    plt.title('learning to place precisly timed output spikes (random input, %d sec window, %d axons)' %(stimulus_duration_sec, num_axons), fontsize=24)
    plt.ylabel('accuracy at 1ms precision (AUC)', fontsize=24)
    plt.xlabel('num requested spikes to place', fontsize=24)

#%% Number of precisly placed spikes as function of num multiple connections

processed_res_curves = {}

for model_type in model_type_list:

    connections_per_axon = []
    num_almost_perfectly_placed_spikes = []

    for res_dict in all_results_curves.values():
        if model_type == res_dict['model_type']:
            connections_per_axon.append(res_dict['connections_per_axon'])
            num_almost_perfectly_placed_spikes.append(res_dict['num_almost_perfectly_placed_spikes'])

    connections_per_axon = np.array(connections_per_axon)
    num_almost_perfectly_placed_spikes = np.array(num_almost_perfectly_placed_spikes)

    sorted_inds = np.argsort(connections_per_axon)

    connections_per_axon = connections_per_axon[sorted_inds]
    num_almost_perfectly_placed_spikes = num_almost_perfectly_placed_spikes[sorted_inds]

    dict_key = model_type
    processed_res_curves[dict_key] = {}

    processed_res_curves[dict_key]['connections_per_axon'] = connections_per_axon
    processed_res_curves[dict_key]['num_almost_perfectly_placed_spikes'] = num_almost_perfectly_placed_spikes

    print('-----')
    print(model_type)
    print(processed_res_curves[dict_key])
    print('-------------------')


if show_plots:
    plt.figure(figsize=(30,12))
    for key, value in processed_res_curves.items():
        plt.plot(value['connections_per_axon'], value['num_almost_perfectly_placed_spikes'], label=key)
    plt.title('learning to place precisly timed output spikes (random input, %d sec window, %d axons)' %(stimulus_duration_sec, num_axons), fontsize=24)
    plt.xlabel('num multiple connections per axon', fontsize=24)
    plt.ylabel('num almost perfectly placed spikes', fontsize=24)
    plt.legend(loc='upper left', fontsize=23)


#%% Save results for Nicer presentation later on

# collect main script parameters
script_main_params = {}
script_main_params['num_axons']                     = num_axons
script_main_params['stimulus_duration_sec']         = stimulus_duration_sec
script_main_params['min_time_between_spikes_ms']    = min_time_between_spikes_ms
script_main_params['refreactory_time_constant']     = refreactory_time_constant
script_main_params['spike_safety_range_ms']         = spike_safety_range_ms
script_main_params['negative_subsampling_fraction'] = negative_subsampling_fraction
script_main_params['num_random_iter']               = len(random_iterations_list)
script_main_params['use_interaction_terms']         = use_interaction_terms
script_main_params['interactions_degree']           = interactions_degree

script_main_params['almost_prefect_accuracy_AUC_threshold'] = almost_prefect_accuracy_AUC_threshold
script_main_params['instantanious_input_spike_probability'] = instantanious_input_spike_probability
script_main_params['connections_per_axon_list']  = connections_per_axon_list
script_main_params['req_num_output_spikes_list'] = req_num_output_spikes_list
script_main_params['tau_rise_range_FF']  = tau_rise_range_FF
script_main_params['tau_decay_range_FF'] = tau_decay_range_FF
script_main_params['tau_rise_range_IF']  = tau_rise_range_IF
script_main_params['tau_decay_range_IF'] = tau_decay_range_IF
script_main_params['v_reset']     = v_reset
script_main_params['v_threshold'] = v_threshold
script_main_params['current_to_voltage_mult_factor'] = current_to_voltage_mult_factor

# collect main script results
script_results_dict = {}
script_results_dict['script_main_params']   = script_main_params
script_results_dict['all_results_dicts']    = all_results_dicts
script_results_dict['all_results_curves']   = all_results_curves
script_results_dict['processed_res_curves'] = processed_res_curves

# filename
if use_interaction_terms:
    results_filename = 'FF_vs_IF_capacity_comparision_interactions_degree_%d_num_axons_%d__sim_duration_sec_%d__num_mult_conn_%d__rand_rep_%d.pickle' %(interactions_degree, num_axons, stimulus_duration_sec, len(connections_per_axon_list), len(random_iterations_list))
else:
    results_filename = 'FF_vs_IF_capacity_comparision__num_axons_%d__sim_duration_sec_%d__num_mult_conn_%d__rand_rep_%d.pickle' %(num_axons, stimulus_duration_sec, len(connections_per_axon_list), len(random_iterations_list))

# pickle everythin
pickle.dump(script_results_dict, open(data_folder + results_filename, "wb"))

#%% Load the saved pickle just to check it's OK

loaded_script_results_dict = pickle.load(open(data_folder + results_filename, "rb" ))

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

script_duration_min = (time.time() - start_time) / 60
print('-----------------------------------')
print('finished script! took %.1f minutes' %(script_duration_min))
print('-----------------------------------')
