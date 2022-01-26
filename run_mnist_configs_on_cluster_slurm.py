import os
import time


def mkdir_p(dir_path):
    '''make a directory (dir_path) if it doesn't exist'''
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


script_name    = 'MNIST_classification_LR_IF_FF_interactions.py'
output_log_dir = '/filter_and_fire_neuron/logs/'

# all experiment configs to run (will be form a grid of all combinations, so number of experiments will explode if not careful)
positive_digit_list = [0,1,2,3,4,5,6,7,8,9]
connections_per_axon_list = [1,2,5,10]
temporal_extent_factor_numerator_list = [1,2,3,4,5]
temporal_extent_factor_denumerator_list = [1,2]
release_probability_list = [0.5,1.0]
num_positive_training_samples_list = [16,32,64,128,256,512,1024,2048,4096]

num_random_seeds = 2
start_seed = 123456

partition_argument_str = "-p ss.q,elsc.q"
timelimit_argument_str = "-t 1-18:00:00"
CPU_argument_str = "-c 1"
RAM_argument_str = "--mem 64000"
CPU_exclude_nodes_str = "--exclude=ielsc-60,ielsc-108,ielsc-109"

temp_jobs_dir = os.path.join(output_log_dir, 'temp/')
mkdir_p(temp_jobs_dir)

random_seed = start_seed
for positive_digit in positive_digit_list:
    for connections_per_axon in connections_per_axon_list:
        for temporal_extent_factor_numerator in temporal_extent_factor_numerator_list:
            for temporal_extent_factor_denumerator in temporal_extent_factor_denumerator_list:
                for release_probability in release_probability_list:
                    for num_positive_training_samples in num_positive_training_samples_list:
                        for exp_index in range(num_random_seeds):
                            random_seed = random_seed + 1

                            # job and log names
                            digit_multconn_str = 'digit_%d_mult_connections_%d' %(positive_digit, connections_per_axon)
                            job_name = '%s_%s_randseed_%d' %(script_name[:-3], digit_multconn_str, random_seed)
                            log_filename = os.path.join(output_log_dir, "%s.log" %(job_name))
                            job_filename = os.path.join(temp_jobs_dir , "%s.job" %(job_name))

                            # write a job file and run it
                            with open(job_filename, 'w') as fh:
                                fh.writelines("#!/bin/bash\n")
                                fh.writelines("#SBATCH --job-name %s\n" %(job_name))
                                fh.writelines("#SBATCH -o %s\n" %(log_filename))
                                fh.writelines("#SBATCH %s\n" %(partition_argument_str))
                                fh.writelines("#SBATCH %s\n" %(timelimit_argument_str))
                                fh.writelines("#SBATCH %s\n" %(CPU_argument_str))
                                fh.writelines("#SBATCH %s\n" %(RAM_argument_str))
                                fh.writelines("#SBATCH %s\n" %(CPU_exclude_nodes_str))
                                fh.writelines("python3.6 -u %s %s %s %s %s %s %s %s\n" %(script_name, random_seed, positive_digit, connections_per_axon,
                                                                                         temporal_extent_factor_numerator, temporal_extent_factor_denumerator,
                                                                                         release_probability, num_positive_training_samples))

                            os.system("sbatch %s" %(job_filename))
                            time.sleep(0.1)
