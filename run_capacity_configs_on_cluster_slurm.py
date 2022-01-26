import os
import time


def mkdir_p(dir_path):
    '''make a directory (dir_path) if it doesn't exist'''
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


script_name    = 'FF_vs_IF_capacity_comparison_interactions.py'
output_log_dir = '/filter_and_fire_neuron/logs/'

# tiny experiment configs to run (will be form a grid of all combinations, so number of experiments will explode if not careful)
num_axons_list = [100]
use_interaction_terms_list = [False]
interactions_degree_list = [2]

# full experiment (axons, interaction degree)
num_axons_list = [50,100,200,300,400]
use_interaction_terms_list = [False]
interactions_degree_list = [2]

num_random_seeds = 1
start_seed = 123456

partition_argument_str = "-p ss.q,elsc.q"
timelimit_argument_str = "-t 0-22:00:00"
CPU_argument_str = "-c 1"
RAM_argument_str = "--mem 32000"
CPU_exclude_nodes_str = "--exclude=ielsc-58,ielsc-60,ielsc-108,ielsc-109"

temp_jobs_dir = os.path.join(output_log_dir, 'temp/')
mkdir_p(temp_jobs_dir)

random_seed = start_seed
for num_axons in num_axons_list:
    for use_interaction_terms in use_interaction_terms_list:
        for interactions_degree in interactions_degree_list:
            for exp_index in range(num_random_seeds):
                random_seed = random_seed + 1

                # job and log names
                axons_interactions_str = 'num_axons_%d_interactions_%s_degree_%d' %(num_axons, use_interaction_terms, interactions_degree)
                job_name = '%s_%s_randseed_%d' %(script_name[:-3], axons_interactions_str, random_seed)
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
                    fh.writelines("python3.6 -u %s %s %s %s %s\n" %(script_name, random_seed, num_axons, use_interaction_terms, interactions_degree))

                os.system("sbatch %s" %(job_filename))
                time.sleep(0.2)
