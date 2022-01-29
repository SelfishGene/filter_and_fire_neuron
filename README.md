# The Filter and Fire (F&F) Neuron Model
This repo contains the code behind the work "[Multiple Synaptic Contacts combined with Dendritic Filtering enhance Spatio-Temporal Pattern Recognition capabilities of Single Neurons](https://www.biorxiv.org/content/10.1101/2022.01.28.478132v1)"  

## Multiple Synaptic Contacts combined with Dendritic Filtering enhance Spatio-Temporal Pattern Recognition capabilities of Single Neurons  
David Beniaguev, Sapir Shapira, Idan Segev, Michael London

**Abstract**: *A cortical neuron typically makes multiple synaptic contacts on the dendrites of a post-synaptic target neuron. The functional implications of this apparent redundancy are unclear. The dendritic location of a synaptic contact affects the time-course of the somatic post-synaptic potential (PSP) due to dendritic cable filtering. Consequently, a single pre-synaptic axonal spike results with a PSP composed of multiple temporal profiles. Here, we developed a "filter-and-fire" (F&F) neuron model that captures these features and show that the memory capacity of this neuron is threefold larger than that of a leaky integrate-and-fire (I&F) neuron, when trained to emit precisely timed output spikes for specific input patterns. Furthermore, the F&F neuron can learn to recognize spatio-temporal input patterns, e.g., MNIST digits, where the I&F model completely fails. Multiple synaptic contacts between pairs of cortical neurons are therefore an important feature rather than a bug and can serve to reduce axonal wiring requirements.*

<img width="1161" alt="Overview_of_F F_neuron_model" src="https://user-images.githubusercontent.com/11506338/151635189-1e6bfe6f-78a5-4c7e-92a4-0599601697c3.PNG">

## Resources
Open Access version of Paper: [biorxiv.org/content/10.1101/2022.01.28.478132v1](https://www.biorxiv.org/content/10.1101/2022.01.28.478132v1)  
Data required for full replication: [kaggle.com/selfishgene/fiter-and-fire-paper](https://www.kaggle.com/selfishgene/fiter-and-fire-paper)  
Introductory Notebook: [kaggle.com/selfishgene/f-f-introduction-figure-fig-1](https://www.kaggle.com/selfishgene/f-f-introduction-figure-fig-1)  
Notebook with replication of main results 1: [kaggle.com/selfishgene/f-f-capacity-figure-fig-2](https://www.kaggle.com/selfishgene/f-f-capacity-figure-fig-2)  
Notebook with replication of main results 2: [kaggle.com/selfishgene/f-f-mnist-figure-fig-3](https://www.kaggle.com/selfishgene/f-f-mnist-figure-fig-3)  
Notebooks for full replication of all figures: [kaggle.com/selfishgene/fiter-and-fire-paper/code](https://www.kaggle.com/selfishgene/fiter-and-fire-paper/code)  


## Increased capacity of F&F vs I&F 
<img width="1161" alt="Capacity_vs_multiple_contacts_compact" src="https://user-images.githubusercontent.com/11506338/151635194-af23b7d3-bb7a-48c9-aaeb-f05648cd4e64.PNG">

- Use `create_capacity_figure_Fig2.py` to replicate Figure 2 in the manuscript
  - All major parameters are documented inside the file using comments  
  - All necessary files are under the folder `results_data_capacity\`
- Use `FF_vs_IF_capacity_comparison_interactions.py` to recreate all files in `results_data_capacity\`
  - All major parameters are documented inside the file using comments  
  - Use `run_capacity_configs_on_cluster_slurm.py` to send jobs to a slurm cluster


## Single Neurons as Spatio-Temporal Pattern Recognizers
<img width="1003" alt="MNIST_classifying_digit_3_compact" src="https://user-images.githubusercontent.com/11506338/151635198-3b65239f-505c-46e3-8ec1-8ddc7931e52d.PNG">

- Use `create_MNIST_figure_Fig3.py` to replicate Figure 3 in the manuscript
  - All major parameters are documented inside the file using comments  
  - All necessary files are under the folder `results_data_mnist\`. large files are on [the dataset](https://www.kaggle.com/selfishgene/fiter-and-fire-paper) on kaggle
- Use `MNIST_classification_LR_IF_FF_interactions.py` to recreate all files in `results_data_mnist\`
  - All major parameters are documented inside the file using comments  
  - Use `run_mnist_configs_on_cluster_slurm.py` to send jobs to a slurm cluster

## PSPs of a realistic detailed biophysical Layer 5 Cortical Pyramidal Neuron
<img width="1040" alt="L5PC_morphology_PSPs" src="https://user-images.githubusercontent.com/11506338/151635200-a8288feb-0365-4c86-91ad-2d87dcc3e7b8.PNG">

- Visit [this link](https://www.kaggle.com/selfishgene/f-f-l5pc-psps-fig-s2) to replicate Figure S2 in the manuscript
- All necessary simulation data are in the file `sim_results_excitatory.p` in [the dataset](https://www.kaggle.com/selfishgene/fiter-and-fire-paper) on kaggle 


## Acknowledgements
We thank all lab members of the Segev and London Labs for many fruitful discussions and valuable feedback regarding this work.
In particular we would like to thank [Sapir Shapira](https://github.com/ssapir) that skillfully collected all data and created Figure S2 in the paper.


If you use this code or dataset, please cite the following two works:  

1. David Beniaguev, Sapir Shapira, Idan Segev and Michael London. "Multiple Synaptic Contacts combined with Dendritic Filtering enhance Spatio-Temporal Pattern Recognition capabilities of Single Neurons
." bioRxiv 2022.01.28.478132; doi: https://doi.org/10.1101/2022.01.28.478132


