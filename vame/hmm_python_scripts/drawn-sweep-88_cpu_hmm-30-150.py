# drawn-sweep-88_cpu_hmm-30-150

import vame
config = '/work/wachslab/aredwine3/VAME_working/config_sweep_drawn-sweep-88_cpu_hmm-30-150.yaml'
train_new_model = True
new_latent_vectors = False
run_type = 'cpu'

# Run your program
vame.pose_segmentation(config, train_new_model, new_latent_vectors, run_type)
