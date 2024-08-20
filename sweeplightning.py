import os
import sys
import pprint
import wandb
from lightning_sdk import Studio, Machine

def setup_wandb_sweep():
    wandb.login()

    sweep_config = {
        'method': 'random',  # grid, random, bayes
        'run_cap': 30,
        'metric': {
            'name': 'loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'optimizer': {
                'values': ['adam', 'sgd']
            },
            'fc_layer_size': {
                'values': [128, 256, 512]
            },
            'dropout': {
                'values': [0.3, 0.4, 0.5]
            },
            'epochs': {
                'value': 1
            },
            'learning_rate': {
                'distribution': 'uniform',
                'min': 0,
                'max': 0.1
            },
            'batch_size': {
                'distribution': 'q_log_uniform_values',
                'q': 8,
                'min': 32,
                'max': 256,
            }
        }
    }

    pprint.pprint(sweep_config)

    projectname = "pytorch-sweeps-demo"
    sweep_id = wandb.sweep(sweep_config, project=projectname)
    print(f"Sweep ID: {sweep_id}")
    return sweep_id, projectname

def run_workers(sweep_id, projectname, num_workers):
    if not os.path.exists('worker.py'):
        print("Error: worker.py not found in the current directory.")
        sys.exit(1)

    # start studio
    s = Studio("hyperparameter search")
    s.start()

    # use the jobs plugin
    jobs_plugin = s.installed_plugins["jobs"]

    for i in range(num_workers):
        cmd = f'python worker.py -s={sweep_id} -p={projectname}'
        print(f"Running command: {cmd}")
        jobs_plugin.run(cmd, name=f"worker-{i}", machine=Machine.A10G)

def main():
    num_workers = 3  # You can adjust this value as needed
    sweep_id, projectname = setup_wandb_sweep()
    run_workers(sweep_id, projectname, num_workers)

if __name__ == "__main__":
    main()