import os
import subprocess
import wandb
import pprint
import sys

def setup_wandb_sweep():
    wandb.login()

    sweep_config = {
        'method': 'random',  # grid, random, bayes
        'run_cap': 2,
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

def run_worker(sweep_id, projectname):
    if not os.path.exists('worker.py'):
        print("Error: worker.py not found in the current directory.")
        sys.exit(1)

    cmd = ['python', 'worker.py', f'-s={sweep_id}', f'-p={projectname}']
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running worker: {e}")
        sys.exit(1)

def main():
    sweep_id, projectname = setup_wandb_sweep()

    num_workers = 1
    for i in range(num_workers):
        print(f"Starting worker {i+1}/{num_workers}")
        run_worker(sweep_id, projectname)

if __name__ == "__main__":
    main()