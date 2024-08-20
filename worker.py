import argparse
# from train import train
import wandb

from activated_notebook_importer import import_notebook

# Import a notebook
trainer_module = import_notebook('train.ipynb')


def run_worker(sweep_id, project):
    # Initialize a new wandb run
    wandb.agent(sweep_id, function=trainer_module.train, project=project)

def main():
    parser = argparse.ArgumentParser(description="Run a WandB sweep worker")
    parser.add_argument("-s", "--sweep_id", help="The ID of the WandB sweep to run", required=True)
    parser.add_argument("-p", "--project", help="The name of the WandB project", required=True)
    parser.add_argument('-n", ')
    args = parser.parse_args()

    # Login to WandB
    wandb.login()

    # Run the worker with the provided sweep ID and project name
    run_worker(args.sweep_id, args.project)

if __name__ == "__main__":
    main()