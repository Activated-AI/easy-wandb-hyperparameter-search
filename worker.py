import argparse
import train
import wandb


run_method = 'script'

def run_worker(sweep_id, project):
    train_function = train.train

    if run_method == 'notebook':  
        from activated_notebook_importer import import_notebook

        # Import the train function from the notebook.
        trainer_module = import_notebook('train.ipynb')
        train_function = trainer_module.train

    wandb.agent(sweep_id, function=train_function, project=project)


def main():
    parser = argparse.ArgumentParser(description="Run a WandB sweep worker")
    parser.add_argument("-s", "--sweep_id", help="The ID of the WandB sweep to run", required=True)
    parser.add_argument("-p", "--project", help="The name of the WandB project", required=True)
    args = parser.parse_args()

    # Login to WandB
    wandb.login()

    # Run the worker with the provided sweep ID and project name
    run_worker(args.sweep_id, args.project)

if __name__ == "__main__":
    main()