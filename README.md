# Easy WandB Hyperparameter Search

This repository demonstrates how to use Weights & Biases (WandB) for hyperparameter optimization in a PyTorch-based machine learning project. It includes a simple MNIST classifier and utilizes WandB's sweep functionality to find optimal hyperparameters. Additionally, it supports running hyperparameter sweeps on Lightning AI.

## Project Structure

- `worker.py`: Runs the WandB agent for hyperparameter search.
- `train.py`: Contains the main training loop and model definition.
- `sweep.py`: Sets up and initiates the WandB sweep, including starting a worker.
- `sweeplightning.py`: Sets up and runs WandB sweep using Lightning AI.
- `train.ipynb`: Jupyter notebook version of the training script.

## Installation

1. Install the required packages:
   ```
   pip install wandb torch torchvision
   ```

2. (Optional) If you want to use Jupyter notebooks for training:
   ```
   pip install activated-notebook-importer
   ```

3. Sign up for a WandB account and login:
   ```
   wandb login
   ```

## Usage

1. Start a hyperparameter sweep and a worker:
   ```
   python sweep.py
   ```

   This command sets up the sweep and starts a worker on your local machine.

2. (Optional) To run additional workers on other machines:
   ```
   python worker.py -s <SWEEP_ID> -p <PROJECT_NAME>
   ```

   Replace `<SWEEP_ID>` with the ID generated by `sweep.py` and `<PROJECT_NAME>` with your WandB project name.

3. To run a sweep using Lightning AI:
   ```
   python sweeplightning.py
   ```

   This will set up the sweep and run multiple workers on Lightning AI's infrastructure. By default, it creates 3 worker jobs, each running on a separate A10G machine. You can adjust the number of workers by modifying the `num_workers` variable in the `main()` function of `sweeplightning.py`.

4. (Optional) To use your local machine alongside Lightning AI workers:
   - After running `sweeplightning.py`, use the generated sweep ID to start a local worker:
     ```
     python worker.py -s <SWEEP_ID> -p <PROJECT_NAME>
     ```
   This allows your local machine to contribute to the hyperparameter search alongside the Lightning AI machines.

## Using Scripts vs Notebooks for Training

This project supports using either a Python script (`train.py`) or a Jupyter notebook (`train.ipynb`) for the training process. The choice between these is controlled in the `worker.py` file:

```python
run_method = 'script'  # Change this to 'notebook' to use train.ipynb
```

When using the notebook method, the project utilizes the Activated Notebook Importer to import the `train.ipynb` file as a Python module. This requires installing the `activated-notebook-importer` package.

The Activated Notebook Importer allows you to directly use your working notebook as the training code. This eliminates the need to export your code to a separate script, reducing code duplication and making it easier to iterate on your model within the notebook environment.

## Lightning AI Configuration

The `sweeplightning.py` script uses Lightning AI's Studio and Jobs plugin to distribute the hyperparameter search across multiple machines:

- It creates a Studio instance named "hyperparameter search".
- It uses the Jobs plugin to spawn multiple worker jobs.
- Each worker runs on an A10G machine (configurable via the `Machine` enum in the script).
- The number of concurrent workers is set to 3 by default but can be easily adjusted.

To modify the number of machines or machine type:

1. Change the `num_workers` variable in the `main()` function to adjust the number of concurrent jobs.
2. Modify the `Machine` enum value in the `jobs_plugin.run()` call to change the machine type for each worker.

## Customization

- Modify `sweep.py` or `sweeplightning.py` to adjust the hyperparameter search space.
- Edit `train.py` or `train.ipynb` to change the model architecture or training process.

## Features

- Supports both script (`train.py`) and notebook (`train.ipynb`) training methods.
- Utilizes WandB for experiment tracking and hyperparameter optimization.
- Implements a simple CNN for MNIST classification.
- Supports running distributed hyperparameter sweeps on Lightning AI infrastructure.
- Configurable number of concurrent workers and machine types on Lightning AI.
- Optional support for using Jupyter notebooks as training modules, reducing code duplication.
- Ability to combine local and cloud resources for hyperparameter search.

## Contributing

Feel free to fork this repository and submit pull requests with improvements or additional features.

## License

[MIT License](https://opensource.org/licenses/MIT)