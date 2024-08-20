# Easy WandB Hyperparameter Search

This repository contains a demonstration of how to set up and perform hyperparameter searches using Weights & Biases (WandB) on a PyTorch MNIST classifier. It serves as an example of how to integrate WandB for hyperparameter tuning in your own projects.

## Project Structure

- `worker.py`: Manages the WandB agent and runs the training process.
- `train.py`: Contains the main training logic and model definition.
- `sweep.py`: Sets up and initiates the WandB sweep.
- `train.ipynb`: Jupyter notebook version of the training script.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- wandb
- activated-notebook-importer (optional, for notebook support)

Install the required packages using:

```
pip install torch torchvision wandb
```

If you plan to use the Activated Notebook Importer feature, also install:

```
pip install activated-notebook-importer
```

## Usage

1. Set up your WandB account and log in:

```
wandb login
```

2. Run the sweep:

```
python sweep.py
```

This script will:
- Set up a WandB sweep with predefined hyperparameters
- Start a worker to run the sweep

3. Monitor your runs on the WandB dashboard.

## Customization

- Modify `sweep.py` to change the hyperparameters and their ranges.
- Adjust the model architecture or training process in `train.py`.
- Use `train.ipynb` for interactive development and testing.

## Lightning Studio Jobs API Integration

This demo can be easily adapted to work with the Lightning Studio Jobs API. For more information on how to integrate with Lightning Studio, please refer to the [Lightning AI documentation](https://lightning.ai/docs/overview/studios/jobs).

## Activated Notebook Importer

This project demonstrates the use of [Activated Notebook Importer](https://github.com/Activated-AI/activated-notebook-importer). This allows you to keep your code in a Jupyter notebook instead of porting it to a script for hyperparameter search. 

To use this feature:
1. Ensure you have installed the package: `pip install activated-notebook-importer`
2. Keep your training code in `train.ipynb`
3. Set `run_method = 'notebook'` in `worker.py`

## Notes

- The current setup uses a subset of the MNIST dataset for faster iteration. Modify the `build_dataset` function in `train.py` to use the full dataset.
- The `worker.py` script supports running from both `.py` files and Jupyter notebooks. Set the `run_method` variable to 'script' or 'notebook' as needed.

## Contributing

Feel free to open issues or submit pull requests for any improvements or bug fixes.

## License

This project is open-source and available under the MIT License.