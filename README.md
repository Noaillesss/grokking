# grokking

This repository contains the code necessary to reproduce [grokking](https://arxiv.org/abs/2201.02177) phenomenom. We modify our codes based on this [repository](https://github.com/danielmamay/grokking).

## Installation

Use Python 3.9 or later. For stability, our experiments work in `python==3.12.7`. Firstly, the `torch` installations need to be tailored to your machine's specific Cuda version. You can get the details in [PyTorch](https://pytorch.org/) website. Then run the following code:
```bash
pip install -r requirements.txt
```

## Experiments

You can use different architectures, different optimizers, different train/test fraction and different random seeds to train the model by setting the arguments. Also you can change the hyperparameters in `config.yaml`. The following is an example:
```bash
python -m main --architecture transformer --optimizer adamw --training_fraction 0.5 --random_seed 42 --length 2
```

Then, you will get two figures saved in `./figures/{architecture_name}` plotting the train/val accuracy/loss.

The arguments parser list is the following:

| Args | Description |
|------|-------------|
| architecture | `str`: The model architecture of the experiments. All the choices is referred to `config.yaml`. Default `transformer`. |
| training_fraction | `float`: The training data fraction. Default 0.5 |
| optimizer | `str`: Use different optimizers or regularization techniques. Default `adamw`. |
| random_seed | `int`: Use different random seeds. Default 42. |
| length | `int`: The length of modular sum operation. Default 2. |

## Source code description

Files in this repository include python scripts to run the experiments and
config files necessary for the script to execute. We listed below
each of them and their description.

| File names | Description |
|------------|-------------|
| `grokking/data.py` | Prepares the training and testing data for modular sum. |
| `grokking/model.py` | Implements different neural network (transformer, MLP, LSTM). |
| `grokking/training.py` | Implements the train and validation steps. |
| `main.py` | Main scripts that loads config and starts the training. |
| `plot_fig1.py` | Plot the steps to validation accuracy > 99% for different architectures and different training data fraction. |
| `plot_fig2.py` | Plot the best validation accuracy for different optimizers and different training data fraction during $10^4$ epochs. |
| `config.yaml` | Stores the hyperparameters we use in our experiments. |

## Supplementary

All the supplementary materials are available at (https://disk.pku.edu.cn/link/AA4A118796F462468381EBA19E78E19BC1)

## References

Code:

* [openai/grok](https://github.com/openai/grok)
* [danielmamay/grokking](https://github.com/danielmamay/grokking)

Paper:

* [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)