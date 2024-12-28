# grokking

## Installation

Use Python 3.9 or later. For stability, our experiments work in `python==3.12.7`. Firstly, the `torch` installations need to be tailored to your machine's specific Cuda version. You can get the details in [PyTorch](https://pytorch.org/) website. Then run the following code:
```bash
pip install -r requirements.txt
```

## Experiments

You can use different architectures, different optimizers, different train/test fraction and different random seeds to train the model by setting the arguments. Also you can change the hyperparameters in `config.yaml`. The following is an example:
```bash
python -m main --architecture transformer --optimizer adamw --training_fraction 0.5 --random_seed 42
```

Then, you will get two figures saved in `./figures` plotting the train/val accuracy/loss.