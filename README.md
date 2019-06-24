# Backpropagation

Implementation of a fully connected neural network from scratch using numpy.

Stratified k-cross validation and SGD, Momentum and Adam optimizers implemented.

Project report for UFRGS INF01017 Aprendizado de Máquina (Machine Learning) – 2019/1 [Relatório.pdf](https://github.com/LucasAlegre/backpropagation/blob/master/Relat%C3%B3rio.pdf).

## Install Requirements

```
pip3 install -r requirements.txt
```

## Run

```
usage: python3 backpropagation.py [-h] [-s SEED] [-d DATA] [-c CLASS_COLUMN]
                          [-sep SEP] [-k NUM_FOLDS] [-e EPOCHS]
                          [-mb BATCH_SIZE] [-drop DROP [DROP ...]] -nn NN
                          [NN ...] [-w WEIGHTS] [-alpha ALPHA] [-beta BETA]
                          [-regularization REGULARIZATION] [-numerical]
                          [-opt OPT] [-log]

Backpropagation - Aprendizado de Máquina 2019/1 UFRGS

optional arguments:
  -h, --help            show this help message and exit
  -s SEED               The random seed. (default: None)
  -d DATA               The dataset .csv file. (default: datasets/wine.csv)
  -c CLASS_COLUMN       The column of the .csv to be predicted. (default:
                        class)
  -sep SEP              .csv separator. (default: ,)
  -k NUM_FOLDS          The number of folds used on cross validation.
                        (default: 10)
  -e EPOCHS             Amount of epochs for training the neural network.
                        (default: 100)
  -mb BATCH_SIZE        Mini-batch size used for training. (default: None)
  -drop DROP [DROP ...]
                        Columns to drop from .csv. (default: [])
  -nn NN [NN ...]       Neural Network structure. (default: None)
  -w WEIGHTS            Initial weights. (default: None)
  -alpha ALPHA          Learning rate. (default: 0.001)
  -beta BETA            Efective direction rate used on the Momentum Method.
                        (default: 0.9)
  -regularization REGULARIZATION
                        Regularization factor. (default: 0.0)
  -numerical            Calculate the gradients numerically. (default: False)
  -opt OPT              Optimizer [SGD, Momentum, Adam]. (default: SGD)
  -log                  Generate log file. (default: False)

```

## Authors

* **Lucas Alegre** - [LucasAlegre](https://github.com/LucasAlegre)
* **Bruno Santana** - [bsmlima](https://github.com/bsmlima)
* **Pedro Perrone** - [pedroperrone](https://github.com/pedroperrone)



