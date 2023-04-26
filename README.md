# Approximate non-linear model predictive control with safety-augmented neural networks
Implementation of safety-augmentation and three numerical benchmark examples (stirtank reactor, quadcopter, and chain mass system). The paper describing the theory can be found [on arXiv](https://arxiv.org/abs/2304.09575).

## Requirements
You need `acados` to run parts of this code.
Please follow [the official acados installation instructions](https://docs.acados.org/installation/index.html). This code was tested with `acados v0.1.9`.

You can install other Python dependencies via pip:
```bash
pip3 install -r examples/requirements.txt
pip3 install -r soeampc/requirements.txt
```

## Numerical Examples
You find the numerical examples from the paper in the `examples` folder. Each example has it's own `README.md` file with instructions how to run them:
- [Two state stir tank reactor](examples/stirtank/README.md)
- [Ten state quadcopter](examples/quadcopter/README.md)
- [Chain mass system](examples/chain_mass/README.md)

## Downloading precomputed datasets and pretrained NNs
You can download the training and testing datasets used in the paper together with the pretrained model from [Zenodo](https://doi.org/10.5281/zenodo.7846094).

Extract the datasets into the `examples/{system}/datasets/` folder, e.g., for the quadcopter example, you should get an `examples/quadcopter/datasets/quadcopter_N_9600000` folder.

Extract the pretrained neural networks into the `examples/{system}/models/` folder, e.g., for the quadcopter example, you should get an `examples/quadcopter/models/10-200-400-600-600-400-200-30_mu=0.12_20230104-232806` folder.

## Running inside Docker
The Dockerfile in this repository allows you to run the code without installing acados or other Python dependencies natively.
To use the Container, you first have build it by running:
```
docker build -t soeampc .
```
Next, run the container and mount this repository:
```
docker run -it --name soeampc -v $(pwd):/soeampc soeampc bash
```
You can now run all the example commands inside the container.