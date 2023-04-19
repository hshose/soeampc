# Approximate non-linear model predictive control with safety-augmented neural networks
Implementation of safety-augmentation and three numerical benchmark examples (stirtank reactor, quadcopter, and chain mass system).

Paper describing the theory can be found [here](https://arxiv.org/abs/).

## Requirements
You need an acados installation to run parts of this code.
Please follow [acados installation instructions](https://docs.acados.org/installation/index.html). This code was tested with acados v0.1.9.

Install other Python dependencies via pip:
`
pip3 install -r examples/requirements.txt
pip3 install -r soeampc/requirements.txt
`

## Examples
You find the numerical examples with their own instructions in the `examples` folder:
- [Two state stir tank reactor](examples/stirtank/README.md)
- [Ten state quadcopter](examples/quadcopter/README.md)
- [Chain mass system](examples/chain_mass/README.md)

## Downloading precomputed datasets and pretrained NNs
You can download the training and testing datasets from [here](10.5281/zenodo.7846094).

Extract the datasets into the `examples/{system}/datasets/` folder, e.g., for the quadcopter example, you should get an `examples/quadcopter/datasets/quadcopter_N_9600000` folder.

Extract the NNs into the `examples/{system}/models/` folder, e.g., for the quadcopter example, you should get an `examples/quadcopter/models/10-200-400-600-600-400-200-30_mu=0.12_20230104-232806` folder.

## Running inside Docker
First, build the container by running:
```
docker build -t soeampc .
```
Next, run the container and mount this repository:
```
docker run -it --name soeampc -v $(pwd):/soeampc soeampc bash
```
You can run all the example commands inside the container.