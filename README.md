Very Minimal Grakn + Pytorch Geometric Example
==============================================

:bangbang: Disclaimer:
These ideas have made it into **TypeDB ML** by now. Please Use that.
https://github.com/vaticle/typedb-ml
https://blog.vaticle.com/link-prediction-knowledge-graph-pytorch-geometric-f35917320806

Copied directly from the diagnosis example in kglib.

This is just example code. The actual repository containing the pytorch geometric "port" of kgcn is in:

https://github.com/jorenretel/grakn-pytorch-geometric

"grakn-pytorch-geometric" is a separate repo because I did not want to mix
library and example code.

There are only a few scripts in here ():

* diagnosis_geometric_and_lightning.py: replicating KGCN in Pytorch Geometric, using Pytoch Lightning to have less verbose training code and many other goodies.
* diagnosis_pytorch_geometric_minimal.py: a very minimal example using the graph convolution from Kipf and Welling 2017. Does not make much sense in terms of machine learning (node and edge types are not embedded properly for example.) but is showing the bare minimum of code.
* diagnosis_pytorch_geometric_gkcn.py: replicating kgcn.
* diagnosis_dgl.py: did not pursue this for now.
* about_this_graph.py:facts about node and edge (attributes) in diagnosis graph used in the examples.


## Environment
Here I used conda to create an environment. To recreate my setup
clone the following 4 repositories and run create the environment
using the environment-local.yml file:

```
git clone https://github.com/graknlabs/kglib.git
git clone https://github.com/jorenretel/grakn-dataloading.git
git clone https://github.com/jorenretel/grakn-pytorch-geometric.git
git clone https://github.com/jorenretel/grakn-pytorch-example.git

cd grakn-pytorch-example

conda env create -f environment-local.yml
```

### Note:
I was not able to install grakn-client and grakn-kglib both from pypi
because of conflicting dependencies. I have kglib "installed" in editable
mode (by cloning from github and adding a setup.py) without the
dependencies like tensorflow installed. You probably have it installed though.
If not this is the setup.py I added to the kglib directory:

```
from setuptools import setup

setup(
    name='kglib',
    version='',
    packages=['kglib'],
    url='',
    license='',
    author='grakn',
    author_email='',
    description=''
)
```


## Run the example

Activate enviroment:
```
conda activate grakn-pytorch-geometric
```

Prepare the grakn database (only once):
```
python populate_database/prepare.py
```

and run the example (with pytorch lightning to  organize the training process):

```
python diagnosis_geometric_and_lightning.py
```

When running this specific example, logs should be written to tensorboard. To look at them:

```
tensorboard --logdir lightning_logs
```


