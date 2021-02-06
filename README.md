Very Minimal Grakn + Pytorch Geometric Example
==============================================

Copied directly from the diagnosis example in kglib and importing
as much code as possible from there. 

Discaimer: machine learning makes no sense now. For now the graph convolution
done and loss is not the same one at all. We can change this to be
more like the original graphnets example. However, it was more about
exploring the code and seeing which part of kglib would potentially
be nice to have decoupled from tensorflow and graphnets.

For now there are only two files here:

* diagnosis_*.py (start here. Scripts that create dataloader, model and training loop)
* transforms (some functions to transform the netwokx graph in kglib to
  Pytorch Geometric Data object)

## Environment
Here I used conda to create an environment. To recreate my setup
clone the following 3 repositories and run create the environment
using the environment-local.yml file:

```
git clone https://github.com/graknlabs/kglib.git
git clone https://github.com/jorenretel/grakn-dataloading.git
git clone https://github.com/jorenretel/grakn-pytorch-geometric.git
git clone https://github.com/jorenretel/grakn-pytorch-example.git

cd grakn-pytorch-example

conda env create -f environment-local.yml
```

"grakn-pytorch-geometric" is a separate repo because I did not want to mix
library and example code.

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

Example scripts:

* diagnosis_pytorch_geometric_gcn.py: one of the simplest graph convolutions (Kipf and Welling 2017).
* diagnosis_pytorch_geometric_custom.py: something custom, including edge features, just as a test.
* diagnosis_dgl.py: starting to look into DGL (might be the better library for Heterogeneous Graphs).


