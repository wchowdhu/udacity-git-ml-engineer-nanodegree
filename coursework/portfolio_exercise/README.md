# Portfolio Exercise 

Create a custom Python package to compute the Gaussian and Binomial distributions and upload it to [Pypi](https://pypi.org/) package repository. This directory contains:

* A `setup.py` file, which is required in order to `pip install` the package
* A folder called `wchowdhu_distributions_v2` which is the name of the Python package

# Install
This exercise requires Python 3.6 and the following Python libraries installed   
  - [matplotlib](https://matplotlib.org/)

# Run

##### To upload a new package to [PyPi](https://pypi.org/):
```
cd portfolio_excercise
python setup.py sdist
pip install twine

twine upload dist/*
pip install wchowdhu_distributions_v2
```
*Note: The name of the package should be unique, as `wchowdhu_distributions_v2` already exists in the PyPi repository, change the name of the package to something else before running the above commands on the command line. That also means changing the information in `setup.py` and the folder name.*

##### To import the package:
Since the package is already uploaded to the PyPi repository, to import and use the package run the following commands in your terminal:
```
pip install wchowdhu_distributions_v2
```

Then start the Python interpreter from the terminal typing:
```
python
```

Then within the Python interpreter, you can use the package:
```
from wchowdhu_distributions_v2 import Gaussian
gaussian_one = Gaussian(25, 2)
gaussian_one.mean
gaussian_one + gaussian_one
```
