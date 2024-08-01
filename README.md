# PySATL_NMVM_Module

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![GitHub License](https://img.shields.io/github/license/Engelsgeduld/PySATL_NMVM_Module?style=for-the-badge)

![Workflow](https://github.com/Engelsgeduld/PySATL_NMVM_Module/actions/workflows/main.yaml/badge.svg)


Python package for statistical analysis of Normal Mean or/and Variance Mixtures.

#### Contributors:
* Andreev Sergey - [andreev-sergej](https://github.com/andreev-sergej)
* Knyazev Dmitrii - [Engelsgeduld](https://github.com/Engelsgeduld)

# Installation
For usage:
```
pip install requirements.txt
```
For contributing:
```
pip install requirements.dev.txt
```

# Requirements
Usage:
* `numpy~=1.26.4`
* `scipy~=1.13.1`
* `matplotlib~=3.8.4`
* `numba~=0.59.0`
* `mpmath~=1.3.0`

Development:
* `mypy~=1.10.0`
* `black~=24.4.2`
* `isort~=5.13.2`
* `mpmath~=1.3.0`
* `pytest~=7.4.4`



# Theoretical basis

#### Normal Mean Mixture
$Y_{NMM}$ is called Normal Mean Mixture (NMM) if it can be represented in classical form with parameters $\alpha$, $\beta$, $\gamma$ $\in \R$ and mixing density function $g(x)$ as:\
\
$Y_{NMM} = \alpha + \beta \cdot \xi + \gamma \cdot N$, where $N \sim \mathcal{N}(0, 1), \xi \sim g(x)$ and $\xi \perp N$

#### Normal Variance Mixture
$Y_{NVM}$ is called Normal Variance Mixture (NVM) if it can be represented in classical form with parameters $\alpha$, $\beta$, $\gamma$ $\in \R$ and mixing density function $g(x)$ as:\
\
$Y_{NVM} = \alpha + \gamma \cdot \sqrt{\xi} \cdot N$, where $N \sim \mathcal{N}(0, 1), \xi \sim g(x)$ and $\xi \perp N$

#### Normal Mean-Variance Mixture 
$Y_{NMVM}$ is called Normal Mean-Variance Mixture (NMVM) if it can be represented in classical form with parameters $\alpha$, $\beta$, $\gamma$ $\in \R$ and mixing density function $g(x)$ as:\
\
$Y_{NMVM}$ = $\alpha  + \beta \cdot \xi +  \gamma \cdot \sqrt{\xi} \cdot N$, where $N \sim \mathcal{N}(0, 1), \xi \sim g(x)$ and $\xi \perp N$

Problem with classical representation is that it is not unique. So usually it is more convenient to make some substitutions and get canonical representation of mixture.

### Canonical representations
$Y_{NMM}(\xi, \sigma) = \xi + \sigma \cdot N$ \
$Y_{NVM}(\xi, \alpha) = \alpha + \sqrt{\xi} \cdot N$ \
$Y_{NMVM}(\xi, \alpha, \mu) = \alpha + \mu \cdot \xi + \sqrt{\xi} \cdot N$\
where $\alpha, \mu, \sigma \in \R$; $N \sim \mathcal{N}(0, 1); \xi \sim g(x)$

# Mixture sample generation:
Classes are implemented in directory [**src.mixtures**](https://github.com/Engelsgeduld/PySATL_NMVM_Module/tree/main/src/mixtures) .

There are three classes of mixtures:
* [NormalMeanMixtures](https://github.com/Engelsgeduld/PySATL_NMVM_Module/blob/main/src/mixtures/nm_mixture.py)
* [NormalVarianceMixtures](https://github.com/Engelsgeduld/PySATL_NMVM_Module/blob/main/src/mixtures/nv_mixture.py)
* [NormalMeanVarianceMixtures](https://github.com/Engelsgeduld/PySATL_NMVM_Module/blob/main/src/mixtures/nmv_mixture.py)

One can select mixture representation (classical or canonical) by specifying *mixture_form* parameter.
## Usage
Example of mixture object creation:
```Python
from src.mixtures.nm_mixture import *
from scipy.stats import norm

mixture = NormalMeanVarianceMixtures("classical", alpha=1.2, beta=2.2, gamma=1, distribution=norm)
```


Example of mixture sample generation:
```Python
from src.generators.nm_generator import NMGenerator
from src.mixtures.nmv_mixture import *
from scipy.stats import expon

generator = NMGenerator()
mixture = NormalMeanMixtures("classical", alpha=1.6, beta=4, gamma=1, distribution=expon)
sample = generator.classical_generate(mixture, 5000)
```

Histogram of *sample* values:
![usage_example_1](images/usage_example_1.svg)

# Calculation of standard statistical characteristics of mixture:
* Calculation of pobability density function
* Cumulative distribution function
* Moments
* Standard deviation and expected value.
-notebook_link-

# Parameter estimation algorithms for Normal Mean or/and Variance Mixtures:
* ## Semiparametric estimation of parameter $\mu$ in Normal Mean-Variance mixture.
    

* ## Estimation of mixing density g(x) for given $\mu$ in Normal Mean-Variance mixture.