# An Evolutionary Algorithm for Learning Interpretable Ensembles of Classifiers

This repository contains the source code of the algorithm presented in paper _An Evolutionary Algorithm for Learning 
Interpretable Ensembles of Classifiers_.

The code implements an Estimation of Distribution Algorithm (EDA), a kind of evolutionary algorithm, to optimize 
hyper-parameters of a set of base classifiers, namely J48, CART, PART, RIPPER, and Decision Table.

All base classifiers are Weka implementations. The EDA is implemented in Python.

If you find this paper useful, please cite in your work:

```bibtex
@incollection{cagnini2020pbil,
  title = {An Evolutionary Algorithm for Learning Interpretable Ensembles of Classifiers},
  author = {Henry E. L. Cagnini and Alex A. Freitas and Rodrigo C. Barros},
  year = {2020},	
  publisher = {Springer},	
  pages = {18--33},	
  booktitle = {Intelligent Systems},
  doi = {10.1007/978-3-030-61377-8_2},	
  url = {https://doi.org/10.1007%2F978-3-030-61377-8_2}
}
```

## Installation

This repository uses Anaconda as the default Python. You can download Anaconda 
[here](https://www.anaconda.com/products/individual).

Follow these steps to set up the development environment for the algorithm:

1. Create a new conda environment: `conda create --name pbil`
2. Activate it: `conda activate pbil`
3. Install libraries: `conda install --file installation/conda_libraries.txt -c conda-forge` 
4. Install graphviz:

    i. On Ubuntu 16.04: `apt-get install graphviz libgraphviz-dev pkg-config`
    
    ii. On Windows: `conda install -c alubbock pygraphviz=1.5 graphviz=2.41`, then add path to graphviz installation to PATH variable: `<folder_to_anaconda_installation>/Anaconda3/pkgs/graphviz-2.41-0/Scripts`    
    
5. Install JRE and JDK. The correct JDK version is jdk-8u221-linux-x64.tar.gz. Tutorial available 
[here](https://www.javahelps.com/2017/09/install-oracle-jdk-9-on-linux.html).  
6. Install pip libraries: `pip install -r installation/pip_libraries.txt` (NOTE: this might require installing 
Visual Studio with Python tools on Windows)
7. Replace Weka from python-weka-wrapper library with provided Weka. This is needed since SimpleCart is not provided 
with default Weka. On Weka, simply installing it as an additional package makes it available in the GUI; however the 
wrapper still won't see it.

    i. On Ubuntu:  `cp installation/weka.jar <folder_to_anaconda_installation>/anaconda3/envs/pbil/lib/python3.7/site-packages/weka/lib/`
    ii. On Windows: `copy installation/weka.jar <folder_to_anaconda_installation>/Anaconda3/envs/pbil/Lib/site-packages/weka/lib/`
    
8. Unpack datasets. 

    a. On Ubuntu:
    
        i. Install 7zip: `apt-get install p7zip-full`
        ii. Create folder for datasets: `mkdir keel_datasets_10fcv`  
        iii. Unpack it: `7z x installation/keel_datasets_10fcv.7z -o.`
        
    b. On Windows:
    
        i. Install 7zip: https://www.7-zip.org/download.html
        ii. Add path to installation to PATH variable: `C:\Program Files\7-Zip`
        iii. Create folder for datasets: `mkdir keel_datasets_10fcv`
        iv. Unpack it: `7z x installation\keel_datasets_10fcv.7z -o.`
        
9. Create metadata path: `mkdir metadata`

## Running
  
### Running the experiments in the paper
  
Once installed, you can call the script `main.py` in root folder to run the tests that were used in the paper.
A successful call to the algorithm in the command line looks as follows:

```
python main.py --datasets-path keel_datasets_10fcv --datasets-names german --metadata-path metadata 
--resources-path resources --n-jobs 1 --heap-size 4g --n-generations 2 --n-individuals 10 --n-samples 1 
--learning-rate 0.5 --selection-share 0.5 
```
  
### Just using the algorithm as-is

On the other hand, if you do not want to reproduce the experiments presented in the paper, you can use the algorithm 
as-is:

```python
from mPBIL.pbil.model import PBIL
from utils import read_datasets
from weka.core import jvm

jvm.start(max_heap_size='4g')  # using 4GB of heap size for JVM

train_data = read_datasets('keel_datasets_10fcv\\sonar\\sonar-10-1tra.arff')
test_data = read_datasets('keel_datasets_10fcv\\sonar\\sonar-10-1tst.arff')

pbil = PBIL(resources_path='mPBIL/resources', train_data=train_data, n_generations=2, n_individuals=10)
overall, last = pbil.run(1)
print(overall.predict(test_data))
jvm.stop()
```
