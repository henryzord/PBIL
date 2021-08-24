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

1. Install Java (JRE and JDK) from Oracle. The correct JDK version is **jdk-8u261-linux-x64.tar.gz** 
   
   On Linux (tested on Ubuntu 16.04):

   `tar -xf jdk-8u261-linux-x64.tar.gz`
   
   Then add the following lines to `~/.bashrc`:
   
   ``` 
   export PATH=:~/jdk1.8.0_261/bin:$PATH
   JAVA_HOME=~/jdk1.8.0_261/jre
   ```

2. If on Windows, install Visual Studio 14.0 with Python Tools. Add the path to the Visual Studio installation as an
exception in your Anti-virus software.

3. Create a new conda environment: 
   
   `conda create --name pbil python==3.6.13`
   
4. Activate it: 
   
   `conda activate pbil`
   
5. Install conda libraries: 
   
   * On Linux (tested on Ubuntu 16.04):

   ```bash
   conda install --file installation/conda_linux_libraries.txt -c conda-forge
   ```
   
   * On Windows (tested on Windows 10):   
   
   ```bash
   conda install --file installation/conda_windows_libraries.txt -c conda-forge
   conda install -c alubbock pygraphviz=1.5 graphviz=2.41
   ```

6. Install pip libraries: 
   
   `pip install -r installation/pip_libraries.txt` 
   
7. Replace Weka from `python-weka-wrapper` library with provided Weka (in installation directory). This is needed since 
SimpleCart is not provided with default Weka, and some functionalities are added to the default .jar. Here the .jar is provided,
however the source code is [here](https://github.com/henryzord/WekaCustom/tree/comparative). 
   
  On Linux (tested on Ubuntu 16.04):
  
  ```bash
  cp installation/weka.jar <folder_to_anaconda_installation>/anaconda3/envs/pbil/lib/python3.6/site-packages/weka/lib/
  ```
    
  On Windows (tested on Windows 10):
    
  ```bash
  copy installation\weka.jar <folder_to_anaconda_installation>\Anaconda3\envs\pbil\Lib\site-packages\weka\lib\
  ```

8. Install mPBIL:

   ```bash
   cd PBIL
   conda activate pbil
   pip install .
   ```

9. Unpack datasets 

   On Linux (tested on Ubuntu 16.04):
    
      Install 7zip: 
         
      `apt-get install p7zip-full`
    
      Create folder for datasets: 
    
      `mkdir keel_datasets_10fcv`  
      
      Unpack it: 
    
      `7z x installation/keel_datasets_10fcv.7z -o.`

   ---

    On Windows (tested on Windows 10):
    
      Install 7zip from [here](https://www.7-zip.org/download.html)
      
      Add path to installation to PATH variable: 
      
      `C:\Program Files\7-Zip`
 
      Create folder for datasets: 
 
      `mkdir keel_datasets_10fcv`
      
      Unpack it: 
 
      `7z x installation\keel_datasets_10fcv.7z -o.`
        
10. Create metadata path: `mkdir metadata`

## Running
  
### Running the experiments in the paper

Once installed, you can call the script `main.py` script under mPBIL folderto run the tests that were used in the paper.
A successful call to the algorithm in the command line looks as follows:

```
python main.py --datasets-path keel_datasets_10fcv --datasets-names german --metadata-path metadata 
--resources-path resources --n-jobs 1 --heap-size 4g --n-generations 2 --n-individuals 10 --n-samples 1 
--learning-rate 0.5 --selection-share 0.5 
```

### Running nested cross-validation

You can call the bash script `nestedcv_pbil.sh` for running the nested cross-validation proposed in the 
[EDNEL](https://github.com/henryzord/ednel) algorithm, which is compared to PBIL.

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
