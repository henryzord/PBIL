from distutils.core import setup

# PBIL_modulename = Extension('AUTOCVE.AUTOCVE',
#                                sources=['src/main.cpp', "src/AUTOCVE.cpp", "src/grammar.cpp", "src/solution.cpp",
#                                         "src/population.cpp", "src/python_interface.cpp", "src/population_ensemble.cpp",
#                                         "src/utility.cpp"],
#                                include_dirs=[numpy.get_include()],
#                                language='c++',
#                                extra_compile_args=['/std:c++latest' if os.name == 'nt' else '--std=c++11']
#                                # extra_compile_args=['-O0']  # TODO remove when deploying
#                                )

setup(
    name="mPBIL",
    version="1.0",
    packages=["mPBIL", "mPBIL.utils", "mPBIL.pbil"],
    package_dir={'mPBIL.utils': 'mPBIL/utils', 'mPBIL.pbil': 'mPBIL/pbil', 'mPBIL': 'mPBIL'},
    py_modules=[
        "mPBIL.pbil.evaluations", "mPBIL.pbil.generation", "mPBIL.pbil.individuals", "mPBIL.pbil.integration",
        "mPBIL.pbil.model", "mPBIL.pbil.ptypes", "mPBIL.pbil.registry",
    ],
    description="PBIL is an EDA for optimizing interpretable classifiers hyper-parameters",
    # ext_modules=[PBIL_modulename],
    install_requires=['graphviz==0.11.1', 'pygraphviz==1.5', 'javabridge==1.0.18', 'python-weka-wrapper3==0.1.7',
                      'matplotlib>=2.2.3', 'networkx==2.2', 'numpy>=1.16.1', 'pandas>=0.23.4', 'scipy>=1.2.0',
                      'scikit-learn>=0.20.2', 'deap==1.2.2', 'tensorflow==1.13.1', 'tensorboard==1.13.1',
                      'psutil==5.6.2',

    ],
    author="Henry Cagnini",
    author_email="henry.cagnini@gmail.com",
    keywords="AUTO-ML, Machine Learning, Estimation of Distribution Algorithm",
    license="MIT License",
    url="https://github.com/henryzord/PBIL",
    package_data={'mPBIL': ['mPBIL/resources/*']},
    # include_package_data=True,
)

