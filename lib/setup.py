
import setuptools 


setuptools.setup(
      version='1.0',
      description='Dataset creation for symbolic expression experiments',
      author='SymposiumOrganization',
      url='https://github.com/SymposiumOrganization/EQLearner',
      name="eqlearner",
      packages=setuptools.find_packages('src'),
      package_dir={'': 'src'},
      install_requires=[
          'numpy','sympy','graphviz','torch', 'sklearn',
          'pytest','nptyping','pandas', 'deprecated', 'tqdm'
      ]
     )