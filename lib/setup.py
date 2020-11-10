
import setuptools 


setuptools.setup(
      version='1.0',
      description='Python Distribution Utilities',
      author='Greg Ward',
      author_email='gward@python.net',
      url='https://www.python.org/sigs/distutils-sig/',
      name="eq_learner",
      packages=setuptools.find_packages('src'),
      package_dir={'': 'src'},
      install_requires=[
          'numpy','sympy','graphviz','torch', 'sklearn'
      ]
     )