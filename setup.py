from setuptools import setup, find_packages

INSTALL_DEPS = ['numpy',
                'scipy',
                'matplotlib',
                'torch'
               ]
TEST_DEPS = ['pytest']
DEV_DEPS = []

setup(name='mastermind',
      version='0.0.0',
      url='https://github.com/gilsh2/final_ai_project',
      license='MIT',
      author='Gil Lapid Shafriri',
      author_email='gilsh@microsoft.com',
      description='Various strategies for playing the mastermind board game.',
      packages=find_packages(exclude=['tests', 'Images', 'sounds']),
      long_description=open('README.md').read(),
      zip_safe=False,
      install_requires=INSTALL_DEPS,

      python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',

      # List additional groups of dependencies here (e.g. development
      # dependencies). You can install these using the following syntax,
      # for example:
      # $ pip install -e .[dev,test]
      extras_require={
          'dev': DEV_DEPS,
          'test': TEST_DEPS,
      },
     )
