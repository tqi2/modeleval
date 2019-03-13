from setuptools import setup, find_packages

with open('requirements.txt') as fp:
  install_requires = fp.read()

setup(
    name='modeleval',
    version='0.1.0',
    url='https://github.com/tqi2/model-evaluation',
    license='MIT',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.6',
    author='Tian (Luke) Qi',
    author_email='tqi2@dons.usfca.edu',
    description='A Python 3 library for a easier machine learning model evaluation',
    keywords='machine-learning evaluation',
    zip_safe=False
)