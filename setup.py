from setuptools import setup, find_packages

setup(
    name='modeleval',
    version='0.1.6',
    url='https://github.com/tqi2/model-evaluation',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy','pandas','sklearn','matplotlib','IPython'],
    python_requires='>=3.6',
    author='Tian (Luke) Qi',
    author_email='tqi2@dons.usfca.edu',
    description='A Python 3 library for a easier machine learning model evaluation',
    keywords='machine-learning evaluation',
    zip_safe=False
)