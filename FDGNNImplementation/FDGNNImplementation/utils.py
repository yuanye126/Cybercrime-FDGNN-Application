from setuptools import setup, find_packages

setup(
    name='gcn_trainer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchmetrics',
        'tqdm',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for training GCN models with regularization',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/gcn_trainer',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
