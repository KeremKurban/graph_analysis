from setuptools import setup, find_packages

setup(
    name='graph_analysis',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'graph_analysis = graph_analysis.master:main',
        ],
    },
    install_requires=[
        'numpy',
        'scipy',
        'igraph',
        'pandas',
        'networkx',
        'matplotlib'
    ]
)
