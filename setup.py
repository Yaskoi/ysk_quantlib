from setuptools import setup, find_packages

setup(
    name='ysk_quantlib',
    version='0.1.1',
    description='Quant finance library: pricing, greeks, hedging, stats',
    author='Yassine Housseine',
    author_email='yassine.housseine2@gmail.com',
    url='https://github.com/yassine-housseine/ysk_quantlib',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'statsmodels',
        'arch'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ]
)

# It's not really a package, it's just a collection of scripts that I use for my own purposes. 

# I'm sharing it here because it might be useful to someone else.

# End of file setup.py
