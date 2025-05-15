from setuptools import setup, find_packages

setup(
    name='ysk_quantlib',
    version='0.1.0',
    author_email='yassine.housseine2@gmail.com',
    description='Custom quantitative finance library for hedging and derivatives pricing',
    author='Yassine Housseine',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib'
    ],
    include_package_data=True,
    zip_safe=False
)

# It's not really a package, it's just a collection of scripts that I use for my own purposes. 

# I'm sharing it here because it might be useful to someone else.

# End of file setup.py