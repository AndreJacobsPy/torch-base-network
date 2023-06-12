from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A base class for easier development of PyTorch models'
LONG_DESCRIPTION = 'A package that makes it easy for developers to create PyTorch models easily'

setup(
    name="torch-base-network",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="Andre Jacobs",
    author_email="andrejacobs365@gmail.com",
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    keywords='PyTorch',
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
    ]
)