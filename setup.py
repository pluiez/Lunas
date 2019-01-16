import sys

from setuptools import setup, find_packages

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for thseq.')

with open('requirements.txt') as r:
    requires = [l.strip() for l in r]

setup(
    name='Lunas',
    version='0.2.0',
    author='Seann Zhang',
    author_email='pluiefox@live.com',
    packages=find_packages(),
    url='https://github.com/pluiez/lunas',
    license='LICENSE.txt',
    description='A data processing pipeline and iterator with minimal dependencies '
                'for machine learning.',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',

)
