import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="accurity-automated-mappings",
    version="0.1",
    description="Accurity Data Science Project",
    author="Miro Mlynarik",
    author_email="miroslav.mlynarik@simplity.eu",
    url="https://gitlab.internal.cloudity.lan/ps/accurity",
    packages=find_packages(),
    long_description=read('README.md'),
)
