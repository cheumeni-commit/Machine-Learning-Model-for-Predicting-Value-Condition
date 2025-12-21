from setuptools import setup, find_packages

with open("./README.md") as fp:
     long_description = fp.read()

with open("./requirements.txt") as fp:
     dependencies = [line.strip() for line in fp.readlines()]


setup(name="Predicting Value Condition",
      version="0.1",
      description="Predicting Value Condition",
      long_description=long_description,
      author="Jean-Michel Cheumeni",
      author_email="cheumenijean@yahoo.fr",
      packages=find_packages(),
      install_requires=dependencies,
)