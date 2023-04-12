# The setup.py is a Python script that is used ensure that the program is intalled correctly
# It includes choices and metadata about the program, such as 
# the package name, version, author, license, minimal dependencies, entry points, data files, and so on.

from setuptools import find_packages, setup
from typing import List


hyphen_e_dot = '-e .'
def get_requirements(file_path: str) -> List[str]:
    """
    this function returns the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines() # readlines() method returns a list containing each line in the file as a list item
        requirements = [req.replace("\n", "") for req in requirements] # replace \n with blank

        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)


setup(
    name='bulding-load-prediction',
    version='0.0.1',
    author='Shuo Li',
    author_email='shuo.li0208@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)

