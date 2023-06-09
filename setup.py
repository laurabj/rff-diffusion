from os import path

# try:  # for pip >= 10
from pip._internal.req import parse_requirements
from setuptools import find_packages, setup

# except ImportError:  # for pip <= 9.0.3
#     from pip.req import parse_requirements

here = path.abspath(path.dirname(__file__))
install_reqs = parse_requirements(here + "/requirements.txt", session=False)

# Catering for latest pip version
try:
    reqs = [str(ir.req) for ir in install_reqs]
except:
    reqs = [str(ir.requirement) for ir in install_reqs]

git_reqs = [
    # "jaxutils @ git+https://github.com/shreyaspadhy/jaxutils"
]


setup(
    name="rff_diffusions",
    version="0.0.1",
    description="Repository for RFF Diffusions.",
    packages=find_packages(),
    license="MIT",
    install_requires=None,  # git_reqs + reqs,
)
