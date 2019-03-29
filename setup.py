import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="intent_classifier",
    version="0.2.1",
    description="An Intent Classifier For Chatbot",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/aitrek/intent_classifier",
    author="aitrek",
    author_email="aitrek.zh@gmail.com",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "scipy", "PyMySQL", "scikit-learn==0.20.3"]
)
