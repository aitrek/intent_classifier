from setuptools import setup, find_packages

setup(
    name="intent_classifier",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "scipy", "PyMySQL", "scikit-learn==0.20.3"]
)
