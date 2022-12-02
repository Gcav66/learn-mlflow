from setuptools import find_packages, setup

setup(
    name="dagster_wine_example",
    packages=find_packages(exclude=["dagster_wine_example_tests"]),
    install_requires=[
        "dagster",
        "mlflow",
        "pandas",
        "numpy",
        "scikit-learn"
    ],
    extras_require={"dev": ["dagit", "pytest"]},
)
