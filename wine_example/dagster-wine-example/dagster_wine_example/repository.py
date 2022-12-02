from dagster import load_assets_from_package_module, repository

from dagster_wine_example import assets


@repository
def dagster_wine_example():
    return [load_assets_from_package_module(assets)]
