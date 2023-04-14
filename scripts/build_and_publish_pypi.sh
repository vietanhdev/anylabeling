# Build package
python -m build --no-isolation --outdir wheels_dist

# Publish to Pypi
twine upload --skip-existing wheels_dist/*
