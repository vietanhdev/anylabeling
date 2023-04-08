# Build package
python -m build --no-isolation

# Publish to Pypi
twine upload --skip-existing dist/*