find . -type f -name "*.py" \
    -not -name "resources.py" \
    -not -name "./build/*" \
    -not -name "./dist/*" \
    | xargs pylint