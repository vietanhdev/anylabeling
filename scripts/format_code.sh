find . -type f -name "*.py" \
    -not -name "resources.py" \
    -not -name "./build/*" \
    -not -name "./dist/*" \
    | xargs black -l 79 --experimental-string-processing

find . -type f -name "*.py" \
    -not -name "resources.py" \
    -not -name "./build/*" \
    -not -name "./dist/*" \
    | xargs isort