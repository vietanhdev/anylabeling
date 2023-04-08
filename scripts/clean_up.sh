find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf
while [ -n "$(find . -depth -type d -empty -print -exec rmdir {} +)" ]; do :; done