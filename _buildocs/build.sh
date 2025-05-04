cp index.rst ./source/index.rst
sphinx-apidoc -f -o ./source ../src
sphinx-build -b html -c . ./source ./built
touch ./built/.nojekyll
mv ./built ../docs