make clean
#git rm -r _build/html
sphinx-apidoc -f -o . ../python/DeepDeconv
make html
#git commit -m "Suppressed Doc"
#git add _build/html
#git add _build/doctrees/
#git commit -m "Added Doc"

