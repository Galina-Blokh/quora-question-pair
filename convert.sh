for i in *.ipynb **/*.ipynb **/**/*.ipynb; do 
    echo "$i"
    jupyter nbconvert  "$i" "$i"
    jupyter nbconvert --to python "$i" "$i"
done
