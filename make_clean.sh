echo
read -p "Do you want to remove ALL generated files [yN]? " -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Removing:"
    find . -maxdepth 3 -type d -name "c_generated_code" -printf '\t%p\n' -exec rm -rf {} \;
    find . -maxdepth 3 -type d -name "__pycache__" -printf '\t%p\n' -exec rm -rf {} \;
    find . -maxdepth 3 -name "acados_ocp_*.json" -printf '\t%p\n' -exec rm {} \;
fi

echo
read -p "Do you want to remove ALL datasets [yN]? " -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Removing:"
    find . -maxdepth 3 -type d -name "datasets" -printf '\t%p\n' -exec rm -rf {} \;
fi
