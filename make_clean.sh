# all
find . -maxdepth 3 -type d -name "datasets" -exec rm -rf {} \;
find . -maxdepth 3 -type d -name "c_generated_code" -exec rm -rf {} \;
find . -maxdepth 3 -type d -name "__pycache__" -exec rm -rf {} \;

find . -maxdepth 3 -name "acados_ocp_*.json" -exec rm {} \;
