# sudo apt-get install python3-venv # for linux systems
python3 -m venv ML
source ML/bin/activate
pip3 install -r requirements.txt
jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user
