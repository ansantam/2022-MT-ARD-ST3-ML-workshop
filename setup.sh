git clone https://github.com/ansantam/2022-MT-ARD-ST3-ML-workshop.git
# sudo apt-get install python3-venv # for linux systems
cd 2022-MT-ARD-ST3-ML-workshop
python3 -m venv ML
source ML/bin/activate
pip3 install -r requirements.txt
jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user
