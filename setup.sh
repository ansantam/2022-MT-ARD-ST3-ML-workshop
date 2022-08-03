ACCENT_START="\033[0;35m"
ACCENT_END="\033[0m"
OUT_PREFIX="${ACCENT_START}[WORKSHOP SETUP]${ACCENT_END}"

# Check if there is a correctly installed Anaconda available, otherwise install our own
if [ ! conda --version &> /dev/null ]; then
    echo "${OUT_PREFIX} Could not find Anaconda -> Installing our own ..."
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge-$(uname)-$(uname -m).sh
else
    echo "${OUT_PREFIX} Detected existing Anaconda installation"
fi

# Check if the workshop environment is setup, set it up if not, otherwise just activate it
# if [ ! ${ conda env list | grep mt-ard-st3-ml-workshop > /dev/null; $? } ]; then
if ! { conda env list | grep "mt-ard-st3-ml-workshop"; } >/dev/null 2>&1; then
    echo "${OUT_PREFIX} Creating conda environment ..."
    conda env create -f environment.yaml

    echo "${OUT_PREFIX} Activating environment ..."
    conda activate mt-ard-st3-ml-workshop

    echo "${OUT_PREFIX} Installing Jupyter extensions ..."
    jupyter contrib nbextension install --user
    jupyter nbextensions_configurator enable --user
else
    echo "${OUT_PREFIX} Found workshop conda environment -> Activating environment ..."
    conda activate mt-ard-st3-ml-workshop
fi

echo "${OUT_PREFIX} Finished setting up the workshop environment! 🎉"
