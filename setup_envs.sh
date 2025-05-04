unset CONDA_PREFIX
cd .. 
python -m venv denv
source denv/bin/activate
cd diffusion_constrain/src/rust_dfa
maturin develop
cd ../../
pip install -r requirements.txt
pip install transformers==4.38.2 
deactivate 


# unset CONDA_PREFIX
# cd ..
# python -m venv denv2
# source denv2/bin/activate
# cd diffusion_constrain/src/rust_dfa
# maturin develop
# cd ../../
# pip install -r requirements.txt
# pip install transformers==4.46.2
# deactivate 

# cd ..
# python -m venv denv3
# source denv3/bin/activate
# cd diffusion_constrain/src/rust_dfa
# maturin develop
# cd ../../
# pip install -r requirements.txt
# pip install transformers==4.44.2
# deactivate 


