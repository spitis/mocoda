# MoCoDA

This is code for **MoCoDA: Model-based Counterfactual Data Augmentation**, by Silviu Pitis, Elliot Creager, Ajay Mandlekar, Animesh Garg, published in proceedings of NeurIPS 2022.

Toy Navigation Offline Data is auto generated by the script. Fetch Sweep Offline Data: https://drive.google.com/file/d/1beDmzn2w7yn0MHE7FlW_hJO02rWSJGWw/view?usp=sharing

First run the `augmented_offline_X.py` script, then the corresponding `batchrl_X.py` script. See `run_toy_experiments.sh` for commands for the navigation environment. 

For the batchrl scripts, you need to have mrl on your PATH: https://github.com/spitis/mrl (commit 249d652d07bbb5ecc63c2fd37ebd6ec05f4c9607 should work).

Python 3.9 with torch 1.10.1 and numpy 1.20.3 works, but probably other versions do too. 