wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
conda create -c conda-forge -n ptychodus tike ptychonn gladier gladier-tools --file ptychodus/requirements-dev.txt
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate ptychodus

python -m pip install funcx-endpoint
cp ptychodus/ptychodus/ptychodusFuncXPolarisConfig.py ~/.funcx/default/config.py
vi ~/.funcx/default/config.py
funcx-endpoint configure/start/restart/stop

See https://docs.globus.org/how-to/guest-collection-share-and-access/

qstat | grep shenke
cd ~/.funcx/default/HighThroughputExecutor/worker_logs/submit_scripts/
tail -f ~/.funcx/default/endpoint.log
