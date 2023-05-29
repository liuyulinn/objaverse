export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

pip install -r requirements.txt

cd /yulin/unlimited3d

git clone 
python3 distributed.py --input-models-path 
ln -s /mnt /