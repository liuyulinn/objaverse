export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

cd /mnt/data
git clone https://github.com/liuyulinn/objaverse.git

cd objaverse
pip install -r requirements.txt

cp /rclip3d/final/split/lvis_final.json lvis_final.json
ln -s /datasets-slow1/Objaverse/rawdata/hf-objaverse-v1/glbs data

blenderproc run objaverse1.py --object-path "data/000-000/000074a334c541878360457c672b6c2e.glb"
#blenderproc run objaverse1.py --object-path "000074a334c541878360457c672b6c2e.glb"

#"data/000-000/000074a334c541878360457c672b6c2e.glb"

python3 run.py --input-models-path lvis_final.json --start 331
