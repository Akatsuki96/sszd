python3 synthetic_comparison.py 'sszd' 50 100000 --sszd-l 50 --sszd-dirtype "orthogonal" --out-file "sszd_qr_50" --reps 10 --target "plnonconv" --sszd-alpha 0.5  &
python3 synthetic_comparison.py 'sszd' 50 100000 --sszd-l 25 --sszd-dirtype "orthogonal" --out-file "sszd_qr_25" --reps 10 --target "plnonconv" --sszd-alpha 0.5 
wait
python3 synthetic_comparison.py 'sszd' 50 100000 --sszd-l 1 --sszd-dirtype "gaussian" --out-file "fd_gauss_1" --reps 10 --target "plnonconv" --sszd-alpha 5e-2 &
python3 synthetic_comparison.py 'sszd' 50 100000 --sszd-l 1 --sszd-dirtype "spherical" --out-file "fd_sp_1" --reps 10  --sszd-alpha 16.0 --target "plnonconv" --sszd-alpha 5e-2 &
python3 synthetic_comparison.py 'sszd' 50 100000 --sszd-l 1 --sszd-dirtype "coordinates" --out-file "fd_co_1" --reps 10 --sszd-alpha 0.5 --sszd-normalize 1 --target "plnonconv" --sszd-alpha 5e-2 & 
python3 synthetic_comparison.py 'sszd' 50 100000 --sszd-l 50 --sszd-dirtype "gaussian" --out-file "fd_gauss_50" --reps 10 --sszd-alpha 0.002 --target "plnonconv" &
python3 synthetic_comparison.py 'sszd' 50 100000 --sszd-l 50 --sszd-dirtype "spherical" --out-file "fd_sp_50" --reps 10 --sszd-alpha 0.25 --target "plnonconv" &
python3 synthetic_comparison.py 'sszd' 50 100000 --sszd-l 50 --sszd-dirtype "coordinates" --out-file "fd_co_50" --reps 10  --sszd-alpha 0.5 --sszd-normalize 1 --target "plnonconv" &
wait
python3 synthetic_comparison.py 'ds'  50 100000 --ds-theta 0.9 --ds-rho 2.0 --ds-min-alpha 2e-10 --ds-max-alpha 10.0 --ds-alpha 1.0 --ds-l 1 --ds-dirtype 'spherical' --ds-constant 10.0 --reps 10 --out-file 'probds_sp_2' --target "plnonconv" &
python3 synthetic_comparison.py 'ds'  50 100000 --ds-theta 0.9 --ds-rho 2.0  --ds-min-alpha 2e-10 --ds-max-alpha 10.0 --ds-alpha 1.0 --ds-l 1 --ds-dirtype 'orthogonal' --ds-constant 10.0 --reps 10 --out-file 'probds_ortho_2' --target "plnonconv" &
python3 synthetic_comparison.py 'sds' 50 100000 --ds-theta 0.9 --ds-rho 2.0  --ds-min-alpha 2e-10 --ds-max-alpha 10.0 --ds-alpha 1.0 --ds-l 1 --ds-dirtype 'spherical' --ds-constant 10.0 --sds-dirtype "orthogonal" --sds-r 25 --reps 10 --out-file 'probds_sketch_ortho_sp_25_2' --target "plnonconv" &
python3 synthetic_comparison.py 'sds' 50 100000 --ds-theta 0.9 --ds-rho 2.0  --ds-min-alpha 2e-10 --ds-max-alpha 10.0 --ds-alpha 1.0 --ds-l 1 --ds-dirtype 'orthogonal' --ds-constant 10.0 --sds-dirtype "orthogonal" --sds-r 25 --reps 10 --out-file 'probds_sketch_ortho_ortho_25_2' --target "plnonconv" &
wait
python3 synthetic_comparison.py 'stp' 50 100000 --out-file "stp" --reps 10 --target "plnonconv" --stp-alpha 1.0 &
python3 synthetic_comparison.py 'cobyla' 50 100000 --out-file "cobyla" --reps 10 --target "plnonconv"
