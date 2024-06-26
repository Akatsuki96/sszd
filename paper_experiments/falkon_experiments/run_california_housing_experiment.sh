nohup python3 california_housing_experiment.py "sszd" 500 --sszd-alpha 50.0 --sszd-l 4 --sszd-h 1e-5 --sszd-dirtype 'orthogonal' --reps 10 --out-file "sszd_4_orth" &
nohup python3 california_housing_experiment.py "sszd" 500 --sszd-alpha 50.0 --sszd-l 9 --sszd-h 1e-5 --sszd-dirtype 'orthogonal' --reps 10 --out-file "sszd_9_orth" &
wait
nohup python3 california_housing_experiment.py 'ds' 500   --ds-theta 0.9 --ds-rho 2.0  --ds-min-alpha 1e-9 --ds-max-alpha 100.0  --ds-alpha 5.0 --ds-l 1 --ds-dirtype 'spherical'  --ds-constant 0.01 --reps 10 --out-file 'probds_sp_2' &
nohup python3 california_housing_experiment.py 'ds' 500   --ds-theta 0.9 --ds-rho 2.0  --ds-min-alpha 1e-9 --ds-max-alpha 100.0  --ds-alpha 5.0 --ds-l 1 --ds-dirtype 'orthogonal' --ds-constant 0.01 --reps 10 --out-file 'probds_ortho_2' &
nohup python3 california_housing_experiment.py 'sds' 500  --ds-theta 0.9 --ds-rho 2.0  --ds-min-alpha 1e-9 --ds-max-alpha 100.0  --ds-alpha 5.0 --ds-l 1 --ds-dirtype 'spherical'  --sds-dirtype 'orthogonal' --sds-r 4 --ds-constant 0.01 --reps 10 --out-file 'probds_sketch_sp_2' &
nohup python3 california_housing_experiment.py 'sds' 500  --ds-theta 0.9 --ds-rho 2.0  --ds-min-alpha 1e-9 --ds-max-alpha 100.0  --ds-alpha 5.0 --ds-l 1 --ds-dirtype 'orthogonal' --sds-dirtype 'orthogonal' --sds-r 4 --ds-constant 0.01 --reps 10 --out-file 'probds_sketch_ortho_2' &
wait
nohup python3 california_housing_experiment.py "sszd" 500 --sszd-alpha 500.0 --sszd-l 1 --sszd-h 1e-5 --sszd-dirtype 'spherical' --reps 10 --out-file "sp_1" &
nohup python3 california_housing_experiment.py "sszd" 500 --sszd-alpha 50.0 --sszd-l 9 --sszd-h 1e-5 --sszd-dirtype 'spherical' --reps 10 --out-file "sp_9" &
nohup python3 california_housing_experiment.py "sszd" 500 --sszd-alpha 50.0 --sszd-l 1 --sszd-h 1e-5 --sszd-dirtype 'gaussian' --reps 10 --out-file "g_1"  &
nohup python3 california_housing_experiment.py "sszd" 500 --sszd-alpha 5.0 --sszd-l 9 --sszd-h 1e-5 --sszd-dirtype 'gaussian' --reps 10 --out-file "g_9"  &
nohup python3 california_housing_experiment.py "sszd" 500 --sszd-alpha 50.0 --sszd-l 1 --sszd-h 1e-5 --sszd-dirtype 'coordinates' --reps 10 --out-file "co_1"  &
nohup python3 california_housing_experiment.py "sszd" 500 --sszd-alpha 50.0 --sszd-l 9 --sszd-h 1e-5 --sszd-dirtype 'coordinates' --reps 10 --out-file "co_9"  &
wait
nohup python3 california_housing_experiment.py "stp" 500  --stp-alpha 1.0  --reps 10 --out-file "stp" &
