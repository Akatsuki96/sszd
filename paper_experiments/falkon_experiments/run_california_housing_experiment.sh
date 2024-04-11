nohup python3 california_housing_experiment.py "sszd" 500 --sszd-alpha 10.0 --sszd-l 4 --sszd-h 1.0 --sszd-dirtype 'orthogonal' --reps 10 --out-file "sszd_4_orth" > out_sszd4 &
nohup python3 california_housing_experiment.py "sszd" 500 --sszd-alpha 10.0 --sszd-l 9 --sszd-h 1.0 --sszd-dirtype 'orthogonal' --reps 10 --out-file "sszd_9_orth" > out_sszd9 &
wait
nohup python3 california_housing_experiment.py 'ds' 500  --ds-theta 0.5 --ds-rho 2.0  --ds-min-alpha 1e-9 --ds-max-alpha 100.0 --ds-alpha 10.0 --ds-l 1 --ds-dirtype 'spherical' --ds-constant 0.1 --reps 10 --out-file 'probds_sp_2' &
python3 california_housing_experiment.py 'ds' 500  --ds-theta 0.5 --ds-rho 2.0  --ds-min-alpha 1e-9 --ds-max-alpha 100.0 --ds-alpha 10.0 --ds-l 1 --ds-dirtype 'orthogonal' --ds-constant 0.1 --reps 10 --out-file 'probds_ortho_2'
