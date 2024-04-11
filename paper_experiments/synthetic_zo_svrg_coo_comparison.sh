python3 synthetic_comparison.py 'sszd' 50 100000 --sszd-l 50 --sszd-dirtype "orthogonal" --out-file "sszd_qr_50" --reps 10 --sszd-alpha 5e-1 &
python3 synthetic_comparison.py 'isszd' 50 100000 --sszd-l 50 --sszd-dirtype "orthogonal" --out-file "isszd_qr_50" --reps 10 --sszd-alpha 1e-1 --isszd-m 200 &
python3 synthetic_comparison.py 'zo_svrg_coo' 50 100000  --out-file "zo_svrg_coo_50" --zo-svrg-alpha 5e-3 --zo-svrg-h 1e-9 --zo-svrg-m 10 --reps 10 &
python3 synthetic_comparison.py 'zo_svrg_coo_rand' 50 100000  --out-file "zo_svrg_coo_rand_50_1" --zo-svrg-alpha 5e-3 --zo-svrg-h 1e-9 --zo-svrg-m 10 --zo-svrg-l 1 --reps 10 &
python3 synthetic_comparison.py 'zo_svrg_coo_rand' 50 100000  --out-file "zo_svrg_coo_rand_50_25" --zo-svrg-alpha 5e-3 --zo-svrg-h 1e-9 --zo-svrg-m 10 --zo-svrg-l 25 --reps 10 &
wait
python3 synthetic_comparison.py 'sszd' 50 100000 --target 'plconv' --sszd-l 50 --sszd-dirtype "orthogonal" --out-file "sszd_qr_50" --reps 10 --sszd-alpha 5e-1 &
python3 synthetic_comparison.py 'isszd' 50 100000 --target 'plconv' --sszd-l 50 --sszd-dirtype "orthogonal" --out-file "isszd_qr_50" --reps 10 --sszd-alpha 1e-1 --isszd-m 200 &
python3 synthetic_comparison.py 'zo_svrg_coo' 50 100000 --target 'plconv'  --out-file "zo_svrg_coo_50" --zo-svrg-alpha 5e-3 --zo-svrg-h 1e-9 --zo-svrg-m 10 --reps 10 &
python3 synthetic_comparison.py 'zo_svrg_coo_rand' 50 100000 --target 'plconv'  --out-file "zo_svrg_coo_rand_50_1" --zo-svrg-alpha 5e-3 --zo-svrg-h 1e-9 --zo-svrg-m 10 --zo-svrg-l 1 --reps 10 &
python3 synthetic_comparison.py 'zo_svrg_coo_rand' 50 100000 --target 'plconv'  --out-file "zo_svrg_coo_rand_50_25" --zo-svrg-alpha 5e-3 --zo-svrg-h 1e-9 --zo-svrg-m 10 --zo-svrg-l 25 --reps 10 &
wait
python3 synthetic_comparison.py 'sszd' 50 100000 --target 'plnonconv' --sszd-l 50 --sszd-dirtype "orthogonal" --out-file "sszd_qr_50" --reps 10 --sszd-alpha 1.0 &
python3 synthetic_comparison.py 'isszd' 50 100000 --target 'plnonconv' --sszd-l 50 --sszd-dirtype "orthogonal" --out-file "isszd_qr_50" --reps 10 --sszd-alpha 1.0 --isszd-m 200 &
python3 synthetic_comparison.py 'zo_svrg_coo' 50 100000 --target 'plnonconv'  --out-file "zo_svrg_coo_50" --zo-svrg-alpha 5e-3 --zo-svrg-h 1e-9 --zo-svrg-m 10 --reps 10 &
python3 synthetic_comparison.py 'zo_svrg_coo_rand' 50 100000 --target 'plnonconv'  --out-file "zo_svrg_coo_rand_50_1" --zo-svrg-alpha 5e-3 --zo-svrg-h 1e-9 --zo-svrg-m 10 --zo-svrg-l 1 --reps 10 &
python3 synthetic_comparison.py 'zo_svrg_coo_rand' 50 100000 --target 'plnonconv'  --out-file "zo_svrg_coo_rand_50_25" --zo-svrg-alpha 5e-3 --zo-svrg-h 1e-9 --zo-svrg-m 10 --zo-svrg-l 25 --reps 10 &
python3 synthetic_comparison.py 'zo_svrg_coo_rand' 50 100000 --target 'plnonconv'  --out-file "zo_svrg_coo_rand_50_50" --zo-svrg-alpha 5e-3 --zo-svrg-h 1e-9 --zo-svrg-m 10 --zo-svrg-l 50 --reps 10 &
