@echo off
REM Regenerate QMIX/VDN JSON from training jsonl, run heuristic on hard, print dual markdown tables.
cd /d "%~dp0.."
python experiments\materialize_eval_from_training_jsonl.py --tier both
python eval_heuristic_3d.py --policy random --env-config configs\envs\grid_3d_hard.yaml --episodes 50 --seed 0 --output-json experiments\results_eval\random_hard_s0.json
python eval_heuristic_3d.py --policy random --env-config configs\envs\grid_3d_hard.yaml --episodes 50 --seed 1 --output-json experiments\results_eval\random_hard_s1.json
python eval_heuristic_3d.py --policy random --env-config configs\envs\grid_3d_hard.yaml --episodes 50 --seed 2 --output-json experiments\results_eval\random_hard_s2.json
python eval_heuristic_3d.py --policy greedy --env-config configs\envs\grid_3d_hard.yaml --episodes 50 --seed 0 --output-json experiments\results_eval\greedy_hard_s0.json
python eval_heuristic_3d.py --policy greedy --env-config configs\envs\grid_3d_hard.yaml --episodes 50 --seed 1 --output-json experiments\results_eval\greedy_hard_s1.json
python eval_heuristic_3d.py --policy greedy --env-config configs\envs\grid_3d_hard.yaml --episodes 50 --seed 2 --output-json experiments\results_eval\greedy_hard_s2.json
python experiments\aggregate_eval_table.py --spec experiments\aggregate_dual_spec.json --skip-missing
pause
