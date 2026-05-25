@echo off
cd /d "%~dp0.."
echo Random policy x3 seeds...
python eval_heuristic_3d.py --policy random --env-config configs\envs\grid_3d_medium.yaml --episodes 50 --seed 0 --output-json experiments\results_eval\random_medium_s0.json
if errorlevel 1 exit /b 1
echo OK random_medium_s0.json
python eval_heuristic_3d.py --policy random --env-config configs\envs\grid_3d_medium.yaml --episodes 50 --seed 1 --output-json experiments\results_eval\random_medium_s1.json
if errorlevel 1 exit /b 1
echo OK random_medium_s1.json
python eval_heuristic_3d.py --policy random --env-config configs\envs\grid_3d_medium.yaml --episodes 50 --seed 2 --output-json experiments\results_eval\random_medium_s2.json
if errorlevel 1 exit /b 1
echo OK random_medium_s2.json
echo Greedy policy x3 seeds...
python eval_heuristic_3d.py --policy greedy --env-config configs\envs\grid_3d_medium.yaml --episodes 50 --seed 0 --output-json experiments\results_eval\greedy_medium_s0.json
if errorlevel 1 exit /b 1
python eval_heuristic_3d.py --policy greedy --env-config configs\envs\grid_3d_medium.yaml --episodes 50 --seed 1 --output-json experiments\results_eval\greedy_medium_s1.json
if errorlevel 1 exit /b 1
python eval_heuristic_3d.py --policy greedy --env-config configs\envs\grid_3d_medium.yaml --episodes 50 --seed 2 --output-json experiments\results_eval\greedy_medium_s2.json
if errorlevel 1 exit /b 1
echo Done. Run: python experiments\aggregate_eval_table.py --spec experiments\aggregate_medium_spec.json
