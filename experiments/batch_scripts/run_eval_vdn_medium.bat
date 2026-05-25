@echo off
cd /d "%~dp0.."
echo Eval VDN medium - seed 0 1 2 (greedy eval, no epsilon flag in this eval_3d.py). Verify checkpoints in vdn_medium_checkpoint_note.md
python eval_3d.py --checkpoint experiments\checkpoints\vdn3d_L3_H6_W6_uavs2_obs008_1774861056.pt --env-config configs\envs\grid_3d_medium.yaml --episodes 50 --seed 0 --output-json experiments\results_eval\vdn_medium_s0.json
if errorlevel 1 exit /b 1
python eval_3d.py --checkpoint experiments\checkpoints\vdn3d_L3_H6_W6_uavs2_obs008_1774876572.pt --env-config configs\envs\grid_3d_medium.yaml --episodes 50 --seed 1 --output-json experiments\results_eval\vdn_medium_s1.json
if errorlevel 1 exit /b 1
python eval_3d.py --checkpoint experiments\checkpoints\vdn3d_L3_H6_W6_uavs2_obs008_1774892113.pt --env-config configs\envs\grid_3d_medium.yaml --episodes 50 --seed 2 --output-json experiments\results_eval\vdn_medium_s2.json
if errorlevel 1 exit /b 1
echo Done. Then: python experiments\aggregate_eval_table.py --spec experiments\aggregate_medium_spec.json
