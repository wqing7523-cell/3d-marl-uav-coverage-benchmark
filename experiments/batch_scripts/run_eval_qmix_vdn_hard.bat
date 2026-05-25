@echo off
REM Greedy eval (50 episodes). Overwrites experiments/results_eval/qmix_hard_s*.json and vdn_hard_s*.json used by aggregate_hard_spec.json.
REM Checkpoint filenames from experiments/logs/qmix3d.log (seed order matches hard_qmix_vdn.jsonl). See experiments/qmix_vdn_hard_checkpoint_note.md.
cd /d "%~dp0.."
python eval_3d.py --checkpoint experiments\checkpoints\qmix3d_L4_H7_W7_uavs3_obs012_1775082129.pt --env-config configs\envs\grid_3d_hard.yaml --episodes 50 --seed 0 --output-json experiments\results_eval\qmix_hard_s0.json
python eval_3d.py --checkpoint experiments\checkpoints\qmix3d_L4_H7_W7_uavs3_obs012_1775126118.pt --env-config configs\envs\grid_3d_hard.yaml --episodes 50 --seed 1 --output-json experiments\results_eval\qmix_hard_s1.json
python eval_3d.py --checkpoint experiments\checkpoints\qmix3d_L4_H7_W7_uavs3_obs012_1775171227.pt --env-config configs\envs\grid_3d_hard.yaml --episodes 50 --seed 2 --output-json experiments\results_eval\qmix_hard_s2.json
python eval_3d.py --checkpoint experiments\checkpoints\vdn3d_L4_H7_W7_uavs3_obs012_1775256641.pt --env-config configs\envs\grid_3d_hard.yaml --episodes 50 --seed 0 --output-json experiments\results_eval\vdn_hard_s0.json
python eval_3d.py --checkpoint experiments\checkpoints\vdn3d_L4_H7_W7_uavs3_obs012_1775284777.pt --env-config configs\envs\grid_3d_hard.yaml --episodes 50 --seed 1 --output-json experiments\results_eval\vdn_hard_s1.json
python eval_3d.py --checkpoint experiments\checkpoints\vdn3d_L4_H7_W7_uavs3_obs012_1775316135.pt --env-config configs\envs\grid_3d_hard.yaml --episodes 50 --seed 2 --output-json experiments\results_eval\vdn_hard_s2.json
echo Done. When checkpoints are missing, run materialize_hard_eval_from_jsonl.bat first for Table-3-aligned metrics.
pause
