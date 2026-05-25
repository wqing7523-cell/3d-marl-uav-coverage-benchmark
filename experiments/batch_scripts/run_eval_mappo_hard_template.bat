@echo off
REM 与 medium 主表一致：50 episodes、随机策略评估 --stochastic、输出含 coverage_mean / success_mean / steps_mean
REM 将下面三个 --checkpoint 换成你训练得到的 mappo3d_L4_H7_W7_uavs3_obs012_时间戳.pt（与 seed 0/1/2 一一对应）
cd /d "%~dp0.."

set CKPT0=experiments\checkpoints\REPLACE_WITH_mappo3d_L4_seed0.pt
set CKPT1=experiments\checkpoints\REPLACE_WITH_mappo3d_L4_seed1.pt
set CKPT2=experiments\checkpoints\REPLACE_WITH_mappo3d_L4_seed2.pt

python eval_mappo_3d.py --checkpoint %CKPT0% --env-config configs\envs\grid_3d_hard.yaml --episodes 50 --seed 0 --stochastic --output-json experiments\results_eval\mappo_hard_s0.json
python eval_mappo_3d.py --checkpoint %CKPT1% --env-config configs\envs\grid_3d_hard.yaml --episodes 50 --seed 1 --stochastic --output-json experiments\results_eval\mappo_hard_s1.json
python eval_mappo_3d.py --checkpoint %CKPT2% --env-config configs\envs\grid_3d_hard.yaml --episodes 50 --seed 2 --stochastic --output-json experiments\results_eval\mappo_hard_s2.json

echo 然后运行: python experiments\aggregate_eval_table.py --spec experiments\aggregate_dual_spec.json
pause
