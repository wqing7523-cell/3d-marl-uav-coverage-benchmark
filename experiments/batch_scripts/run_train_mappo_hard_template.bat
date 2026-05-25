@echo off
REM 难例：grid_3d_hard（4x7x7, 3 UAV, obs 0.12）。三轮 seed 一条命令串起来（失败则停止）。
cd /d "%~dp0.."

python train_mappo_3d.py --env-config configs\envs\grid_3d_hard.yaml --algo-config configs\algos\mappo.yaml --seed 0 && python train_mappo_3d.py --env-config configs\envs\grid_3d_hard.yaml --algo-config configs\algos\mappo.yaml --seed 1 && python train_mappo_3d.py --env-config configs\envs\grid_3d_hard.yaml --algo-config configs\algos\mappo.yaml --seed 2

if errorlevel 1 echo 某一步失败，请检查日志 experiments\logs\mappo3d.log
echo 完成后在 experiments\checkpoints\ 找 mappo3d_L4_H7_W7_uavs3_obs012_*.pt，填入 run_eval_mappo_hard_template.bat
pause
