@echo off
cd /d "%~dp0.."
python experiments\aggregate_eval_table.py --spec experiments\aggregate_dual_spec.json --skip-missing
pause
