@echo off
cd /d "%~dp0.."
echo Writing qmix/vdn *_medium_* and *_hard_* from training JSONL (Tables 1 and 3 aligned)...
python experiments/materialize_eval_from_training_jsonl.py --tier both
echo Done.
pause
