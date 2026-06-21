@echo off
cd /d "C:\Users\Darsh Veer Singh\Documents\GitHub\traffic_violation_detect\cctv_system"
"venv\Scripts\python.exe" -u train\finetune_triple.py 100 > runs\triple_train.log 2>&1
