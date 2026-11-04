@echo off
rem Runs the first real benchmark in stages (one process each, so memory is freed between stages).
rem Close Chrome / VS Code first; about 3 GB RAM free. Run from the project root:  scripts\run_all.cmd
rem Results are saved in evals\results\. Uses qwen3:1.7b (low-RAM dev model); the model name is stored in each result file.
set REGRAG_LLM_MODEL=qwen3:1.7b
set PY=.venv\Scripts\python

echo === 1/5 golden: classic vs hybrid, no gate ===
%PY% scripts\run_eval.py --suite golden --pipelines classic hybrid --gate none --tag golden_nogate || goto :fail
echo === 2/5 safety suite: hybrid, no gate ===
%PY% scripts\run_eval.py --suite safety --pipelines hybrid --gate none --tag safety_nogate || goto :fail
echo === 3/5 gate: llm_judge ===
%PY% scripts\run_eval.py --suite golden --pipelines hybrid --gate llm_judge --tag gate_llm_judge || goto :fail
echo === 4/5 gate: nli + judge (+ number check) ===
%PY% scripts\run_eval.py --suite golden --pipelines hybrid --gate nli_plus_judge --tag gate_nli_judge || goto :fail
%PY% scripts\run_eval.py --suite golden --pipelines hybrid --gate nli_plus_judge --check-numbers --tag gate_nli_judge_numbers || goto :fail
echo === 5/5 query rewriting on, no gate ===
%PY% scripts\run_eval.py --suite golden --pipelines hybrid --gate none --rewrite --tag golden_rewrite || goto :fail
echo ALL DONE. Tell Claude.
exit /b 0
:fail
echo STOPPED: a stage failed. Copy the last lines and show them to Claude.
exit /b 1
