@echo off
echo ========================================================
echo   Running All Adaptive FL Framework Unit Test Suites
echo ========================================================
cd /d "%~dp0\adaptive-fl-framework"
set PYTHON="..\.venv\Scripts\python.exe"
if not exist %PYTHON% (
    set PYTHON=python
)

echo [1/7] Testing Phase 1: Baseline Architecture and Loops...
%PYTHON% tests\test_phase1_baseline.py
if errorlevel 1 goto error

echo [2/7] Testing Phase 2: Differential Privacy Engine...
%PYTHON% tests\test_phase2_dp.py
if errorlevel 1 goto error

echo [3/7] Testing Phase 3: Secure Aggregation (SecAgg+)...
%PYTHON% tests\test_phase3_secagg.py
if errorlevel 1 goto error

echo [4/7] Testing Adaptive Privacy Controller...
%PYTHON% tests\test_adaptive_controller.py
if errorlevel 1 goto error

echo [5/7] Testing Phase 7: Evaluation and MIA Privacy Auditor...
%PYTHON% tests\test_phase7_evaluation.py
if errorlevel 1 goto error

echo [6/7] Testing Phase 8: Datasets and Model Factory...
%PYTHON% tests\test_phase8_dataset_model.py
if errorlevel 1 goto error

echo [7/7] Testing Phase 9: Experiment Runner and Configs...
%PYTHON% tests\test_phase9_experiment_runner.py
if errorlevel 1 goto error

echo ========================================================
echo   [SUCCESS] ALL TEST SUITES PASSED CLEANLY!
echo ========================================================
goto end

:error
echo [FAIL] A test suite failed. Please check output above.

:end
pause
