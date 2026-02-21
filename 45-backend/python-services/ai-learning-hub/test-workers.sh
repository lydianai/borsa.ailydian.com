#!/bin/bash
# Quick worker syntax test

echo "🧪 Testing all workers..."
echo ""

workers=(
    "workers/rl_agent_worker.py"
    "workers/online_learning_worker.py"
    "workers/multi_agent_worker.py"
    "workers/automl_worker.py"
    "workers/nas_worker.py"
    "workers/meta_learning_worker.py"
    "workers/federated_worker.py"
    "workers/causal_ai_worker.py"
    "workers/regime_detection_worker.py"
    "workers/explainable_ai_worker.py"
)

for worker in "${workers[@]}"; do
    echo -n "Testing $worker... "
    ./venv/bin/python3 -m py_compile "$worker" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ OK"
    else
        echo "❌ FAILED"
    fi
done

echo ""
echo "Testing orchestrator.py... "
./venv/bin/python3 -m py_compile orchestrator.py
if [ $? -eq 0 ]; then
    echo "✅ OK"
else
    echo "❌ FAILED"
fi

echo ""
echo "Testing services/data_collector.py... "
./venv/bin/python3 -m py_compile services/data_collector.py
if [ $? -eq 0 ]; then
    echo "✅ OK"
else
    echo "❌ FAILED"
fi

echo ""
echo "Testing services/service_integrator.py... "
./venv/bin/python3 -m py_compile services/service_integrator.py
if [ $? -eq 0 ]; then
    echo "✅ OK"
else
    echo "❌ FAILED"
fi

echo ""
echo "✅ All syntax tests completed!"
