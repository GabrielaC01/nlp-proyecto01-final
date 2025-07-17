# Activar el entorno virtual
source .venv/bin/activate

echo "Ejecutando benchmarks..."
mkdir -p benchmarks/results
python benchmarks/bench_translation.py
echo "Benchmark completado"
