<h2 align="center">
<p>Proyecto 1 - Decodificadores avanzados para Seq2Seq</p>
</h2>

## Descripción
Este proyecto implementa un modelo Transformer Seq2Seq para tareas de traducción automática (EN → ES) y compara distintas técnicas de decodificación: Greedy, Beam Search, Top-k, Top-p y Diverse Beam Search. El objetivo es analizar el impacto de cada estrategia en la calidad de traducción (BLEU), latencia y consumo de memoria, mediante un entorno modular y reproducible.

## Estructura del repositorio

- `src/`: Implementación del modelo Transformer, decodificadores y utilidades
- `tests/`: Pruebas unitarias para funciones clave del sistema
- `benchmarks/`: Scripts y resultados de evaluación (BLEU, latencia, memoria)
- `exposicion.ipynb`: Cuaderno explicativo 
- `requirements.txt`: Lista de dependencias del proyecto
---

## Cómo ejecutar

1. Clonar el repositorio
   ```bash
   git clone https://github.com/GabrielaC01/nlp-proyecto01-final.git
   cd nlp-proyecto01-final
  
2. Crear y activar un entorno virtual
    ```bash
   python3 -m venv .venv
   source .venv/bin/activate

3. Instalar dependencias
   ```bash
   pip install -r requirements.txt

4. Ejecutar pruebas
    ```bash
    pytest --cov=src --cov-report=html
    ```
    Luego abre el archivo htmlcov/index.html en tu navegador para ver el reporte
  
5. Dar permisos y correr los benchmarks:
   ```bash
   chmod +x benchmarks/run_bench.sh
   ./benchmarks/run_bench.sh

6. Abrir `exposicion.ipynb` para la demo
  
7. Enlace al [video](https://drive.google.com/drive/folders/1dvLogBQQJu3rxOzpUUsyqv2yzCJbdOoR?usp=sharing)

## Autor
* Gabriela Colque  

