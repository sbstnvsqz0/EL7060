Pasos a seguir para correr código:
1. Copiar carpeta CREMA-D a carpeta T2.
2. Instalar requirements.txt 
3. Correr demo.ipynb para ver que todo funcione.
4. Correr comando python main.py para poder entrenar en environment con librerias instaladas.

Ejemplo para ejecutar main.py:

python main.py --epochs 1 --batch_size 64 --hidden_size 128 --num_lstm_layers 1 --num_mlp_layers 1 --learning_rate 0.001 --dropout 0.1 --name "modelo"

Ejemplo para ejecutar optimizer.py:

python optimizer.py