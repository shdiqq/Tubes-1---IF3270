from function.generateModel import generate_model
import numpy as np

while True :
  model = str(input("Masukkan model yang ingin digunakan (Sialakan cek nama file yang ada pada folder model): "))

  if (model == 'ReLU'):
    break
  elif (model == 'sigmoid'):
    break
  else :
    print("Input yang diberikan salah")

ffnn = generate_model(filename=f"model/{model}.json")

while True :
  n_instance = str(input("Masukkan jumlah instance yang akan diprediksi: "))

  if (int(n_instance) <= len(ffnn.input_layer[0].X) and int(n_instance) >= 1):
    break
  else :
    print("Input yang diberikan salah")

ffnn.predict(int(n_instance))

