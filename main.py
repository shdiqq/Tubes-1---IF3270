from function.generateModel import generate_model
import numpy as np

while True :
  model = str(input("Masukkan model yang ingin digunakan (Silakan cek nama file yang ada pada folder model): "))

  if (model == 'relu'):
    break
  elif (model == 'sigmoid'):
    break
  elif (model == 'softmax'):
    break
  elif (model == 'linear'):
    break
  elif (model == 'multilayer'):
    break
  else :
    print("Input yang diberikan salah")

ffnn = generate_model(filename=f"model/{model}.json")

while True :
  n_instance = str(input("Masukkan jumlah instance yang akan diprediksi: "))

  if (int(n_instance) <= len(ffnn.input_layer[0].input_data) and int(n_instance) >= 1):
    print("=========================================================")
    break
  else :
    print("Input yang diberikan salah")

ffnn.forward_propagation(int(n_instance))
ffnn.printListActivationValue()
ffnn.accuracy()
