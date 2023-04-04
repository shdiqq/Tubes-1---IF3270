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

ffnn = generate_model(filename=f"file/{model}.json")
