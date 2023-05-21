from function.generateModel import generate_model

quit = False

while (not quit):

  while True :
    model = str(input("Masukkan model yang ingin digunakan (Silakan cek nama file yang ada pada folder model): "))
    filePath = f"model/{model}.json"
    ffnn = generate_model(filePath)

    if (ffnn == False):
        print("File tidak ditemukan pada folder model! Pastikan file yang diinput sudah berada pada folder tersebut!")
    else :
      break

  ffnn.information()
  ffnn.forward_propagation()
  ffnn.accuracy()
  ffnn.visualize()

  print("Apakah anda ingin keluar? (Y/N)")
  while (True):
    inputUser = input(">>> ")
    if (inputUser.upper() == 'Y'):
      print("Keluar dari program...")
      quit = True
      break
    elif (inputUser.upper() == 'N'):
      break
    else:
      print("input anda salah! Lakukan input kembali")