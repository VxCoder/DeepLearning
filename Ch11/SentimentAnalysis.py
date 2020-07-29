from tensorflow import keras
from LSTM import MODEL_SAVE_PATH

def main():
    model = keras.models.load_model(MODEL_SAVE_PATH)
    while True:
        print("==>")
        line = input()
        if line == 'exit':
            break



if __name__ == "__main__":
    main()