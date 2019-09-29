from utils.utilities import *

def saveStandardizeMatrixes():
    myMean = np.mean(X_train, axis=0)[:, :]
    myStd = np.std(X_train, axis=0)[:, :]
    sample = ""

    with open('meanMatrix', 'w') as file:
        for i in range(len(myMean)):
            for j in range(len(myMean[0])):
                sample += str(myMean[i][j]) + " "
            sample += "\n"
        file.write(sample)
    print("--- meanMatrix EXPORTED ---")

    sample = ""
    with open('stdMatrix', 'w') as file:
        for i in range(len(myStd)):
            for j in range(len(myStd[0])):
                sample += str(myStd[i][j]) + " "
            sample += "\n"
        file.write(sample)
    print("--- stdMatrix EXPORTED ---")

if __name__ == "__main__":
    X_train, labels_train, list_ch_train = read_data(data_path="./UCIHAR/", split="train")  # Train
    X_test, labels_test, list_ch_test = read_data(data_path="./UCIHAR/", split="test")      # Test
    saveStandardizeMatrixes()