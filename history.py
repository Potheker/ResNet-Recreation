import csv

labels = ["train_loss", "train_error_rate", "val_loss", "val_error_rate"]

def write_csv(n, residual, arr):
    if not len(arr) == 4:
        return 1
    with open(f"./history/{n}_{residual}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        for i, row in enumerate(arr):
            writer.writerow([labels[i]] + row)
    return 0

def read_csv(n, residual):
    res = []
    with open(f"./history/{n}_{residual}.csv", "r", newline="") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            res.append([float(x) for x in row[1:]])
    return res

