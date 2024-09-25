import torch


with open("../../Dataset/Formula/math.txt", "r") as f:
    dict = {}
    formulas = f.readlines()
    count = 0
    for formula in formulas:
        ls = formula.split(" ")
        for token in ls:
            token = token.strip()
            if token not in dict.keys() and len(token) > 0:
                dict[token] = count
                count += 1
with open("math.txt", "w") as f:
    for token, value in dict.items():
        f.write(f"{token} {value}\n")
