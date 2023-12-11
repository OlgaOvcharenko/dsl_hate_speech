from sklearn.metrics import auc


f = open("logs/fast_text.txt", "r")
vals = []
for x in f:
    print(x)

    if "Weighted" in x or "Binary" in x or "Macro" in x:
        tmp = x.split(" ")
        p, r = tmp[2].replace(",", "").replace("(", ""), tmp[3].replace(",", "").replace("(", "")
        p, r = float(p), float(r)
        auc_score = auc(r, p)
        print(f"{tmp[0]} {tmp[1]} AUPRC: {auc_score}")



