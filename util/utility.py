
def cut_range(n, batch):
    cuts = n // batch
    cuts = [i*batch for i in range(cuts)]
    cuts.append(n)
    return cuts