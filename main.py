from perceptron import *

def main():
    pcptrn = Perceptron(eta=0.1, n_iter=7)
    pcptrn.fit()
    pcptrn.plot()
    pcptrn.plot_errors()

main()