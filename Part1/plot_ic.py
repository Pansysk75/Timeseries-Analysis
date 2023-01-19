import matplotlib.pyplot as plt
import numpy as np

result_files = [
    "eruption1989.dat_ic_smalln.txt", 
    "eruption1989.dat_ic.txt","eruption2000.dat_ic.txt","eruption2011.dat_ic.txt"]

for file_name in result_files:
    p, error,aic,bic,fpe = np.genfromtxt(file_name)
    # print(np.fromfile("eruption1989.dat_ic.txt", sep="\n"))


    lner = np.log(error)
    plt.figure(figsize=(4,8))
    plt.subplot(3,1,1)
    plt.title("Error")
    plt.plot(p, lner, label="Error term", linewidth=2)
    plt.xlabel("p")
    plt.legend()
    # plt.show()

    plt.subplot(3,1,2)
    plt.title("Complexity Penalty")
    plt.plot(p, aic-lner, "c-", label="Penalty in AIC", linewidth=2)
    plt.plot(p, bic-lner, "g-",label="Penalty in BIC", linewidth=2)
    plt.plot(p, np.log(fpe)-lner,  "k--", label="Penalty in ln(FPE)", linewidth=1.5)
    plt.xlabel("p")
    plt.legend()

    # fpe = np.log(1 + 2*p/(n-p))
    # plt.figure(figsize=(4,4))
    plt.subplot(3,1,3)
    plt.title("Information Criteria")
    plt.plot(p, aic, "c-", label=r"$AIC$", linewidth=2)
    plt.plot(p, bic, "g-",label=r"$BIC$", linewidth=2)
    plt.plot(p, np.log(fpe),  "k--", label=r"$ln(FPE)$", linewidth=1.5)
    plt.xlabel("p")
    plt.legend()


    plt.tight_layout()
    plt.savefig("plots/" + file_name + ".png", dpi=200)
    plt.show()
