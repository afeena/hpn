import re
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", nargs="+")

    args = parser.parse_args()
    runs = {}
    run = 0



    for log_f in args.fname:
        with open(log_f, "r") as f:
            for line in f:
                #see if it's a start of the new training
                if re.match(r"Namespace.*", line):
                    if run > 0:
                        runs[hpn] = perf

                    hpn = ".".join(re.search(r'hpnfile=\'(.*?)\', ', line).group().split("=")[-1].split("/")[-1].split(".")[0:-1])
                    print(hpn)
                    epoch = 0
                    perf = {}
                    run += 1
                if re.match("Testing: classification accuracy*", line):
                    acc = float(line.split("-")[-1])
                    perf[epoch] = acc
                if re.match(r"^---$", line):
                    epoch+=1
        runs[hpn] = perf

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    for key, vals in runs.items():
        t = key
        x = list(vals.keys())
        y = list(vals.values())
        ax.plot(x, y, label=t)
    ax.legend()

    plt.show()