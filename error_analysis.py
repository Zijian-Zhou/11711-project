from util import Results

reproduce = Results("/Users/seanchang/Downloads/golds_new.txt")
reproduce.write_H("results/reproduction")
reproduce.write_T("results/ref")
#reproduce.write_n_random(100, "results/reproduce-sample-100.txt")

MLE = Results("/Users/seanchang/Downloads/mle.txt")
MLE.write_H("results/MLE")
#MLE.write_n_random(100, "results/MLE-sample-100.txt")

entropy1 = Results("/Users/seanchang/Downloads/golds_ent1.txt")
entropy1.write_H("results/entropy-alpha-1")
#entropy1.write_n_random(100, "results/entropy-alpha-1-sample-100.txt")

entropyn05 = Results("/Users/seanchang/Downloads/golds_ent_alpha-0.5.txt")
entropyn05.write_H("results/entropy-alpha-n0.5")
entropyn05.write_n_random(100, "results/entropy-alpha-n0.5-sample-100.txt")