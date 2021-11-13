from util import Results

reproduce = Results("generate_out/gold-s-gpu.txt")
reproduce.write_H("results/reproduction")
reproduce.write_T("results/ref")
reproduce.write_n_random(100, "results/sample-100.txt")

pretrain = Results("../Baseline/pretrain/gold-s.txt")
pretrain.write_H("results/pretrain")