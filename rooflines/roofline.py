import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import csv
sns.set_theme()

# Peak FP16 Tflops
peak_tflops_per_sec = peak_tflops
# Peak BW TiB/s
peak_tib_per_sec = peak_bw / 1024.
peak_tb_per_sec = peak_tib_per_sec * ((1024. / 1000)**3)

def compute_ai(M, N, K):
    ai = (M * N * K) / (M * K + K * N + M * N)
    return ai

def tflops_per_sec(ai):
    return min(peak_tflops_per_sec, ai * peak_tb_per_sec)

def plot_roofline(ax):
    arith_intensity = np.arange(0, 10**5, 1)
    arith_intensity2 = np.arange(0, 10**3, 1)
    plt.plot(arith_intensity, [tflops_per_sec(x) for x in arith_intensity], 'b-', linewidth=2)
    plt.plot(arith_intensity, peak_tflops_per_sec * np.ones(np.size(arith_intensity)), 'b--', linewidth=2)
    plt.plot(arith_intensity2, [x * peak_tb_per_sec for x in arith_intensity2], 'b--', linewidth=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Arithmetic intensity [FLOP/byte]')
    ax.set_ylabel('Performance [TFLOP/s]')
    ax.grid(alpha=0.7, linestyle='--')

def read_stats():
    with open('stats.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        shapes = []
        flops = []
        for i, row in enumerate(reader):
            if i == 0: continue
            shapes.append(row[0])
            flops.append(row[2])
        return shapes, flops

def plot_data(shapes, flops):
    color = ['r', 'g', 'b', 'k', 'y']
    sym = ['*', 'o', '+']
    i = 0
    j = 0
    for shape, flop in zip(shapes, flops):
        M, N, K = [int(x) for x in shape.split('x')]
        flop = float(flop)
        if flop < 1e-5:
            continue
        plt.plot(compute_ai(M, N, K), flop, color[i] + sym[j], label=f'{M}x{N}x{K}', markersize=10)
        i = (i + 1) % len(color)
        j = (j + 1) % len(sym)


fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(1, 1, 1)
plot_roofline(ax)

shapes, flops = read_stats()
plot_data(shapes, flops)

plt.legend(loc='best')
plt.savefig('rooflines.png', dpi=300)
