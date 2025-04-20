
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import csv

# Monte Carlo SIS simulation
def run_sis_mc(G, beta, mu, n_rep=50, rho0=0.2, t_max=1000, t_trans=900):
    N = G.number_of_nodes()
    nodes = list(G.nodes())
    avg_rho = 0.0
    for rep in range(n_rep):
        # initialize state: 1 infected, 0 susceptible
        state = {node: 1 if random.random() < rho0 else 0 for node in nodes}
        rho_time = []
        for t in range(t_max):
            new_state = state.copy()
            # recovery
            for node, s in state.items():
                if s == 1 and random.random() < mu:
                    new_state[node] = 0
            # infection
            for node, s in state.items():
                if s == 0:
                    for nbr in G.neighbors(node):
                        if state[nbr] == 1 and random.random() < beta:
                            new_state[node] = 1
                            break
            state = new_state
            rho_t = sum(state.values()) / N
            if t >= t_trans:
                rho_time.append(rho_t)
        avg_rho += np.mean(rho_time)
    return avg_rho / n_rep

# MMCA theoretical prediction
def run_mmca(G, beta, mu, tol=1e-6, max_iter=10000):
    N = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    rho = np.random.rand(N)
    for _ in range(max_iter):
        rho_new = np.zeros_like(rho)
        # compute infection probability per node
        for i in range(N):
            prod = 1.0
            for j in range(N):
                if A[i, j] == 1:
                    prod *= (1 - beta * rho[j])
            rho_new[i] = (1 - rho[i]) * (1 - prod) + (1 - mu) * rho[i]
        if np.linalg.norm(rho_new - rho) < tol:
            break
        rho = rho_new
    return rho.mean()

if __name__ == "__main__":
    # parameters
    N = 1000
    betas = np.arange(0.0, 0.301, 0.01)
    mus = [0.2, 0.4]
    network_types = [
        ('ER', 4),
        ('ER', 6),
        ('BA', 4),
        ('BA', 6),
    ]
    results = {}

    for mu in mus:
        plt.figure(figsize=(8, 6))
        for net_name, k in network_types:
            # generate network
            if net_name == 'ER':
                p = k / (N - 1)
                G = nx.erdos_renyi_graph(N, p)
            else:  # BA
                m = k // 2
                G = nx.barabasi_albert_graph(N, m)

            rhos_mc = []
            rhos_th = []
            for beta in betas:
                rho_mc = run_sis_mc(G, beta, mu)
                rhos_mc.append(rho_mc)
                rho_th = run_mmca(G, beta, mu)
                rhos_th.append(rho_th)
                print(f"mu={mu}, {net_name}<k>={k}, beta={beta:.2f} -> mc={rho_mc:.4f}, th={rho_th:.4f}")

            # save results CSV
            csv_file = f"results_{net_name}_k{k}_mu{mu}.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['beta', 'rho_mc', 'rho_mmca'])
                writer.writerows(zip(betas, rhos_mc, rhos_th))

            # plot Monte Carlo
            plt.plot(betas, rhos_mc, label=f"{net_name} <k>={k} MC")
            # plot MMCA
            plt.plot(betas, rhos_th, '--', label=f"{net_name} <k>={k} MMCA")

        plt.title(f"SIS epidemic diagram (mu={mu})")
        plt.xlabel(r"Infection probability $\beta$")
        plt.ylabel(r"Stationary fraction infected $\rho$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"sis_epidemic_mu{mu}.png")
        plt.close()

    print("Simulation and theoretical predictions complete. Results saved to CSV and PNG files.")



