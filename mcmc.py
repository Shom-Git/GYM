import torch
import matplotlib.pyplot as plt

def log_prob(x: torch.Tensor) -> torch.Tensor:
    # Unnormalized log density of N(0, 1)
    return -0.5 * x**2

def metropolis(num_samples=5000, step_size=0.5):
    samples = []
    x = torch.zeros(1)  # initial state (1D)

    for _ in range(num_samples):
        proposal = x + step_size * torch.randn_like(x)

        # log acceptance ratio
        log_alpha = log_prob(proposal) - log_prob(x)
        alpha = torch.exp(torch.clamp(log_alpha, max=0))

        # accept/reject
        if torch.rand(1) < alpha:
            x = proposal

        samples.append(x.clone())

    return torch.cat(samples)

# run sampler
samples = metropolis()

# -------- Stats --------
print("Mean:", samples.mean().item())
print("Std:", samples.std().item())
print("Min:", samples.min().item())
print("Max:", samples.max().item())

# -------- Plot Histogram --------
plt.figure(figsize=(6,4))
plt.hist(samples.numpy(), bins=50, density=True)
plt.title("Histogram of Samples (Should Approximate N(0,1))")
plt.xlabel("x")
plt.ylabel("density")
plt.savefig("mcmc_hist.png")
plt.show()
