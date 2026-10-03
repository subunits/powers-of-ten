"""Powers of Ten, applied to AI scaling (PyTorch).

1. Film frames: L(n) = 10^n m, with a variable zoom rate r(t)
2. Fit a scaling law L(N) = E + A / N^alpha by LBFGS in log space
3. Extrapolate one more decade, and check how much that depends on noise
3b. Check how extrapolation error grows with distance, for a correctly specified
    model and for a misspecified one (the data has a term the model lacks)
4. Project compute growth over time
"""
import math
import torch

dt = torch.float64
LN10 = math.log(10.0)

# ---------------------------------------------------------------- 1. Film
n = torch.arange(-16, 25, dtype=dt)            # -16 ... 24
length = 10.0 ** n                             # meters
print(f"Frames: {len(n)}, span = {int(n[-1] - n[0])} orders of magnitude")

# Zoom rate r(t) in decades/s: fastest at the start and end, slowest at the
# middle of the leg (t = 120 s). The cosine integrates to zero over one full
# period, so the total is still 0.1 * 240 = 24 decades.
T = 240.0
t = torch.linspace(0, T, 2401, dtype=dt)
r = 0.1 * (1 + 0.5 * torch.cos(2 * math.pi * t / T))

# s(t) = integral of r, via the trapezoid rule (cumsum * dt is only a Riemann sum)
s = torch.cat([torch.zeros(1, dtype=dt), torch.cumulative_trapezoid(r, t)])

# Check against the closed form: s(t) = 0.1 * (t + 0.5 * T/(2 pi) * sin(2 pi t / T))
s_exact = 0.1 * (t + 0.5 * T / (2 * math.pi) * torch.sin(2 * math.pi * t / T))
assert torch.allclose(s, s_exact, atol=1e-6), "integration error too large"

L_t = 10.0 ** s                                # L(t) = 10^s(t)
edge_speed = LN10 * r * L_t                    # dL/dt = ln10 * r * L
print(f"After {T:.0f} s: L = {L_t[-1]:.3e} m, edge speed = {edge_speed[-1]:.3e} m/s")
print(f"Zoom rate range: {r.min():.3f} to {r.max():.3f} decades/s "
      f"(slowest at t = {t[r.argmin()]:.0f} s)")

# ------------------------------------------------- 2. Fit a scaling law
E_true, A_true, alpha_true = 1.69, 406.4, 0.34
N_obs = 10.0 ** torch.linspace(6, 9.5, 12, dtype=dt)        # 1M ... ~3B params
x = torch.log10(N_obs) - 6.0                   # decades above 1M params (conditioning)
N_new = 3e10                                   # ~10x beyond the data
x_new = torch.tensor([math.log10(N_new) - 6.0], dtype=dt)
truth_new = E_true + A_true / N_new ** alpha_true


def make_data(seed, noise=0.005):
    g = torch.Generator().manual_seed(seed)
    clean = E_true + A_true / N_obs ** alpha_true
    eps = torch.randn(clean.shape, generator=g, dtype=dt)
    return clean * (1 + noise * eps)


def fit(loss_obs):
    raw_E = torch.tensor(0.0, dtype=dt, requires_grad=True)
    a = torch.tensor(math.log(5.0), dtype=dt, requires_grad=True)   # log A' at N=1e6
    raw_alpha = torch.tensor(-1.0, dtype=dt, requires_grad=True)

    def model(x):
        E = torch.nn.functional.softplus(raw_E)
        alpha = torch.nn.functional.softplus(raw_alpha)
        return E + torch.exp(a - alpha * LN10 * x), E, alpha

    opt = torch.optim.LBFGS([raw_E, a, raw_alpha], lr=1.0, max_iter=500,
                            tolerance_grad=1e-12, tolerance_change=1e-14,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        pred, _, _ = model(x)
        loss = torch.mean((torch.log(pred) - torch.log(loss_obs)) ** 2)  # log-space error
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        _, E, alpha = model(x)
        pred_new, _, _ = model(x_new)
        final = torch.mean((torch.log(model(x)[0]) - torch.log(loss_obs)) ** 2)

    def predict(x_query):
        with torch.no_grad():
            return model(x_query)[0]
    return E.item(), alpha.item(), pred_new.item(), final.item(), predict


E_fit, alpha_fit, pred_new, final_loss, _ = fit(make_data(seed=0))
print(f"\nFit (seed 0): E = {E_fit:.3f} (true {E_true}), "
      f"alpha = {alpha_fit:.3f} (true {alpha_true}), final log-MSE = {final_loss:.2e}")

# ------------------------------------------ 3. Extrapolate one decade out
reducible = pred_new - E_fit
print(f"Predicted loss at {N_new:.0e} params: {pred_new:.3f} (truth {truth_new:.3f})")
print(f"Reducible part: {reducible:.3f}; one more decade multiplies it by "
      f"{10 ** -alpha_fit:.3f}")

# E and alpha trade off against each other with 12 points and 3.5 decades, so
# check how much a single fit depends on the noise draw.
res = torch.tensor([fit(make_data(s_))[:3] for s_ in range(50)], dtype=dt)
mean, std = res.mean(0), res.std(0)
print("\nSpread over 50 noise seeds (mean +/- std):")
print(f"  E         = {mean[0]:.3f} +/- {std[0]:.3f}")
print(f"  alpha     = {mean[1]:.3f} +/- {std[1]:.3f}")
print(f"  loss@{N_new:.0e} = {mean[2]:.3f} +/- {std[2]:.3f}  (truth {truth_new:.3f})")
print(f"  corr(E, alpha) = {torch.corrcoef(res[:, :2].T)[0, 1]:.2f}")

# ------------------- 3b. Extrapolation error vs distance, and misspecification
# The data above is generated from the exact formula being fitted, so the fit is
# expected to do well. Here we also add a second power-law term the 3-parameter
# model cannot represent, to see what misspecification does to extrapolation.
def true_clean(N, misspecified):
    out = E_true + A_true / N ** alpha_true
    return out + 60.0 / N ** 0.15 if misspecified else out


def sweep(misspecified, n_seeds=50):
    print(f"\nExtrapolation error vs distance "
          f"({'misspecified: extra 60/N^0.15 term' if misspecified else 'correctly specified'}):")
    print(f"  {'N':>7} {'decades out':>12} {'mean |err| %':>13} {'max |err| %':>12}")
    for N in [3e10, 3e11, 3e12, 3e13]:
        xq = torch.tensor([math.log10(N) - 6.0], dtype=dt)
        t_val = true_clean(torch.tensor(N, dtype=dt), misspecified).item()
        errs = []
        for sd in range(n_seeds):
            g = torch.Generator().manual_seed(sd)
            y = true_clean(N_obs, misspecified) * (
                1 + 0.005 * torch.randn(N_obs.shape, generator=g, dtype=dt))
            p = fit(y)[4](xq).item()
            errs.append(abs(p - t_val) / t_val * 100)
        e = torch.tensor(errs)
        print(f"  {N:7.0e} {math.log10(N / N_obs[-1].item()):12.1f} "
              f"{e.mean():13.2f} {e.max():12.2f}")


sweep(misspecified=False)
sweep(misspecified=True)

# ----------------------------- 4. Compute growth as a zoom in decades/year
growth_per_year = 4.5                          # Epoch AI: ~4-5x/yr historically; an assumption, not a forecast
r_year = math.log10(growth_per_year)
years = torch.arange(0, 6, dtype=dt)
compute = 1e24 * 10.0 ** (r_year * years)      # FLOPs, starting at 1e24
print(f"\nCompute rate: {r_year:.2f} decades/yr, doubling time = "
      f"{math.log10(2) / r_year * 12:.1f} months")
for y, c in zip(years.tolist(), compute.tolist()):
    print(f"  year {int(y)}: {c:.2e} FLOPs")
