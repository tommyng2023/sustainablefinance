# ============================================
# SUSTAINABLE FINANCE APP – FULL VERSION
# ESG + CML + Utility Curves
# ============================================

# ------------------------------
# IMPORTS (MUST BE FIRST)
# ------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------
# USER INPUTS
# ------------------------------

print("==== Sustainable Finance Portfolio App ====\n")

# Asset 1
r1 = float(input("Asset 1 Expected Return (%) [e.g., 5]: ")) / 100
sd1 = float(input("Asset 1 Standard Deviation (%) [e.g., 9]: ")) / 100
esg1 = float(input("Asset 1 ESG Rating (0–100): "))

# Asset 2
r2 = float(input("\nAsset 2 Expected Return (%) [e.g., 12]: ")) / 100
sd2 = float(input("Asset 2 Standard Deviation (%) [e.g., 20]: ")) / 100
esg2 = float(input("Asset 2 ESG Rating (0–100): "))

# Market Inputs
rho = float(input("\nCorrelation between Asset 1 and 2 [-1 to 1]: "))
r_free = float(input("Risk-Free Rate (%) [e.g., 2]: ")) / 100

# Preferences
gamma = float(input("\nRisk Aversion (γ) [e.g., 5]: "))
lambda_esg = float(input("ESG Preference Weight (λ) [0–1]: "))

# ------------------------------
# FUNCTIONS
# ------------------------------

def portfolio_return(w):
   return w * r1 + (1 - w) * r2

def portfolio_sd(w):
   return np.sqrt(
       w**2 * sd1**2 +
       (1-w)**2 * sd2**2 +
       2 * rho * w * (1-w) * sd1 * sd2
   )

def portfolio_esg(w):
   return w * esg1 + (1 - w) * esg2

# Mean–Variance Utility
def utility_mv(ret, sd):
   return ret - 0.5 * gamma * sd**2

# ESG-Adjusted Utility
def utility_esg(ret, sd, esg):
   return (1 - lambda_esg) * utility_mv(ret, sd) + lambda_esg * (esg / 100)

# ------------------------------
# CALCULATIONS
# ------------------------------

weights = np.linspace(0, 1, 1000)

returns = []
risks = []
esg_scores = []
utilities_mv = []
utilities_esg = []
sharpe_ratios = []

for w in weights:
   ret = portfolio_return(w)
   sd = portfolio_sd(w)
   esg = portfolio_esg(w)

   returns.append(ret)
   risks.append(sd)
   esg_scores.append(esg)

   utilities_mv.append(utility_mv(ret, sd))
   utilities_esg.append(utility_esg(ret, sd, esg))

   if sd > 0:
       sharpe_ratios.append((ret - r_free) / sd)
   else:
       sharpe_ratios.append(-np.inf)

# Convert to arrays
returns = np.array(returns)
risks = np.array(risks)
utilities_mv = np.array(utilities_mv)
utilities_esg = np.array(utilities_esg)

# ------------------------------
# OPTIMAL PORTFOLIOS
# ------------------------------

# Mean–Variance Optimal
idx_mv = np.argmax(utilities_mv)
w_mv = weights[idx_mv]
ret_mv = returns[idx_mv]
sd_mv = risks[idx_mv]

# ESG Optimal
idx_esg = np.argmax(utilities_esg)
w_esg = weights[idx_esg]
ret_esg = returns[idx_esg]
sd_esg = risks[idx_esg]
esg_opt = esg_scores[idx_esg]

# Tangency Portfolio (CML)
idx_tan = np.argmax(sharpe_ratios)
w_tan = weights[idx_tan]
ret_tan = returns[idx_tan]
sd_tan = risks[idx_tan]

# ------------------------------
# OUTPUT TABLE
# ------------------------------

print("\n==== Portfolio Comparison ====\n")

df = pd.DataFrame({
   "Portfolio": ["Mean-Variance", "ESG-Optimal", "Tangency"],
   "Weight Asset 1 (%)": [
       round(w_mv*100,2),
       round(w_esg*100,2),
       round(w_tan*100,2)
   ],
   "Weight Asset 2 (%)": [
       round((1-w_mv)*100,2),
       round((1-w_esg)*100,2),
       round((1-w_tan)*100,2)
   ]
})

print(df.to_string(index=False))

print("\nESG Optimal Portfolio Characteristics:")
print(f"Expected Return: {ret_esg*100:.2f}%")
print(f"Risk (Std Dev): {sd_esg*100:.2f}%")
print(f"Portfolio ESG Score: {esg_opt:.2f}")
print(f"Mean-Variance Utility: {utilities_mv[idx_esg]:.4f}")
print(f"ESG Utility: {utilities_esg[idx_esg]:.4f}")

# ------------------------------
# PLOT
# ------------------------------

plt.figure(figsize=(10,7))

# Efficient Frontier
plt.plot(risks, returns, label="Efficient Frontier", linewidth=2)

# Capital Market Line
if sd_tan > 0:
   sigma_range = np.linspace(0, max(risks)*1.2, 200)
   ret_cml = r_free + (ret_tan - r_free)/sd_tan * sigma_range
   plt.plot(sigma_range, ret_cml, linestyle="--", label="Capital Market Line")

# Mark portfolios
plt.scatter(sd_mv, ret_mv, s=120, label="MV Optimum")
plt.scatter(sd_esg, ret_esg, s=120, label="ESG Optimum")
plt.scatter(sd_tan, ret_tan, s=180, marker="*", label="Tangency Portfolio")
plt.scatter(0, r_free, s=100, marker="s", label="Risk-Free Asset")

# ------------------------------
# MEAN–VARIANCE INDIFFERENCE CURVE
# ------------------------------

U_mv_star = utility_mv(ret_mv, sd_mv)
sigma_curve = np.linspace(0, max(risks)*1.2, 200)
mu_mv_curve = U_mv_star + 0.5 * gamma * sigma_curve**2

plt.plot(sigma_curve, mu_mv_curve,
        linestyle=":",
        linewidth=2,
        label="MV Indifference Curve")

# ------------------------------
# ESG INDIFFERENCE CURVE
# ------------------------------

U_esg_star = utility_esg(ret_esg, sd_esg, esg_opt)

if lambda_esg < 1:
   mu_esg_curve = (
       (U_esg_star - lambda_esg*(esg_opt/100))/(1-lambda_esg)
       + 0.5 * gamma * sigma_curve**2
   )

   plt.plot(sigma_curve, mu_esg_curve,
            linestyle="-.",
            linewidth=2,
            label="ESG Indifference Curve")

plt.xlabel("Risk (Standard Deviation)")
plt.ylabel("Expected Return")
plt.title("Sustainable Portfolio Optimisation\n(MV vs ESG with CML)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
