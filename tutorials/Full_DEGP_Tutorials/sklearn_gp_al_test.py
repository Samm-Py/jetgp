import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats.qmc import Sobol
# ---------------------------
# True function
# ---------------------------
def true_function(X, alg=np):
    x = X[:, 0]
    return alg.sin(x)

# ---------------------------
# Settings
# ---------------------------
lb_x = 0.2
ub_x = 5.0
num_candidate_pts = 500
num_initial_points = 4
num_points_to_add = 1

domain_bounds = [(lb_x, ub_x)]
lb = np.array([b[0] for b in domain_bounds])
ub = np.array([b[1] for b in domain_bounds])

# --- Stage 1: Candidate points using Sobol sequence ---
sampler = Sobol(d=len(domain_bounds), scramble=True)
coarse_points_unit = sampler.random(n=1024)

lb = np.array([b[0] for b in domain_bounds])
ub = np.array([b[1] for b in domain_bounds])
X_candidates = lb + coarse_points_unit * (ub - lb)

# Candidate points
X_domain= np.linspace(lb_x, ub_x, num_candidate_pts).reshape(-1,1)

# Initialize training points
X_train = np.linspace(lb_x, ub_x, num_initial_points).reshape(-1,1)
y_train = true_function(X_train).reshape(-1,1)

# GP setup
alpha = 0.0
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=.85, length_scale_bounds=(1e-2, 10.0))
gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=250, normalize_y=False)
gp.fit(X_train, y_train)

# Helper function: compute variance reduction for all candidates vectorized
def compute_imse_reduction_vectorized(gp, X_train, X_candidates, X_domain, alpha=0.0):
    """
    Vectorized IMSE reduction across a full domain for each candidate point,
    using the exact variance update formula.
    """
    X_candidates = np.sort(X_candidates, axis = 0)
    # --- Kernel matrices for training data ---
    K_train = gp.kernel_(X_train) + alpha * np.eye(len(X_train))   # n x n
    K_train_inv = np.linalg.inv(K_train)

    # Cross-covariance domain <-> train
    K_domain_train = gp.kernel_(X_domain, X_train)                # Nd x n
    K_domain_domain_diag = np.diag(gp.kernel_(X_domain))          # Nd

    # Current variance at domain points
    sigma_old2 = K_domain_domain_diag - np.sum(
        K_domain_train @ K_train_inv * K_domain_train, axis=1
    )  # Nd

    # Candidate-related quantities
    K_cand_train = gp.kernel_(X_candidates, X_train)              # Nc x n
    v_cand = K_train_inv @ K_cand_train.T                         # n x Nc

    denom = np.diag(gp.kernel_(X_candidates)) - np.sum(
        K_cand_train * v_cand.T, axis=1
    ) + alpha  # Nc

    # --- Cross-covariances ---
    K_domain_cand = gp.kernel_(X_domain, X_candidates)            # Nd x Nc
    correction = (K_domain_train @ v_cand)                        # Nd x Nc

    # Numerator of variance update: (k(x, x*) - k(x, X) K^-1 k(X, x*))^2
    numerator = (K_domain_cand - correction) ** 2                 # Nd x Nc

    # Variance reduction at each domain point for each candidate
    delta_sigma2 = numerator / denom[None, :]                     # Nd x Nc

    # IMSE reduction = average variance reduction over domain
    delta_imse = delta_sigma2.mean(axis=0)                        # Nc

    return delta_imse, sigma_old2

# ---------------------------
# Active learning loop
# ---------------------------
imse_history = []
imse_matrix = []

for iter_idx in range(num_points_to_add):
    gp.fit(X_train, y_train)
    
    # Compute IMSE reduction vectorized
    delta_imse, sigma_old2 = compute_imse_reduction_vectorized(gp, X_train, X_candidates, X_domain, alpha)
    plt.plot(delta_imse)
    # Store
    imse_history.append(np.max(delta_imse))
    imse_matrix.append(delta_imse)
    
    # Select candidate with maximum IMSE reduction
    best_idx = np.argmax(delta_imse)
    X_next = X_candidates[best_idx].reshape(1,-1)
    y_next = true_function(X_next).reshape(-1,1)
    
    # Add to training set
    X_train = np.vstack([X_train, X_next])
    y_train = np.vstack([y_train, y_next])

# Final GP fit
gp.fit(X_train, y_train)

# ---------------------------
# Plot max IMSE reduction per iteration
# ---------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, num_points_to_add+1), imse_history, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Max IMSE reduction")
plt.title("IMSE reduction over iterations")
plt.grid(True)
plt.show()

# ---------------------------
# Heatmap of IMSE reduction for all candidates
# ---------------------------
imse_matrix = np.array(imse_matrix)
plt.figure(figsize=(12,6))
plt.imshow(imse_matrix, aspect='auto', extent=[lb_x, ub_x, num_points_to_add, 1], cmap='viridis')
plt.colorbar(label='IMSE reduction')
plt.xlabel('Candidate point x')
plt.ylabel('Iteration')
plt.title('IMSE reduction for all candidate points at each iteration')
plt.show()