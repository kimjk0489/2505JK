import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning, DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.optim import optimize_acqf
from scipy.spatial import ConvexHull

st.set_page_config(page_title="Bayesian Optimization")
st.title("Slurry 조성 최적화 : Bayesian Optimization")

#CSV_PATH = "C:\Dev\PythonProject\Data\BO_Slurry_data.csv"
#df = pd.read_csv(CSV_PATH)
CSV_URL = "https://github.com/kimjk0489/2505JK/blob/main/BO_Slurry_data.csv"
df = pd.read_csv(CSV_URL)

x_cols = ["Graphite", "Carbon_black", "CMC", "SBR", "Solvent"]
y_cols = ["Yield_stress", "Viscosity"]

X_raw = df[x_cols].values
Y_raw = df[y_cols].values
graphite_idx = x_cols.index("Graphite")
graphite_wt_values = X_raw[:, graphite_idx].reshape(-1, 1)
Y_raw_extended = np.hstack([Y_raw, graphite_wt_values])

x_scaler = MinMaxScaler()
x_scaler.fit(X_raw)
X_scaled = x_scaler.transform(X_raw)

train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw_extended, dtype=torch.double)

train_y_hv = train_y.clone()
train_y_hv[:, 1] = -train_y_hv[:, 1]

ref_point = [0.0, -10.0, 20.0]
partitioning = NondominatedPartitioning(ref_point=torch.tensor(ref_point, dtype=torch.double), Y=train_y_hv)

model = SingleTaskGP(train_x, train_y_hv)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

acq_func = qExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point,
    partitioning=partitioning
)

bounds = torch.tensor([[0.0] * len(x_cols), [1.0] * len(x_cols)], dtype=torch.double)
candidate_scaled, _ = optimize_acqf(acq_func, bounds=bounds, q=1, num_restarts=10, raw_samples=128)
candidate_wt = x_scaler.inverse_transform(candidate_scaled.detach().cpu().numpy())[0]
candidate_wt = candidate_wt / np.sum(candidate_wt) * 100

# 조성 Table 출력
st.subheader("qEHVI를 통한 최적화 조성 추천")
table_composition = pd.DataFrame({
    "Graphite": [f"{candidate_wt[x_cols.index('Graphite')]:.2f} wt%"],
    "Carbon_black": [f"{candidate_wt[x_cols.index('Carbon_black')]:.2f} wt%"],
    "CMC": [f"{candidate_wt[x_cols.index('CMC')]:.2f} wt%"],
    "SBR": [f"{candidate_wt[x_cols.index('SBR')]:.2f} wt%"],
    "Solvent": [f"{candidate_wt[x_cols.index('Solvent')]:.2f} wt%"]
})
st.dataframe(table_composition, hide_index=True)

X_predict = x_scaler.transform(candidate_wt.reshape(1, -1))
X_tensor = torch.tensor(X_predict, dtype=torch.double)
posterior = model.posterior(X_tensor)  # 예측 분포 획득
pred_mean = posterior.mean.detach().cpu().numpy()[0]
pred_std = posterior.variance.sqrt().detach().cpu().numpy()[0]

# 평균값
yield_pred = pred_mean[0]
visc_pred = -pred_mean[1]  # 최소화를 위해 부호 반전
graphite_pred = pred_mean[2]

# 표준편차
yield_std = pred_std[0]
visc_std = pred_std[1]

# 95% 신뢰구간 계산 (1.96 * std)
yield_ci_lower = yield_pred - 1.96 * yield_std
yield_ci_upper = yield_pred + 1.96 * yield_std
visc_ci_lower = visc_pred - 1.96 * visc_std
visc_ci_upper = visc_pred + 1.96 * visc_std

# 예측값 Table (깔끔한 표 형식으로 출력)
st.subheader("추천 조성의 유변학적 물성 예측 (95% 신뢰구간)")

# 값 계산
yield_ci = (yield_pred - 1.96 * yield_std, yield_pred + 1.96 * yield_std)
visc_ci = (visc_pred - 1.96 * visc_std, visc_pred + 1.96 * visc_std)

# 테이블 생성
results_df = pd.DataFrame({
    "Property": ["Yield Stress (Pa)", "Viscosity (Pa·s)"],
    "Predicted": [f"{yield_pred:.2f}", f"{visc_pred:.3f}"],
    "Std Dev": [f"±{yield_std:.2f}", f"±{visc_std:.3f}"],
    "95% CI": [f"[{yield_ci[0]:.2f}, {yield_ci[1]:.2f}]",
               f"[{visc_ci[0]:.3f}, {visc_ci[1]:.3f}]"]
})
st.dataframe(results_df, hide_index=True)


pareto_mask = is_non_dominated(train_y_hv)
train_y_vis_plot = train_y_hv.clone()
train_y_vis_plot[:, 1] = -train_y_vis_plot[:, 1]
pareto_points = train_y_vis_plot[pareto_mask].numpy()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(train_y_vis_plot[:, 1], train_y_vis_plot[:, 0], train_y_vis_plot[:, 2],
           color='gray', alpha=0.7, label='Data', s=30, depthshade=True)
ax.scatter(pareto_points[:, 1], pareto_points[:, 0], pareto_points[:, 2],
           color='red', edgecolors='black', s=90, marker='o', depthshade=True, label='Pareto Front')
ax.scatter(visc_pred, yield_pred, graphite_pred,
           color='yellow', edgecolors='black', s=200, marker='^', label='Candidate')

if len(pareto_points) >= 4:
    try:
        hull = ConvexHull(pareto_points)
        for simplex in hull.simplices:
            tri = pareto_points[simplex]
            ax.plot_trisurf(tri[:, 1], tri[:, 0], tri[:, 2],
                            color='pink', alpha=0.4, edgecolor='gray', linewidth=1.2)
    except Exception as e:
        st.warning(f"Convex Hull 실패: {e}")

ax.set_xlabel("Viscosity [Pa.s] (↓)", fontsize=12, labelpad=10)
ax.set_ylabel("Yield Stress [Pa] (↑)", fontsize=12, labelpad=10)
ax.set_zlabel("Graphite wt% (↑)", fontsize=12, labelpad=15)
ax.set_zlim(20, 40)
ax.zaxis.set_ticks(np.arange(20, 45, 5))
ax.view_init(elev=25, azim=135)
ax.legend()
ax.grid(True)
plt.tight_layout()
st.pyplot(fig)

st.subheader("2D 시각화: 각 조성 목표 간 관계")

# 복원된 값
yield_stress_vals = train_y_hv[:, 0].numpy()
viscosity_vals = -train_y_hv[:, 1].numpy()  # 부호 복원
graphite_vals = train_y_hv[:, 2].numpy()
is_pareto = is_non_dominated(train_y_hv).numpy()

# 시각화 함수
def plot_2d(x, y, xlabel, ylabel, x_cand, y_cand, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, c='gray', alpha=0.5, label='All Data')
    ax.scatter(x[is_pareto], y[is_pareto], c='red', label='Pareto Front', edgecolors='black')
    ax.scatter(x_cand, y_cand, c='yellow', marker='^', edgecolors='black', s=100, label='Candidate')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# 1. Yield Stress vs Viscosity
plot_2d(viscosity_vals, yield_stress_vals,
        "Viscosity [Pa·s]", "Yield Stress [Pa]",
        visc_pred, yield_pred,
        "Yield Stress vs. Viscosity")

# 2. Yield Stress vs Graphite
plot_2d(graphite_vals, yield_stress_vals,
        "Graphite wt%", "Yield Stress [Pa]",
        graphite_pred, yield_pred,
        "Yield Stress vs. Graphite")

# 3. Viscosity vs Graphite
plot_2d(graphite_vals, viscosity_vals,
        "Graphite wt%", "Viscosity [Pa·s]",
        graphite_pred, visc_pred,
        "Viscosity vs. Graphite")

# Pareto front에 해당하는 데이터 추출
pareto_indices = np.where(pareto_mask.numpy())[0]
pareto_X = X_raw[pareto_indices]                   # 조성 원본
pareto_Y = Y_raw_extended[pareto_indices]          # 물성 원본

# DataFrame 구성
pareto_df = pd.DataFrame(pareto_X, columns=x_cols)
pareto_df["Yield stress (Pa)"] = pareto_Y[:, 0]
pareto_df["Viscosity (Pa.s)"] = pareto_Y[:, 1]
pareto_df.insert(0, "실험 번호", pareto_indices+1)

pareto_df.index = np.arange(1, len(pareto_df) + 1)
pareto_df.index.name = "Index"

# Streamlit에서 표로 출력
st.subheader("Pareto Front를 구성하는 실험 데이터")
st.dataframe(pareto_df.round(3), hide_index=False)

#st.subheader("\U0001F4CA 5-Fold Cross Validation (RMSE)")
#kf = KFold(n_splits=5, shuffle=True, random_state=42)
#rmse_dict = {"Yield_stress": [], "Viscosity": [], "Graphite_wt%": []}
#X_tensor = torch.tensor(X_scaled, dtype=torch.double)
#Y_tensor = torch.tensor(Y_raw_extended, dtype=torch.double)

#for train_idx, test_idx in kf.split(X_tensor):
    #X_train, Y_train = X_tensor[train_idx], Y_tensor[train_idx]
    #X_test, Y_test = X_tensor[test_idx], Y_tensor[test_idx]
    #Y_train_mod = Y_train.clone()
    #Y_train_mod[:, 1] = -Y_train_mod[:, 1]
    #model_cv = SingleTaskGP(X_train, Y_train_mod)
    #mll_cv = ExactMarginalLogLikelihood(model_cv.likelihood, model_cv)
    #fit_gpytorch_mll(mll_cv)
    #preds = model_cv.posterior(X_test).mean.detach().numpy()
    #preds[:, 1] = -preds[:, 1]
    #for i, target in enumerate(["Yield_stress", "Viscosity", "Graphite_wt%"]):
        #rmse = np.sqrt(mean_squared_error(Y_test[:, i], preds[:, i]))
        #rmse_dict[target].append(rmse)

#rmse_df = pd.DataFrame({
    #"Target": list(rmse_dict.keys()),
    #"RMSE Mean": [np.mean(v) for v in rmse_dict.values()],
    #"RMSE Std": [np.std(v) for v in rmse_dict.values()]
#})
#st.dataframe(rmse_df)

hv_log_path = "C:\Dev\PythonProject\Data\hv_tracking_bo.csv"
hv_list = []
ref_point_fixed = torch.tensor([0.0, -10.0, 20.0], dtype=torch.double)

for i in range(1, len(train_y_hv) + 1):
    current_Y = train_y_hv[:i].clone()
    try:
        bd = DominatedPartitioning(ref_point=ref_point_fixed, Y=current_Y.clone().detach())
        hv = bd.compute_hypervolume().item()
    except Exception as e:
        hv = float('nan')
        st.warning(f"{i}번째 계산 중 에러: {e}")
    hv_list.append({"iteration": i, "hv": hv})

hv_df = pd.DataFrame(hv_list)
hv_df.to_csv(hv_log_path, index=False)

fig_hv, ax_hv = plt.subplots(figsize=(8, 4))
ax_hv.plot(hv_df["iteration"], hv_df["hv"], marker='o', color='red')
ax_hv.set_xlabel("Iteration")
ax_hv.set_ylabel("Hypervolume")
ax_hv.set_title("3D Hypervolume Progress Over Iterations")
ax_hv.set_xticks(np.arange(1, hv_df["iteration"].max() + 1, 3))
ax_hv.grid(True)
st.pyplot(fig_hv)
