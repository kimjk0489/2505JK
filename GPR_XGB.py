import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch

st.set_page_config(page_title="XGBoost vs GPR 예측 비교")
st.title("새로운 조성값 예측 비교 (XGBoost vs GPR)")

# 데이터 로딩
data_path = "BO_slurry_data.csv"
df = pd.read_csv(data_path)

x_cols = ["Graphite", "Carbon_black", "CMC", "SBR", "Solvent"]
y_cols = ["Yield_stress", "n", "K", "Viscosity"]

X_raw = df[x_cols].values
Y_raw = df[["Yield_stress", "Viscosity"]].values
graphite_idx = x_cols.index("Graphite")
graphite_wt_values = X_raw[:, graphite_idx].reshape(-1, 1)
Y_raw_extended = np.hstack([Y_raw, graphite_wt_values])

# 데이터 정규화 (MinMax)
scaler = MinMaxScaler()
scaler.fit(X_raw)
X_scaled = scaler.transform(X_raw)

# XGBoost 모델을 한 번만 학습
model_xgb_yield = XGBRegressor(random_state=42, verbosity=0)
model_xgb_yield.fit(X_raw, df["Yield_stress"])

model_xgb_visc = XGBRegressor(random_state=42, verbosity=0)
model_xgb_visc.fit(X_raw, df["Viscosity"])

# GPR 모델도 한 번만 학습
train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw_extended, dtype=torch.double)
train_y[:, 1] = -train_y[:, 1]  # Viscosity 음수화

model_gp = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(model_gp.likelihood, model_gp)
fit_gpytorch_mll(mll)

# 사용자 입력
st.subheader("새로운 조성 입력")
graphite = st.number_input("Graphite (wt%)", value=10.0)
carbon_black = st.number_input("Carbon black (wt%)", value=5.0)
cmc = st.number_input("CMC (wt%)", value=0.5)
sbr = st.number_input("SBR (wt%)", value=1.0)
solvent = st.number_input("Solvent (wt%)", value=83.5)

if st.button("예측 실행 (XGBoost & GPR)"):
    # 입력값
    new_input = np.array([[graphite, carbon_black, cmc, sbr, solvent]])

    # XGBoost 예측 (모델 재학습 X)
    yield_xgb_pred = model_xgb_yield.predict(new_input)[0]
    visc_xgb_pred = model_xgb_visc.predict(new_input)[0]

    # GPR 예측 (모델 재학습 X)
    X_scaled_new = scaler.transform(new_input)
    X_tensor_new = torch.tensor(X_scaled_new, dtype=torch.double)
    posterior = model_gp.posterior(X_tensor_new)
    pred_gp = posterior.mean.detach().cpu().numpy()[0]
    yield_pred_gp = pred_gp[0]
    visc_pred_gp = -pred_gp[1]

    # 비교 표 출력 (Graphite는 예측X → 사용자 입력값으로 표시)
    st.subheader("XGBoost vs GPR 예측 비교")
    compare_df = pd.DataFrame({
        "Parameter": ["Yield_stress (Pa)", "Viscosity (Pa.s)", "Graphite wt% (입력값)"],
        "XGBoost Prediction": [
            f"{yield_xgb_pred:.3f}",
            f"{visc_xgb_pred:.3f}",
            f"{graphite:.2f}"
        ],
        "GPR Prediction": [
            f"{yield_pred_gp:.3f}",
            f"{visc_pred_gp:.3f}",
            f"{graphite:.2f}"
        ]
    })
    st.dataframe(compare_df, hide_index=True)
