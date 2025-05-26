import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Hypervolume tracking")

# CSV 파일 로딩
hv_df_blue = pd.read_csv("hv_tracking_ga_bo.csv")   # GA+BO 결과
hv_df_red = pd.read_csv("hv_tracking_bo.csv")       # BO만 사용한 결과

# ✅ 정적 최종 그래프
st.subheader("최종 하이퍼볼륨 비교 그래프")
fig_final, ax_final = plt.subplots(figsize=(8, 4))
ax_final.plot(hv_df_blue["iteration"], hv_df_blue["hv"], marker='o', label='GA+BO (blue)')
ax_final.plot(hv_df_red["iteration"], hv_df_red["hv"], marker='o', color='red', label='BO (red)')
ax_final.set_xlabel("Iteration")
ax_final.set_ylabel("Hypervolume")
ax_final.set_title("Final Hypervolume Comparison")
ax_final.set_xticks(np.arange(1, max(len(hv_df_blue), len(hv_df_red)) + 1, 3))
ax_final.grid(True)
ax_final.legend()
st.pyplot(fig_final)
