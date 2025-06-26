import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import os

st.set_page_config(page_title="Hypervolume Tracking")

# CSV 파일 로딩
hv_df_blue = pd.read_csv("C:/Dev/PythonProject/Data/hv_tracking_ga_bo.csv")   # GA+BO 결과
hv_df_red = pd.read_csv("C:/Dev/PythonProject/Data/hv_tracking_bo.csv")       # BO만 사용한 결과

iterations = hv_df_blue["iteration"].values
hv_blue = hv_df_blue["hv"].values
hv_red = hv_df_red["hv"].values

# 정적인 하이퍼볼륨 비교 그래프 생성
fig_final, ax_final = plt.subplots(figsize=(8, 4))
ax_final.plot(hv_df_blue["iteration"], hv_df_blue["hv"], marker='o', label='GA+BO (blue)')
ax_final.plot(hv_df_red["iteration"], hv_df_red["hv"], marker='o', color='red', label='BO (red)')
ax_final.set_xlabel("Iteration")
ax_final.set_ylabel("Hypervolume")
ax_final.set_title("Final Hypervolume Comparison")
ax_final.set_xticks(np.arange(1, max(len(hv_df_blue), len(hv_df_red)) + 1, 3))
ax_final.grid(True)
ax_final.legend()

# 저장
plt.tight_layout()
final_img_path = "final_hypervolume_comparison.png"
fig_final.savefig(final_img_path, dpi=300)
plt.close(fig_final)

# Streamlit에서 정적인 그래프 출력
st.subheader("최종 하이퍼볼륨 비교 그래프")
st.image(final_img_path, use_container_width=True)
