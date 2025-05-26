import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import streamlit as st
import os

st.set_page_config(page_title="Hypervolume tracking")

# CSV 파일 로딩
hv_df_blue = pd.read_csv("hv_tracking_ga_bo.csv")   # GA+BO 결과
hv_df_red = pd.read_csv("hv_tracking_bo.csv")       # BO만 사용한 결과

iterations = hv_df_blue["iteration"].values
hv_blue = hv_df_blue["hv"].values
hv_red = hv_df_red["hv"].values

# 애니메이션 생성
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, iterations[-1] + 1)
ax.set_ylim(0, max(hv_blue.max(), hv_red.max()) * 1.1)
ax.set_xlabel("Iteration")
ax.set_ylabel("Hypervolume")
ax.set_title("Hypervolume Comparison: GA+BO vs BO")
line_blue, = ax.plot([], [], 'o-', label="GA+BO", color='blue')
line_red, = ax.plot([], [], 'o-', label="BO only", color='red')
ax.legend()
ax.grid(True)

def update(frame):
    line_blue.set_data(iterations[:frame], hv_blue[:frame])
    line_red.set_data(iterations[:frame], hv_red[:frame])
    return line_blue, line_red

ani = animation.FuncAnimation(fig, update, frames=len(iterations)+1, interval=400, blit=True)

# GIF로 저장
gif_path = "hypervolume_comparison.gif"
#ani.save(gif_path, writer='pillow', fps=5)
plt.close(fig)  # Streamlit에 중복 출력 방지

# ✅ Streamlit에서 GIF 표시
st.subheader("하이퍼볼륨 비교 (GA+BO vs BO only)")
with open(gif_path, "rb") as f:
    gif_bytes = f.read()
st.image(gif_bytes, caption="Hypervolume over Iterations", use_container_width=True)

# 마지막 프레임 기준 정적인 비교 그래프 저장
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
#fig_final.savefig("final_hypervolume_comparison.png", dpi=300)

# Streamlit에서 정적인 최종 그래프 출력
st.subheader("최종 하이퍼볼륨 비교 그래프")
st.image("final_hypervolume_comparison.png", use_container_width=True)


