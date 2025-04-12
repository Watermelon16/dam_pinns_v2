# Tối ưu hóa triển khai PINNs cho tính toán mặt cắt đập bê tông

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import base64
from io import BytesIO
import sqlite3
import threading
from contextlib import contextmanager
import streamlit as st
import plotly.graph_objects as go

# Thêm class mạng PINNs
device = "cuda" if torch.cuda.is_available() else "cpu"

class OptimalParamsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3), nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net(x)
        # Giới hạn đầu ra chính xác như trong file gốc
        n = out[:, 0] * 0.4             # n ∈ [0, 0.4]
        m = out[:, 1] * 3.5 + 0.5       # m ∈ [0.5, 4.0]
        xi = out[:, 2] * 0.99 + 0.01    # xi ∈ (0.01, 1]
        return n, m, xi

# Hàm tính vật lý dùng PINNs
def compute_physics(n, xi, m, H, gamma_bt, gamma_n, f, C, a1):
    # Chuyển đổi các tham số đầu vào thành tensors nếu chúng là scalars
    # Điều này đảm bảo tất cả các phép tính đều trả về tensors
    if not isinstance(H, torch.Tensor):
        H = torch.tensor(H, dtype=torch.float32, device=device)
    if not isinstance(gamma_bt, torch.Tensor):
        gamma_bt = torch.tensor(gamma_bt, dtype=torch.float32, device=device)
    if not isinstance(gamma_n, torch.Tensor):
        gamma_n = torch.tensor(gamma_n, dtype=torch.float32, device=device)
    if not isinstance(f, torch.Tensor):
        f = torch.tensor(f, dtype=torch.float32, device=device)
    if not isinstance(C, torch.Tensor):
        C = torch.tensor(C, dtype=torch.float32, device=device)
    if not isinstance(a1, torch.Tensor):
        a1 = torch.tensor(a1, dtype=torch.float32, device=device)
    
    B = H * (m + n * (1 - xi))
    G1 = 0.5 * gamma_bt * m * H**2
    G2 = 0.5 * gamma_bt * n * H**2 * (1 - xi)**2
    G = G1 + G2
    W1 = 0.5 * gamma_n * H**2
    W2_1 = gamma_n * n * (1 - xi) * xi * H**2
    W2_2 = 0.5 * gamma_n * n * H**2 * (1 - xi)**2
    W2 = W2_1 + W2_2
    Wt = 0.5 * gamma_n * a1 * H * (m * H + n * H * (1 - xi))
    P = G + W2 - Wt
    lG1 = H * (m / 6 - n * (1 - xi) / 2)
    lG2 = H * (m / 2 - n * (1 - xi) / 6)
    lt  = H * (m + n * (1 - xi)) / 6
    l2  = H * m / 2
    l22 = H * m / 2 + H * n * (1 - xi) / 6
    l1  = H / 3
    M0 = -G1 * lG1 - G2 * lG2 + Wt * lt - W2_1 * l2 - W2_2 * l22 + W1 * l1
    sigma = P / B - 6 * M0 / B**2
    Fct = f * (G + W2 - Wt) + C * H * (m + n * (1 - xi))
    Fgt = 0.5 * gamma_n * H**2
    K = Fct / Fgt
    A = 0.5 * H**2 * (m + n * (1 - xi)**2)
    return sigma, K, A, G, W1, W2, Wt, Fct, Fgt, B, M0, G1, G2, W2_1, W2_2, lG1, lG2, lt, l2, l22, l1, P

# Hàm mất mát cải tiến
def loss_function(sigma, K, A, Kc, alpha):
    k_factor = 1.0  # Có thể điều chỉnh lên 1.05 nếu cần dư ổn định
    K_min = Kc * k_factor
    BIG_PENALTY = 1e5
    
    penalty_K = BIG_PENALTY * torch.clamp(K_min - K, min=0)**2
    penalty_sigma = sigma**2
    objective = alpha * A
    return penalty_K.mean() + 100 * penalty_sigma.mean() + objective.mean()

# Hàm tối ưu hóa cải tiến với cơ chế hội tụ sớm
def optimize_dam_section(H, gamma_bt, gamma_n, f, C, Kc, a1, max_iterations=5000, convergence_threshold=1e-6, patience=50):
    alpha = 0.01  # hệ số phạt diện tích
    model = OptimalParamsNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False)

    data = torch.ones((1, 1), device=device)
    
    # Lưu lịch sử tối ưu hóa
    loss_history = []
    best_loss = float('inf')
    best_params = None
    patience_counter = 0
    
    for epoch in range(max_iterations):
        optimizer.zero_grad()
        n, m, xi = model(data)
        sigma, K, A = compute_physics(n, xi, m, H, gamma_bt, gamma_n, f, C, a1)[:3]
        loss = loss_function(sigma, K, A, Kc, alpha)
        loss.backward()
        optimizer.step()
        
        # Cập nhật learning rate
        scheduler.step(loss)
        
        # Lưu lịch sử loss
        current_loss = loss.item()
        loss_history.append(current_loss)
        
        # Kiểm tra điều kiện hội tụ
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = (n.detach().clone(), m.detach().clone(), xi.detach().clone())
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Kiểm tra điều kiện dừng sớm
        if patience_counter >= patience:
            print(f"Dừng sớm tại epoch {epoch} do không cải thiện sau {patience} vòng lặp")
            break
            
        # Kiểm tra hội tụ dựa trên giá trị loss
        if epoch > 100 and abs(loss_history[-1] - loss_history[-100]) < convergence_threshold:
            print(f"Đã hội tụ tại epoch {epoch} với sai số {abs(loss_history[-1] - loss_history[-100])}")
            break
            
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    # Sử dụng tham số tốt nhất nếu có
    if best_params is not None:
        n, m, xi = best_params
    else:
        model.eval()
        n, m, xi = model(data)
        
    # Tính toán các đại lượng vật lý với tham số tối ưu
    sigma, K, A, G, W1, W2, Wt, Fct, Fgt, B, M0, G1, G2, W2_1, W2_2, lG1, lG2, lt, l2, l22, l1, P = compute_physics(n, xi, m, H, gamma_bt, gamma_n, f, C, a1)
    
    # Tính độ lệch tâm
    e = B/2 - M0/P
    
    # Số vòng lặp thực tế đã thực hiện
    actual_iterations = epoch + 1

    # Hàm trợ giúp để xử lý cả tensor và float
    def get_value(x):
        if isinstance(x, torch.Tensor):
            return x.item()
        return float(x)

    return {
        'H': H,
        'gamma_bt': gamma_bt,
        'gamma_n': gamma_n,
        'f': f,
        'C': C,
        'Kc': Kc,
        'a1': a1,
        'n': get_value(n),
        'm': get_value(m),
        'xi': get_value(xi),
        'A': get_value(A),
        'K': get_value(K),
        'sigma': get_value(sigma),
        'G': get_value(G),
        'G1': get_value(G1),
        'G2': get_value(G2),
        'W1': get_value(W1),
        'W2': get_value(W2),
        'W2_1': get_value(W2_1),
        'W2_2': get_value(W2_2),
        'Wt': get_value(Wt),
        'Fct': get_value(Fct),
        'Fgt': get_value(Fgt),
        'B': get_value(B),
        'e': get_value(e),
        'M0': get_value(M0),
        'P': get_value(P),
        'lG1': get_value(lG1),
        'lG2': get_value(lG2),
        'lt': get_value(lt),
        'l2': get_value(l2),
        'l22': get_value(l22),
        'l1': get_value(l1),
        'iterations': actual_iterations,  # Số vòng lặp thực tế
        'max_iterations': max_iterations, # Số vòng lặp tối đa
        'loss_history': loss_history,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# ==== Biểu đồ lực dùng plotly có tương tác ====
def create_force_diagram_plotly(H, n, m, xi):
    B = H * (m + n * (1 - xi))
    lG1 = H * (m / 6 - n * (1 - xi) / 2)
    lG2 = H * (m / 2 - n * (1 - xi) / 6)
    lt  = H * (m + n * (1 - xi)) / 6
    l2  = H * m / 2
    l22 = H * m / 2 + H * n * (1 - xi) / 6
    l1  = H / 3
    mid = B / 2

    x0 = 0
    x1 = n * H * (1 - xi)
    x3 = x1
    x4 = x3 + m * H
    y0 = 0
    y3 = H

    x = [x0, x1, x1, x1, x4, x0]
    y = [y0, H * (1 - xi), H, H, y0, y0]

    fig = go.Figure()

    # Hình mặt cắt
    fig.add_trace(go.Scatter(x=x, y=y, fill='toself', mode='lines', line=dict(color='gray'), name='Mặt cắt'))

    def add_arrow(x, y, dx, dy, label):
        if label in ['W1']:
            fig.add_annotation(
                x=x, y=y,
                ax=x + dx, ay=y + dy,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=3, arrowsize=1.0, arrowwidth=2, arrowcolor='red')
            fig.add_annotation(
                x=x + dx, y=y + dy+1,
                text=f"{label} = {{:.1f}} T/m".format(np.random.uniform(50, 200)),
                showarrow=False,
                font=dict(size=12, color='black')
            )
        elif label in ['Wt']:
            fig.add_annotation(
                x=x, y=y,
                ax=x + dx, ay=y + dy,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=3, arrowsize=1.0, arrowwidth=2, arrowcolor='red')
            fig.add_annotation(
                x=x + dx, y=y + dy,
                text=f"{label} = {{:.1f}} T/m".format(np.random.uniform(50, 200)),
                showarrow=False,
                font=dict(size=12, color='black')
            )
        else:
            fig.add_annotation(
                x=x + dx, y=y + dy,
                ax=x, ay=y,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,arrowhead=3, arrowsize=1.0, arrowwidth=2, arrowcolor='red')
            fig.add_annotation(
                x=x, y=y,
                text=f"{label} = {{:.1f}} T/m".format(np.random.uniform(50, 200)),
                showarrow=False,
                font=dict(size=12, color='black'))

    add_arrow(mid - lG1, H / 3, 0, -5, 'G1')
    add_arrow(mid - lG2, H * (1 - xi) / 3, 0, -5, 'G2')
    add_arrow(mid - lt, 0, 0, -5, 'Wt')
    add_arrow(mid - l2, H * (1 - xi) + xi * H / 2, 0, -5, "W'2")
    add_arrow(mid - l22, 2/3 * H * (1 - xi), 0, -5, 'W"2')
    add_arrow(x0 - 2, l1, -5, 0, 'W1')

    
    # Màu đồng bộ áp lực nước (nhạt)
    water_color = 'rgba(223, 242, 255, 0.9)'
    water_line = dict(color='rgba(0,0, 255, 0.1)', dash='solid')

    # W1 - tam giác: từ (0,0) lên (0,H) rồi xuống (-H,0)
    fig.add_trace(go.Scatter(
        x=[0, 0, -0.5*H, 0],
        y=[0, H, 0, 0],
        fill='toself',
        mode='lines',
        line=water_line,
        fillcolor=water_color,
        name='W1'))

    # W'2 - hình chữ nhật đối xứng quanh gốc
    fig.add_trace(go.Scatter(
        x=[0, 0, x1, x1, 0],
        y=[H * (1 - xi), H, H, H * (1 - xi), H * (1 - xi)],
        fill='toself',
        mode='lines',
        line=water_line,
        fillcolor=water_color,
        name="W'2"))

    # W"2 - tam giác đối xứng quanh gốc
    fig.add_trace(go.Scatter(
        x=[0, x1, 0, 0],
        y=[H * (1 - xi), H * (1 - xi), 0, H * (1 - xi)],
        fill='toself',
        mode='lines',
        line=water_line,
        fillcolor=water_color,
        name='W"2'))

    # Wt - tam giác thấm nằm dưới với góc alpha1 (đáy nghiêng)
    fig.add_trace(go.Scatter(
        x=[x0, x4, x0],
        y=[0, 0, -0.3 * H],
        fill='toself',
        mode='lines',
        line=water_line,
        fillcolor=water_color,
        name='Wt'))

    # Ghi chú thông số động theo hình dạng đập
    fig.add_annotation(x=x1 * 0.7, y=H * (1 - xi) * 0.55, text="n", showarrow=False, font=dict(size=18, color='black', family='Arial Black'))
    fig.add_annotation(x=(x1 + x4) / 2, y=H * 0.6, text="m", showarrow=False, font=dict(size=18, color='black', family='Arial Black'))
    fig.add_annotation(x=x1 * 1.1, y=H * (1 - xi / 2), text="ξ", showarrow=False, font=dict(size=18, color='black', family='Arial Black'))

    fig.update_layout(
        title=f"Sơ đồ lực và phân bố áp lực (H = {H} m)",
        xaxis_title="Khoảng cách(m)",
        yaxis_title="Cao độ (m)",
        width=800,
        height=600,
        showlegend=False,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        plot_bgcolor='white')

    return fig

# Hàm tạo biểu đồ hàm mất mát
def plot_loss_curve(loss_history):
    """
    Tạo biểu đồ hàm mất mát từ lịch sử tối ưu hóa
    
    Parameters:
    -----------
    loss_history : list
        Danh sách giá trị hàm mất mát theo từng vòng lặp
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Biểu đồ Plotly tương tác
    """
    import plotly.graph_objects as go
    import numpy as np
    
    # Tạo mảng chỉ số vòng lặp
    epochs = np.arange(len(loss_history))
    
    # Tạo biểu đồ
    fig = go.Figure()
    
    # Thêm đường biểu diễn hàm mất mát
    fig.add_trace(go.Scatter(
        x=epochs,
        y=loss_history,
        mode='lines',
        name='Hàm mất mát',
        line=dict(color='#0066cc', width=2)
    ))
    
    # Cấu hình chung
    fig.update_layout(
        title='Biểu đồ hàm mất mát theo vòng lặp',
        xaxis_title='Vòng lặp',
        yaxis_title='Giá trị hàm mất mát',
        width=850,
        height=500,
        plot_bgcolor='white',
        hovermode='x unified'
    )
    
    # Thêm lưới
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        type='log'  # Sử dụng thang logarit cho trục y
    )
    
    return fig

import streamlit as st
import plotly.graph_objects as go
import math

# --- Hàm tạo biểu đồ mặt cắt thực tế đúng theo sơ đồ 1–8 ---
def create_actual_dam_profile(H_opt, n, m, xi, H_total, B_top):
    B = H_opt * (m + n * (1 - xi))
    H = H_opt

    x1, y1 = 0, 0
    x2, y2 = B - m * H, (1 - xi) * H
    x3, y3 = x2, H
    x4, y4 = B, 0
    x5, y5 = x2, H_total
    x6, y6 = x5 + B_top, H_total
    x8, y8 = x6, 0
    slope_34 = (y4 - y3) / (x4 - x3)
    y7 = y3 + slope_34 * (x6 - x3)
    x7 = x6

    x_poly = [x1, x2, x3, x5, x6, x7, x4, x1]
    y_poly = [y1, y2, y3, y5, y6, y7, y4, y1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_poly, y=y_poly, fill='toself', mode='lines', line=dict(color='gray')))

    fig.add_trace(go.Scatter(x=[x7, x3], y=[y7, y3], mode='lines', line=dict(dash='dot', color='black'), showlegend=False))

    # Kích thước Hₜ
    fig.add_annotation(x=x1 - 4.5, y=H_total / 2, text=f"Hₜ = {H_total:.2f} m", showarrow=False, font=dict(size=14))
    fig.add_shape(type="line", x0=x1 - 2.0, y0=0, x1=x1 - 2.0, y1=H_total, line=dict(width=1))
    fig.add_shape(type="line", x0=x1 - 3.0, y0=0, x1=x1 - 1.0, y1=0, line=dict(width=1))
    fig.add_shape(type="line", x0=x1 - 3.0, y0=H_total, x1=x1 - 1.0, y1=H_total, line=dict(width=1))

    # Kích thước B
    fig.add_annotation(x=(x1 + x4) / 2, y=-4.0, text=f"B = {B:.2f} m", showarrow=False, font=dict(size=14))
    fig.add_shape(type="line", x0=x1, y0=-3.0, x1=x4, y1=-3.0, line=dict(width=1))
    fig.add_shape(type="line", x0=x1, y0=-4, x1=x1, y1=-2.0, line=dict(width=1))
    fig.add_shape(type="line", x0=x4, y0=-4, x1=x4, y1=-2.0, line=dict(width=1))

    # Kích thước Bđ
    fig.add_annotation(x=(x5 + x6) / 2, y=H_total + 3.5, text=f"Bđ = {B_top:.2f} m", showarrow=False, font=dict(size=14))
    fig.add_shape(type="line", x0=x5, y0=H_total + 2.0, x1=x6, y1=H_total + 2.0, line=dict(width=1))
    fig.add_shape(type="line", x0=x5, y0=H_total + 1.0, x1=x5, y1=H_total + 3.0, line=dict(width=1))
    fig.add_shape(type="line", x0=x6, y0=H_total + 1.0, x1=x6, y1=H_total + 3.0, line=dict(width=1))

    # Hệ số n tại đoạn 1–2
    angle_n = math.degrees(math.atan2(y2 - y1, x2 - x1))
    fig.add_annotation(x=(x1 + x2)/2-2, y=(y1 + y2)/2, text=f"n = {n:.2f}", textangle=-angle_n, showarrow=False, font=dict(size=14))

    # Hệ số m tại đoạn 3–4
    angle_m = math.degrees(math.atan2(y4 - y3, x4 - x3))
    fig.add_annotation(x=(x3 + x4)/2+2, y=(y3 + y4)/2, text=f"m = {m:.2f}", textangle=-angle_m, showarrow=False, font=dict(size=14))

    fig.update_layout(
        title="Mặt cắt thực tế của đập bê tông trọng lực",
        xaxis_title="Chiều rộng (m)",
        yaxis_title="Chiều cao (m)",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        showlegend=False,
        width=800,
        height=600,
        plot_bgcolor='white'
    )

    return fig

# --- Giao diện Streamlit ---
st.header("📐 Tạo mặt cắt thực tế từ mặt cắt tối ưu")

if 'result' not in st.session_state:
    st.warning("Vui lòng tính toán mặt cắt tối ưu trước!")
else:
    result = st.session_state['result']

    with st.form("actual_profile_form"):
        st.markdown("#### Nhập thông số đập thực tế")
        H_total = st.number_input("Chiều cao đập thực tế Hₜ (m)", min_value=result['H'], value=result['H'] + 10.0, step=1.0)
        B_top = st.number_input("Chiều rộng đỉnh đập Bđ (m)", min_value=1.0, value=5.0, step=0.5)
        submitted = st.form_submit_button("Vẽ mặt cắt thực tế")

    if submitted:
        fig = create_actual_dam_profile(
            H_opt=result['H'],
            n=result['n'],
            m=result['m'],
            xi=result['xi'],
            H_total=H_total,
            B_top=B_top
        )
        st.plotly_chart(fig, use_container_width=True)
