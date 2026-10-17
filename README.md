---
title: Urban Water Level Depth Monitor
emoji: 🌊
colorFrom: blue
colorTo: cyan
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Real-time Urban Flood Depth Estimation via Dual YOLOv8
---

# 🌊 AI-Powered Urban Water Level & Traffic Depth Monitor

Hệ thống giám sát mực nước ngập đô thị thời gian thực tích hợp đường ống AI kép (**Dual YOLOv8**):
- **Mô hình 1: YOLOv8s-seg** huấn luyện trên bộ dữ liệu `master-flood-road-segmentation` (10,892 mẫu) đạt mAP@50 phân đoạn mặt nước 45.82%.
- **Mô hình 2: YOLOv8s** phát hiện phương tiện giao thông (xe máy, ô tô, SUV, xe buýt, xe tải).
- **Scale Calibration & Depth Engine**: Tự động tính toán tỷ lệ scale $k = L_0 / L_{BB}$ và độ sâu ngập nước thực tế $h_{water}$ tại vị trí từng bánh xe mà không cần cảm biến vật lý.

### 📐 Công Thức Tính Độ Sâu:
1. $k = \frac{L_0}{L_{BB}} \quad (\text{cm / pixel})$
2. $h_{water} = h_{submerged\_pixel} \times k \quad (\text{cm})$
3. Phân loại:
   - **SHALLOW FLOOD**: $h_{visible\_pixel} > \frac{1}{2} H_{BB}$
   - **DEEP FLOOD**: $h_{visible\_pixel} \le \frac{1}{2} H_{BB}$
