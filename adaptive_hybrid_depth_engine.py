#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
🌊 ADAPTIVE HYBRID FLOOD DEPTH ESTIMATION ENGINE (TAB 5)
Kết hợp 2 Phương pháp Đo Lường Thủy Văn Đô Thị Bổ Trợ Nhau:
1. Dựa trên Phương tiện: Tái tạo Bounding Box chìm (Amodal BBox) theo tỷ lệ bất biến
   QCVN 14 & 09:2015/BGTVT, giải quyết triệt để "Nghịch lý BBox bị cắt cụt" khi bánh xe chìm.
2. Dựa trên Vật làm mốc (Thủy chuẩn vật thể cố định): Sử dụng thước đo phối cảnh LUT O(1).
3. Bộ điều phối Trọng tài Thích ứng (Adaptive Arbiter):
   - Giờ cao điểm (Traffic Active): Đo theo lốp xe + Tự học chênh cao tim đường Delta_h.
   - Giờ vắng xe / Đêm khuya (Traffic Idle): Đo theo mốc vỉa hè + Bù trừ Delta_h.
   - Ngập cực hạn (>70cm): Phát hiện xe nổi bồng bềnh (Buoyancy Check) -> Chuyển sang mốc cao tầng.
4. Giao diện HUD X-Ray trực quan: Hộp nổi xanh lá, Hộp chìm cam X-Ray, Vạch tiếp đất vàng dạ quang.
================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import cv2
import time
import math
import torch

from fixed_object_gauge import VirtualStaffGauge

INFER_DEVICE = 0 if torch.cuda.is_available() else "cpu"


@dataclass
class HybridDepthReport:
    master_depth_cm: float
    source: str                 # 'vehicle_wheel', 'virtual_gauge', or 'emergency_pole'
    confidence_pct: int
    alert_level: int            # 0 to 5
    alert_title: str
    traffic_advice: str
    active_vehicles_count: int = 0
    vehicle_depth_cm: Optional[float] = None
    gauge_depth_cm: Optional[float] = None
    road_offset_delta_h: float = 6.5
    has_floating_suspect: bool = False
    details: Dict = field(default_factory=dict)


class VehicleAmodalDepthEstimator:
    """
    Ước lượng độ sâu ngập dựa trên Phương tiện giải quyết Nghịch lý BBox cắt cụt.
    Khi bánh xe bị ngập, bộ phát hiện đối tượng cắt cụt BBox tại mép nước (y2).
    Thuật toán dùng chiều rộng ngang bất biến (W_box) và tỷ lệ chuẩn theo luật Việt Nam
    để khôi phục vị trí tiếp đất của lốp xe (y_ground) dưới mặt nước.
    """

    # Tỷ lệ khung xe chuẩn theo Quy chuẩn Kỹ thuật Quốc gia QCVN 14 & 09:2015/BGTVT
    CANONICAL_RATIOS_BY_VIEW = {
        'motorcycle': {
            'front_rear': 1.60,   # Honda Wave/Vision: Rộng ~68-70cm, Cao ~110-112cm (W/H < 0.85)
            'quarter': 1.10,      # Góc 45 độ (0.85 <= W/H <= 1.35)
            'side': 0.60          # Dài ~190cm, Cao ~110cm (W/H > 1.35)
        },
        'car': {
            'front_rear': 0.84,   # Sedan Vios/Accent: Rộng ~173-175cm, Cao ~145-147cm (W/H < 1.40)
            'quarter': 0.55,      # (1.40 <= W/H <= 2.20)
            'side': 0.32          # Dài ~440-450cm, Cao ~145cm (W/H > 2.20)
        },
        'truck': {
            'front_rear': 1.18,   # Xe tải nhẹ Thaco/Kia: Rộng ~175-185cm, Cao ~210-235cm
            'quarter': 0.60,
            'side': 0.38
        },
        'bus': {
            'front_rear': 1.26,   # Xe buýt đô thị: Rộng ~250cm, Cao ~315cm
            'quarter': 0.62,
            'side': 0.30
        },
        'default': {
            'front_rear': 1.40,
            'quarter': 0.90,
            'side': 0.50
        }
    }

    # Chiều cao thực tế toàn thân chuẩn (cm) từ mặt đất đến nóc/gương
    CANONICAL_REAL_HEIGHTS = {
        'motorcycle': 110.0,
        'car': 147.0,
        'truck': 235.0,
        'bus': 315.0,
        'default': 120.0
    }

    # Ngưỡng chịu tải thủy lực tối đa trước khi phương tiện bị nổi bồng bềnh hoặc mất bám (cm)
    BUOYANCY_LIMITS = {
        'motorcycle': 50.0,  # Xe máy mất bám/cuốn trôi ở mức >50cm
        'car': 60.0,         # Ô tô con nổi bồng bềnh ở mức >55-60cm
        'truck': 110.0,
        'bus': 120.0,
        'default': 50.0
    }

    def __init__(self, camera_pitch_deg: float = 15.0, min_box_width_px: float = 40.0, max_scale_cm_per_px: float = 3.0):
        self.track_history = {}     # {track_id: [depth_measurements]}
        self.bbox_smoothing = {}    # {track_id: (x1, y1, x2, y2)}
        self.camera_pitch_deg = float(camera_pitch_deg)
        self.min_box_width_px = float(min_box_width_px)
        self.max_scale_cm_per_px = float(max_scale_cm_per_px)

    def classify_view_orientation(self, vehicle_class: str, visible_w: float, visible_h: float) -> str:
        """
        Phân loại góc nhìn phương tiện dựa trên tỷ lệ hộp nhìn thấy được.
        Ngưỡng được tinh chỉnh chịu ngập (Flood-tolerant): Khi xe bị ngập, H_vis giảm làm aspect = W / H_vis tăng,
        ngưỡng mở rộng giúp phân loại chính xác ngay cả khi xe đang ngập lội nước.
        """
        aspect = visible_w / max(1.0, visible_h)
        if vehicle_class in ('motorcycle', 'bicycle'):
            if aspect < 1.05:
                return 'front_rear'
            elif aspect <= 1.65:
                return 'quarter'
            else:
                return 'side'
        elif vehicle_class in ('car', 'suv'):
            if aspect < 1.75:
                return 'front_rear'
            elif aspect <= 2.60:
                return 'quarter'
            else:
                return 'side'
        else:
            if aspect < 1.35:
                return 'front_rear'
            elif aspect <= 2.20:
                return 'quarter'
            else:
                return 'side'

    def estimate_single_vehicle(self, bbox: Tuple[int, int, int, int], vehicle_class: str,
                                flood_mask: Optional[np.ndarray] = None,
                                track_id: Optional[int] = None,
                                camera_pitch_deg: Optional[float] = None) -> Dict[str, Any]:
        """
        Tính toán độ sâu ngập của một phương tiện đơn lẻ (Chuẩn hóa độ phân giải & Camera):
        1. Làm mượt tọa độ BBox theo thời gian (Temporal EMA Smoothing)
        2. Bù trừ góc chúc camera trên cao (Pitch Angle Compensation)
        3. Cắt lọc bóng phản chiếu mặt nước (Reflection Snapping)
        4. Kiểm tra độ tin cậy cự ly pixel (Min-Pixel Footprint Gate)
        """
        raw_x1, raw_y1, raw_x2, raw_y2 = bbox

        # 1. Làm mượt tọa độ BBox theo thời gian qua EMA (Khử rung giật do nén video H.264 & sóng nước)
        if track_id is not None:
            if track_id in self.bbox_smoothing:
                prev_b = self.bbox_smoothing[track_id]
                alpha = 0.70  # Trọng số khung hình hiện tại
                x1 = alpha * raw_x1 + (1.0 - alpha) * prev_b[0]
                y1 = alpha * raw_y1 + (1.0 - alpha) * prev_b[1]
                x2 = alpha * raw_x2 + (1.0 - alpha) * prev_b[2]
                y2 = alpha * raw_y2 + (1.0 - alpha) * prev_b[3]
            else:
                x1, y1, x2, y2 = raw_x1, raw_y1, raw_x2, raw_y2
            self.bbox_smoothing[track_id] = (x1, y1, x2, y2)
        else:
            x1, y1, x2, y2 = raw_x1, raw_y1, raw_x2, raw_y2

        w_box = max(2.0, float(x2 - x1))
        h_vis = max(2.0, float(y2 - y1))

        v_cls = vehicle_class.lower()
        if 'motor' in v_cls or 'bike' in v_cls:
            mapped_cls = 'motorcycle'
        elif 'truck' in v_cls:
            mapped_cls = 'truck'
        elif 'bus' in v_cls:
            mapped_cls = 'bus'
        elif 'car' in v_cls or 'sedan' in v_cls or 'suv' in v_cls:
            mapped_cls = 'car'
        else:
            mapped_cls = 'default'

        orientation = self.classify_view_orientation(mapped_cls, w_box, h_vis)
        ratio_table = self.CANONICAL_RATIOS_BY_VIEW.get(mapped_cls, self.CANONICAL_RATIOS_BY_VIEW['default'])
        canonical_ratio = ratio_table.get(orientation, 1.0)
        h_real = self.CANONICAL_REAL_HEIGHTS.get(mapped_cls, 120.0)

        # 2. Bù trừ góc chúc camera trên cao (Camera Pitch Compensation)
        pitch = float(camera_pitch_deg) if camera_pitch_deg is not None else self.camera_pitch_deg
        pitch_rad = math.radians(max(0.0, min(75.0, pitch)))
        cos_pitch = max(0.25, math.cos(pitch_rad))
        effective_canonical_ratio = canonical_ratio / cos_pitch

        # Tính toán chiều cao toàn thân dự kiến (pixel) và vị trí tiếp đất ảo dưới nước
        expected_full_h_px = w_box * effective_canonical_ratio
        y_ground = int(round(y1 + expected_full_h_px))

        # Thang đo mm/pixel bất biến
        px_to_cm_scale = h_real / max(1.0, expected_full_h_px)

        # 3. Kiểm tra độ tin cậy cự ly pixel (Min-Pixel Footprint Gate): Loại bỏ xe quá xa ở hậu cảnh
        is_distance_reliable = bool(w_box >= self.min_box_width_px and px_to_cm_scale <= self.max_scale_cm_per_px)

        # 4. Tìm mực nước waterline & Khử bóng phản chiếu mặt nước (Reflection Snapping)
        has_water = False
        waterline_y = y2

        if flood_mask is not None:
            # Quét mặt nạ nước trong vùng chân xe
            img_h, img_w = flood_mask.shape[:2]
            cx_min = max(0, int(x1))
            cx_max = min(img_w, int(x2))
            cy_min = max(0, int(y1 + h_vis * 0.4))
            cy_max = min(img_h, max(int(y2 + 20), int(y_ground + 10)))

            if cx_max > cx_min and cy_max > cy_min:
                roi_mask = flood_mask[cy_min:cy_max, cx_min:cx_max]
                water_pixels = np.where(roi_mask > 0.3)
                if len(water_pixels[0]) > 10:
                    has_water = True
                    # Waterline là vị trí pixel nước cao nhất (y nhỏ nhất) tiếp xúc với xe
                    waterline_y = int(cy_min + np.min(water_pixels[0]))

                    # Reflection Snapping: Nếu BBox bị bóng soi nước kéo dài quá waterline >15px, cắt gọn về ngấn nước
                    if y2 > waterline_y + 15:
                        y2 = float(waterline_y)
                        h_vis = max(2.0, float(y2 - y1))

        # Áp dụng Van Khóa 3 Lớp (Strict 3-Gate Trigger Filter)
        # Gate 1: Kiểm tra tiếp xúc vùng nước flood_mask
        # Gate 2: Kiểm tra tỷ lệ co rút chiều cao H_vis / H_full_px < 0.95 (lốp xe bị che khuất)
        shrinkage_ratio = h_vis / max(1.0, expected_full_h_px)

        is_submerged = False
        if flood_mask is not None:
            # Nếu có mặt nạ nước: Bắt buộc chân xe phải chạm nước VÀ có hiện tượng co rút hộp
            if has_water and shrinkage_ratio < 0.95:
                is_submerged = True
        else:
            # Nếu không có mặt nạ nước: Phải có sự co rút rõ rệt so với chuẩn QCVN
            if shrinkage_ratio < 0.95:
                is_submerged = True

        if is_submerged:
            submerged_px = max(0.0, float(y_ground - waterline_y))
            raw_depth_cm = submerged_px * px_to_cm_scale
        else:
            submerged_px = 0.0
            raw_depth_cm = 0.0

        # Khử nhiễu: Nếu độ ngập < 3cm coi như đường chỉ ướt hoặc sai số nhẹ
        if raw_depth_cm < 3.0:
            raw_depth_cm = 0.0
            is_submerged = False

        # Gate 3: Kiểm tra ngưỡng nổi bồng bềnh (Buoyancy Limit)
        buoyancy_limit = self.BUOYANCY_LIMITS.get(mapped_cls, 50.0)
        is_floating = (raw_depth_cm > buoyancy_limit)

        # Lọc trung vị theo track nếu có ID
        final_depth_cm = raw_depth_cm
        if track_id is not None:
            history = self.track_history.setdefault(track_id, [])
            history.append(raw_depth_cm)
            if len(history) > 10:
                history.pop(0)
            final_depth_cm = float(np.median(history))

        # Điểm tiếp đất hiển thị: nếu không ngập thì đáy tiếp đất trùng đáy nhìn thấy y2
        effective_y_ground = int(y_ground) if (final_depth_cm > 0 and is_submerged) else int(y2)

        return {
            'class': mapped_cls,
            'orientation': orientation,
            'bbox_visible': (int(x1), int(y1), int(x2), int(y2)),
            'bbox_submerged': (int(x1), int(y2), int(x2), effective_y_ground),
            'y_ground': int(y_ground),
            'waterline_y': int(waterline_y),
            'depth_cm': float(round(final_depth_cm, 1)),
            'scale_cm_per_px': float(round(px_to_cm_scale, 3)),
            'is_floating': is_floating,
            'buoyancy_limit': buoyancy_limit,
            'has_water_contact': has_water,
            'is_submerged': is_submerged,
            'shrinkage_ratio': float(round(shrinkage_ratio, 2)),
            'is_distance_reliable': is_distance_reliable,
            'camera_pitch_deg': float(round(pitch, 1))
        }


class AdaptiveHybridArbiter:
    """
    Bộ Điều Phối Trọng Tài Thích Ứng:
    Hợp nhất số đo giữa Phương tiện và Thước đo Mốc cố định.
    Tự học độ chênh cao mặt đường (Delta_h) và điều chuyển kịch bản linh hoạt.
    """

    WARNING_LEVELS = [
        (0, "An toàn - Đường khô ráo", (34, 197, 94), "Bình thường"),
        (1, "Ngập nhẹ (5 - 15 cm)", (234, 179, 8), "Chú ý giảm tốc"),
        (2, "Ngập trung bình (15 - 25 cm)", (249, 115, 22), "Ngập nửa bánh xe"),
        (3, "Ngập sâu nguy hiểm (25 - 45 cm)", (239, 68, 68), "Nguy cơ chết máy"),
        (4, "Báo động khẩn cấp (45 - 70 cm)", (168, 85, 247), "Hạn chế lưu thông"),
        (5, "Ngập cực hạn (> 70 cm)", (220, 38, 38), "Cấm đường khẩn cấp")
    ]

    def __init__(self, default_road_offset_delta_h: float = 0.0):
        self.vehicle_estimator = VehicleAmodalDepthEstimator()
        self.delta_h = float(default_road_offset_delta_h)
        self.learned_offsets = []
        self.last_valid_vehicle_depth = None
        self.last_valid_gauge_depth = None
        self.prev_flood_mask: Optional[np.ndarray] = None

    def get_alert_info(self, depth_cm: float) -> Tuple[int, str, Tuple[int, int, int], str]:
        """Phân loại 6 cấp độ cảnh báo thủy văn đô thị."""
        if depth_cm < 5.0:
            lvl, title, color, advice = self.WARNING_LEVELS[0]
        elif depth_cm < 15.0:
            lvl, title, color, advice = self.WARNING_LEVELS[1]
        elif depth_cm < 25.0:
            lvl, title, color, advice = self.WARNING_LEVELS[2]
        elif depth_cm < 45.0:
            lvl, title, color, advice = self.WARNING_LEVELS[3]
        elif depth_cm < 70.0:
            lvl, title, color, advice = self.WARNING_LEVELS[4]
        else:
            lvl, title, color, advice = self.WARNING_LEVELS[5]
        return lvl, title, color, advice

    def arbitrate(self, vehicle_detections: List[Dict[str, Any]],
                  virtual_gauge: Optional[VirtualStaffGauge] = None,
                  gauge_depth_cm: Optional[float] = None) -> HybridDepthReport:
        """
        Thực hiện điều phối logic:
        1. Kiểm tra xe bám đất (Active)
        2. Nếu không có xe -> Chuyển sang mốc ảo (Idle)
        3. Nếu xe nổi trôi -> Chuyển sang mốc cao tầng (Takeover)
        """
        # Lọc các xe ngập hợp lệ, ưu tiên xe có cự ly tin cậy (không quá xa ở chân trời)
        reliable_veh_depths = [v['depth_cm'] for v in vehicle_detections if not v.get('is_floating', False) and v.get('depth_cm', 0.0) > 0.0 and v.get('is_distance_reliable', True)]
        if not reliable_veh_depths:
            # Fallback nếu toàn bộ xe trong khung hình đều ở xa
            reliable_veh_depths = [v['depth_cm'] for v in vehicle_detections if not v.get('is_floating', False) and v.get('depth_cm', 0.0) > 0.0]

        valid_veh_depths = reliable_veh_depths
        has_floating = any(v.get('is_floating', False) for v in vehicle_detections)

        # Xác định độ sâu mốc (nếu có)
        g_depth = None
        if gauge_depth_cm is not None and gauge_depth_cm > 0.0:
            g_depth = float(gauge_depth_cm)
        elif virtual_gauge is not None:
            if hasattr(virtual_gauge, 'smoothed_fused_z'):
                g_depth = float(virtual_gauge.smoothed_fused_z)
            elif getattr(virtual_gauge, 'is_configured', False):
                g_depth = float(virtual_gauge.last_measured_cm)


        # ==============================================================
        # KỊCH BẢN 1: Có phương tiện đang di chuyển hợp lệ trên đường
        # ==============================================================
        if len(valid_veh_depths) > 0 and not has_floating:
            # Lấy trung vị độ sâu các xe đang lội nước
            master_depth = float(np.median(valid_veh_depths))
            source = f"vehicle_wheel ({len(valid_veh_depths)} xe)"
            conf = 90

        # ==============================================================
        # KỊCH BẢN 2: Ngập cực hạn xe bị nổi bồng bềnh / trôi dạt (>60-70cm)
        # ==============================================================
        elif has_floating:
            if g_depth is not None:
                master_depth = g_depth
                source = "emergency_pole_takeover (Xe bi noi)"
                conf = 85
            else:
                master_depth = 75.0
                source = "emergency_buoyancy_estimate"
                conf = 70

        # ==============================================================
        # KỊCH BẢN 3: Đường vắng không có xe / Đêm khuya (Traffic Idle Mode)
        # ==============================================================
        elif g_depth is not None:
            # Đo trực tiếp theo mạng mốc thủy chuẩn cao trình tuyệt đối SCSG-Net [H_start -> H_end]
            master_depth = g_depth
            source = f"virtual_gauge ({g_depth:.1f} cm)"
            conf = 85
        else:
            # Mặc định an toàn
            master_depth = 0.0
            source = "no_traffic_no_gauge"
            conf = 95

        lvl, title, color, advice = self.get_alert_info(master_depth)

        return HybridDepthReport(
            master_depth_cm=float(round(master_depth, 1)),
            source=source,
            confidence_pct=int(conf),
            alert_level=lvl,
            alert_title=title,
            traffic_advice=advice,
            active_vehicles_count=len(vehicle_detections),
            vehicle_depth_cm=float(round(np.median(valid_veh_depths), 1)) if len(valid_veh_depths) > 0 else None,
            gauge_depth_cm=float(round(g_depth, 1)) if g_depth is not None else None,
            road_offset_delta_h=float(round(self.delta_h, 1)),
            has_floating_suspect=has_floating,
            details={
                'num_vehicles_detected': len(vehicle_detections),
                'valid_vehicle_depths': valid_veh_depths,
                'delta_h_learned': self.delta_h
            }
        )


def render_hybrid_xray_hud(img_bgr: np.ndarray,
                           vehicle_reports: List[Dict[str, Any]],
                           virtual_gauge: Optional[VirtualStaffGauge],
                           report: HybridDepthReport,
                           flood_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Vẽ Giao diện X-Ray HUD Tổng hợp Thời gian thực:
    - Mặt nạ phân đoạn nước bán trong suốt (Cyan Tint BGR [235, 140, 40] + Shoreline Contour)
    - BBox xe nổi (Xanh lá)
    - BBox xe chìm bán trong suốt (Cam X-Ray)
    - Vạch tiếp đất ảo dưới nước (Vàng dạ quang GROUND: -X cm)
    - Thước đo thủy chuẩn mốc cố định (GAUGE: Y cm)
    - Banner cảnh báo thời gian thực trên cùng
    """
    hud = img_bgr.copy()
    h_img, w_img = img_bgr.shape[:2]

    # 0. Vẽ Phân Đoạn Mặt Nạ Nước (Water Segmentation Mask & Shorelines)
    if flood_mask is not None:
        try:
            m_max = float(np.max(flood_mask))
            thresh = 0.25 if m_max <= 1.0 else 30.0
            water_bin = (flood_mask > thresh).astype(np.uint8)
            if np.any(water_bin > 0):
                # Lớp màu xanh nước biển bán trong suốt (Cyan BGR: [235, 140, 40])
                water_tint = np.zeros_like(hud)
                water_tint[water_bin > 0] = [235, 140, 40]
                hud = cv2.addWeighted(hud, 1.0, water_tint, 0.35, 0)

                # Vẽ đường viền ranh giới mặt nước (Shorelines)
                contours, _ = cv2.findContours(water_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_cnts = [c for c in contours if cv2.contourArea(c) > 40]
                if valid_cnts:
                    cv2.drawContours(hud, valid_cnts, -1, (255, 230, 0), 1, cv2.LINE_AA)
        except Exception as e:
            print(f"⚠️ Lỗi vẽ mặt nạ nước: {e}")

    overlay = hud.copy()

    # 1. Vẽ các phương tiện & X-Ray BBox
    for v in vehicle_reports:
        vx1, vy1, vx2, vy2 = v['bbox_visible']
        sx1, sy2, sx2, sy_ground = v['bbox_submerged']
        depth = v['depth_cm']
        v_cls = v['class']
        v_orient = v['orientation']
        is_float = v['is_floating']

        # Hộp chìm màu cam trong suốt (chỉ hiển thị khi xe thực sự bị ngập lội nước)
        if sy_ground > vy2 and depth > 0:
            cv2.rectangle(overlay, (sx1, vy2), (sx2, sy_ground), (0, 140, 255), -1)
            cv2.rectangle(hud, (sx1, vy2), (sx2, sy_ground), (0, 165, 255), 1, cv2.LINE_AA)
            cv2.line(hud, (sx1 - 6, sy_ground), (sx2 + 6, sy_ground), (0, 255, 255), 1, cv2.LINE_AA)
            ground_txt = f"-{depth:.1f}cm"
            cv2.putText(hud, ground_txt, (sx1, min(h_img - 10, sy_ground + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        # Hộp nổi màu xanh lá (hoặc đỏ nếu xe bồng bềnh)
        box_color = (0, 0, 255) if is_float else (0, 255, 0)
        cv2.rectangle(hud, (vx1, vy1), (vx2, vy2), box_color, 1, cv2.LINE_AA)

        # Nhãn thông số trên nóc xe
        tag = f"{v_cls.upper()}"
        if depth > 0:
            tag += f": {depth:.1f}cm"
        cv2.putText(hud, tag, (vx1, max(15, vy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

    # Pha trộn overlay bán trong suốt
    cv2.addWeighted(overlay, 0.35, hud, 0.65, 0, hud)

    # 2. Vẽ Thước đo Mốc Cố Định (Hỗ trợ cả Mạng lưới Đa Mốc Bậc Thang SCSG-Net lẫn Mốc Đơn)
    if virtual_gauge is not None:
        if hasattr(virtual_gauge, 'render_multi_on_frame'):
            res = virtual_gauge.render_multi_on_frame(hud, water_mask=flood_mask, draw_hud=False, raw_frame_bgr=img_bgr)
            hud = res[0]
        elif getattr(virtual_gauge, 'is_configured', False):
            bx, by = virtual_gauge.p_bottom
            tx, ty = virtual_gauge.p_top
            g_depth = virtual_gauge.last_measured_cm
            gauge_width = int(getattr(virtual_gauge, 'strip_width', 20))
            w_half = max(2, gauge_width // 2)

            dx = tx - bx
            dy = ty - by
            length_px = math.hypot(dx, dy)
            ratio_h = min(1.0, max(0.0, g_depth / max(1.0, virtual_gauge.height_cm)))

            if length_px >= 5.0:
                ux, uy = dx / length_px, dy / length_px
                nx, ny = -uy, ux

                p1 = (int(bx - nx * w_half), int(by - ny * w_half))
                p2 = (int(bx + nx * w_half), int(by + ny * w_half))
                p3 = (int(tx + nx * w_half), int(ty + ny * w_half))
                p4 = (int(tx - nx * w_half), int(ty - ny * w_half))
                poly = np.array([p1, p2, p3, p4], dtype=np.int32)

                gauge_overlay = hud.copy()
                cv2.fillPoly(gauge_overlay, [poly], (25, 30, 40))
                cv2.addWeighted(gauge_overlay, 0.45, hud, 0.55, 0, hud)
                cv2.polylines(hud, [poly], True, (255, 255, 0), 1, cv2.LINE_AA)

                if ratio_h > 0.01:
                    mx1 = int(bx - nx * w_half + dx * ratio_h)
                    my1 = int(by - ny * w_half + dy * ratio_h)
                    mx2 = int(bx + nx * w_half + dx * ratio_h)
                    my2 = int(by + ny * w_half + dy * ratio_h)
                    sub_poly = np.array([p1, p2, (mx2, my2), (mx1, my1)], dtype=np.int32)
                    sub_overlay = hud.copy()
                    cv2.fillPoly(sub_overlay, [sub_poly], (0, 0, 220))
                    cv2.addWeighted(sub_overlay, 0.40, hud, 0.60, 0, hud)
                    cv2.line(hud, (mx1 - 4, my1), (mx2 + 4, my2), (0, 0, 255), 1, cv2.LINE_AA)
                    wy = int(by + dy * ratio_h)
                else:
                    wy = by
            else:
                wy = by

            cv2.circle(hud, (bx, by), 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(hud, (tx, ty), 4, (0, 215, 255), -1, cv2.LINE_AA)

            gauge_lbl = f"{g_depth:.1f} cm"
            cv2.putText(hud, gauge_lbl, (min(w_img - 150, max(10, bx + w_half + 8)), max(25, wy - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)

    # 3. Top Banner HUD đã được loại bỏ hoàn toàn theo yêu cầu để khung hình video 100% thông thoáng
    # Toàn bộ thông số mực nước ngập, cấp cảnh báo, độ tin cậy và offset tim đường
    # được hiển thị đầy đủ và chuyên nghiệp ở Bảng Thống Kê và Khung Cảnh Báo HTML bên ngoài.
    return hud


def refine_flood_segmentation_mask(
    raw_mask: np.ndarray,
    h: int,
    w: int,
    conf_thresh: float = 0.25,
    prev_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Tinh chỉnh mặt nạ phân đoạn nước ngập độ chính xác cao:
    1. Chuẩn hóa kích thước (w, h)
    2. Cắt bỏ nhiễu vùng trời / mái hiên (12% đỉnh ảnh)
    3. Đóng hình thái học (Morphological Closing) lấp lỗ thủng sóng nước & xóa đốm nhiễu < 200px
    4. Làm mượt theo thời gian (Temporal EMA) chống nhấp nháy giữa các frame
    """
    if raw_mask.shape[:2] != (h, w):
        mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        mask = raw_mask.copy()

    # 1. Cắt bỏ vùng trời / mái nhà ở 12% phía trên cùng của khung hình
    sky_cutoff = int(h * 0.12)
    if sky_cutoff > 0:
        mask[:sky_cutoff, :] = 0.0

    # 2. Lọc hình thái học & xóa đốm nhiễu nhỏ rời rạc
    bin_mask = (mask >= conf_thresh).astype(np.uint8) * 255
    if np.any(bin_mask > 0):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed_bin = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

        # Chỉ giữ lại các vùng nước liên thông có diện tích >= 200 pixel
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_bin, connectivity=8)
        clean_bin = np.zeros_like(closed_bin, dtype=np.float32)
        for lbl_idx in range(1, num_labels):
            if stats[lbl_idx, cv2.CC_STAT_AREA] >= 200:
                clean_bin[labels == lbl_idx] = 1.0

        # Giữ độ mềm xác suất bên trong vùng hợp lệ và làm mịn biên bằng Gaussian nhẹ
        refined = np.maximum(mask * clean_bin, clean_bin * float(conf_thresh + 0.05))
        refined = cv2.GaussianBlur(refined, (5, 5), 0)
    else:
        refined = np.zeros((h, w), dtype=np.float32)

    # 3. Làm mượt theo thời gian (Temporal EMA) nếu cùng độ phân giải
    if prev_mask is not None and prev_mask.shape == refined.shape:
        refined = 0.65 * refined + 0.35 * prev_mask

    return np.clip(refined, 0.0, 1.0)


def process_single_frame_hybrid(frame_bgr: np.ndarray,
                                gauge: Optional[VirtualStaffGauge],
                                arbiter: AdaptiveHybridArbiter,
                                flood_model: Any,
                                vehicle_model: Any = None,
                                flood_conf: float = 0.25,
                                veh_conf: float = 0.35,
                                mode: str = "Cột mốc") -> Tuple[np.ndarray, Any, str, HybridDepthReport]:
    """
    Xử lý một khung hình hoàn chỉnh qua pipeline đo độ sâu theo Thủy chuẩn Cột mốc.
    """
    import pandas as pd
    from fixed_object_gauge import process_single_frame_gauge

    h, w = frame_bgr.shape[:2]

    # 1. Dự đoán mặt nạ nước độ phân giải cao trên GPU (retina_masks=True + Spatial-Temporal Refinement)
    flood_mask = np.zeros((h, w), dtype=np.float32)
    if flood_model is not None:
        try:
            f_res = flood_model.predict(
                frame_bgr,
                conf=float(flood_conf),
                iou=0.55,
                imgsz=640,
                retina_masks=True,
                device=INFER_DEVICE,
                verbose=False
            )[0]
            prev_m = getattr(arbiter, "prev_flood_mask", None) if arbiter is not None else None
            if f_res.masks is not None and len(f_res.masks.data) > 0:
                combined_gpu = torch.amax(f_res.masks.data, dim=0)
                m_np = combined_gpu.cpu().numpy()
                flood_mask = refine_flood_segmentation_mask(
                    m_np, h, w, conf_thresh=float(flood_conf), prev_mask=prev_m
                )
            elif prev_m is not None and prev_m.shape == (h, w):
                # Giảm dần mượt mà khi mất tín hiệu đột ngột 1 frame
                flood_mask = np.where(prev_m * 0.35 >= float(flood_conf), prev_m * 0.35, 0.0).astype(np.float32)
            if arbiter is not None:
                arbiter.prev_flood_mask = flood_mask
        except Exception as e:
            print(f"⚠️ Lỗi suy luận Flood Model: {e}")

    # 2. Đo theo mốc thủy chuẩn cố định
    g_measured_cm = None
    g_status = "UNCONFIGURED"
    g_rule_desc = ""
    active_gauge = gauge
    if active_gauge is not None:
        if hasattr(active_gauge, 'measure_all') and hasattr(active_gauge, 'fuse_water_level'):
            # Là MultiVirtualStaffGaugeManager (Đa mốc bậc thang SCSG-Net)
            try:
                all_res = active_gauge.measure_all(water_mask=flood_mask, frame_bgr=frame_bgr)
                fusion_res = active_gauge.fuse_water_level(all_res)
                g_measured_cm = fusion_res['fused_z'] if fusion_res['status'] != 'NO_FLOOD' else 0.0
                g_status = fusion_res['status']
                g_rule_desc = fusion_res.get('rule_desc', '')
            except Exception as e:
                print(f"⚠️ Lỗi đo mạng lưới đa mốc: {e}")
        elif getattr(active_gauge, 'is_configured', False):
            try:
                _, g_measured_cm, _, g_status, _ = process_single_frame_gauge(
                    frame_bgr, active_gauge, flood_model, flood_conf=flood_conf
                )
            except Exception as e:
                print(f"⚠️ Lỗi đo mốc thủy chuẩn: {e}")

    # 3. Trọng tài điều phối kết quả (chỉ sử dụng thủy chuẩn cột mốc)
    report = arbiter.arbitrate([],
                               virtual_gauge=active_gauge,
                               gauge_depth_cm=g_measured_cm)

    # 4. Vẽ HUD X-Ray
    hud_bgr = render_hybrid_xray_hud(frame_bgr,
                                     [],
                                     active_gauge,
                                     report,
                                     flood_mask=flood_mask)
    hud_rgb = cv2.cvtColor(hud_bgr, cv2.COLOR_BGR2RGB)

    # 5. Tạo DataFrame thống kê
    alert_tag = report.alert_title.split(':')[1].strip() if ':' in report.alert_title else report.alert_title
    if active_gauge is not None:
        if hasattr(active_gauge, 'gauges'):
            cfg_count = sum(1 for g in active_gauge.gauges.values() if g.is_configured)
            if cfg_count > 0 and g_measured_cm is not None:
                if g_rule_desc:
                    gauge_desc = g_rule_desc
                else:
                    gauge_desc = f"Mạng lưới {cfg_count} mốc SCSG-Net: {g_measured_cm:.1f} cm ({g_status})"
            else:
                gauge_desc = "Chưa thiết lập mốc nào"
        elif getattr(active_gauge, 'is_configured', False) and g_measured_cm is not None:
            gauge_desc = f"Cột mốc: {g_measured_cm:.1f}cm"
        else:
            gauge_desc = "Chưa thiết lập mốc"
    else:
        gauge_desc = "Chưa thiết lập mốc"

    df = pd.DataFrame([{
        "Nguồn Đo Lường": report.source,
        "Độ Sâu Ngập": f"{report.master_depth_cm:.1f} cm",
        "Cấp Báo Động": f"Level {report.alert_level}: {alert_tag}",
        "Độ Tin Cậy": f"{report.confidence_pct}%",
        "Thủy Chuẩn Mốc": gauge_desc,
    }])

    # 6. Tạo HTML Cảnh báo
    colors_hex = {
        0: "#22c55e",
        1: "#eab308",
        2: "#f97316",
        3: "#ef4444",
        4: "#a855f7",
        5: "#dc2626"
    }
    alarm_color = colors_hex.get(report.alert_level, "#ef4444")

    alarm_html = f"""
    <div style="background:#1e293b; color:white; padding:10px 14px; border-radius:8px; border-left:5px solid {alarm_color}; display:flex; justify-content:space-between; align-items:center;">
        <span style="color:{alarm_color}; font-size:15px; font-weight:700;">🚨 {report.alert_title}</span>
        <span style="font-size:16px; font-weight:700;">Mực nước: <span style="color:#38bdf8;">{report.master_depth_cm:.1f} cm</span></span>
    </div>
    """

    return hud_rgb, df, alarm_html, report


def process_video_tab5_hybrid(video_path: str,
                              gauge: Any,
                              arbiter: AdaptiveHybridArbiter,
                              flood_model: Any,
                              vehicle_model: Any,
                              flood_conf: float = 0.25,
                              veh_conf: float = 0.35,
                              output_dir: Optional[str] = None,
                              progress=None,
                              mode: str = "Cả 2") -> Optional[str]:
    """
    Xử lý toàn bộ tệp Video tải lên với Hệ Thống Lai Tab 5 và xuất file MP4.
    """
    from fixed_object_gauge import get_clean_filepath

    clean_path = get_clean_filepath(video_path)
    if not clean_path:
        return None

    cap = cv2.VideoCapture(clean_path)
    if not cap.isOpened():
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

    target_w, target_h = (1280, 720) if (w > 1280 or h > 720) else (w, h)
    out_dir = output_dir if output_dir is not None else "."
    output_path = f"{out_dir}/output_tab5_hybrid_{int(time.time())}.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 12, (target_w, target_h))

    frame_idx = 0
    max_frames_to_export = min(total_frames, 500)

    while cap.isOpened() and frame_idx < max_frames_to_export:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        if (frame_idx % 2) != 0:
            continue

        if progress is not None:
            progress(frame_idx / max_frames_to_export, desc=f"Đang xuất video Tab 5 ({frame_idx}/{max_frames_to_export} frames)...")

        hud_rgb, _, _, _ = process_single_frame_hybrid(
            frame, gauge, arbiter, flood_model, vehicle_model,
            flood_conf=flood_conf, veh_conf=veh_conf, mode=mode
        )
        hud_bgr = cv2.cvtColor(hud_rgb, cv2.COLOR_RGB2BGR)
        if (hud_bgr.shape[1], hud_bgr.shape[0]) != (target_w, target_h):
            hud_bgr = cv2.resize(hud_bgr, (target_w, target_h))

        out.write(hud_bgr)

    cap.release()
    out.release()
    return output_path

