import numpy as np
import cv2

def visualize_optical_flow(flow: np.ndarray, convert_to_bgr=False) -> np.ndarray:
    """
    可视化光流，输入为形状 [H, W, 2] 的 numpy 数组，返回 RGB 图像。

    Args:
        flow (np.ndarray): 光流图像，形状为 [H, W, 2]，最后一个维度是 (dx, dy)。
        convert_to_bgr (bool): 如果为 True，则返回 BGR 图像（适用于 OpenCV 显示）。

    Returns:
        np.ndarray: RGB（或BGR）格式的可视化图像，形状为 [H, W, 3]，uint8。
    """
    h, w = flow.shape[:2]
    flow_map = np.zeros((h, w, 3), dtype=np.uint8)

    # 计算光流的极坐标（magnitude: 幅值，angle: 角度）
    dx = flow[..., 0]
    dy = flow[..., 1]
    magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)

    # 使用 HSV 映射方向和大小
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)        # H: 方向（0~180）
    hsv[..., 1] = 255                                 # S: 饱和度
    hsv[..., 2] = np.clip((magnitude * 32), 0, 255).astype(np.uint8)  # V: 强度（可调节倍数）

    flow_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR if convert_to_bgr else cv2.COLOR_HSV2RGB)
    return flow_map