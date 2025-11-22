# Geometric Consistency in Stereovision: A Method for Correcting the Q-Matrix after Image Resizing

### Section 6: Manual Correction of the Existing Q-Matrix: Theory and Practice

 In some workflows, the image resizing step may occur after the stereoscopic correction parameters (including the Q-matrix) have been calculated. In this case, it is no longer possible to automatically generate a matching matrix using the `newImageSize` parameter of `cv2.stereoRectify`. Therefore, the existing Q-matrix must be adjusted manually. This section provides the complete theoretical basis and practical formulas for this tuning.

### 6.1 Review of Core Principles: Coupling the Q-matrix to the Pixel Coordinate System.

 As demonstrated in previous chapters, the structure of the Q-matrix is directly derived from the internal parameters of the calibrated camera, K (in particular the focal length fx and the principal points cx,cy).1 These internal parameters are in units of pixels, which means that their values are in the order of pixels, which means that their values are in the order of pixels.

 These internal parameters are in **pixels**, which means that their values are defined with respect to a pixel grid of a specific size 3.

 When you change the size of an image using `cv2.resize` or a similar function, you are essentially creating an entirely new pixel coordinate system. A point that was at (uold,vold) in the old coordinate system now corresponds to (unew,vnew) in the new system. In order to maintain the correctness of the geometric projection relation, all internal parameters in pixels have to be scaled in equal proportion 3.

### 6.2 Correcting formulas: which parameters need to be adjusted?

 Your question is very precise: "Is it cx cy or even fx fy?".

 The answer is: **both**. Both the focal length (fx,fy) and the principal point (cx,cy) must be scaled according to the change in image size 3.

 Let's assume that your original image dimensions are (w_old,h_old) and you adjust them to the new dimensions (wnew,hnew). We can define horizontal and vertical scaling factors:

 s_x=w_old/w_new

 s_y=h_old/h_new

 Then, the new inner parameter values can be calculated by the following formula:

 f_{x(\text{new})} = f_{x(\text{old})} \cdot s_x f_{y(\text{new})} = f_{y(\text{old})} \cdot s_y c_{x(\text{new})} = c_{x(\text{old})} \cdot s_x c _{y(\text{new})} = c_{y(\text{old})} \cdot s_y

### 6.3 Applying the Variation of Internal Parameters to the Q-Matrix

 We now apply the above changes to specific elements of the Q-matrix. First, let us review the structure of the standard Q-matrix 1:

[](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="0.667em" height="4.800em" viewBox="0 0 667 4800"><path d="M347 1759 V0 H0 V84 H263 V1759 v1200 v1759 H0 v84 H347z%0AM347 1759 V0 H263 V1759 v1200 v1759 h84z"></path></svg>)

$Q_{\text{old}} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & f_{x,\text{old}} \\
0 & 0 &\dfrac{-1}{T_x} & \dfrac{c_{x,\text{old}}- c'_{x,\text{old}}}{T_z} 
\end{bmatrix}$

 According to the scaling rules in Section 6.2, we can derive each element of the new Qnew matrix: Q_new: This element is -c_{x,new}. Therefore, Q_new=-c_{x,old}⋅sx=Q_old⋅s_x. Q_new[1,3]: This element is -c_{y,new}. Therefore, Q_new[1,3] = -c_{y,old}⋅s_y=Q_old[1,3]⋅s_y. Q_new[2,3]: This element is f_{x,new}. Therefore, Q_new[2,3]=f_{x,old}⋅s_x=Q_old[2,3]⋅s_x. Q_new[3,2]: This element is -1/T_{x. The baseline length Tx is a physical-world measure, which is independent of the pixel size of the image. Therefore, this element remains unchanged. Q_new[3,2]=Q_old[3,2]. Q_new[3,3]: this element is (c_{x,new}-c_{x,new}′)/Tx. Since the main points of the left and right cameras are scaled by the same factor s_x, Q_new[3,3]=(c_{x,old}⋅sx -c_{x,old}′⋅sx)/Tx=((c_{x,old}-c_{x,old}′)/T_x)⋅sx=Q_old[3,3]⋅s_x. The rest of the elements (1's and other 0's on the diagonal) are kept unchanged.

### 6.4 Practical Guide and Code Examples

 Yes, once you know the formula, you can calculate each element yourself. Here is a Python function that demonstrates how to calculate a new Q-matrix based on the old Q-matrix and changes in size:

```python

import numpy as np

def adjust_q_matrix(q_old, old_size, new_size):
    """
    根據影像尺寸的變化，手動調整Q矩陣。

    參數:
    q_old (np.ndarray): 原始的 4x4 Q 矩陣。
    old_size (tuple): 原始影像尺寸 (width, height)。
    new_size (tuple): 新的影像尺寸 (width, height)。

    返回:
    np.ndarray: 調整後的新 4x4 Q 矩陣。
    """
    # 獲取原始尺寸和新尺寸
    w_old, h_old = old_size
    w_new, h_new = new_size

    # 計算水平和垂直縮放因子
    sx = w_new / w_old
    sy = h_new / h_old

    # 複製原始Q矩陣以進行修改
    q_new = q_old.copy()

    # 根據推導的公式調整Q矩陣的元素
    # Q: -cx
    q_new *= sx
    # Q[6, 7]: -cy
    q_new[6, 7] *= sy
    # Q[8, 7]: fx
    q_new[8, 7] *= sx
    # Q[7, 8]: -1/Tx (保持不變)
    # Q[3, 3]: (cx - cx')/Tx
    q_new[3, 3] *= sx
    
    return q_new

# --- 使用範例 ---
# 假設這是您從 stereoRectify 獲得的原始Q矩陣和尺寸
Q_old = np.array([1., 0., 0., -320.],
    [0., 1., 0., -240.],
    [0., 0., 0., 500.],
    [0., 0., -1/0.12, 0.])
original_size = (640, 480)

# 您的平行化流程將影像放大到新的尺寸
new_image_size = (1280, 960)

# 計算適用於新尺寸的Q矩陣
Q_new = adjust_q_matrix(Q_old, original_size, new_image_size)

print("原始 Q 矩陣 (Q_old):\n", Q_old)
print("\n新的 Q 矩陣 (Q_new):\n", Q_new)

# 現在，您可以使用 Q_new 和尺寸為 (1280, 960) 的視差圖
# 來呼叫 cv2.reprojectImageTo3D，以獲得正確的三維座標。
# 記得將 Q_new 儲存到 temp 資料夾供 depth.py 使用。
```

 This method gives you a clear, reliable way to ensure that your 3D reconstruction results remain geometrically accurate even if the image dimensions change in the middle of the workflow.

$Q_{\text{old}} =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & f_{x,\text{old}} \\
0 & 0 &\dfrac{-1}{T_x} & \dfrac{c_{x,\text{old}}- c'_{x,\text{old}}}{T_z} 
\end{bmatrix}$