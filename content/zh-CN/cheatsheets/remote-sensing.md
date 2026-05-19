---
title: "遥感与地理信息名词速查表"
date: 2026-05-19
summary: "遥感物理机制、多光谱与高光谱波段、常用遥感指数及空间数据处理的快速查询手册。支持实时搜索与过滤。"
type: "cheatsheets"
---

## 物理基础与波段 | Physical Foundations & Bands

### 电磁波谱 | Electromagnetic Spectrum
- **定义**: 按波长或频率顺序排列的电磁波序列。遥感传感器主要利用可见光、近红外、短波红外、热红外和微波波段进行观测。
- **大白话**: “光的色彩大谱表”。人类肉眼只能看到彩虹这极窄的一段可见光，而遥感相机的眼睛却能看清肉眼看不到的红外线、微波等，从而透视地表隐藏的秘密。
- **怎么实现**: 利用不同材质的探测器敏感到不同波长范围的光子能量，将其转换为电信号，并记录为不同波段的数字量化值（DN值）。
- **代码**:
  ```python
  # 遥感影像波段通常读取为多通道矩阵 (Channels, Height, Width)
  import numpy as np

  # 模拟一个 4 波段的遥感影像矩阵 (蓝, 绿, 红, 近红外)
  simulated_image = np.random.randint(0, 65535, size=(4, 512, 512), dtype=np.uint16)
  blue, green, red, nir = simulated_image[0], simulated_image[1], simulated_image[2], simulated_image[3]
  ```

### 光谱签名 | Spectral Signature
- **定义**: 某种特定地物在电磁波谱中不同波段的反射率随波长变化的特征曲线。它是遥感识别和分类地物的主要物理基础。
- **大白话**: “地物的光谱身份证”。比如绿色健康植物在红光区吸收强、近红外区反射极强，这构成了独特的“红边”特征，能让我们在卫星上瞬间认出植被。
- **怎么实现**: 传感器测量每个像素地表在各窄通道的反射能量，消除大气影响后，绘制反射率关于波长的二维连续曲线。
- **代码**:
  ```python
  import matplotlib.pyplot as plt

  # 模拟典型植被的光谱反射曲线
  wavelengths = [450, 550, 650, 850, 1600, 2200] # 单位 nm
  vegetation_reflectance = [0.05, 0.15, 0.08, 0.50, 0.30, 0.10] # 反射率百分比

  plt.plot(wavelengths, vegetation_reflectance, '-o', label='Vegetation Signature')
  plt.xlabel('Wavelength (nm)')
  plt.ylabel('Reflectance')
  plt.legend()
  ```

### 四大分辨率 | Resolutions
- **定义**: 遥感影像性能的四个核心度量维度：
  - **空间分辨率**: 像素对应的地面实际尺寸大小；
  - **光谱分辨率**: 传感器通道数量的多寡及波段宽窄程度；
  - **辐射分辨率**: 记录信号的明暗细分级别（比特深度）；
  - **时间分辨率**: 对同一地区进行重访观测的时间周期。
- **大白话**:
  - **空间**: 照片能放大看多清（是看清整栋大楼，还是看清楼顶的猫）；
  - **光谱**: 传感器对“色彩”分得有多细（是黑白，普通彩色，还是上百个通道的高光谱）；
  - **辐射**: 画面明暗过渡多细腻（是只有黑白 2 档，还是 65536 阶灰色级别）；
  - **时间**: 卫星多久重访拍一次照片。
- **怎么实现**: 通过镜头的焦距和轨道高度设计空间分辨率；通过分光元件分光决定光谱范围；通过 AD 转换器的位数决定辐射灰度（如 8bit/12bit/16bit）；通过卫星轨道周期和星座部署控制重访周期。
- **代码**:
  ```python
  # 不同的比特深度对应不同的辐射分辨率
  bit_depth = 12
  max_val = 2 ** bit_depth - 1 # 4095 个明暗级别
  print(f"{bit_depth}-bit 图像可记录的最高亮度灰度级为: {max_val}")
  ```

### 高程模型 | DEM / DTM / DSM
- **定义**: 表示地表高程特征的不同数字化模型：
  - **DEM (数字高程模型)**: 广义的地面高程统计模型；
  - **DTM (数字地形模型)**: 剥离了植被和建筑物等人工设施后，光秃秃的裸露地表起伏模型；
  - **DSM (数字表面模型)**: 包含了地表建筑物、植被等所有自然和人工附着物顶端的最表层高程模型。
- **大白话**:
  - **DSM (表面高程)**: 卫星从上往下看，碰到的第一层“物理高度”，包括屋顶和树梢；
  - **DTM (地形高程)**: 把房子和森林用铲子推平，只剩下泥土和山石地表本身的起伏高度；
  - **nDSM (归一化高程)**: 两者相减 ($DSM - DTM$)，正好得出纯粹的“房子有多高、树木有多高”。
- **怎么实现**: 利用激光雷达 (LiDAR) 发射激光脉冲，依靠首回波提取 DSM，末回波或地面滤波算法剔除树木房屋生成 DTM。亦可利用光学立体像元对进行视差匹配重构三维点云。
- **代码**:
  ```python
  import numpy as np

  # 计算归一化数字表面模型 (nDSM) 以获得地面物体的净高度
  def calculate_ndsm(dsm, dtm):
      ndsm = dsm - dtm
      # 滤除由于噪声等因素引起的负值净高
      ndsm[ndsm < 0] = 0
      return ndsm
  ```

### 合成孔径雷达 | SAR
- **定义**: 一种主动式微波传感器成像雷达。利用雷达天线与地表的相对运动，结合相干信号处理，合成一个超大等效天线（孔径），从而获取高分辨率微波图像。
- **大白话**: “全天候穿透式透视相机”。它自己往地面发射微波并接收回波，不需要太阳光，也不怕云雾阻挡。刮风下雨、白天黑夜都能看清地面的粗糙程度、含水量以及三维形变。
- **怎么实现**: 卫星在轨道上边飞边发射微波，并连续记录接收回波的振幅与相位。在计算机中对一整条飞行轨迹上收到的回波进行积分累加，等效于拥有一个长达几公里的物理天线进行极精细的成像。
- **代码**:
  ```python
  import numpy as np

  # 将 SAR 的复数回波 (Complex Data) 转化为强度值并转换为对数分贝 (dB)
  def complex_to_db(complex_data):
      intensity = np.abs(complex_data) ** 2
      # 防止零值导致 log 溢出
      intensity[intensity == 0] = 1e-6
      return 10 * np.log10(intensity)
  ```

## 常用遥感指数 | Remote Sensing Indices

### 归一化植被指数 | NDVI
- **定义**: 衡量植被覆盖状况、叶面积指数和生长活力的最常用指数，计算公式为：$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$
- **大白话**: “草木体检器”。因为活植被对近红外（NIR）反射极强，对红光（Red）吸收极强，计算两者的差比就能算出 -1 到 1 的指数值，越接近 1 说明草木越绿、越茂密。
- **怎么实现**: 用影像的近红外波段与红光波段进行栅格计算：两波段相减得到的差，除以两波段相加的和。
- **公式**: $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$
- **代码**:
  ```python
  import numpy as np

  def calculate_ndvi(nir, red):
      # 避免分母为零
      denom = nir + red
      denom[denom == 0.0] = 1e-5
      ndvi = (nir - red) / denom
      return np.clip(ndvi, -1.0, 1.0)
  ```

### 归一化水体指数 | NDWI
- **定义**: 提取冰雪、湖泊、河流等地表开放水体的关键指数，经典的麦克费特斯（McFeeters）公式为：$$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$$
- **大白话**: “水体剥离定位仪”。水对绿光有一定反射，但对近红外（NIR）几乎完全吸收。两者的差比运算可以压制植被和土壤，把亮闪闪的水库和河流抠出来。
- **怎么实现**: 用绿光波段和近红外波段进行归一化差值计算，通常 NDWI 大于 0 的区域被识别为水体。
- **公式**: $$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$$
- **代码**:
  ```python
  import numpy as np

  def calculate_ndwi(green, nir):
      denom = green + nir
      denom[denom == 0.0] = 1e-5
      return (green - nir) / denom
  ```

## 数据处理与校正 | Data Processing & Corrections

### 正射校正 | Orthorectification
- **定义**: 消除因传感器倾斜投影及地表起伏引起的倾斜和投影形变，将遥感影像改造成具有绝对正射投影几何特性的校正过程。
- **大白话**: “把斜拍照拉直”。因为卫星可能侧着身子拍照，或者山顶近、山谷远，导致照片倾斜形变。正射校正就是结合地面海拔高程（DEM），强行把斜的照片“摊平”，变成绝对垂直俯视且能精确测量面积和距离的标准地图。
- **怎么实现**: 结合影像自带的相机姿态方程（RPC 参数）和数字高程模型（DEM），反算出新正射网格上的每个坐标在源图像中的像素索引，并用双线性插值进行像素值填充。
- **代码**:
  ```python
  # 概念流程伪代码
  def orthorectify_pixel(src_img, rpc_coefficients, dem_height, target_lat, target_lon):
      # 1. 结合经纬度和高程算得源图像对应的像素位置 (u, v)
      # u, v = rpc_forward(rpc_coefficients, target_lat, target_lon, dem_height)
      # 2. 从源图上双线性重采样该像素值
      # val = bilinear_sample(src_img, u, v)
      # return val
      pass
  ```

### 大气校正 | Atmospheric Correction
- **定义**: 消除太阳辐射在进入大气、到达地面、反射再穿过大气到达传感器这整个过程中，因大气中的气体、水汽、气溶胶等散射和吸收造成的干扰，反演地表真实反射率。
- **大白话**: “去掉天空的灰霾滤镜”。太阳光穿透空气时会产生散射（比如晴天散射蓝光，雾霾天散射灰光）。大气校正就是把这层“半透明面纱”擦干净，露出地面物体的真实物理本色。
- **怎么实现**: 建立大气的物理和化学参数模型（利用 6S 或 MODTRAN 辐射传输方程），计算出大气程辐射与透过率，从而将辐亮度转化为地表反射率（Surface Reflectance）。
- **代码**:
  ```python
  # 简化的大气校正公式：SR = (Radiance - PathRadiance) / Transmittance
  def atmospheric_correction(radiance, path_radiance, transmittance):
      sr = (radiance - path_radiance) / transmittance
      return np.clip(sr, 0.0, 1.0)
  ```

### 空间插值 | Spatial Interpolation
- **定义**: 利用已知离散样点的观测值来预测周围其他未观测位置数值的数学网格化过程，常见方法包括反距离权重法（IDW）、克里金法（Kriging）。
- **大白话**: “地图脑补术”。假如你在城市里设立了 10 个气象站测气温，怎么画出全市的无缝气温地图？插值就是根据邻近地点的实测温度，按照“越近影响越大”的物理法则，把全市所有空白网格的温度给算出来。
- **怎么实现**:
  - **IDW (反距离权重)**: 未知点的值是周围已知样点的加权平均值，权重与距离的二次方（或多次方）成反比。
  - **Kriging (克里金)**: 不仅考虑距离，还利用半变异函数分析样点间的空间方位和相关性，进行最优无偏空间估计。
- **代码**:
  ```python
  import numpy as np

  def simple_idw(known_coords, known_values, target_coord, power=2):
      # known_coords shape: (N, 2), known_values shape: (N,)
      distances = np.linalg.norm(known_coords - target_coord, axis=1)
      # 避免距离为 0 导致除以零
      distances[distances == 0] = 1e-5
      weights = 1.0 / (distances ** power)
      return np.sum(weights * known_values) / np.sum(weights)
  ```

### 混合像元分解 | Mixed Pixel Linear Unmixing
- **定义**: 解决遥感像元因分辨率有限而包含多种地物的问题。通过解混算法计算像素中每种纯净地物（端元，Endmember）所占的面积比例（丰度，Abundance）。
- **大白话**: “像素调色盘拆解”。如果卫星拍下的一个像素点对应地面 30米×30米，里面不可能只有一种东西。它混合了 60% 小麦、30% 水泥路和 10% 泥土。分解算法就是把这个混合出来的“调色盘颜色”还原出各成分的原始比例。
- **怎么实现**: 假定像素光谱为各端元光谱反射率的线性组合：$y = M s + e$。在丰度大于等于 0 且和为 1 的物理约束下，使用最小二乘优化算法解算出丰度比例向量 $s$。
- **代码**:
  ```python
  import numpy as np
  from scipy.optimize import minimize

  def linear_unmixing(pixel_spectrum, endmembers):
      # pixel_spectrum: (B,) 目标像素波段反射率, endmembers: (B, N) 各端元反射率 (N 个端元, B 个波段)
      # 目标函数：最小化重构误差的平方和
      def objective(abundance):
          reconstructed = np.dot(endmembers, abundance)
          return np.sum((pixel_spectrum - reconstructed) ** 2)

      # 限制条件：丰度必须在 0 到 1 之间
      bounds = [(0, 1) for _ in range(endmembers.shape[1])]
      # 限制条件：丰度和必须为 1
      constraints = {'type': 'eq', 'fun': lambda a: np.sum(a) - 1.0}

      init_abundance = np.ones(endmembers.shape[1]) / endmembers.shape[1]
      result = minimize(objective, init_abundance, bounds=bounds, constraints=constraints)
      return result.x # 返回每个端元的丰度百分比
  ```

