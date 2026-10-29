# FlapSim 气动力（aero）实现、推导与验证

> 本文保留完整代码实现与验证细节。用于论文正文的理想悬停简化推导见 [hover_simplification.md](hover_simplification.md)，其中在明确假设下取 $a_{U1}=a_{U0}=0$。

本文整理当前 `flapsim/fluid` 的实际实现。这里的 “aero” 不是一套模型，而是两套用途不同的模型：

1. `fix=True`：`MuJoCoAeroAdapter` 调用 `aero.py`，逐翼、逐仿真步计算准定常气动力；
2. `fix=False`：`AveragedHoverAero` 使用由固定基座扫频辨识出的周期平均力和力矩。

二者在 `SimCore.__init__` 中二选一，并不会同时工作。下文先完整推导逐翼模型，再说明周期平均模型及其辨识关系。

---

## 1. 文件与运行时调用链

关键文件如下：

| 文件 | 作用 |
|---|---|
| `flapsim/fluid/aero.py` | 与 MuJoCo 解耦的纯气动计算：速度、攻角、法向力、压力中心和 wrench |
| `flapsim/fluid/mujoco_adapter.py` | 从 `MjData` 读取状态、处理左右翼坐标系、把 wrench 写入 `xfrc_applied` |
| `flapsim/fluid/averaged_hover.py` | 自由飞行时使用的周期平均悬停模型 |
| `flapsim/config/Wing.json` | 翼几何积分、空气密度及肩部几何参数 |
| `flapsim/config/AeroOptions.py` | 缩放、限幅、扑翼平面角和偏离角等运行参数 |
| `flapsim/sim_core.py` | 根据 `fix` 选择气动模型并挂接到环境 |
| `flapsim/environment/environment.py` | 每步先更新控制，再施加气动力，最后调用 `mujoco.mj_step` |

每个固定基座仿真步的顺序是

$$
\text{controller}
\rightarrow \text{joint state}
\rightarrow \texttt{MuJoCoAeroAdapter.apply\_forces}
\rightarrow \texttt{aero\_step}
\rightarrow \texttt{wing\_aero}
\rightarrow \texttt{world\_wrench}
\rightarrow \texttt{xfrc\_applied}
\rightarrow \texttt{mj\_step}.
$$

`TrajectoryController` 规定扑翼角为

$$
\psi(t)=dA\sin(2\pi ft+\varphi_0),\qquad
\dot\psi(t)=dA(2\pi f)\cos(2\pi ft+\varphi_0),
$$

其中 $d\in\{-1,+1\}$ 是左右翼方向，默认振幅 $A=1.5\ \mathrm{rad}$。

---

## 2. 坐标、状态与符号

### 2.1 机身状态

设 $R_{WB}$ 是机身坐标到世界坐标的旋转矩阵。MuJoCo 给出的世界系速度先被变换到机身系：

$$
\boldsymbol v_B=R_{WB}^{T}\boldsymbol v_W=(u,v,w)^T,
\qquad
\boldsymbol\omega_B=R_{WB}^{T}\boldsymbol\omega_W=(p,q,r)^T.
$$

实现会把 $\lVert\boldsymbol\omega_B\rVert$ 限制到 `angular_velocity_limit`，默认 $5000\ \mathrm{rad/s}$。线速度不做同类限幅。

### 2.2 翼运动变量

- $s=+1$：左翼；$s=-1$：右翼；
- $\psi,\dot\psi$：拍动角及角速度；
- $\theta,\dot\theta$：羽化/翼俯仰角及角速度；
- $\phi,\dot\phi$：偏离角，代码名为 `deviation_angle/rate`；
- $\Phi,\dot\Phi$：扑翼平面角，代码名为 `stroke_plane_angle/rate`；
- $d_0$：半肩宽；$d_s$：扑翼平面相对机身的偏置；
- $r_w\in[0,R]$：翼展向坐标。为避免与机身偏航角速度 $r$ 混淆，本文把翼展坐标写成 $r_w$。

气动翼局部坐标约定为：`+X` 是翼法向，`+Y` 是展向/羽化轴。局部压力中心写成

$$
\boldsymbol x_{cp}^{A}=
\begin{bmatrix}0&s\,r_{cp}&-c_{cp}\end{bmatrix}^{T},
\qquad
\boldsymbol F^{A}=
\begin{bmatrix}F_N&0&0\end{bmatrix}^{T}.
$$

`mujoco_adapter.py` 用固定矩阵把 URDF 翼体坐标对齐到这个气动坐标，并对右翼做镜像匹配。

---

## 3. 翼几何参数为何是这些积分

设实际局部弦长为 $c(r_w)$，平均弦长为

$$
\bar c=\frac{1}{R}\int_0^R c(r_w)\,dr_w.
$$

引入无量纲变量

$$
\hat r=\frac{r_w}{R},\qquad
\hat c(\hat r)=\frac{c(R\hat r)}{\bar c}.
$$

`r00` 到 `r33` 实际表示弦长加权的径向矩：

$$
r_{kk}=\int_0^1\hat c(\hat r)\hat r^k\,d\hat r,
\qquad k=0,1,2,3.
$$

`z_cp0` 到 `z_cp2` 表示弦长平方加权的矩：

$$
z_{cp,k}=\int_0^1\hat c^2(\hat r)\hat r^k\,d\hat r,
\qquad k=0,1,2.
$$

为什么法向力使用 $r_{kk}$，而气动俯仰力矩使用 $z_{cp,k}$？因为翼条面积为 $c(r_w)dr_w$，压力中心到前缘的距离又与 $c(r_w)$ 成正比。因此

$$
dF_N\propto c(r_w)dr_w,\qquad
dM_{aero}\propto c^2(r_w)dr_w.
$$

当前翼是 $70\times20\ \mathrm{mm}$ 的矩形翼，所以 $\hat c=1$，从而

$$
r_{00}=1,quad r_{11}=\frac12,quad r_{22}=\frac13,quad r_{33}=\frac14,
$$

$$
z_{cp,0}=1,quad z_{cp,1}=\frac12,quad z_{cp,2}=\frac13.
$$

这正是 `Wing.json` 中的数值。其他主要参数为

$$
R=0.07\ \mathrm m,\quad \bar c=0.02\ \mathrm m,\quad
\rho=1.225\ \mathrm{kg/m^3},
$$

$$
d_0=0.01395\ \mathrm m,\qquad d_s=0.0189913\ \mathrm m.
$$

`examples/notebooks/02_wing_parameters.ipynb` 用翼平面多边形截线和 Gauss–Legendre 积分生成这些参数。其压力矩公式还要求羽化轴/前缘定义与参数积分一致；若翼型相对该轴存在不可忽略的弦向偏置，完整压力矩会多出代码目前没有保留的偏置项。

---

## 4. 局部相对速度的构造

### 4.1 最关键的结构：速度关于翼展坐标是仿射函数

刚体上一点的速度满足

$$
\boldsymbol V(r_w)=\boldsymbol V_O+\boldsymbol\Omega\times\boldsymbol r(r_w).
$$

翼展位置 $\boldsymbol r(r_w)$ 对 $r_w$ 是线性的，因此投影到翼的两个气动方向后仍为一次函数。代码把它们写成

$$
U_i(r_w)=U_{i1}r_w+U_{i0},\qquad
U_o(r_w)=U_{o1}r_w+U_{o0}.
$$

这里 $U_i$ 是主要的翼面内来流分量，$U_o$ 是出平面/法向来流分量。所有机身平移、机身转动、拍动、偏离运动和几何偏置最终都收集到四个系数中。

令 $S_x=\sin x,C_x=\cos x$，当前代码的完整表达式为

$$
\begin{aligned}
U_{o1}={}&s(pC_\psi C_\Phi-\dot\Phi S_\psi)
          +qS_\psi+rC_\psi S_\Phi+\dot\phi,\\
U_{o0}={}&s[-(u+qd_s)C_\phi S_\Phi
          -rd_0S_\phi S_\psi C_\Phi
          -(v-pd_s)S_\phi C_\psi\\
&\qquad+wS_\phi S_\psi S_\Phi
          +pd_0C_\phi C_\Phi]
          +(u+qd_s)S_\phi S_\psi C_\Phi\\
&\qquad+rd_0C_\phi S_\Phi+wC_\phi C_\Phi
          +pd_0S_\phi S_\psi S_\Phi,\\
U_{i1}={}&s[pS_\phi S_\psi C_\Phi+rC_\phi C_\Phi
          +\dot\Phi S_\phi C_\psi]
          -pC_\phi S_\Phi-qS_\phi C_\psi\\
&\qquad+rS_\phi S_\psi S_\Phi+\dot\psi C_\phi,\\
U_{i0}={}&s[rd_0C_\psi C_\Phi-(v-pd_s)S_\psi-wC_\psi S_\Phi]
          -(u+qd_s)C_\psi C_\Phi-pd_0C_\psi S_\Phi.
\end{aligned}
$$

默认 $\phi=\Phi=\dot\phi=\dot\Phi=0$ 时，它们化简为

$$
\begin{aligned}
U_{o1}&=spC_\psi+qS_\psi, & U_{o0}&=spd_0+w,\\
U_{i1}&=sr+\dot\psi, &
U_{i0}&=s[rd_0C_\psi-(v-pd_s)S_\psi]-(u+qd_s)C_\psi.
\end{aligned}
$$

若机身完全静止，则进一步得到

$$
U_i(r_w)=\dot\psi r_w,\qquad U_o(r_w)=0,
$$

这给出扑翼速度随展向半径线性增加的标准结果。

### 4.2 为什么使用固定的展向代表点

代码定义

$$
r_{cp}=R\frac{r_{33}}{r_{22}}.
$$

证明如下：纯旋转扑翼时 $U^2\propto r_w^2$，力分布满足

$$
dF\propto c(r_w)r_w^2dr_w.
$$

所以该工况下的展向压力中心为

$$
\frac{\int_0^Rr_w c(r_w)r_w^2dr_w}
     {\int_0^Rc(r_w)r_w^2dr_w}
=R\frac{r_{33}}{r_{22}}.
$$

矩形翼得到 $r_{cp}=3R/4=0.0525\ \mathrm m$。代码在所有工况下都用这个固定代表点计算攻角。若 $U_{i0},U_{o0}$ 很大，真实合力中心一般会移动；当前实现没有动态更新展向 CoP，这是准定常降阶假设之一。

---

## 5. 攻角、法向力系数和压力中心拟合

在固定代表点处计算

$$
U_i^*=U_{i1}r_{cp}+U_{i0},\qquad
U_o^*=U_{o1}r_{cp}+U_{o0}.
$$

流向修正角和攻角为

$$
\delta_\alpha=\arctan\!\left(\frac{U_o^*}{U_i^*}\right),
$$

$$
\alpha=\theta+\operatorname{sgn}(U_i^*)\frac{\pi}{2}-\delta_\alpha.
$$

当 $|U_i^*|\le10^{-12}$ 时，代码直接令 $\delta_\alpha=0$，而不是调用 `atan2`。因此在换向点附近，这个角度定义是带人为分支处理的。

法向力系数是经验拟合：

$$
C_N(\alpha)=1.8\sin(2\alpha)\cos\alpha
+1.95\sin\alpha
-1.5\cos(2\alpha)\sin\alpha.
$$

弦向压力中心占局部弦长的比例是

$$
d_{cp}(\alpha)=0.46-0.332\cos\alpha
-0.037\cos(3\alpha)-0.013\cos(5\alpha).
$$

这两个式子是模型输入的经验关系，不能由代码中的运动学从第一性原理推出。能严格验证的是它们的结构性质：

$$
C_N(-\alpha)=-C_N(\alpha),\qquad
d_{cp}(-\alpha)=d_{cp}(\alpha).
$$

也就是说，反转攻角会反转法向力系数，但压力中心比例保持不变。

---

## 6. 从翼条积分推导闭式法向力

局部速度平方为

$$
U^2(r_w)=U_i^2(r_w)+U_o^2(r_w).
$$

代入两个一次函数并展开：

$$
U^2(r_w)=a_{U2}r_w^2+a_{U1}r_w+a_{U0},
$$

其中

$$
\begin{aligned}
a_{U2}&=U_{i1}^2+U_{o1}^2,\\
a_{U1}&=2(U_{i1}U_{i0}+U_{o1}U_{o0}),\\
a_{U0}&=U_{i0}^2+U_{o0}^2.
\end{aligned}
$$

准定常翼条法向力为

$$
dF_N=\frac12\rho C_N(\alpha)c(r_w)U^2(r_w)dr_w.
$$

模型用代表点攻角，即认为同一时刻整片翼的 $C_N$ 相同。于是

$$
\begin{aligned}
F_N
&=\frac12\rho C_N\int_0^Rc(r_w)
(a_{U2}r_w^2+a_{U1}r_w+a_{U0})dr_w\\
&=\frac12\rho\bar c C_N
\left(a_{U2}R^3r_{22}+a_{U1}R^2r_{11}+a_{U0}Rr_{00}\right).
\end{aligned}
$$

这就是 `force_raw`。之后乘 `lift_scale`，再裁剪到

$$
F_N^{clip}=\operatorname{clip}(F_N,-F_{max},F_{max}),
$$

其中 Python 默认 $F_{max}=5\ \mathrm N$。

静止机身、纯扑动时 $U=|\dot\psi|r_w$，所以

$$
F_N=\frac12\rho\bar c C_N\dot\psi^2R^3r_{22}.
$$

在相同拍动相位和振幅下 $\dot\psi\propto f$，故

$$
F_N\propto f^2.
$$

这也是后面周期平均模型采用 $L=k_f f^2$ 的理论依据；其系数仍需用完整模型辨识，因为被动羽化使 $C_N$ 随周期变化。

---

## 7. 气动俯仰矩、弦向 CoP 与旋转阻尼

### 7.1 压力产生的气动俯仰矩

局部压力作用点距参考边的距离近似为 $d_{cp}c(r_w)$。因此翼条力矩为

$$
dM_{aero}=-d_{cp}(\alpha)c(r_w)dF_N.
$$

积分得到

$$
M_{aero}=-\frac12\rho\bar c^2C_Nd_{cp}
\left(a_{U2}R^3z_{cp,2}
+a_{U1}R^2z_{cp,1}
+a_{U0}Rz_{cp,0}\right).
$$

代码再乘 `moment_scale`。注意 `M_aero` 没有被独立写入 MuJoCo；它被换算为一个等效弦向压力中心：

$$
c_{cp}=-\frac{M_{aero}}{F_N^{raw}}.
$$

只要未触发力限幅，就有

$$
-c_{cp}F_N=M_{aero},
$$

所以在局部坐标中由压力中心力产生的 $y$ 轴矩恰好等于 `M_aero`。若 $F_N^{raw}=0$，代码令 $c_{cp}=0$。

### 7.2 羽化旋转阻尼

实现使用

$$
M_{rd}^{raw}
=-\frac18\rho|\dot\theta|\dot\theta
\left(5R\bar c^4z_{rd}\right).
$$

再乘 `moment_scale` 并裁剪到 $\pm0.05\ \mathrm{N\,m}$。其耗散性可直接证明：

$$
P_{rd}=M_{rd}\dot\theta
=-K|\dot\theta|\dot\theta^2\le0,
\qquad K\ge0.
$$

因此该项不会向羽化自由度主动注入能量。当前矩形翼参数为 $z_{rd}=1$。

### 7.3 从 $M_{aero}$、CoP 和 $M_{rd}$ 到单翼完整 wrench

这一段是几个量之间最容易混淆的地方。首先要区分它们各自改变什么：

| 量 | 是否改变合力大小 | 是否改变作用点 | 是否增加独立力矩 |
|---|---:|---:|---:|
| $F_N$ | 是 | 否 | 否 |
| $M_{aero}$ | 否 | 通过 $c_{cp}$ 改变 | 不直接施加 |
| $r_{cp}$、$c_{cp}$ | 否 | 是 | 通过力臂间接产生 |
| $M_{rd}$ | 否 | 否 | 是 |

因此，$M_{aero}$ 和 $M_{rd}$ 都不会再生成一份额外升力。单翼始终只有一个局部法向力

$$
\boldsymbol F_A=
\begin{bmatrix}F_N&0&0\end{bmatrix}^{T}.
$$

$M_{aero}$ 的作用是决定这个力在弦向哪里施加，$M_{rd}$ 则是在羽化轴上额外叠加一个纯力矩。计算流程可写成

$$
\boxed{
(F_N,M_{aero},M_{rd})
\longrightarrow
(r_{cp},c_{cp})
\longrightarrow
(\boldsymbol F_A,\boldsymbol r_{cp,A})
\longrightarrow
(\boldsymbol F_W,\boldsymbol\tau_W)
}.
$$

对第 $k$ 个翼，$k\in\{L,R\}$，令 $s_L=+1$、$s_R=-1$。压力中心相对翼体坐标原点的位置为

$$
\boldsymbol r_{cp,A,k}
=
\begin{bmatrix}
0\\
s_k r_{cp}\\
-c_{cp,k}
\end{bmatrix}.
$$

但 MuJoCo 中的外力矩必须关于该翼的质心计算。定义翼质心相对翼体原点的位置

$$
\boldsymbol r_{com,A,k}
=R_{WA,k}^{T}
\left(
\boldsymbol x_{com,W,k}-\boldsymbol x_{body,W,k}
\right),
$$

则压力中心相对质心的真实力臂是

$$
\boldsymbol\ell_{A,k}
=\boldsymbol r_{cp,A,k}-\boldsymbol r_{com,A,k}
=
\begin{bmatrix}
\ell_x&\ell_y&\ell_z
\end{bmatrix}^{T}.
$$

局部力的叉乘可以完全展开：

$$
\begin{aligned}
\boldsymbol\tau_{F,A,k}
&=\boldsymbol\ell_{A,k}\times\boldsymbol F_{A,k}\\
&=
\begin{bmatrix}
\ell_x\\
\ell_y\\
\ell_z
\end{bmatrix}
\times
\begin{bmatrix}
F_{N,k}\\
0\\
0
\end{bmatrix}\\
&=
\begin{bmatrix}
0\\
\ell_zF_{N,k}\\
-\ell_yF_{N,k}
\end{bmatrix}.
\end{aligned}
$$

这个结果说明：

1. 弦向力臂 $\ell_z$ 产生局部 $y$ 轴俯仰矩；
2. 展向力臂 $\ell_y$ 产生局部 $z$ 轴力矩；
3. 单独的局部法向力不会通过这个力臂产生局部 $x$ 轴矩。

如果暂时忽略翼体原点与质心在弦向的偏移，即取

$$
\ell_z=-c_{cp,k},
$$

那么

$$
\tau_{F,A,k,y}
=-c_{cp,k}F_{N,k}
=M_{aero,k}.
$$

这就是“用 CoP 隐式实现 $M_{aero}$”的严格含义。若翼体原点与质心不重合，则

$$
\ell_z=-c_{cp,k}-r_{com,A,k,z},
$$

于是关于质心的 $y$ 轴矩还会包含

$$
-r_{com,A,k,z}F_{N,k},
$$

它是把参考点从翼体原点平移到质心时必然出现的力矩，不是重复计算气动俯仰矩。

旋转阻尼是一个与压力中心力臂无关的纯力矩：

$$
\boldsymbol\tau_{rd,A,k}
=
\begin{bmatrix}
0\\
M_{rd,k}\\
0
\end{bmatrix}.
$$

因此单翼局部总力矩为

$$
\boldsymbol\tau_{A,k}
=\boldsymbol\tau_{F,A,k}
+\boldsymbol\tau_{rd,A,k}
=
\begin{bmatrix}
0\\
\ell_zF_{N,k}+M_{rd,k}\\
-\ell_yF_{N,k}
\end{bmatrix}.
$$

再利用旋转矩阵保持叉乘的性质

$$
R(\boldsymbol a\times\boldsymbol b)
=(R\boldsymbol a)\times(R\boldsymbol b),
$$

可把单翼 wrench 旋转到世界系：

$$
\boldsymbol F_{W,k}=R_{WA,k}\boldsymbol F_{A,k},
\qquad
\boldsymbol\tau_{W,k}=R_{WA,k}\boldsymbol\tau_{A,k}.
$$

这与代码先求世界系 CoP、再在世界系计算叉乘完全等价。

### 7.4 从左右单翼到 $F_{\mathrm{lift,total}}$

左右翼纯气动函数给出的世界系合力是向量和，而不是绝对值之和：

$$
\boldsymbol F_{\mathrm{pure,total}}
=\boldsymbol F_{W,L}+\boldsymbol F_{W,R}
=R_{WA,L}
\begin{bmatrix}F_{N,L}\\0\\0\end{bmatrix}
+R_{WA,R}
\begin{bmatrix}F_{N,R}\\0\\0\end{bmatrix}.
$$

如果把世界系 $+z$ 定义为竖直向上，则真正的瞬时总升力是

$$
F_{\mathrm{lift,total}}^{W}
=\boldsymbol e_z^{T}\boldsymbol F_{\mathrm{applied,total}}
=F_{\mathrm{applied,total},z}.
$$

当前适配器因为 STL 法向约定会翻转两翼的力，所以

$$
\boldsymbol F_{\mathrm{applied},k}=-\boldsymbol F_{W,k},
$$

从而

$$
\boldsymbol F_{\mathrm{applied,total}}
=-\left(\boldsymbol F_{W,L}+\boldsymbol F_{W,R}\right)
=-\boldsymbol F_{\mathrm{pure,total}}.
$$

这正是适配器最终保存到诊断量 $F_{\mathrm{world}}$ 的值。因而在当前模型里，判断机器人受到的实际竖直气动力应使用

$$
\boxed{
F_{\mathrm{lift,total}}^{W}
=\text{diagnostics}[F_{\mathrm{world}}]_z
}.
$$

代码中的标量

$$
\text{lift\_total}=|F_{N,L}|+|F_{N,R}|
$$

含义不同。它只是在两个气动翼局部坐标中，把法向力大小相加。它不包含以下信息：

1. 翼法向相对世界竖直方向的倾斜；
2. 左右翼水平分量是相消还是相加；
3. 适配器对 STL 法向做的整体翻转；
4. $M_{aero}$、CoP 和 $M_{rd}$ 产生的任何力矩。

令两翼实际法向的世界系竖直投影为

$$
\gamma_k
=\boldsymbol e_z^{T}R_{WA,k}
\begin{bmatrix}1\\0\\0\end{bmatrix},
$$

则实际总竖直气动力可明确写成

$$
F_{\mathrm{lift,total}}^{W}
=-\gamma_LF_{N,L}-\gamma_RF_{N,R}.
$$

只有在两翼法向都与竖直方向对齐、符号约定也已经统一时，才可能近似写成

$$
F_{\mathrm{lift,total}}^{W}
\approx |F_{N,L}|+|F_{N,R}|.
$$

所以代码里的 $\text{lift\_total}$ 更适合用作“气动载荷强度”诊断量，不能在任意姿态下直接当作世界系升力。

接口级数值检查也验证了这条关系：每一步左右翼 `xfrc_applied[:, :3]` 的向量和与适配器最终的 `diagnostics["F_world"]` 完全一致，当前检查的差值范数为 0。

### 7.5 左右翼总力矩如何定义

两个翼的力矩不能在任意参考点下直接相加。代码写入每个翼的力矩是关于该翼质心的。若希望得到关于机器人参考点 $O$ 的整机气动总力矩，应使用

$$
\boldsymbol\tau_{O,\mathrm{total}}
=\sum_{k\in\{L,R\}}
\left[
\boldsymbol\tau_{\mathrm{applied},k}
+(\boldsymbol x_{com,W,k}-\boldsymbol x_{O,W})
\times\boldsymbol F_{\mathrm{applied},k}
\right].
$$

当前适配器保持纯函数算出的力矩、只翻转力：

$$
\boldsymbol\tau_{\mathrm{applied},k}
=\boldsymbol\tau_{W,k},
\qquad
\boldsymbol F_{\mathrm{applied},k}
=-\boldsymbol F_{W,k}.
$$

因此适配器之后的实际 wrench 必须按上面两个量直接理解，不能再假设它等价于“翻转后的力仍然作用在原 CoP”。MuJoCo 会分别把左右翼 wrench 写入对应翼体，并通过关节约束把它们传递为整机广义力。

---

## 8. 从局部力到 MuJoCo 世界系 wrench

设 $R_{WA}$ 为气动翼坐标到世界坐标的旋转矩阵。纯函数 `world_wrench` 计算

$$
\boldsymbol x_{cp}^{W}=\boldsymbol x_{body}^{W}
+R_{WA}\boldsymbol x_{cp}^{A},
$$

$$
\boldsymbol F^{W}=R_{WA}\boldsymbol F^{A},
$$

$$
\boldsymbol\tau_F^{W}=
(\boldsymbol x_{cp}^{W}-\boldsymbol x_{com}^{W})\times\boldsymbol F^{W},
$$

$$
\boldsymbol\tau_{rd}^{W}=R_{WA}
\begin{bmatrix}0&M_{rd}&0\end{bmatrix}^{T},
\qquad
\boldsymbol\tau^{W}=\boldsymbol\tau_F^{W}+\boldsymbol\tau_{rd}^{W}.
$$

最终 wrench 顺序遵循 MuJoCo：

$$
\boldsymbol w=
[F_x,F_y,F_z,\tau_x,\tau_y,\tau_z]^T.
$$

`read_only=True` 时所有诊断量照常计算，但返回给 `xfrc_applied` 的 wrench 为零。

### 8.1 适配器的实际法向修正

STL 翼法向与气动模型的有向法向相反。`MuJoCoAeroAdapter` 在 `world_wrench` 计算完成后执行

$$
\boldsymbol F_{applied}=-\boldsymbol F_{pure},\qquad
\boldsymbol\tau_{applied}=\boldsymbol\tau_{pure}.
$$

也就是说，只翻转前三个力分量，不翻转已经由原力计算出的 `r×F` 力矩，也不翻转阻尼矩。这个行为是有意用于当前 URDF/STL 坐标约定，但它意味着适配器最终注入的 wrench 不再严格等价于“把翻转后的力施加在同一个 CoP”。诊断中的 `F_world` 也被翻转，以匹配实际施加的力。

---

## 9. 左右翼处理与对称性

`aero_step` 对左翼使用 $s=+1$，右翼使用 $s=-1$。适配器还对右翼拍动角和角速度使用 `stroke_sign=-1`，以把 URDF 镜像关节统一到同一气动符号约定。羽化角符号由 `pitch_axis_y` 决定，当前适配器传入 $+1$。

理想镜像状态下，左右翼的法向力在各自局部系中可以符号相反，但经镜像旋转和适配器法向修正后，竖直分量相加，侧向分量和部分力矩相消。代码诊断量

$$
\texttt{lift\_total}=|F_{N,L}|+|F_{N,R}|
$$

只是局部法向力绝对值之和，不等于严格的世界系竖直升力；实际世界合力应读取 `diagnostics["F_world"]`。

---

## 10. 周期平均悬停模型

自由飞行 `fix=False` 不使用上述逐翼瞬时模型，而使用 `AveragedHoverAero`。同时环境把机器人动力学等效为质量 $m=0.03\ \mathrm{kg}$ 的单刚体；子部件只保留极小的数值惯量，扑翼关节速度也不参与动态能量注入。

### 10.1 平均升力

模型写成

$$
L(f)=k_f f^2,
\qquad
k_f=\frac{0.14858349039827917}{30^2}
=1.6509276711\times10^{-4}\ \mathrm{N/Hz^2}.
$$

机体系推力方向由平台命令决定：

$$
\boldsymbol n_B=
\frac{[-\delta_r,-\delta_p,1]^T}
{\sqrt{1+\delta_r^2+\delta_p^2}},
\qquad
\boldsymbol F_{thrust}^{W}=R_{WB}L\boldsymbol n_B.
$$

其中 $\delta_p=\texttt{pitch\_joint}$，$\delta_r=\texttt{roll\_joint}$。小角度下

$$
\boldsymbol n_B\approx[-\delta_r,-\delta_p,1]^T,
$$

所以平台倾斜线性地产生水平推力。

世界系水平阻尼为

$$
\boldsymbol F_{xy}^{W}\leftarrow
\boldsymbol F_{xy}^{W}-k_v\boldsymbol v_{xy}^{W},
\qquad k_v=0.06\ \mathrm{N\,s/m}.
$$

### 10.2 平均控制力矩

辨识力矩随频率平方缩放：

$$
\lambda_f=\left(\frac{f}{42}\right)^2.
$$

机体系力矩为

$$
\boldsymbol\tau_B=
\lambda_f
\begin{bmatrix}
k_{roll}\delta_p\\
-k_{pitch}\delta_r\\
0
\end{bmatrix}
-D_\omega\boldsymbol\omega_B,
$$

其中

$$
k_{roll}=\frac{0.01055}{20},\quad
k_{pitch}=\frac{0.01144}{20}\quad[\mathrm{N\,m/rad}],
$$

$$
D_\omega=\operatorname{diag}(2,2,1)\times10^{-4}
\quad[\mathrm{N\,m\,s/rad}].
$$

URDF 命名和物理轴是交叉的：`pitch_joint` 主要控制滚转矩，`roll_joint` 主要控制俯仰矩。因此上式存在交叉项，而不是同名角度控制同名轴。

### 10.3 悬停频率的一致性

30 g 等效质量的重力为

$$
mg=0.03\times9.81=0.2943\ \mathrm N.
$$

由 $k_f f_h^2=mg$ 得

$$
f_h=\sqrt{\frac{mg}{k_f}}=42.2213\ \mathrm{Hz}.
$$

这与 `HoveringController` 的标称 `hover_frequency=42 Hz` 一致。42 Hz 时模型升力为 $0.291224\ \mathrm N$，比重力低约 $1.05\%$，剩余静差由高度 PID 的频率修正补偿。

---

## 11. 数值验证

以下验证在当前仓库代码和当前 `Wing.json` 上执行，单位均为 SI。

### 11.1 纯函数闭式积分验证

选取代表状态

$$
\psi=0.37,\ \dot\psi=92,\ \theta=0.48,\ \dot\theta=-11,
$$

$$
\boldsymbol v_B=(0.3,-0.2,0.1),\quad
\boldsymbol\omega_B=(0.7,-0.5,0.4).
$$

用 200001 个展向点直接数值积分翼条公式，并与 `wing_aero` 的几何矩闭式公式比较：

| 检查项 | 代码闭式结果 | 数值积分结果 | 相对误差 |
|---|---:|---:|---:|
| $F_N$ | 0.0351425732167 N | 0.0351425732172 N | $1.36\times10^{-11}$ |
| $M_{aero}$ | -0.000406734027976 N·m | -0.000406734027981 N·m | $1.36\times10^{-11}$ |

结论：从仿射速度展开到 `rxx/z_cpx` 几何矩的闭式积分实现正确，误差仅来自数值求积。

其他代数性质验证结果：

| 性质 | 结果 |
|---|---:|
| $r_{cp}=3R/4$ | 0.05250000000000003 m |
| $-M_{aero}/F_N=c_{cp}$ | 两边均为 0.0115738260106 m |
| $M_{rd}\dot\theta$ | $-1.14133\times10^{-5}\ \mathrm W<0$ |
| 15 Hz 到 30 Hz 的同相位力比 | 4.0 |
| $C_N$ 奇函数最大残差（1001 点） | 0.0 |
| $d_{cp}$ 偶函数最大残差（1001 点） | 0.0 |

### 11.2 完整固定基座仿真扫频验证

分别设置 20、30、40 Hz，运行 0.6 s，舍弃前 0.3 s，对适配器实际世界系竖直力取平均：

| 频率 | 平均 $F_z$ | $F_z/f^2$ |
|---:|---:|---:|
| 20 Hz | 0.0661819985 N | $1.65455\times10^{-4}$ N/Hz² |
| 30 Hz | 0.1486352297 N | $1.65150\times10^{-4}$ N/Hz² |
| 40 Hz | 0.2639114107 N | $1.64945\times10^{-4}$ N/Hz² |

30 Hz 结果与平均模型使用的辨识点 $0.1485834904\ \mathrm N$ 相差约 $0.035\%$。三点的 $F_z/f^2$ 基本恒定，证明当前被动羽化稳态下用 $L=k_ff^2$ 降阶是合理的局部近似。

### 11.3 自由悬停验证

从 $z=0.5\ \mathrm m$ 启动自由模型并仿真 2.5 s：

| 指标 | 结果 |
|---|---:|
| 最终高度 | 0.479012 m |
| 高度范围 | 0.470056–0.500000 m |
| 最后 0.5 s 平均高度 | 0.475627 m |
| 最后 0.5 s 平均频率 | 42.234855 Hz |

结论：平均模型保持有界并收敛到标称悬停频率附近；有限时间内仍存在约 2.4 cm 高度偏差，所以这项结果证明的是模型与控制器的定性稳定和量级一致，不应解读为严格零静差或全飞行包线验证。

---

## 12. 已证明的内容、模型假设与实现边界

### 12.1 可以由实现严格证明或数值复核的内容

1. 刚体运动学使两个局部速度分量关于展向坐标为一次函数；
2. 速度平方因此是二次多项式，可以用预计算几何矩精确完成展向积分；
3. 未限幅时，弦向 CoP 产生的力矩与 `M_aero` 代数等价；
4. 旋转阻尼始终满足 $M_{rd}\dot\theta\le0$；
5. 纯扑动同相位法向力随 $f^2$ 缩放；当前完整固定基座扫频也支持这一近似；
6. 周期平均模型的标称悬停频率与 30 g 重力平衡一致。

### 12.2 不能称为第一性原理证明的内容

1. $C_N(\alpha)$ 和 $d_{cp}(\alpha)$ 是经验拟合，其准确性需要实验或更高保真 CFD 数据验证；
2. 用单一代表点攻角代替展向变化攻角是一项准定常降阶假设；
3. 固定 $r_{cp}=Rr_{33}/r_{22}$ 只在速度平方主要由 $r_w^2$ 项主导时严格成立；
4. 当前模型不显式包含尾迹捕获、附加质量、翼间干扰、动态失速和地面效应；
5. 周期平均模型仅由有限工作点辨识，不能自动外推到任意振幅、频率、姿态和大速度飞行。

### 12.3 阅读和修改代码时应特别注意的边界

1. `Wing.json` 中虽然带有 `options`，但当前 `MuJoCoAeroAdapter` 直接构造 `AeroOptions()`，没有加载 JSON 里的 options。因此 JSON 中的 `force_limit=2.0` 当前不生效，实际默认值是 Python 数据类中的 5.0 N。
2. `force_raw` 被限幅，但 `M_aero` 不限幅，`chord_CoP` 又由未限幅的 `force_raw` 计算。触发力限幅后，`r×F_clipped` 不再等于原来的 `M_aero`。
3. 适配器只翻转最终力，不翻转已计算力矩；因此应区分纯函数 wrench 和实际注入 MuJoCo 的 wrench。
4. 攻角使用 `atan(U_o/U_i)` 且在 $U_i\approx0$ 时强制修正角为零，换向点附近不具备 `atan2` 的完整象限连续性。
5. `diagnostics["lift_total"]` 是两翼局部法向力绝对值之和；判断悬停应使用世界系 `F_world[2]`，不能直接把它当作竖直升力。
6. `fix=True/False` 不只是是否固定根部，还会切换气动模型和整机质量/惯量建模方式。比较两种仿真时必须把这项模型变化考虑在内。

---

## 13. 总结

当前逐翼 aero 的核心可以概括为：

$$
\boxed{
\text{刚体运动学}
\Rightarrow U_i,U_o\text{ 关于 }r_w\text{ 为一次函数}
\Rightarrow U^2\text{ 为二次函数}
\Rightarrow \text{用翼几何矩闭式积分 }F_N,M_{aero}
\Rightarrow \text{CoP+wrench 写入 MuJoCo}
}
$$

周期平均模型则把上述完整周期响应进一步降阶为

$$
\boxed{L=k_ff^2+\text{倾斜方向}+\text{线性阻尼}+\text{辨识控制力矩}}
$$

闭式积分、压力中心恒等式、阻尼耗散性和频率平方律均已通过代数或数值验证。模型的主要不确定性不在积分实现，而在经验气动系数、单点攻角、固定展向压力中心以及从固定基座辨识到自由飞行的外推范围。
