# FlapSim aerodynamics (aero): implementation, derivation and validation

*Translated from the Chinese original (`docs/zh/derivations/quasi_steady_aero.md`). It documents the implementation as it stood before 2026-10-28; the pitching-moment sign and the removed force flip are described in [quasi_steady_model.md](quasi_steady_model.md).*

> This document keeps the complete code-level implementation and validation details. The simplified ideal-hover derivation intended for the main text of the paper is in [hover_simplification.md](hover_simplification.md), where $a_{U1}=a_{U0}=0$ is taken under explicit assumptions.

This document collects the actual implementation in `flapsim/fluid` at the time of writing (the per-wing engine now lives in `flapsim/aero/`). Here "aero" is not one model but two models with different purposes:

1. `fix=True`: `MuJoCoAeroAdapter` calls `aero.py` and computes quasi-steady aerodynamic forces wing by wing at every simulation step;
2. `fix=False`: `AveragedHoverAero` uses the cycle-averaged force and moment identified from a fixed-base frequency sweep.

One of the two is selected in `SimCore.__init__`; they never run at the same time. Below we first derive the per-wing model in full, then describe the cycle-averaged model and its identification relationships.

---

## 1. Files and the runtime call chain

The key files are:

| File | Role |
|---|---|
| `flapsim/fluid/aero.py` | Pure aerodynamic computation decoupled from MuJoCo: velocities, angle of attack, normal force, centre of pressure and wrench |
| `flapsim/fluid/mujoco_adapter.py` (now a thin compatibility wrapper around `flapsim/aero/base.py`, `AeroEngine`, and the `aero/quasi_steady` plugin in `flapsim/aero/stack.py`) | Reads state from `MjData`, handles the left/right wing frames, writes the wrench into `xfrc_applied` |
| `flapsim/fluid/averaged_hover.py` (now `flapsim/aero/cycle_averaged.py`) | Cycle-averaged hover model used in free flight |
| `flapsim/config/Wing.json` | Wing geometry integrals, air density and shoulder geometry parameters |
| `flapsim/config/AeroOptions.py` | Runtime parameters: scaling, clipping, stroke-plane angle and deviation angle |
| `flapsim/sim_core.py` | Selects the aero model according to `fix` and attaches it to the environment |
| `flapsim/environment/environment.py` (now `flapsim/core/simulation.py`, class `Simulation`) | Each step: update the controller first, then apply the aerodynamic forces, finally call `mujoco.mj_step` |

The order of operations in each fixed-base simulation step is

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

`TrajectoryController` (now `flapsim/actuators/sinusoidal_stroke.py`) prescribes the stroke angle as

$$
\psi(t)=dA\sin(2\pi ft+\varphi_0),\qquad
\dot\psi(t)=dA(2\pi f)\cos(2\pi ft+\varphi_0),
$$

where $d\in\{-1,+1\}$ is the left/right wing direction and the default amplitude is $A=1.5\ \mathrm{rad}$.

---

## 2. Coordinates, state and notation

### 2.1 Body state

Let $R_{WB}$ be the rotation matrix from body coordinates to world coordinates. The world-frame velocities given by MuJoCo are first transformed into the body frame:

$$
\boldsymbol v_B=R_{WB}^{T}\boldsymbol v_W=(u,v,w)^T,
\qquad
\boldsymbol\omega_B=R_{WB}^{T}\boldsymbol\omega_W=(p,q,r)^T.
$$

The implementation clips $\lVert\boldsymbol\omega_B\rVert$ to `angular_velocity_limit`, default $5000\ \mathrm{rad/s}$. The linear velocity is not clipped in the same way.

### 2.2 Wing motion variables

- $s=+1$: left wing; $s=-1$: right wing;
- $\psi,\dot\psi$: stroke angle and stroke rate;
- $\theta,\dot\theta$: feathering / wing pitch angle and its rate;
- $\phi,\dot\phi$: deviation angle, named `deviation_angle/rate` in the code;
- $\Phi,\dot\Phi$: stroke-plane angle, named `stroke_plane_angle/rate` in the code;
- $d_0$: half shoulder width; $d_s$: offset of the stroke plane relative to the body;
- $r_w\in[0,R]$: spanwise coordinate. To avoid confusion with the body yaw rate $r$, this document writes the spanwise coordinate as $r_w$.

The wing aerodynamic frame convention is: `+X` is the wing normal, `+Y` is the spanwise/feathering axis. The local centre of pressure is written as

$$
\boldsymbol x_{cp}^{A}=
\begin{bmatrix}0&s\,r_{cp}&-c_{cp}\end{bmatrix}^{T},
\qquad
\boldsymbol F^{A}=
\begin{bmatrix}F_N&0&0\end{bmatrix}^{T}.
$$

`mujoco_adapter.py` uses a fixed matrix to align the URDF wing-body frame with this aerodynamic frame, and mirrors it for the right wing.

---

## 3. Why the wing geometry parameters are these integrals

Let the actual local chord be $c(r_w)$ and the mean chord be

$$
\bar c=\frac{1}{R}\int_0^R c(r_w)\,dr_w.
$$

Introduce the dimensionless variables

$$
\hat r=\frac{r_w}{R},\qquad
\hat c(\hat r)=\frac{c(R\hat r)}{\bar c}.
$$

`r00` to `r33` are in fact chord-weighted radial moments:

$$
r_{kk}=\int_0^1\hat c(\hat r)\hat r^k\,d\hat r,
\qquad k=0,1,2,3.
$$

`z_cp0` to `z_cp2` are moments weighted by the chord squared:

$$
z_{cp,k}=\int_0^1\hat c^2(\hat r)\hat r^k\,d\hat r,
\qquad k=0,1,2.
$$

Why does the normal force use $r_{kk}$ while the aerodynamic pitching moment uses $z_{cp,k}$? Because the strip area is $c(r_w)dr_w$, and the distance from the centre of pressure to the leading edge is also proportional to $c(r_w)$. Hence

$$
dF_N\propto c(r_w)dr_w,\qquad
dM_{aero}\propto c^2(r_w)dr_w.
$$

The current wing is a $70\times20\ \mathrm{mm}$ rectangular wing, so $\hat c=1$ and therefore

$$
r_{00}=1,\quad r_{11}=\frac12,\quad r_{22}=\frac13,\quad r_{33}=\frac14,
$$

$$
z_{cp,0}=1,\quad z_{cp,1}=\frac12,\quad z_{cp,2}=\frac13.
$$

These are exactly the values in `Wing.json`. The other main parameters are

$$
R=0.07\ \mathrm m,\quad \bar c=0.02\ \mathrm m,\quad
\rho=1.225\ \mathrm{kg/m^3},
$$

$$
d_0=0.01395\ \mathrm m,\qquad d_s=0.0189913\ \mathrm m.
$$

`examples/notebooks/02_wing_parameters.ipynb` generates these parameters from polygon cross-sections of the wing planform and Gauss–Legendre quadrature. Its pressure-moment formula also requires the feathering-axis/leading-edge definition to be consistent with the parameter integrals; if the planform has a non-negligible chordwise offset relative to that axis, the full pressure moment acquires an offset term that the code does not currently retain.

---

## 4. Construction of the local relative velocity

### 4.1 The key structural fact: velocity is an affine function of the spanwise coordinate

The velocity of a point on a rigid body satisfies

$$
\boldsymbol V(r_w)=\boldsymbol V_O+\boldsymbol\Omega\times\boldsymbol r(r_w).
$$

The spanwise position $\boldsymbol r(r_w)$ is linear in $r_w$, so after projection onto the two aerodynamic directions of the wing it remains a first-order function. The code writes them as

$$
U_i(r_w)=U_{i1}r_w+U_{i0},\qquad
U_o(r_w)=U_{o1}r_w+U_{o0}.
$$

Here $U_i$ is the main in-plane inflow component and $U_o$ is the out-of-plane / normal inflow component. All body translation, body rotation, stroke motion, deviation motion and geometric offsets are ultimately collected into these four coefficients.

Writing $S_x=\sin x,C_x=\cos x$, the complete expressions in the current code are

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

With the defaults $\phi=\Phi=\dot\phi=\dot\Phi=0$ they reduce to

$$
\begin{aligned}
U_{o1}&=spC_\psi+qS_\psi, & U_{o0}&=spd_0+w,\\
U_{i1}&=sr+\dot\psi, &
U_{i0}&=s[rd_0C_\psi-(v-pd_s)S_\psi]-(u+qd_s)C_\psi.
\end{aligned}
$$

If the body is completely stationary, one further obtains

$$
U_i(r_w)=\dot\psi r_w,\qquad U_o(r_w)=0,
$$

which gives the standard result that the flapping velocity increases linearly with spanwise radius.

### 4.2 Why a fixed spanwise representative point is used

The code defines

$$
r_{cp}=R\frac{r_{33}}{r_{22}}.
$$

The proof is as follows: for pure rotational flapping $U^2\propto r_w^2$, and the force distribution satisfies

$$
dF\propto c(r_w)r_w^2dr_w.
$$

So the spanwise centre of pressure for this case is

$$
\frac{\int_0^Rr_w c(r_w)r_w^2dr_w}
     {\int_0^Rc(r_w)r_w^2dr_w}
=R\frac{r_{33}}{r_{22}}.
$$

For a rectangular wing $r_{cp}=3R/4=0.0525\ \mathrm m$. The code uses this fixed representative point to compute the angle of attack in all operating conditions. When $U_{i0},U_{o0}$ are large the true centre of the resultant force generally moves; the current implementation does not update the spanwise CoP dynamically, which is one of the quasi-steady reduced-order assumptions.

---

## 5. Angle of attack, normal-force coefficient and centre-of-pressure fits

At the fixed representative point compute

$$
U_i^*=U_{i1}r_{cp}+U_{i0},\qquad
U_o^*=U_{o1}r_{cp}+U_{o0}.
$$

The flow-correction angle and the angle of attack are

$$
\delta_\alpha=\arctan\!\left(\frac{U_o^*}{U_i^*}\right),
$$

$$
\alpha=\theta+\operatorname{sgn}(U_i^*)\frac{\pi}{2}-\delta_\alpha.
$$

When $|U_i^*|\le10^{-12}$ the code sets $\delta_\alpha=0$ directly instead of calling `atan2`. Near stroke reversal this angle definition therefore contains an artificial branch.

The normal-force coefficient is an empirical fit:

$$
C_N(\alpha)=1.8\sin(2\alpha)\cos\alpha
+1.95\sin\alpha
-1.5\cos(2\alpha)\sin\alpha.
$$

The chordwise centre of pressure as a fraction of the local chord is

$$
d_{cp}(\alpha)=0.46-0.332\cos\alpha
-0.037\cos(3\alpha)-0.013\cos(5\alpha).
$$

These two expressions are empirical relations supplied as model inputs; they cannot be derived from first principles using the kinematics in the code. What can be verified rigorously are their structural properties:

$$
C_N(-\alpha)=-C_N(\alpha),\qquad
d_{cp}(-\alpha)=d_{cp}(\alpha).
$$

That is, reversing the angle of attack reverses the normal-force coefficient, while the centre-of-pressure fraction stays the same.

---

## 6. Closed-form normal force from the strip integral

The local velocity squared is

$$
U^2(r_w)=U_i^2(r_w)+U_o^2(r_w).
$$

Substituting the two first-order functions and expanding:

$$
U^2(r_w)=a_{U2}r_w^2+a_{U1}r_w+a_{U0},
$$

where

$$
\begin{aligned}
a_{U2}&=U_{i1}^2+U_{o1}^2,\\
a_{U1}&=2(U_{i1}U_{i0}+U_{o1}U_{o0}),\\
a_{U0}&=U_{i0}^2+U_{o0}^2.
\end{aligned}
$$

The quasi-steady strip normal force is

$$
dF_N=\frac12\rho C_N(\alpha)c(r_w)U^2(r_w)dr_w.
$$

The model uses the angle of attack at the representative point, i.e. it assumes the whole wing shares one $C_N$ at any instant. Therefore

$$
\begin{aligned}
F_N
&=\frac12\rho C_N\int_0^Rc(r_w)
(a_{U2}r_w^2+a_{U1}r_w+a_{U0})dr_w\\
&=\frac12\rho\bar c C_N
\left(a_{U2}R^3r_{22}+a_{U1}R^2r_{11}+a_{U0}Rr_{00}\right).
\end{aligned}
$$

This is `force_raw`. It is then multiplied by `lift_scale` and clipped to

$$
F_N^{clip}=\operatorname{clip}(F_N,-F_{max},F_{max}),
$$

where the Python default is $F_{max}=5\ \mathrm N$.

For a stationary body and pure flapping $U=|\dot\psi|r_w$, so

$$
F_N=\frac12\rho\bar c C_N\dot\psi^2R^3r_{22}.
$$

At the same stroke phase and amplitude $\dot\psi\propto f$, hence

$$
F_N\propto f^2.
$$

This is also the theoretical basis for the cycle-averaged model's use of $L=k_f f^2$ later on; its coefficient still has to be identified with the full model, because passive feathering makes $C_N$ vary over the cycle.

---

## 7. Aerodynamic pitching moment, chordwise CoP and rotational damping

### 7.1 Aerodynamic pitching moment produced by the pressure

The local pressure acts at a distance of approximately $d_{cp}c(r_w)$ from the reference edge. The strip moment is therefore

$$
dM_{aero}=-d_{cp}(\alpha)c(r_w)dF_N.
$$

Integrating gives

$$
M_{aero}=-\frac12\rho\bar c^2C_Nd_{cp}
\left(a_{U2}R^3z_{cp,2}
+a_{U1}R^2z_{cp,1}
+a_{U0}Rz_{cp,0}\right).
$$

The code then multiplies by `moment_scale`. Note that `M_aero` is not written into MuJoCo independently; it is converted into an equivalent chordwise centre of pressure:

$$
c_{cp}=-\frac{M_{aero}}{F_N^{raw}}.
$$

As long as the force clip is not triggered,

$$
-c_{cp}F_N=M_{aero},
$$

so the $y$-axis moment produced in local coordinates by the force at the centre of pressure equals `M_aero` exactly. If $F_N^{raw}=0$ the code sets $c_{cp}=0$.

### 7.2 Feathering rotational damping

The implementation uses

$$
M_{rd}^{raw}
=-\frac18\rho|\dot\theta|\dot\theta
\left(5R\bar c^4z_{rd}\right).
$$

It is then multiplied by `moment_scale` and clipped to $\pm0.05\ \mathrm{N\,m}$. Its dissipativity can be shown directly:

$$
P_{rd}=M_{rd}\dot\theta
=-K|\dot\theta|\dot\theta^2\le0,
\qquad K\ge0.
$$

Hence this term never actively injects energy into the feathering degree of freedom. For the current rectangular wing $z_{rd}=1$.

### 7.3 From $M_{aero}$, the CoP and $M_{rd}$ to the complete single-wing wrench

This is the part where the quantities are most easily confused. First distinguish what each of them changes:

| Quantity | Changes the magnitude of the resultant force? | Changes the point of application? | Adds an independent moment? |
|---|---:|---:|---:|
| $F_N$ | yes | no | no |
| $M_{aero}$ | no | yes, through $c_{cp}$ | not applied directly |
| $r_{cp}$, $c_{cp}$ | no | yes | indirectly, through the lever arm |
| $M_{rd}$ | no | no | yes |

Therefore neither $M_{aero}$ nor $M_{rd}$ generates an additional share of lift. A single wing always has exactly one local normal force

$$
\boldsymbol F_A=
\begin{bmatrix}F_N&0&0\end{bmatrix}^{T}.
$$

The role of $M_{aero}$ is to decide where along the chord this force is applied; $M_{rd}$ adds a pure moment about the feathering axis. The computation can be written as

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

For wing $k$, $k\in\{L,R\}$, let $s_L=+1$ and $s_R=-1$. The position of the centre of pressure relative to the wing-body frame origin is

$$
\boldsymbol r_{cp,A,k}
=
\begin{bmatrix}
0\\
s_k r_{cp}\\
-c_{cp,k}
\end{bmatrix}.
$$

But external torques in MuJoCo must be taken about the centre of mass of that wing. Define the position of the wing centre of mass relative to the wing-body origin

$$
\boldsymbol r_{com,A,k}
=R_{WA,k}^{T}
\left(
\boldsymbol x_{com,W,k}-\boldsymbol x_{body,W,k}
\right),
$$

then the true lever arm from the centre of mass to the centre of pressure is

$$
\boldsymbol\ell_{A,k}
=\boldsymbol r_{cp,A,k}-\boldsymbol r_{com,A,k}
=
\begin{bmatrix}
\ell_x&\ell_y&\ell_z
\end{bmatrix}^{T}.
$$

The cross product with the local force can be fully expanded:

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

This result says:

1. the chordwise lever arm $\ell_z$ produces a local $y$-axis pitching moment;
2. the spanwise lever arm $\ell_y$ produces a local $z$-axis moment;
3. the local normal force alone does not produce a local $x$-axis moment through this lever arm.

If one temporarily ignores the chordwise offset between the wing-body origin and the centre of mass, i.e. takes

$$
\ell_z=-c_{cp,k},
$$

then

$$
\tau_{F,A,k,y}
=-c_{cp,k}F_{N,k}
=M_{aero,k}.
$$

This is the rigorous meaning of "implementing $M_{aero}$ implicitly through the CoP". If the wing-body origin and the centre of mass do not coincide, then

$$
\ell_z=-c_{cp,k}-r_{com,A,k,z},
$$

so the $y$-axis moment about the centre of mass additionally contains

$$
-r_{com,A,k,z}F_{N,k},
$$

which is the moment that necessarily appears when the reference point is moved from the wing-body origin to the centre of mass; it is not a double count of the aerodynamic pitching moment.

The rotational damping is a pure moment independent of the centre-of-pressure lever arm:

$$
\boldsymbol\tau_{rd,A,k}
=
\begin{bmatrix}
0\\
M_{rd,k}\\
0
\end{bmatrix}.
$$

Hence the total local moment on one wing is

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

Using the fact that rotation matrices preserve the cross product,

$$
R(\boldsymbol a\times\boldsymbol b)
=(R\boldsymbol a)\times(R\boldsymbol b),
$$

the single-wing wrench can be rotated into the world frame:

$$
\boldsymbol F_{W,k}=R_{WA,k}\boldsymbol F_{A,k},
\qquad
\boldsymbol\tau_{W,k}=R_{WA,k}\boldsymbol\tau_{A,k}.
$$

This is exactly equivalent to the code's approach of first computing the world-frame CoP and then taking the cross product in the world frame.

### 7.4 From the left and right single wings to $F_{\mathrm{lift,total}}$

The world-frame resultant given by the pure aerodynamic functions for the left and right wings is a vector sum, not a sum of absolute values:

$$
\boldsymbol F_{\mathrm{pure,total}}
=\boldsymbol F_{W,L}+\boldsymbol F_{W,R}
=R_{WA,L}
\begin{bmatrix}F_{N,L}\\0\\0\end{bmatrix}
+R_{WA,R}
\begin{bmatrix}F_{N,R}\\0\\0\end{bmatrix}.
$$

If world $+z$ is defined as vertically upward, the true instantaneous total lift is

$$
F_{\mathrm{lift,total}}^{W}
=\boldsymbol e_z^{T}\boldsymbol F_{\mathrm{applied,total}}
=F_{\mathrm{applied,total},z}.
$$

Because of the STL normal convention the current adapter flips the forces of both wings, so

$$
\boldsymbol F_{\mathrm{applied},k}=-\boldsymbol F_{W,k},
$$

and hence

$$
\boldsymbol F_{\mathrm{applied,total}}
=-\left(\boldsymbol F_{W,L}+\boldsymbol F_{W,R}\right)
=-\boldsymbol F_{\mathrm{pure,total}}.
$$

This is exactly the value the adapter finally stores in the diagnostic quantity $F_{\mathrm{world}}$. Therefore, in the current model, the actual vertical aerodynamic force on the robot should be judged with

$$
\boxed{
F_{\mathrm{lift,total}}^{W}
=\text{diagnostics}[F_{\mathrm{world}}]_z
}.
$$

The scalar in the code

$$
\text{lift\_total}=|F_{N,L}|+|F_{N,R}|
$$

has a different meaning. It merely adds the magnitudes of the normal forces in the two wing aerodynamic frames. It does not contain the following information:

1. the inclination of the wing normal relative to the world vertical;
2. whether the horizontal components of the left and right wings cancel or add;
3. the global flip the adapter applies for the STL normal;
4. any moment produced by $M_{aero}$, the CoP and $M_{rd}$.

Let the world-frame vertical projection of each wing's actual normal be

$$
\gamma_k
=\boldsymbol e_z^{T}R_{WA,k}
\begin{bmatrix}1\\0\\0\end{bmatrix},
$$

then the actual total vertical aerodynamic force can be written explicitly as

$$
F_{\mathrm{lift,total}}^{W}
=-\gamma_LF_{N,L}-\gamma_RF_{N,R}.
$$

Only when both wing normals are aligned with the vertical and the sign conventions have been unified can it be approximated as

$$
F_{\mathrm{lift,total}}^{W}
\approx |F_{N,L}|+|F_{N,R}|.
$$

So the $\text{lift\_total}$ in the code is better used as an "aerodynamic load intensity" diagnostic; it cannot be treated directly as world-frame lift at arbitrary attitude.

An interface-level numerical check also confirmed this relationship: at every step the vector sum of the left and right `xfrc_applied[:, :3]` equals the adapter's final `diagnostics["F_world"]` exactly; the norm of the difference in the current check is 0.

### 7.5 How the total moment of the left and right wings is defined

The moments of the two wings cannot simply be added at arbitrary reference points. The moment the code writes for each wing is about that wing's centre of mass. To obtain the total aerodynamic moment of the whole robot about a robot reference point $O$, use

$$
\boldsymbol\tau_{O,\mathrm{total}}
=\sum_{k\in\{L,R\}}
\left[
\boldsymbol\tau_{\mathrm{applied},k}
+(\boldsymbol x_{com,W,k}-\boldsymbol x_{O,W})
\times\boldsymbol F_{\mathrm{applied},k}
\right].
$$

The current adapter keeps the moment computed by the pure function and flips only the force:

$$
\boldsymbol\tau_{\mathrm{applied},k}
=\boldsymbol\tau_{W,k},
\qquad
\boldsymbol F_{\mathrm{applied},k}
=-\boldsymbol F_{W,k}.
$$

Hence the actual wrench after the adapter must be understood directly in terms of the two quantities above; it can no longer be assumed equivalent to "the flipped force still acting at the original CoP". MuJoCo writes the left and right wrenches into the corresponding wing bodies and transmits them through the joint constraints as generalised forces on the whole robot.

---

## 8. From the local force to the MuJoCo world-frame wrench

Let $R_{WA}$ be the rotation matrix from the wing aerodynamic frame to the world frame. The pure function `world_wrench` computes

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

The final wrench ordering follows MuJoCo:

$$
\boldsymbol w=
[F_x,F_y,F_z,\tau_x,\tau_y,\tau_z]^T.
$$

With `read_only=True` all diagnostics are computed as usual, but the wrench returned for `xfrc_applied` is zero.

### 8.1 The adapter's actual normal correction

The STL wing normal is opposite to the oriented normal of the aerodynamic model. After `world_wrench` has finished, `MuJoCoAeroAdapter` performs

$$
\boldsymbol F_{applied}=-\boldsymbol F_{pure},\qquad
\boldsymbol\tau_{applied}=\boldsymbol\tau_{pure}.
$$

That is, only the first three force components are flipped; the `r×F` moment already computed from the original force is not flipped, nor is the damping moment. This behaviour was intentional for the current URDF/STL coordinate convention, but it means the wrench the adapter finally injects is no longer strictly equivalent to "applying the flipped force at the same CoP". `F_world` in the diagnostics is also flipped to match the force actually applied. [Translator's note: this force flip was removed on 2026-10-28; see [quasi_steady_model.md](quasi_steady_model.md).]

---

## 9. Left/right wing handling and symmetry

`aero_step` uses $s=+1$ for the left wing and $s=-1$ for the right wing. The adapter also applies `stroke_sign=-1` to the right-wing stroke angle and rate, to bring the mirrored URDF joint into the common aerodynamic sign convention. The sign of the feathering angle is set by `pitch_axis_y`; the current adapter passes $+1$.

In the ideal mirrored state the normal forces of the two wings may have opposite signs in their own local frames, but after the mirrored rotation and the adapter's normal correction the vertical components add while the lateral components and part of the moments cancel. The code diagnostic

$$
\texttt{lift\_total}=|F_{N,L}|+|F_{N,R}|
$$

is only the sum of the absolute local normal forces, not the strict world-frame vertical lift; the actual world resultant should be read from `diagnostics["F_world"]`.

---

## 10. Cycle-averaged hover model

Free flight with `fix=False` does not use the per-wing instantaneous model above but `AveragedHoverAero` (now `flapsim/aero/cycle_averaged.py`). At the same time the environment treats the robot dynamics as a single rigid body of mass $m=0.03\ \mathrm{kg}$; the sub-bodies retain only tiny numerical inertias, and the stroke-joint velocities do not take part in dynamic energy injection.

### 10.1 Mean lift

The model is written as

$$
L(f)=k_f f^2,
\qquad
k_f=\frac{0.14858349039827917}{30^2}
=1.6509276711\times10^{-4}\ \mathrm{N/Hz^2}.
$$

The body-frame thrust direction is set by the plate command:

$$
\boldsymbol n_B=
\frac{[-\delta_r,-\delta_p,1]^T}
{\sqrt{1+\delta_r^2+\delta_p^2}},
\qquad
\boldsymbol F_{thrust}^{W}=R_{WB}L\boldsymbol n_B.
$$

where $\delta_p=\texttt{pitch\_joint}$ and $\delta_r=\texttt{roll\_joint}$. For small angles

$$
\boldsymbol n_B\approx[-\delta_r,-\delta_p,1]^T,
$$

so tilting the plate produces horizontal thrust linearly.

The world-frame horizontal damping is

$$
\boldsymbol F_{xy}^{W}\leftarrow
\boldsymbol F_{xy}^{W}-k_v\boldsymbol v_{xy}^{W},
\qquad k_v=0.06\ \mathrm{N\,s/m}.
$$

### 10.2 Mean control moments

The identified moments scale with the square of frequency:

$$
\lambda_f=\left(\frac{f}{42}\right)^2.
$$

The body-frame moment is

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

where

$$
k_{roll}=\frac{0.01055}{20},\quad
k_{pitch}=\frac{0.01144}{20}\quad[\mathrm{N\,m/rad}],
$$

$$
D_\omega=\operatorname{diag}(2,2,1)\times10^{-4}
\quad[\mathrm{N\,m\,s/rad}].
$$

The URDF naming and the physical axes are crossed: `pitch_joint` mainly controls the roll moment and `roll_joint` mainly controls the pitch moment. The expression above therefore contains cross terms, rather than each named angle controlling the axis of the same name.

### 10.3 Consistency of the hover frequency

The weight of the 30 g equivalent mass is

$$
mg=0.03\times9.81=0.2943\ \mathrm N.
$$

From $k_f f_h^2=mg$,

$$
f_h=\sqrt{\frac{mg}{k_f}}=42.2213\ \mathrm{Hz}.
$$

This agrees with the nominal `hover_frequency=42 Hz` of `HoveringController` (now `flapsim/controllers/hover_pid.py`). At 42 Hz the model lift is $0.291224\ \mathrm N$, about $1.05\%$ below the weight; the remaining steady-state error is compensated by the frequency correction of the altitude PID.

---

## 11. Numerical validation

The following checks were run on the repository code and the `Wing.json` current at the time of writing; all units are SI.

### 11.1 Closed-form integral check of the pure functions

Choose the representative state

$$
\psi=0.37,\ \dot\psi=92,\ \theta=0.48,\ \dot\theta=-11,
$$

$$
\boldsymbol v_B=(0.3,-0.2,0.1),\quad
\boldsymbol\omega_B=(0.7,-0.5,0.4).
$$

Integrate the strip formula numerically with 200001 spanwise points and compare with the closed-form geometric-moment formula in `wing_aero`:

| Check | Closed form (code) | Numerical integration | Relative error |
|---|---:|---:|---:|
| $F_N$ | 0.0351425732167 N | 0.0351425732172 N | $1.36\times10^{-11}$ |
| $M_{aero}$ | -0.000406734027976 N·m | -0.000406734027981 N·m | $1.36\times10^{-11}$ |

Conclusion: the closed-form integration from the affine velocity expansion to the `rxx/z_cpx` geometric moments is implemented correctly; the error comes only from the numerical quadrature.

Other algebraic properties verified:

| Property | Result |
|---|---:|
| $r_{cp}=3R/4$ | 0.05250000000000003 m |
| $-M_{aero}/F_N=c_{cp}$ | both sides 0.0115738260106 m |
| $M_{rd}\dot\theta$ | $-1.14133\times10^{-5}\ \mathrm W<0$ |
| Force ratio at equal phase, 15 Hz to 30 Hz | 4.0 |
| Maximum residual of $C_N$ oddness (1001 points) | 0.0 |
| Maximum residual of $d_{cp}$ evenness (1001 points) | 0.0 |

### 11.2 Full fixed-base simulation frequency sweep

Set 20, 30 and 40 Hz respectively, run for 0.6 s, discard the first 0.3 s, and average the adapter's actual world-frame vertical force:

| Frequency | Mean $F_z$ | $F_z/f^2$ |
|---:|---:|---:|
| 20 Hz | 0.0661819985 N | $1.65455\times10^{-4}$ N/Hz² |
| 30 Hz | 0.1486352297 N | $1.65150\times10^{-4}$ N/Hz² |
| 40 Hz | 0.2639114107 N | $1.64945\times10^{-4}$ N/Hz² |

The 30 Hz result differs from the identification point $0.1485834904\ \mathrm N$ used by the averaged model by about $0.035\%$. $F_z/f^2$ is essentially constant over the three points, showing that the reduction $L=k_ff^2$ is a reasonable local approximation in the current passive-feathering steady state.

### 11.3 Free hover check

Start the free model from $z=0.5\ \mathrm m$ and simulate for 2.5 s:

| Metric | Result |
|---|---:|
| Final altitude | 0.479012 m |
| Altitude range | 0.470056–0.500000 m |
| Mean altitude over the last 0.5 s | 0.475627 m |
| Mean frequency over the last 0.5 s | 42.234855 Hz |

Conclusion: the averaged model stays bounded and converges near the nominal hover frequency; within the finite time there is still an altitude offset of about 2.4 cm, so this result demonstrates qualitative stability and order-of-magnitude consistency of model and controller, and should not be read as strict zero steady-state error or as validation over the full flight envelope.

---

## 12. What has been proven, model assumptions and implementation boundaries

### 12.1 What can be proven rigorously from the implementation or re-checked numerically

1. Rigid-body kinematics make the two local velocity components first-order functions of the spanwise coordinate;
2. the velocity squared is therefore a quadratic polynomial, and the spanwise integral can be completed exactly with precomputed geometric moments;
3. without clipping, the moment produced by the chordwise CoP is algebraically equivalent to `M_aero`;
4. the rotational damping always satisfies $M_{rd}\dot\theta\le0$;
5. the pure-flapping normal force at equal phase scales with $f^2$; the current full fixed-base sweep also supports this approximation;
6. the nominal hover frequency of the cycle-averaged model is consistent with balancing the 30 g weight.

### 12.2 What cannot be called a first-principles proof

1. $C_N(\alpha)$ and $d_{cp}(\alpha)$ are empirical fits whose accuracy needs experimental or higher-fidelity CFD data;
2. replacing the spanwise-varying angle of attack by the angle at a single representative point is a quasi-steady reduced-order assumption;
3. the fixed $r_{cp}=Rr_{33}/r_{22}$ is strictly valid only when the velocity squared is dominated by the $r_w^2$ term;
4. the current model does not explicitly include wake capture, added mass, wing–wing interaction, dynamic stall or ground effect;
5. the cycle-averaged model is identified only at a finite set of operating points and cannot automatically be extrapolated to arbitrary amplitude, frequency, attitude or high-speed flight.

### 12.3 Boundaries to watch when reading and modifying the code

1. Although `Wing.json` carries an `options` block, the current `MuJoCoAeroAdapter` constructs `AeroOptions()` directly and does not load the options from the JSON. Hence the `force_limit=2.0` in the JSON has no effect; the actual default is the 5.0 N in the Python dataclass.
2. `force_raw` is clipped but `M_aero` is not, and `chord_CoP` is computed from the unclipped `force_raw`. Once the force clip triggers, `r×F_clipped` no longer equals the original `M_aero`.
3. The adapter flips only the final force, not the already computed moment; the pure-function wrench and the wrench actually injected into MuJoCo must therefore be distinguished.
4. The angle of attack uses `atan(U_o/U_i)` and forces the correction angle to zero when $U_i\approx0$; near stroke reversal it lacks the full quadrant continuity of `atan2`.
5. `diagnostics["lift_total"]` is the sum of the absolute local normal forces of the two wings; hover should be judged with the world-frame `F_world[2]`, and `lift_total` must not be taken directly as vertical lift.
6. `fix=True/False` does not only fix or free the root; it also switches the aerodynamic model and the way the whole-robot mass/inertia is modelled. This model change must be taken into account when comparing the two simulations.

---

## 13. Summary

The core of the current per-wing aero can be summarised as

$$
\boxed{
\text{rigid-body kinematics}
\Rightarrow U_i,U_o\text{ are first-order in }r_w
\Rightarrow U^2\text{ is quadratic}
\Rightarrow \text{closed-form }F_N,M_{aero}\text{ from the wing geometric moments}
\Rightarrow \text{CoP + wrench written to MuJoCo}
}
$$

The cycle-averaged model further reduces the full periodic response above to

$$
\boxed{L=k_ff^2+\text{tilt direction}+\text{linear damping}+\text{identified control moments}}
$$

The closed-form integration, the centre-of-pressure identity, the dissipativity of the damping and the frequency-squared law have all been verified algebraically or numerically. The main uncertainty of the model lies not in the integration implementation but in the empirical aerodynamic coefficients, the single-point angle of attack, the fixed spanwise centre of pressure, and the range over which the fixed-base identification extrapolates to free flight.
