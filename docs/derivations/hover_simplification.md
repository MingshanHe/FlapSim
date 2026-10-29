# Quasi-steady aerodynamic derivation for a flapping-wing robot in ideal hover

*Translated from the Chinese original (`docs/zh/derivations/hover_simplification.md`). It documents the implementation as it stood before 2026-10-28; the pitching-moment sign and the removed force flip are described in [quasi_steady_model.md](quasi_steady_model.md).*

## Abstract

This document gives a simplified derivation of the flapping-wing aerodynamic model in FlapSim, suitable for the main text of a paper. The object of study is restricted to symmetric hover: the instantaneous translational and angular velocities of the body are zero, the stroke-plane angle and the wing deviation angle are zero, and the main wing motion is described by the stroke angle and the passive feathering angle. Under these conditions the in-plane velocity retains only the stroke velocity, which is proportional to the spanwise coordinate, so the first-order and constant terms of the velocity-squared polynomial both vanish, i.e.

$$
a_{U1}=0,\qquad a_{U0}=0.
$$

On this basis the document derives in turn the single-wing normal force, the aerodynamic pitching moment, the centre of pressure, the rotational damping moment, the single-wing wrench in the world frame, and the total lift of the two wings. Finally it states the condition under which the cycle-averaged lift is proportional to the square of the flapping frequency.

---

## 1. Model assumptions

To obtain clear closed-form expressions usable in a paper, the following assumptions are adopted.

1. The body is approximately stationary at the instant of the aerodynamic computation:

   $$
   \boldsymbol v_B=\boldsymbol 0,\qquad
   \boldsymbol\omega_B=\boldsymbol 0.
   $$

2. The stroke-plane angle and the wing deviation angle are both zero:

   $$
   \Phi=0,\quad\dot\Phi=0,\qquad
   \phi=0,\quad\dot\phi=0.
   $$

3. The left and right wings have identical geometry and move in mirror symmetry.

4. The instantaneous angle of attack along the span is approximated by the angle of attack at the representative centre of pressure, i.e. the whole wing uses the same normal-force coefficient at any instant.

5. The aerodynamic force follows a quasi-steady blade-element model; wake capture, added mass, wing–wing interaction and dynamic stall are not considered explicitly.

6. The derivation uses the raw, unclipped aerodynamic forces and takes `lift_scale = moment_scale = 1`. The centre-of-pressure relation $c_{cp}=-M_{aero}/F_N$ therefore holds exactly.

It must be stressed that $a_{U1}=a_{U0}=0$ is not an identity of the full six-degree-of-freedom model. It holds only under the ideal-hover assumptions above. When the body translates or rotates, or the stroke plane has an offset motion, these two terms are in general non-zero.

---

## 2. Coordinate frames and notation

### 2.1 Coordinate frames

Before reading the formulas that follow, one must distinguish "the components of a vector" from "the actual direction of the vector in the world". This document uses three levels of coordinate frames.

| Symbol | Frame | Definition |
|---|---|---|
| $\{W\}$ | world frame | $+z_W$ vertically upward |
| $\{B\}$ | body frame | fixed to the robot body |
| $\{A_k\}$ | aerodynamic frame of wing $k$ | $+x_A$ is the wing normal, $+y_A$ the spanwise/feathering axis, $-z_A$ points towards the chordwise CoP |

The relationship between the frames is shown in Figure 1.

![Relationship between the world frame, the body frame and the wing aerodynamic frames](../assets/aero_coordinate_frames.svg)

**Figure 1  Relationship between the world frame, the body frame and the wing aerodynamic frames.**

Here $k\in\{L,R\}$ denotes the left and right wing respectively. Define

$$
R_{WB}\in SO(3)
$$

as the rotation matrix from the body frame to the world frame, and

$$
R_{WA,k}\in SO(3)
$$

as the rotation matrix from the aerodynamic frame of wing $k$ to the world frame. Any wing-local vector $\boldsymbol q_{A,k}$ transforms to the world frame as

$$
\boldsymbol q_{W,k}=R_{WA,k}\boldsymbol q_{A,k}.
$$

If $R_{BA,k}$ denotes the rotation matrix from the aerodynamic frame to the body frame, the three satisfy

$$
\boxed{
R_{WA,k}=R_{WB}R_{BA,k}
}.
$$

The coordinate transformation can therefore be understood in two steps:

$$
\{A_k\}
\xrightarrow{\ R_{BA,k}\ }
\{B\}
\xrightarrow{\ R_{WB}\ }
\{W\}.
$$

In the program, $R_{WA,k}$ is determined jointly by the MuJoCo wing-body orientation and the fixed alignment matrix from the URDF/STL frame to the aerodynamic frame. The derivation below only needs the final $R_{WA,k}$.

### 2.2 Why $F_N$ is written in the first component

The column vector

$$
\boldsymbol F_{A,k}
=
\begin{bmatrix}
F_{N,k}\\
0\\
0
\end{bmatrix}
$$

represents "the three components of the force in the aerodynamic frame $\{A_k\}$", in the fixed order

$$
\begin{bmatrix}
x_A\text{ component}\\
y_A\text{ component}\\
z_A\text{ component}
\end{bmatrix}.
$$

So $F_N$ sits in the first position not because it points along the first world axis, nor because it is the "first row" of some matrix. There is only one reason:

$$
\boxed{
\text{this model defines the wing surface normal as the }+x_A\text{ axis of the aerodynamic frame}
}.
$$

The normal force acts only along $+x_A$, so its coordinates in $\{A_k\}$ are necessarily

$$
\boldsymbol F_{A,k}
=F_{N,k}\boldsymbol e_{x_A}
=F_{N,k}
\begin{bmatrix}
1\\0\\0
\end{bmatrix}.
$$

$F_N$ is a signed scalar: $F_N>0$ means the force points along $+x_A$, $F_N<0$ means it points along $-x_A$. There is therefore no need to define separate force vectors for the two directions.

Only after this local vector is transformed to the world frame does one obtain the actual spatial direction:

$$
\boldsymbol F_{W,k}
=R_{WA,k}\boldsymbol F_{A,k}.
$$

Writing the rotation matrix by columns,

$$
R_{WA,k}
=
\begin{bmatrix}
\vert&\vert&\vert\\
\boldsymbol e_{x_A}^{W}
&\boldsymbol e_{y_A}^{W}
&\boldsymbol e_{z_A}^{W}\\
\vert&\vert&\vert
\end{bmatrix},
$$

where $\boldsymbol e_{x_A}^{W}$ denotes "the wing-local $+x_A$ unit axis expressed in the world frame". Then

$$
\begin{aligned}
\boldsymbol F_{W,k}
&=
\begin{bmatrix}
\vert&\vert&\vert\\
\boldsymbol e_{x_A}^{W}
&\boldsymbol e_{y_A}^{W}
&\boldsymbol e_{z_A}^{W}\\
\vert&\vert&\vert
\end{bmatrix}
\begin{bmatrix}
F_{N,k}\\0\\0
\end{bmatrix}\\
&=F_{N,k}\boldsymbol e_{x_A}^{W}.
\end{aligned}
$$

This shows that multiplying by $[F_N,0,0]^T$ simply selects the first column of $R_{WA,k}$ and scales it by $F_N$. Hence:

- in the wing aerodynamic frame, $F_N$ is always the first component;
- in the world frame, $F_N$ can have all three components $x_W$, $y_W$ and $z_W$ at once;
- a change of wing attitude changes the world direction of the force through $R_{WA,k}$, without changing how the local force vector is written.

For example, if at some instant

$$
\boldsymbol e_{x_A}^{W}
=
\begin{bmatrix}
0\\0\\-1
\end{bmatrix},
$$

then

$$
\boldsymbol F_{W,k}
=
\begin{bmatrix}
0\\0\\-F_{N,k}
\end{bmatrix}.
$$

Although $F_N$ is the first component in local coordinates, in world coordinates it lies entirely along $-z_W$. The current adapter subsequently also flips the sign of the force according to the STL normal convention, so the force actually applied is $-\boldsymbol F_{W,k}$.

### 2.3 Left/right wings and the direction of the centre of pressure

This document uses one common aerodynamic frame convention for both wings and handles the mirror relationship with

$$
s_L=+1,\qquad s_R=-1
$$

The local coordinates of the centre of pressure are therefore

$$
\boxed{
\boldsymbol r_{cp,A,k}
=
\begin{bmatrix}
0\\
s_kr_{cp}\\
-c_{cp,k}
\end{bmatrix}
}.
$$

The three components mean, in order:

1. the centre of pressure lies in the wing plane, so its normal coordinate is $0$;
2. the spanwise positions of the left and right wings are $+r_{cp}$ and $-r_{cp}$ respectively;
3. the chordwise centre of pressure lies along the $-z_A$ direction defined here, hence it is written $-c_{cp}$.

### 2.4 Main symbols

| Symbol | Meaning | Unit |
|---|---|---|
| $R$ | single-wing span | m |
| $r$ | spanwise coordinate, $0\le r\le R$ | m |
| $c(r)$ | local chord | m |
| $\bar c$ | mean chord | m |
| $\rho$ | air density | kg/m³ |
| $\psi$, $\dot\psi$ | stroke angle and stroke rate | rad, rad/s |
| $\theta$, $\dot\theta$ | feathering angle and feathering rate | rad, rad/s |
| $U_i$ | in-plane relative velocity | m/s |
| $U_o$ | out-of-plane relative velocity | m/s |
| $\alpha$ | instantaneous angle of attack | rad |
| $C_N$ | normal-force coefficient | — |
| $F_N$ | single-wing local normal force | N |
| $M_{aero}$ | aerodynamic pitching moment from the pressure distribution | N·m |
| $M_{rd}$ | feathering rotational damping moment | N·m |
| $r_{cp}$ | spanwise centre-of-pressure position | m |
| $c_{cp}$ | chordwise centre-of-pressure distance | m |
| $s_k$ | left/right wing sign, $s_L=+1,\ s_R=-1$ | — |

The current rectangular-wing parameters are

$$
R=0.07\ \mathrm m,\qquad
\bar c=0.02\ \mathrm m,\qquad
\rho=1.225\ \mathrm{kg/m^3}.
$$

---

## 3. Wing geometric moments

Define the dimensionless spanwise coordinate and the dimensionless chord:

$$
\hat r=\frac{r}{R},\qquad
\hat c(\hat r)=\frac{c(R\hat r)}{\bar c}.
$$

The mean chord satisfies

$$
\bar c=\frac{1}{R}\int_0^R c(r)\,dr.
$$

The radial geometric moments needed for the normal-force integral are defined as

$$
r_{kk}=\int_0^1\hat c(\hat r)\hat r^k\,d\hat r.
$$

This document mainly uses

$$
r_{22}=\int_0^1\hat c(\hat r)\hat r^2\,d\hat r,
\qquad
r_{33}=\int_0^1\hat c(\hat r)\hat r^3\,d\hat r.
$$

The aerodynamic pitching moment also needs the chord-squared weighted moment:

$$
z_{cp,2}=
\int_0^1\hat c^2(\hat r)\hat r^2\,d\hat r.
$$

For a rectangular wing $\hat c(\hat r)=1$, so

$$
r_{22}=\frac13,\qquad
r_{33}=\frac14,\qquad
z_{cp,2}=\frac13.
$$

---

## 4. Local velocity in ideal hover

### 4.1 The general velocity polynomial

The full model writes the in-plane and out-of-plane velocities as first-order functions of the spanwise coordinate:

$$
U_i(r)=U_{i1}r+U_{i0},
\qquad
U_o(r)=U_{o1}r+U_{o0}.
$$

The local velocity squared can therefore be written as

$$
U^2(r)=U_i^2(r)+U_o^2(r)
=a_{U2}r^2+a_{U1}r+a_{U0},
$$

where

$$
\begin{aligned}
a_{U2}&=U_{i1}^2+U_{o1}^2,\\
a_{U1}&=2(U_{i1}U_{i0}+U_{o1}U_{o0}),\\
a_{U0}&=U_{i0}^2+U_{o0}^2.
\end{aligned}
$$

### 4.2 Simplification under the hover assumptions

When the body velocity, the body angular velocity, the stroke-plane motion and the deviation motion are all zero, the main velocity at any point on the wing comes from the stroke alone:

$$
U_i(r)=\dot\psi r,\qquad
U_o(r)=0.
$$

Comparing with the general form gives

$$
U_{i1}=\dot\psi,\qquad
U_{i0}=0,\qquad
U_{o1}=0,\qquad
U_{o0}=0.
$$

Therefore

$$
\boxed{
a_{U2}=\dot\psi^2,\qquad
a_{U1}=0,\qquad
a_{U0}=0
}.
$$

The velocity squared finally degenerates to

$$
\boxed{
U^2(r)=\dot\psi^2r^2
}.
$$

This expression is the kinematic basis of all the simplified formulas that follow.

---

## 5. Angle of attack and aerodynamic coefficients

### 5.1 Spanwise representative point

In pure stroke motion the local dynamic pressure is proportional to $r^2$, so the spanwise point of application of the resultant force is

$$
r_{cp}
=\frac{\displaystyle\int_0^R r\,c(r)r^2\,dr}
{\displaystyle\int_0^R c(r)r^2\,dr}
=R\frac{r_{33}}{r_{22}}.
$$

For a rectangular wing,

$$
\boxed{
r_{cp}=\frac34R=0.0525\ \mathrm m
}.
$$

### 5.2 Instantaneous angle of attack

At the spanwise representative point,

$$
U_i(r_{cp})=\dot\psi r_{cp},\qquad
U_o(r_{cp})=0.
$$

The inflow deflection angle is therefore zero and the angle of attack simplifies to

$$
\boxed{
\alpha
=\theta+\operatorname{sgn}(\dot\psi)\frac{\pi}{2}
}.
$$

The normal-force coefficient used by the program is

$$
C_N(\alpha)
=1.8\sin(2\alpha)\cos\alpha
+1.95\sin\alpha
-1.5\cos(2\alpha)\sin\alpha.
$$

The chordwise centre-of-pressure fraction is

$$
d_{cp}(\alpha)
=0.46-0.332\cos\alpha
-0.037\cos(3\alpha)
-0.013\cos(5\alpha).
$$

The two relations above belong to the empirical aerodynamic model. This document adopts them directly and does not derive them from first principles of fluid mechanics. For a formal submission, the original experimental or modelling source of these empirical coefficients should be added here; this document does not fabricate references that have not been verified.

---

## 6. Single-wing normal force

The quasi-steady normal force on a spanwise element $dr$ is

$$
dF_N
=\frac12\rho C_N(\alpha)c(r)U^2(r)\,dr.
$$

Substituting

$$
U^2(r)=\dot\psi^2r^2
$$

gives

$$
dF_N
=\frac12\rho C_N(\alpha)c(r)\dot\psi^2r^2\,dr.
$$

Integrating along the span:

$$
\begin{aligned}
F_N
&=\frac12\rho C_N(\alpha)\dot\psi^2
\int_0^R c(r)r^2\,dr\\
&=\frac12\rho\bar c C_N(\alpha)
\dot\psi^2R^3r_{22}.
\end{aligned}
$$

Hence the single-wing normal force in ideal hover is

$$
\boxed{
F_N
=\frac12\rho\bar c R^3r_{22}
C_N(\alpha)\dot\psi^2
}.
$$

For the current rectangular wing $r_{22}=1/3$, so

$$
F_N
=\frac16\rho\bar c R^3
C_N(\alpha)\dot\psi^2.
$$

---

## 7. Aerodynamic pitching moment and chordwise centre of pressure

### 7.1 Aerodynamic pitching moment

Let the distance from the local centre of pressure to the reference edge be $d_{cp}(\alpha)c(r)$. The pitching moment produced by the elemental normal force is

$$
dM_{aero}
=-d_{cp}(\alpha)c(r)\,dF_N.
$$

Substituting the elemental normal force and integrating along the span:

$$
\begin{aligned}
M_{aero}
&=-\frac12\rho C_N(\alpha)d_{cp}(\alpha)\dot\psi^2
\int_0^R c^2(r)r^2\,dr\\
&=-\frac12\rho\bar c^2R^3z_{cp,2}
C_N(\alpha)d_{cp}(\alpha)\dot\psi^2.
\end{aligned}
$$

Hence

$$
\boxed{
M_{aero}
=-\frac12\rho\bar c^2R^3z_{cp,2}
C_N(\alpha)d_{cp}(\alpha)\dot\psi^2
}.
$$

### 7.2 Equivalent chordwise centre of pressure

The model does not apply $M_{aero}$ a second time as an independent moment; instead it converts it into the chordwise point of application of the normal force:

$$
c_{cp}=-\frac{M_{aero}}{F_N}.
$$

Substituting the $F_N$ and $M_{aero}$ obtained above:

$$
\boxed{
c_{cp}
=\bar c\,d_{cp}(\alpha)
\frac{z_{cp,2}}{r_{22}}
}.
$$

For a rectangular wing,

$$
z_{cp,2}=r_{22}=\frac13,
$$

so one further obtains

$$
\boxed{
c_{cp}=\bar c\,d_{cp}(\alpha)
}.
$$

This result shows that, for a rectangular wing under the ideal-hover assumptions, the chordwise centre of pressure equals exactly the local chord times the empirical centre-of-pressure fraction.

---

## 8. Feathering rotational damping

The passive feathering motion produces a damping moment that opposes the feathering rate:

$$
\boxed{
M_{rd}
=-\frac58\rho R\bar c^4z_{rd}
|\dot\theta|\dot\theta
}.
$$

For the current rectangular wing $z_{rd}=1$. Since

$$
M_{rd}\dot\theta
=-\frac58\rho R\bar c^4z_{rd}
|\dot\theta|\dot\theta^2
\le0,
$$

the rotational damping always dissipates mechanical energy.

$M_{aero}$ and $M_{rd}$ play different physical roles:

- $M_{aero}$ is implemented implicitly through the chordwise centre of pressure $c_{cp}$;
- $M_{rd}$ is applied as an independent pure moment about the feathering axis.

---

## 9. Single-wing aerodynamic force in the world frame

### 9.1 Force and centre of pressure in the aerodynamic frame

The normal force of wing $k$ in its aerodynamic frame is

$$
\boldsymbol F_{A,k}
=
\begin{bmatrix}
F_{N,k}\\
0\\
0
\end{bmatrix}.
$$

The position of the centre of pressure relative to the wing-body origin is

$$
\boldsymbol r_{cp,A,k}
=
\begin{bmatrix}
0\\
s_kr_{cp}\\
-c_{cp,k}
\end{bmatrix}.
$$

Its world coordinates are

$$
\boldsymbol x_{cp,W,k}
=\boldsymbol x_{body,W,k}
+R_{WA,k}\boldsymbol r_{cp,A,k}.
$$

### 9.2 Force in the world frame

The single-wing normal force transformed to the world frame is

$$
\boxed{
\boldsymbol F_{W,k}
=R_{WA,k}
\begin{bmatrix}
F_{N,k}\\
0\\
0
\end{bmatrix}
}.
$$

### 9.3 Moment about the wing centre of mass

The world-frame lever arm from the wing centre of mass to the centre of pressure is

$$
\boldsymbol\ell_{W,k}
=\boldsymbol x_{cp,W,k}-\boldsymbol x_{com,W,k}.
$$

The moment produced by the normal force at the centre of pressure is

$$
\boldsymbol\tau_{F,W,k}
=\boldsymbol\ell_{W,k}\times\boldsymbol F_{W,k}.
$$

The rotational damping moment transformed to the world frame is

$$
\boldsymbol\tau_{rd,W,k}
=R_{WA,k}
\begin{bmatrix}
0\\
M_{rd,k}\\
0
\end{bmatrix}.
$$

Hence the total single-wing moment is

$$
\boxed{
\boldsymbol\tau_{W,k}
=\boldsymbol\tau_{F,W,k}
+\boldsymbol\tau_{rd,W,k}
}.
$$

The complete single-wing wrench is

$$
\boxed{
\boldsymbol w_k
=
\begin{bmatrix}
\boldsymbol F_{W,k}\\
\boldsymbol\tau_{W,k}
\end{bmatrix}
}.
$$

---

## 10. Total lift of the two wings

### 10.1 Normal correction

The current STL wing normal is opposite to the oriented normal of the aerodynamic model, so the adapter applies one sign correction to the world-frame aerodynamic force:

$$
\boldsymbol F_{applied,k}=-\boldsymbol F_{W,k}.
$$

This sign correction belongs to the current URDF/STL coordinate convention, not to the quasi-steady aerodynamic theory itself. [Translator's note: this force flip was removed on 2026-10-28; see [quasi_steady_model.md](quasi_steady_model.md).]

### 10.2 World-frame resultant of the two wings

The total aerodynamic force applied to the robot by the left and right wings is

$$
\boxed{
\boldsymbol F_{total}
=\boldsymbol F_{applied,L}
+\boldsymbol F_{applied,R}
}.
$$

Substituting the single-wing expressions:

$$
\boldsymbol F_{total}
=-
R_{WA,L}
\begin{bmatrix}
F_{N,L}\\0\\0
\end{bmatrix}
-
R_{WA,R}
\begin{bmatrix}
F_{N,R}\\0\\0
\end{bmatrix}.
$$

Define the world-frame vertical unit vector

$$
\boldsymbol e_z=
\begin{bmatrix}
0\\0\\1
\end{bmatrix},
$$

then the instantaneous total lift is

$$
\boxed{
F_{\mathrm{lift}}
=\boldsymbol e_z^T\boldsymbol F_{total}
}.
$$

Further defining the projection of the normal of wing $k$ onto the world vertical,

$$
\gamma_k
=\boldsymbol e_z^TR_{WA,k}
\begin{bmatrix}
1\\0\\0
\end{bmatrix},
$$

one obtains

$$
\boxed{
F_{\mathrm{lift}}
=-\gamma_LF_{N,L}
-\gamma_RF_{N,R}
}.
$$

In the fully symmetric state,

$$
F_{N,L}=F_{N,R}=F_N,\qquad
\gamma_L=\gamma_R=\gamma,
$$

so

$$
\boxed{
F_{\mathrm{lift}}=-2\gamma F_N
}.
$$

If the current coordinate convention makes the aerodynamic normal of the wings point along world $-z_W$, then $\gamma=-1$ and

$$
\boxed{
F_{\mathrm{lift}}=2F_N
}.
$$

This step completes the coordinate transformation from the local single-wing normal force to the total lift of the two wings.

---

## 11. Cycle-averaged lift and the frequency-squared law

Let the stroke angle be

$$
\psi(t)=A\sin(2\pi ft),
$$

then

$$
\dot\psi(t)=2\pi fA\cos(2\pi ft).
$$

Substituting into the single-wing normal force:

$$
F_N(t)
=\frac12\rho\bar cR^3r_{22}
C_N[\alpha(t)]
(2\pi fA)^2\cos^2(2\pi ft).
$$

The cycle-averaged lift of the two wings is defined as

$$
\bar F_{\mathrm{lift}}
=\frac{1}{T}
\int_0^T
F_{\mathrm{lift}}(t)\,dt,
\qquad T=\frac1f.
$$

Introducing the dimensionless phase

$$
\xi=2\pi ft,
$$

gives

$$
\bar F_{\mathrm{lift}}
=K_f f^2,
$$

where

$$
\boxed{
K_f
=-\frac{1}{2\pi}
\int_0^{2\pi}
\sum_{k\in\{L,R\}}
\gamma_k(\xi)
\frac12\rho\bar cR^3r_{22}
C_N[\alpha_k(\xi)]
(2\pi A)^2\cos^2\xi
\,d\xi
}.
$$

Hence, as long as the following dimensionless periodic quantities do not change significantly with frequency,

$$
\theta_k(\xi),\qquad
\gamma_k(\xi),\qquad
C_N[\alpha_k(\xi)],
$$

the cycle-averaged lift satisfies

$$
\boxed{
\bar F_{\mathrm{lift}}\propto f^2
}.
$$

FlapSim's cycle-averaged hover model uses

$$
\bar F_{\mathrm{lift}}=k_f f^2,
$$

where $k_f$ is identified from the full fixed-base flapping simulation rather than prescribed directly by the constant-coefficient approximation above.

---

## 12. Summary of the derived results

Under ideal-hover conditions the aerodynamic derivation can be summarised in the following seven steps.

1. Local velocity:

   $$
   U_i(r)=\dot\psi r,\qquad U_o(r)=0.
   $$

2. Velocity-squared coefficients:

   $$
   a_{U2}=\dot\psi^2,\qquad
   a_{U1}=a_{U0}=0.
   $$

3. Single-wing normal force:

   $$
   F_N
   =\frac12\rho\bar cR^3r_{22}
   C_N(\alpha)\dot\psi^2.
   $$

4. Aerodynamic pitching moment:

   $$
   M_{aero}
   =-\frac12\rho\bar c^2R^3z_{cp,2}
   C_N(\alpha)d_{cp}(\alpha)\dot\psi^2.
   $$

5. Centre of pressure:

   $$
   r_{cp}=R\frac{r_{33}}{r_{22}},
   \qquad
   c_{cp}
   =\bar c\,d_{cp}\frac{z_{cp,2}}{r_{22}}.
   $$

6. Single-wing force in the world frame:

   $$
   \boldsymbol F_{applied,k}
   =-R_{WA,k}
   \begin{bmatrix}F_{N,k}\\0\\0\end{bmatrix}.
   $$

7. Total lift of the two wings:

   $$
   F_{\mathrm{lift}}
   =-\gamma_LF_{N,L}-\gamma_RF_{N,R}.
   $$

In the ideal symmetric state with vertical wing normals:

$$
\boxed{
F_{\mathrm{lift}}=2F_N
}.
$$

---

## 13. Range of validity

The formulas in this document are intended for the ideal-hover theoretical derivation in the paper. In the following situations the full velocity model should be restored, and $a_{U1}$ and $a_{U0}$ are in general no longer zero:

1. the body has a significant translational velocity;
2. the body has roll, pitch or yaw angular velocity;
3. the stroke-plane angle or the wing deviation angle is non-zero;
4. the shoulder-offset velocity is not negligible relative to the local inflow;
5. the study concerns rapid manoeuvres, strongly coupled attitude changes or large-range free flight.

The simplified model in this document is therefore suited to a "hover-state aerodynamic modelling" section of the paper, while the full FlapSim implementation is suited to numerical simulation and validation in non-ideal states.
