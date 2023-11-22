# Poisson FDM Solver

A simple framework for setting up electrostatic problems and solving them using the finite difference method.

# Basics
<div align="center">
  <img src="https://user-images.githubusercontent.com/28433296/144881741-0bc4e902-7351-4d13-95db-21d26594d297.png" width="60%"></img>
</div>

In order to set up an electrostatic problem, the grid values of ρ, ε and the boundary conditions can be set.

`drawing_ops` contains functions for drawing basic geometric objects onto a grid. When an input argument is passed to the function, the specified object is drawn onto it.
```python
import drawing_ops as ops
N = 256
U = np.zeros((N, N))
U = ops.plate_capacitor(U, center=(N/2, N/2),
                           length=N/2,
                           distance=N/8,
                           rotation=90)

```
Alternatively, if only geometric properties are specified, the function returns another function that can draw the object onto an input.
```python
N = 256
capacitor = ops.plate_capacitor(center=(N/2, N/2), 
                                length=N/2, 
                                distance=N/8, 
                                rotation=90)
U = capacitor(np.zeros((N, N)))
```
The function `solve_poisson` receives the grid values as input and returns the finite-difference solution of φ.
```python
phi = solve_poisson(U)
```

<div align="center">
  <img src="https://github.com/nipponjo/poisson-fdm/blob/main/docs/poisson-ops.png" width="75%"></img>
</div>



## (1) Plate capacitor
```python
capacitor = ops.plate_capacitor(center=(N/2, N/2), 
                                length=N/2, 
                                distance=N/8, 
                                rotation=90)
U = capacitor(np.zeros((N, N)))
phi = solve_poisson(U)
plot_images([U, phi], ['U', 'φ'], cmap='afmhot')
```
<div align="center">
  <img src="https://github.com/nipponjo/poisson-fdm/blob/main/docs/case1_u_phi.png" width="65%"></img>
</div>

```python
Ey, Ex = np.gradient(phi)
Ex, Ey = -Ex, -Ey
E = np.sqrt(Ex**2 + Ey**2)
```

<div align="center">
  <img src="https://github.com/nipponjo/poisson-fdm/blob/main/docs/case1_E.png" width="50%"></img>
</div>


## (2) Dielectric block between two plates
```python
capacitor = ops.plate_capacitor(center=(N/2-1, N/2-1), 
                                length=N/1.5, 
                                distance=N/8, 
                                rotation=90, 
                                values=[-10, 10])
U = capacitor(np.zeros((N, N)))

Eps = np.ones((N, N))
Eps = ops.rectangle(Eps, center=(N/2-1, N/2-1), 
                         wh=(N//16, N//16), 
                         filled=True, 
                         value=5)

phi = solve_poisson(U=U, Eps=Eps)
```

<div align="center">
  <img src="https://github.com/nipponjo/poisson-fdm/blob/main/docs/case2_rho_eps_phi.png" width="85%"></img>
</div>

<div align="center">
  <img src="https://github.com/nipponjo/poisson-fdm/blob/main/docs/case2_E.png" width="50%"></img>
</div>


## (3) Two oppositely charged circles
```python
charges = ops.composition(
    ops.circle(center=(N/2-N/5, N/2), radius=N/16, filled=True, value=1),
    ops.circle(center=(N/2+N/5, N/2), radius=N/16, filled=True, value=-1)
)
Rho = charges(np.zeros((N, N)))

phi = solve_poisson(U=np.zeros((N, N)), Rho=Rho)
```

<div align="center">
  <img src="https://github.com/nipponjo/poisson-fdm/blob/main/docs/case3_rho_phi.png" width="65%"></img>
</div>

<div align="center">
  <img src="https://github.com/nipponjo/poisson-fdm/blob/main/docs/case3_E.png" width="50%"></img>
</div>


## (4) Charged circle in front of a plate
```python
Rho = ops.circle(np.zeros((N, N)), center=(N/2+N/8, N/2), 
                                   radius=N/32, 
                                   filled=True)
U = ops.line(np.zeros((N, N)), center=(N/2-N/8, N/2), 
                               length=N/2, 
                               rotation=90, 
                               value=1)

phi = solve_poisson(U=U, Rho=Rho)
```

<div align="center">
  <img src="https://github.com/nipponjo/poisson-fdm/blob/main/docs/case4_u_rho_phi.png" width="85%"></img>
</div>

<div align="center">
  <img src="https://github.com/nipponjo/poisson-fdm/blob/main/docs/case4_E.png" width="50%"></img>
</div>


## (5) Charged circle in front of a dielectric half-space

```python
Eps = np.ones((N, N))
Eps[:,:N//2-N//8] = 3
Rho = ops.circle(np.zeros((N, N)), center=(N/2+N/16, N/2), 
                                   radius=N/32, 
                                   filled=True)

phi = solve_poisson(Eps=Eps, Rho=Rho)
```

<div align="center">
  <img src="https://github.com/nipponjo/poisson-fdm/blob/main/docs/case5_eps_rho_phi.png" width="85%"></img>
</div>

<div align="center">
  <img src="https://github.com/nipponjo/poisson-fdm/blob/main/docs/case5_E.png" width="50%"></img>
</div>


<br>
<br>

# Make videos

```python
frames = make_frames(ts, U, stars)
frames_c = colorize_frames(frames['E'], num_levels=30)

plt.figure(figsize=(10, 10))
plot_frames(frames_c, num=10, nrow=5)

make_video('data/E_stars_512_fps20_l30_m15.mp4', frames_c, fps=20, fourcc='h264')
```

## Examples

```python
distance = lambda t: N/5 + N/10*np.sin(2*np.pi*t)
cap = lambda t: ops.plate_capacitor(center=(N/2, N/2), 
                                    length=N/2, 
                                    distance=distance(t), 
                                    values=[-10, 10], 
                                    plate_width=N//50,
                                    rotation=90)
```

https://github.com/nipponjo/poisson-fdm/assets/28433296/af6ac4ae-0a9c-43c5-b08f-c33e3a777552



```python
theta = lambda t: 360*t
arcs2 = lambda t: ops.composition(
  ops.arc(center=(N/2, N/2), radius=N/3, start=0, end=250, rotation=theta(t), value=10),
  ops.arc(center=(N/2, N/2), radius=N/3-N/10, start=0, end=250, rotation=45-theta(t), value=-10),
  ops.arc(center=(N/2, N/2), radius=N/3-2*N/10, start=0, end=250, rotation=90+theta(t), value=10),
  ops.arc(center=(N/2, N/2), radius=N/3-3*N/10, start=0, end=250, rotation=135-theta(t), value=-10)
)
```

https://github.com/nipponjo/poisson-fdm/assets/28433296/e1766860-7ac1-4de3-9cb4-5b58cc4aa0d6




```python
voltage = lambda t, phase: 10*np.sin(2*np.pi*t + phase/180*np.pi)                                        
stars = lambda t: ops.composition(
  ops.star(center=(N/2-N/5, N/2+N/5), N_angles=6, ro=N/10, value=voltage(t, phase=0)),                   
  ops.star(center=(N/2+N/5, N/2+N/5), N_angles=6, ro=N/10, value=voltage(t, phase=120)),                        
  ops.star(center=(N/2, N/2-N/5), N_angles=6, ro=N/10, value=voltage(t, phase=-120)))
```

https://github.com/nipponjo/poisson-fdm/assets/28433296/1b719411-49be-405a-8e8e-321fd35378a5



# References
[Nagel 2012 - Solving the Generalized Poisson Equation Using the
Finite-Difference Method (FDM)](https://my.ece.utah.edu/~ece6340/LECTURES/Feb1/Nagel%202012%20-%20Solving%20the%20Generalized%20Poisson%20Equation%20using%20FDM.pdf)


