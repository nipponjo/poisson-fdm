import sys
# import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from solver import solve_poisson

# ----------------------------------------------------------------------------

###########################
######### GENERAL #########
###########################

def progbar(iterable, length=30, symbol='='):
    """Wrapper generator function for an iterable. 
       Prints a progressbar when yielding an item. \\
       Args:
          iterable: an object supporting iteration
          length: length of the progressbar
    """
    n = len(iterable)
    for i, item in enumerate(iterable):
        steps = int(length/n*(i+1))
        sys.stdout.write('\r')        
        sys.stdout.write(f"[{symbol*steps:{length}}] {(100/n*(i+1)):.1f}%")
        sys.stdout.flush()
        yield item

# ----------------------------------------------------------------------------

###########################
###### VISUALIZATION ######
###########################

def plot_images(images, labels=[], nrow=3, cmap='viridis'):
    n_images = len(images)
    for i, image in enumerate(images):
        plt.subplot(-(n_images//-nrow), nrow, i+1)
        plt.imshow(image, cmap=cmap)
        if labels:
            plt.title(labels[i], fontdict={'fontsize': 22})
        plt.xlabel('x', fontdict={'fontsize': 18})
        plt.ylabel('y', fontdict={'fontsize': 18})
        plt.colorbar(shrink=0.35)

# ----------------------------------------------------------------------------

def make_colored_plot(A, num_levels=None, s=None, mul=1., cmap='afmhot'):  
    s = A.max()-A.min() if s is None else s
    
    cms = cm.get_cmap(cmap) 
    if num_levels is None:
        return cms(mul*(A - A.min()) / s)    

    Ac = np.floor((A - A.min())/s*num_levels) / num_levels
    return cms(mul*Ac)

# ----------------------------------------------------------------------------

def make_frames(values, 
                U, configU=None, configU_t=None,
                Rho=None, configRho=None, configRho_t=None, 
                Eps=None, configEps=None, configEps_t=None, 
                out=['phi', 'E']
                ):
    out_dict = {k: [] for k in out}

    if configU is not None:
        configU_t = lambda _: configU
    if configRho is not None:
        configRho_t = lambda _: configRho
    if configEps is not None:
        configEps_t = lambda _: configEps
    
    Uc, RhoC, EpsC = U, Rho, Eps
    for val in progbar(values):
        if configU_t is not None:
            Uc = configU_t(val)(U.copy())
        if configRho_t is not None:
            RhoC = configRho_t(val)(Rho.copy())
        if configEps_t is not None:
            EpsC = configEps_t(val)(Eps.copy())

        if 'U' in out: 
            out_dict['U'].append(Uc)

        if 'phi' in out:
            phi = solve_poisson(Uc, RhoC, EpsC)
            out_dict['phi'].append(phi)

        if 'E' in out:
            E = get_E_abs(phi)
            out_dict['E'].append(E)
    
    return out_dict

# ----------------------------------------------------------------------------

def colorize_frames(frames, num_levels=None, mul=1.):
    max_val = np.max([frame - np.min(frame) for frame in frames]) 
    frames_c = [make_colored_plot(frame, 
                                  num_levels=num_levels, 
                                  s=max_val, 
                                  mul=mul,
                                  cmap='afmhot')[:,:,:3] 
                                  for frame in frames]
    return frames_c

# ----------------------------------------------------------------------------

def plot_frames(frames, num=20, nrow=5):
    for i, frame in enumerate(frames[::len(frames)//num]):
        plt.subplot(-(num//-nrow), nrow, i+1)
        plt.imshow(frame)

# ----------------------------------------------------------------------------

###########################
######### PHYSICS #########
###########################

def get_E_abs(phi):
    Ey, Ex = np.gradient(phi)
    E = np.sqrt(Ex**2 + Ey**2)
    return E

# ----------------------------------------------------------------------------