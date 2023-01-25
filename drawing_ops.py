import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------------

def rotate(vecs, theta):
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    return vecs @ rot_mat.T

# ----------------------------------------------------------------------------

def arc(input=None,
        center=None,
        xy=None,
        radius=None,
        start=0,
        end=None,
        width=1,
        rotation=0,
        value=1
        ):
    """arc"""
    if isinstance(input, type(None)):
        def fun(input):
            return arc(input=input, center=center, xy=xy,
                       radius=radius, start=start, end=end,
                       width=width, rotation=rotation, value=value)
        return fun

    if not isinstance(radius, (tuple, list, np.ndarray)):
        radius = (radius, radius)

    if isinstance(xy, type(None)):
        xy = (center[0]-radius[0], center[1]-radius[1],
              center[0]+radius[0], center[1]+radius[1])

    input_pil = Image.fromarray(input)
    draw = ImageDraw.Draw(input_pil)
    draw.arc(xy, start=start+rotation, end=end+rotation, width=width, fill=value)
    input = np.array(input_pil)
    return input

# ----------------------------------------------------------------------------

def chord(input=None,
          center=None,
          xy=None,
          radius=None,
          start=0,
          end=None,
          width=1,
          filled=False,
          rotation=0,
          value=1):
    """chord"""
    if isinstance(input, type(None)):
        def fun(input):
            return chord(input, center=center,  xy=xy,
                         radius=radius, start=start,  end=end,
                         width=width, filled=filled,
                         rotation=rotation, value=value)
        return fun

    if isinstance(xy, type(None)):
        xy = (center[0]-radius, center[1]-radius,
              center[0]+radius, center[1]+radius)

    input_pil = Image.fromarray(input)
    draw = ImageDraw.Draw(input_pil)
    if filled:
        draw.chord(xy, start=start+rotation, end=end+rotation, fill=value)
    else:
        draw.chord(xy, start=start+rotation, end=end+rotation, width=width, outline=value)
    input = np.array(input_pil)
    return input

# ----------------------------------------------------------------------------

def circle(input=None,
           center=None,
           xy=None,
           radius=None,
           width=1,
           filled=False,
           value=1
           ):
    """circle"""
    return ellipse(input=input, center=center, xy=xy,
                   radii=(radius, radius), width=width,
                   filled=filled, value=value)

# ----------------------------------------------------------------------------

def composition(*ops):
    """composition"""
    def fun(input):
        for op in ops:
            input = op(input)
        return input
    return fun

# ----------------------------------------------------------------------------

def ellipse(input=None,
            center=None,
            xy=None,
            radii=None,
            width=1,
            filled=False,
            value=1):
    """ellipse"""
    if isinstance(input, type(None)):
        def fun(input):
            return ellipse(input, center=center, xy=xy,
                           radii=radii, width=width,
                           filled=filled, value=value)
        return fun

    # to bounding box
    if isinstance(xy, type(None)):
        xy = [center[0]-radii[0], center[1]-radii[1],
              center[0]+radii[0], center[1]+radii[1]]

    input_pil = Image.fromarray(input)
    draw = ImageDraw.Draw(input_pil)
    if filled:
        draw.ellipse(xy=xy, fill=value)
    else:
        draw.ellipse(xy=xy, outline=value, width=width)
    input = np.array(input_pil)
    return input

# ----------------------------------------------------------------------------

def line(input=None,
         center=None,
         xy=None,
         length=None,
         width=1,
         rotation=0,
         value=1):
    """line"""
    if isinstance(input, type(None)):
        def fun(input):
            return line(input, center=center, xy=xy,
                        length=length, width=width,
                        rotation=rotation, value=value)
        return fun

    rot_deg = rotation/180*np.pi
    if isinstance(xy, type(None)):
        xy = [center[0]-np.cos(rot_deg)*length/2, center[1]-np.sin(rot_deg)*length/2,
              center[0]+np.cos(rot_deg)*length/2, center[1]+np.sin(rot_deg)*length/2]

    input_pil = Image.fromarray(input)
    draw = ImageDraw.Draw(input_pil)
    draw.line(xy=xy, fill=value, width=width)
    input = np.array(input_pil)
    return input

# ----------------------------------------------------------------------------

def plate_capacitor(input=None,
                    center=None,
                    xy=None,
                    length=None,
                    distance=None,
                    plate_width=1,
                    rotation=0,
                    values=[-1, 1]
                    ):
    """plate capacitor"""
    rot_rad = rotation/180*np.pi
    fun = composition(
        line(center=(center[0]+np.sin(rot_rad)*distance/2,
                     center[1]-np.cos(rot_rad)*distance/2),
             xy=xy, length=length, width=plate_width,
             rotation=rotation, value=values[0]),
        line(center=(center[0]-np.sin(rot_rad)*distance/2,
                     center[1]+np.cos(rot_rad)*distance/2),
             xy=xy, length=length, width=plate_width,
             rotation=rotation, value=values[1]))
    if isinstance(input, type(None)):
        return fun
    return fun(input)

# ----------------------------------------------------------------------------

def point(input=None, xy=None, value=1):
    """point"""
    if isinstance(input, type(None)):
        def fun(input):
            return point(input, xy=xy, value=value)
        return fun
    x, y = int(xy[0]), int(xy[1])
    input[y, x] = value
    return input

# ----------------------------------------------------------------------------

def polygon(input=None, xy=None, offset=None, width=1, rotation=0, filled=False, value=1):
    """polygon"""
    if isinstance(input, type(None)):
        def fun(input):
            return polygon(input, xy=xy, offset=offset, width=width,
                           rotation=rotation, filled=filled, value=value)
        return fun

    if not isinstance(xy[0], (list, tuple)):
        xy = list(zip(xy[::2], xy[1::2]))

    if rotation != 0:
        xy = np.array(xy)
        center = np.mean(xy, axis=0, keepdims=True)
        xy_rot = rotate(xy - center, rotation/180*np.pi) + center
        xy = [tuple(xy_) for xy_ in xy_rot]

    if isinstance(offset, (tuple, list, np.ndarray)):
        xy = [(x+offset[0], y+offset[1]) for (x, y) in xy]

    input_pil = Image.fromarray(input)
    draw = ImageDraw.Draw(input_pil)
    if filled:
        draw.polygon(xy=xy, fill=value)
    else:
        return line(input=input, xy=list(xy) + [xy[0]], 
                    rotation=rotation, width=width, value=value)

    input = np.array(input_pil)
    return input

# ----------------------------------------------------------------------------

def rectangle(input=None,
              center=None,
              xy=None,
              wh=None,
              width=1,
              filled=False,
              rotation=0,
              value=1):
    """rectangle"""
    if isinstance(input, type(None)):
        def fun(input):
            return rectangle(input, center=center, xy=xy,
                             wh=wh, width=width, filled=filled,
                             rotation=rotation, value=value)
        return fun

    if isinstance(xy, type(None)):
        xy = [(center[0]-wh[0]/2, center[1]+wh[1]/2), 
              (center[0]-wh[0]/2, center[1]-wh[1]/2), 
              (center[0]+wh[0]/2, center[1]-wh[1]/2),
              (center[0]+wh[0]/2, center[1]+wh[1]/2)]
   
    return polygon(input=input, xy=xy, offset=0, width=width,
                   rotation=rotation, filled=filled, value=value)

# ----------------------------------------------------------------------------

def regular_polygon(input=None,
                    center=None,
                    xy=None,
                    radius=None,
                    n_sides=6,
                    width=1,
                    filled=False,
                    rotation=0,
                    value=1):
    """regular_polygon"""
    if isinstance(input, type(None)):
        def fun(input):
            return regular_polygon(input, center=center, xy=xy,
                             radius=radius, n_sides=n_sides, filled=filled,
                             width=width, rotation=rotation, value=value)
        return fun

    #if isinstance(xy, type(None)):

    points = []
    for i in range(n_sides):
        points.append((radius*np.cos(2*np.pi*i/n_sides), 
                       radius*np.sin(2*np.pi*i/n_sides)))
    
    points = [(p[0]+center[0], p[1]+center[1]) for p in points]

    return polygon(input=input, xy=points, offset=0, width=width,
                    rotation=rotation, filled=filled, value=value)

# ----------------------------------------------------------------------------

def star(input=None,
         center=None,
         n_corners=6,
         ro=None, ri=None, 
         width=1,
         filled=False,
         rotation=0,
         value=1):
    """star"""
    if isinstance(input, type(None)):
        def fun(input):
            return star(input, center=center, n_corners=n_corners,
                        ro=ro, ri=ri, width=width, filled=filled,
                        rotation=rotation, value=value)
        return fun

    ri = ro/2 if ri is None else ri
    points = []
    for i in range(n_corners):
        points.append((ro*np.cos(2*np.pi*i/n_corners), 
                       ro*np.sin(2*np.pi*i/n_corners)))
        points.append((ri*np.cos((2*i+1)*np.pi/n_corners), 
                       ri*np.sin((2*i+1)*np.pi/n_corners)))
    points = [(p[0]+center[0], p[1]+center[1]) for p in points]

    return polygon(input=input, xy=points, offset=0, width=width,
                    rotation=rotation, filled=filled, value=value)

# ----------------------------------------------------------------------------

def text(input=None,
         xy=None,  # anchor coordinate
         text_str=None,
         fontname='arial',
         fontsize=None,
         value=1,
         **kwargs):
    """text"""
    if isinstance(input, type(None)):
        def fun(input):
            return text(input, xy=xy, text_str=text_str,
                        fontsize=fontsize, **kwargs)
        return fun

    font = ImageFont.truetype(fontname, size=fontsize)

    input_pil = Image.fromarray(input)
    draw = ImageDraw.Draw(input_pil)
    draw.text(xy, text_str, font=font, fill=value, **kwargs)
    input = np.array(input_pil)
    return input

# ----------------------------------------------------------------------------
