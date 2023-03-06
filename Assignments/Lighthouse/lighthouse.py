import numpy as np

def get_aspect(ax):
    x0, y0 = ax.transAxes.transform((0, 0)) # lower left in pixels
    x1, y1 = ax.transAxes.transform((1, 1)) # upper right in pixes
    dx0, dy0 = ax.transLimits.inverted().transform((0, 0)) # lower left in pixels
    dx1, dy1 = ax.transLimits.inverted().transform((1, 1)) # upper right in pixes
    
    dx = (x1 - x0)/(dx1-dx0)
    dy = (y1 - y0)/(dy1-dy0)
    return dx/dy


def polar2xy(xy,angle, rad_x, aspect=1):
        rad_y = aspect*rad_x
        return rad_x*np.cos(angle)+xy[0], rad_y*np.sin(angle)+xy[1]
    