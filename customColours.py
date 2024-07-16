from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np

def customCmap(hexArr, repeats):
    """
    hexArr is an arrary of hex strings e.g. ["#3366cc", "#800000"].
    repeats (integer) loops over these e.g. colour1, colour2, colour1, colour2 if = 2.
    Returns a cmap.     
    """
    colours = [to_rgb(hex) for hex in hexArr]
    total = len(hexArr)*repeats
    x = np.linspace(0.0, 1.0, total)

    cdict = {"red" : [], "green" : [], "blue" : []}
    for i in range(total):
        j = i % len(hexArr)
        cdict["red"].append((x[i], colours[j][0], colours[j][0]))
        cdict["green"].append((x[i], colours[j][1], colours[j][1]))
        cdict["blue"].append((x[i], colours[j][2], colours[j][2]))

    return LinearSegmentedColormap('customColours', cdict)