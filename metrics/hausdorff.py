from scipy.spatial.distance import directed_hausdorff

def hausdorff_distance(x, y):
    x = x.cpu().data.numpy()
    u = x.reshape(x.shape[1], -1)
    y = y.cpu().data.numpy()
    v = y.reshape(y.shape[1], -1)
    return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])