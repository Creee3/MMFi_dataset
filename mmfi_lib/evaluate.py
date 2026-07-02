import numpy as np


def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    X0 = X0 / normX
    Y0 = Y0 / normY
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)
    traceTA = s.sum()
    if compute_optimal_scale:
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX
    c = muX - b * np.dot(muY, T)
    return d, Z, T, b, c


def compute_pck(preds, gts, thresholds=(0.2, 0.5)):
    rs = gts[:, 5, :]
    lh = gts[:, 12, :]
    scale = np.sqrt(np.sum(np.square(rs - lh), axis=1))
    scale = np.maximum(scale, 1e-6)
    dist = np.sqrt(np.sum(np.square(preds - gts), axis=2))
    result = {}
    for alpha in thresholds:
        thr = (scale * alpha)[:, np.newaxis]
        correct = (dist < thr)
        pck = correct.mean() * 100.0
        result[f"pck@{int(alpha*100)}"] = pck
    return result

def calulate_error(preds, gts):
    N = preds.shape[0]
    num_joints = preds.shape[1]
    mpjpe = np.mean(np.sqrt(np.sum(np.square(preds - gts), axis=2)))
    pampjpe = np.zeros([N, num_joints])
    for n in range(N):
        frame_pred = preds[n]
        frame_gt = gts[n]
        _, Z, T, b, c = compute_similarity_transform(frame_gt, frame_pred, compute_optimal_scale=True)
        frame_pred = (b * frame_pred.dot(T)) + c
        pampjpe[n] = np.sqrt(np.sum(np.square(frame_pred - frame_gt), axis=1))
    pampjpe = np.mean(pampjpe)
    pck_dict = compute_pck(preds, gts)
    return mpjpe, pampjpe, pck_dict