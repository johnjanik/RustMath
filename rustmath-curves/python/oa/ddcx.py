"""Double-double real + complex elementwise arithmetic on arrays (cupy/numpy).
Used for the O(n^2) per-pair scalar work in MxpSVDStep (Alg 3); the O(n^3) GEMMs
go through ozaki.py. dd-complex is a dict {'reh','rel','imh','iml'} of f64 arrays."""
try:
    import cupy as xp
    _GPU = True
except Exception:
    import numpy as xp
    _GPU = False
from ozaki import two_sum, two_prod, dd_add, dd_sub

# ---- dd real ----
def dd_mul(xh, xl, yh, yl):
    p, e = two_prod(xh, yh); e = e + xh*yl + xl*yh
    return two_sum(p, e)
def dd_div(xh, xl, yh, yl):
    q1 = xh / yh
    ph, pl = dd_mul(q1, xp.zeros_like(q1), yh, yl)
    rh, rl = dd_sub(xh, xl, ph, pl)
    q2 = rh / yh
    ph2, pl2 = dd_mul(q2, xp.zeros_like(q2), yh, yl)
    rh2, rl2 = dd_sub(rh, rl, ph2, pl2)
    q3 = rh2 / yh
    zh, zl = two_sum(q1, q2); zl = zl + q3
    return two_sum(zh, zl)

# ---- dd complex (dict) ----
def cnew(reh, rel, imh, iml): return {'reh': reh, 'rel': rel, 'imh': imh, 'iml': iml}
def czeros(shape):
    z = xp.zeros(shape); return cnew(z, z.copy(), z.copy(), z.copy())
def from_c128(z):  # complex128 array -> dd-complex (lo=0)
    return cnew(xp.real(z).copy(), xp.zeros(z.shape), xp.imag(z).copy(), xp.zeros(z.shape))
def to_c128(a):
    return (a['reh'] + a['rel']) + 1j*(a['imh'] + a['iml'])
def cadd(a, b):
    reh, rel = dd_add(a['reh'], a['rel'], b['reh'], b['rel'])
    imh, iml = dd_add(a['imh'], a['iml'], b['imh'], b['iml'])
    return cnew(reh, rel, imh, iml)
def csub(a, b):
    reh, rel = dd_sub(a['reh'], a['rel'], b['reh'], b['rel'])
    imh, iml = dd_sub(a['imh'], a['iml'], b['imh'], b['iml'])
    return cnew(reh, rel, imh, iml)
def cconj(a):
    return cnew(a['reh'], a['rel'], -a['imh'], -a['iml'])
def cscale_ddreal(a, sh, sl):   # dd-complex * dd-real (broadcast)
    reh, rel = dd_mul(a['reh'], a['rel'], sh, sl)
    imh, iml = dd_mul(a['imh'], a['iml'], sh, sl)
    return cnew(reh, rel, imh, iml)
def cdiv_ddreal(a, sh, sl):     # dd-complex / dd-real (broadcast)
    reh, rel = dd_div(a['reh'], a['rel'], sh, sl)
    imh, iml = dd_div(a['imh'], a['iml'], sh, sl)
    return cnew(reh, rel, imh, iml)
def cmul(a, b):                 # dd-complex * dd-complex
    t1h, t1l = dd_mul(a['reh'], a['rel'], b['reh'], b['rel'])
    t2h, t2l = dd_mul(a['imh'], a['iml'], b['imh'], b['iml'])
    reh, rel = dd_sub(t1h, t1l, t2h, t2l)
    t3h, t3l = dd_mul(a['reh'], a['rel'], b['imh'], b['iml'])
    t4h, t4l = dd_mul(a['imh'], a['iml'], b['reh'], b['rel'])
    imh, iml = dd_add(t3h, t3l, t4h, t4l)
    return cnew(reh, rel, imh, iml)
def cabs2_ddreal(a):           # |a|^2 as dd-real
    rh, rl = dd_mul(a['reh'], a['rel'], a['reh'], a['rel'])
    ih, il = dd_mul(a['imh'], a['iml'], a['imh'], a['iml'])
    return dd_add(rh, rl, ih, il)
def cslice(a, rows, cols):
    return cnew(a['reh'][rows][:, cols], a['rel'][rows][:, cols],
                a['imh'][rows][:, cols], a['iml'][rows][:, cols])
def cget_cols(a, cols):
    return cnew(a['reh'][:, cols], a['rel'][:, cols], a['imh'][:, cols], a['iml'][:, cols])
def cset_cols(a, cols, b):
    for key in ('reh', 'rel', 'imh', 'iml'):
        a[key][:, cols] = b[key]
def cH(a):                     # conjugate transpose
    return cnew(a['reh'].T.copy(), a['rel'].T.copy(), -a['imh'].T.copy(), -a['iml'].T.copy())
def ceye(n):
    z = xp.zeros((n, n)); d = xp.eye(n)
    return cnew(d, z.copy(), z.copy(), z.copy())
def cfrob(a):                  # Frobenius norm (fp64 is enough for a diagnostic)
    return float(xp.sqrt(xp.sum(a['reh']**2 + a['imh']**2)))
