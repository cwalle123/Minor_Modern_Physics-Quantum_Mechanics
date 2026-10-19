"""Imports"""

# External imports
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

# Optional import: SymPy is used to parse Psi(x,0) and to find the
# normalisation constant A exactly. Without it, a numerical fallback is used.
try:
    import sympy as sp
except ImportError:
    sp = None

#################################################################################################################################################################
"""Info"""

# Time evolution of a wave packet for the 1D free particle (V = 0 everywhere),
# using the free-particle equations from Griffiths (Section 2.4):
#
#     Psi(x,t) = 1/sqrt(2 pi) * integral_-inf^inf phi(k) exp(i (k x - hbar k^2 t / (2 m))) dk
#
#     phi(k)   = 1/sqrt(2 pi) * integral_-inf^inf Psi(x,0) exp(-i k x) dx      (Plancherel's theorem)
#
# This is the continuous version of the infinite square well:
#     sum over n        ->  integral over k
#     c_n               ->  phi(k) dk
#     Fourier's trick   ->  Plancherel's theorem
#     |c_n|^2           ->  |phi(k)|^2   (probability density in k, so in momentum p = hbar k)
#
# Workflow:
#   1. You type in Psi(x,0), which may contain an unknown constant A.
#   2. A is found from normalisation: integral_-inf^inf |Psi(x,0)|^2 dx = 1
#      (exactly with SymPy when possible, otherwise numerically).
#   3. phi(k) is computed from Plancherel's theorem.
#   4. Psi(x,t) is built from the integral over k and animated.
#
# Both integrals are evaluated with the fast Fourier transform (FFT) on a large
# x grid. The FFT is just a fast way to do the same integrals as a sum.
#
# Usage:
#   python free_particle.py                              # asks for Psi(x,0)
#   python free_particle.py --psi0 "Piecewise((A, Abs(x) < a), (0, True))"
#   python free_particle.py --psi0 "A*exp(-Abs(x)/a)"
#   python free_particle.py --psi0 "A*exp(-x**2/a**2)"
#   python free_particle.py --psi0 "A*exp(-x**2/a**2)*exp(I*5*x/a)"
#
# Input syntax (SymPy): use x, a, A, pi, I (imaginary unit), sqrt, sin, cos, exp,
# Abs, Heaviside, Piecewise((expr, condition), (expr, True)), ...
#
# Controls in the animation window: SPACE = pause/play, Up/Down = faster/slower,
# R = restart at t = 0. The animation starts again at t = 0 after --tmax.
#
# Units: SI. By default the particle is an electron and the length scale is
# a = 1 nm (change with --m and --a). x is in m, k in m^-1, t in s, E in J
# (also shown in eV), Psi in m^-1/2, |Psi|^2 in m^-1 and |phi|^2 in m.
# a is just a length scale you can use in Psi(x,0) (there are no walls).
# Write lengths in Psi(x,0) in terms of a (e.g. a/3, 0.5*a, 5/a), so the
# same expression works for any length scale.
# Time is also shown as a fraction of the spreading time tau = 2 m Dx0^2 / hbar,
# where Dx0 is the initial width (standard deviation) of the packet. A Gaussian
# packet is sqrt(2) times wider at t = tau.
#
# Psi(x,0) must be normalisable, so it has to go to 0 for large |x|. A single
# plane wave A*exp(I*k*x) is not normalisable, which is why a free particle
# can't be in a stationary state.
#
# Requires: numpy, matplotlib, sympy   (pip install numpy matplotlib sympy)

#################################################################################################################################################################
"""Plot styling"""

FIGURE_SIZE = (11, 7.5)         # inches
FRAME_INTERVAL_MS = 30          # ms between animation frames
SAVE_FPS = 30                   # frames per second when saving to .gif/.mp4
MAX_PLOT_POINTS = 4000          # max points drawn per curve (the full grid is still used for the maths)

# Colours per plotted quantity
COLOUR_RE = 'tab:blue'          # Re Psi
COLOUR_IM = 'tab:orange'        # Im Psi
COLOUR_ABS = 'k'                # +-|Psi| envelope
COLOUR_RHO = 'tab:purple'       # |Psi|^2
COLOUR_RHO_0 = '0.6'            # |Psi(x,0)|^2 reference curve
COLOUR_EXP_X = 'tab:red'        # <x> marker
COLOUR_PHI = 'tab:green'        # |phi(k)|^2

#################################################################################################################################################################
"""Constants"""

HBAR = 1.054571817e-34          # J s   (reduced Planck constant)
ELECTRON_MASS = 9.1093837015e-31  # kg  (electron rest mass)
EV = 1.602176634e-19            # J     (1 electronvolt, for printing energies in eV)

LENGTH_SCALE_DEFAULT = 1e-9     # m     (length scale a, 1 nm)
MASS_DEFAULT = ELECTRON_MASS    # kg    (particle mass, an electron by default)

PROBE_HALF_WIDTH = 50           # in units of a: Psi(x,0) must vanish outside [-50a, 50a]
N_PROBE = 2 ** 16               # grid points on the probe grid (used to measure Psi(x,0) and phi(k))
N_SIM_MAX = 2 ** 20             # max grid points on the simulation grid
POINTS_PER_WIDTH = 40           # simulation grid: at least this many points per initial width Dx0
DOMAIN_PADDING = 1.5            # empty space on each side of the view, as a fraction of the view
                                # width (keeps the FFT's periodic copies away from the view)
VIEW_MARGIN = 4                 # view extends this many packet widths beyond where the packet goes
K_COVERAGE = 1 - 1e-6           # fraction of |phi(k)|^2 the simulation grid must resolve
K_PLOT_COVERAGE = 0.999         # fraction of |phi(k)|^2 shown in the phi(k) panel
PHI_PLOT_WIDTHS = 8             # ... but at most this many widths Dk_eff either side of <k>

T_MAX_DEFAULT = 4.0             # run time in units of tau
SPEED_DEFAULT = 1.0             # animation speed multiplier
FRAMES_PER_TAU = 150            # at speed 1, one spreading time tau takes this many frames

JUMP_RATIO = 5                  # Dk / Dk_eff above which Psi(x,0) is treated as having a jump
EDGE_TOLERANCE = 1e-6           # |Psi(x,0)|^2 at the probe edges, relative to its max, above which a warning is printed

DEFAULT_PSI0 = 'A*exp(-x**2/a**2)*exp(I*5*x/a)'  # used when the user just presses Enter

EXAMPLES = """Examples of Psi(x,0)  (use x, a, A, pi, I, sqrt, sin, cos, exp, Abs, Piecewise, ...):
  Piecewise((A, Abs(x) < a), (0, True))             (Griffiths' example: particle released from -a < x < a)
  A*exp(-Abs(x)/a)                                  (A e^(-|x|/a))
  A*exp(-x**2/a**2)                                 (Gaussian wave packet at rest)
  A*exp(-x**2/a**2)*exp(I*5*x/a)                    (moving Gaussian wave packet, k0 = 5/a)
  Piecewise((A*(a-Abs(x)), Abs(x) < a), (0, True))  (triangle)"""

#################################################################################################################################################################
"""Functions"""

def _integrate(y, x):
    """
    Trapezoidal integral of y over x along the last axis. Uses
    np.trapezoid on newer NumPy and falls back to np.trapz on older NumPy.

    y : array of function values (last axis matches x)
    x : 1D array of grid points

    Returns the integral (scalar, or array if y has extra leading axes).
    """
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x, axis=-1)
    return np.trapz(y, x, axis=-1)

def parse_psi0(expr_str, a_val):
    """
    Turn the user's string for Psi(x,0) into something we can compute with.
    The unknown constant A is set to 1 here; it is found afterwards from
    normalisation.

    With SymPy, the string is parsed symbolically, so the exact expression is
    also kept (for the exact normalisation). Without SymPy, the string is
    evaluated with NumPy functions directly.

    expr_str : Psi(x,0) as a string, e.g. 'A*exp(-x**2/a**2)'
    a_val    : length scale a

    Returns (f_np, sym):
      f_np : function x -> Psi(x,0) with A = 1, as a complex numpy array
      sym  : (expr, x, a) SymPy objects for the exact normalisation, or None
             when SymPy is not available
    """
    if sp is not None:
        x = sp.symbols('x', real=True)  # x runs over the whole real line here
        a, A = sp.symbols('a A', positive=True)
        local = {'x': x, 'a': a, 'A': A, 'I': sp.I, 'pi': sp.pi, 'e': sp.E,
                 'Heaviside': sp.Heaviside, 'Piecewise': sp.Piecewise}
        expr = sp.sympify(expr_str, locals=local)
        expr_A1 = expr.subs(A, 1)
        f = sp.lambdify(x, expr_A1.subs(a, a_val), modules='numpy')

        def f_np(xx):
            # Broadcast in case the expression is constant in x
            return np.asarray(f(xx), dtype=complex) * np.ones_like(xx, dtype=complex)

        return f_np, (expr_A1, x, a)

    # Fallback without SymPy: plain numpy evaluation
    ns = {k: getattr(np, k) for k in
          ('sin', 'cos', 'tan', 'exp', 'sqrt', 'pi', 'abs', 'where',
           'heaviside', 'sinh', 'cosh', 'tanh', 'log')}
    x_cur = [None]  # current x grid, needed by Piecewise to broadcast its branches

    def Piecewise(*pairs):
        conds = [np.broadcast_to(np.asarray(c, dtype=bool), np.shape(x_cur[0])) for _, c in pairs]
        vals = [np.broadcast_to(np.asarray(v, dtype=complex), np.shape(x_cur[0])) for v, _ in pairs]
        return np.select(conds, vals, default=0)

    ns.update(I=1j, j=1j, A=1.0, a=a_val, Abs=np.abs, Piecewise=Piecewise,
              Heaviside=lambda z, h0=0.5: np.heaviside(z, h0))

    def f_np(xx):
        x_cur[0] = xx
        val = eval(expr_str.replace('^', '**'), {'__builtins__': {}}, {**ns, 'x': xx, 'True': True})
        return np.asarray(val, dtype=complex) * np.ones_like(xx, dtype=complex)

    return f_np, None

def exact_normalisation(sym, a_val):
    """
    Try to find A exactly from

        1 = |A|^2 * integral_-inf^inf |f(x)|^2 dx

    where f is Psi(x,0) with A = 1.

    sym   : (expr, x, a) SymPy objects from parse_psi0, or None
    a_val : length scale a (used to check that A evaluates to a number)

    Returns A as a SymPy expression in a, or None if SymPy is missing or
    the integral can't be done in closed form.
    """
    if sym is None:
        return None
    expr, x, a = sym
    try:
        integrand = sp.simplify(expr * sp.conjugate(expr))
        I_sym = sp.integrate(integrand, (x, -sp.oo, sp.oo))
        if I_sym.has(sp.Integral) or not I_sym.is_finite:
            return None
        A_sym = sp.simplify(1 / sp.sqrt(I_sym))
        # Sanity check that it evaluates to a number
        complex(A_sym.subs(a, a_val).evalf())
        return A_sym
    except Exception:
        return None

def numeric_normalisation(f_vals, x):
    """
    Find A numerically from

        A = 1 / sqrt( integral |f(x)|^2 dx )

    f_vals : Psi(x,0) with A = 1, on the x grid
    x      : 1D array of grid points

    Returns A (float).
    """
    norm2 = _integrate(np.abs(f_vals) ** 2, x)
    if not np.isfinite(norm2) or norm2 <= 0:
        raise ValueError('Psi(x,0) has zero or infinite norm -- cannot normalise.')
    return 1.0 / np.sqrt(norm2)

def normalise(f_vals, x, sym, a_val):
    """
    Find the normalisation constant A (exactly if possible, otherwise
    numerically) and print the result.

    f_vals : Psi(x,0) with A = 1, on the probe grid
    x      : probe grid
    sym    : (expr, x, a) SymPy objects from parse_psi0, or None
    a_val  : length scale a

    Returns (A_val, A_text):
      A_val  : A as a float
      A_text : label for A, used in the info box of the animation
    """
    A_num = numeric_normalisation(f_vals, x)
    A_sym = exact_normalisation(sym, a_val)

    print('\n--- Normalisation:  integral_-inf^inf |Psi(x,0)|^2 dx = 1 ---')
    if A_sym is not None:
        A_val = float(sp.re(A_sym.subs(sym[2], a_val).evalf()))
        print(f'  A = {sp.pretty(A_sym, use_unicode=False)}   (exact)')
        print(f'    = {A_val:.10g}   for a = {a_val} m')
        A_text = r'$A = ' + sp.latex(A_sym) + r'$' + f' = {A_val:.6g}'
    else:
        A_val = A_num
        print(f'  A = {A_num:.10g}   (numerical)')
        A_text = f'A = {A_num:.6g}  (numerical)'
    return A_val, A_text

def check_localised(f_vals, probe_half_width):
    """
    Stop if Psi(x,0) hasn't died off at the edges of the probe grid. Then
    it is either not normalisable (like a plane wave), or wider than
    the probe grid (increase --probe), and the results can't be trusted.

    f_vals           : Psi(x,0) on the probe grid
    probe_half_width : half-width of the probe grid, in units of a
    """
    rho = np.abs(f_vals) ** 2
    edge = max(rho[0], rho[-1]) / max(np.max(rho), 1e-300)
    if edge > EDGE_TOLERANCE:
        sys.exit(f'\nPsi(x,0) has not gone to 0 at x = +-{probe_half_width}a, so it can\'t be normalised here.\n'
                 '  - If it never goes to 0 (e.g. a plane wave A*exp(I*k*x)), it is not normalisable at all:\n'
                 '    that is why a free particle can\'t be in a single stationary state.\n'
                 '  - If it is just wide, write lengths in terms of a, or increase --probe.')

def make_grid(x_min, x_max, N):
    """
    Evenly spaced x grid for the FFT and the matching k grid.

    x_min, x_max : ends of the grid (x_max itself is left out, since the
                   FFT treats the grid as periodic)
    N            : number of points

    Returns (x, k), both 1D arrays of length N. k is in FFT order
    (0, positive k, then negative k), use np.fft.fftshift to sort it.
    """
    x = x_min + (x_max - x_min) * np.arange(N) / N
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    return x, k

def get_phi(psi0, x, k):
    """
    phi(k) from Plancherel's theorem:

        phi(k) = 1/sqrt(2 pi) * integral Psi(x,0) exp(-i k x) dx

    On the grid x_j = x_0 + j dx this sum is exactly an FFT, times
    dx/sqrt(2 pi) and a phase exp(-i k x_0) for the grid not starting at 0.

    psi0 : normalised Psi(x,0) on the x grid
    x    : x grid from make_grid
    k    : k grid from make_grid

    Returns phi(k) on the k grid (complex array, FFT order).
    """
    dx = x[1] - x[0]
    return dx / np.sqrt(2 * np.pi) * np.exp(-1j * k * x[0]) * np.fft.fft(psi0)

def omega(k, m, hbar):
    """
    Angular frequency of each plane wave, omega(k) = hbar k^2 / (2 m),
    so that E = hbar omega = hbar^2 k^2 / (2 m).

    k    : wave numbers
    m    : particle mass
    hbar : reduced Planck constant

    Returns omega(k).
    """
    return hbar * k ** 2 / (2 * m)

def psi_at_time(t, phi, x, k, m, hbar):
    """
    Psi(x,t) from the integral over k:

        Psi(x,t) = 1/sqrt(2 pi) * integral phi(k) exp(i (k x - hbar k^2 t / (2 m))) dk

    Each plane wave just picks up the phase exp(-i omega(k) t). The sum over
    k is done with an inverse FFT (the reverse of get_phi).

    t    : time
    phi  : phi(k) from get_phi
    x, k : grids from make_grid
    m    : particle mass
    hbar : reduced Planck constant

    Returns Psi(x,t) on the x grid (complex array).
    """
    N = len(x)
    dk = 2 * np.pi / (N * (x[1] - x[0]))
    phase = np.exp(-1j * omega(k, m, hbar) * t)
    return dk / np.sqrt(2 * np.pi) * N * np.fft.ifft(phi * phase * np.exp(1j * k * x[0]))

def position_stats(psi, x):
    """
    Expectation value and spread of x:

        <x> = integral x |Psi|^2 dx,   Dx = sqrt(<x^2> - <x>^2)

    psi : Psi on the x grid
    x   : x grid

    Returns (<x>, Dx).
    """
    rho = np.abs(psi) ** 2
    ex = _integrate(x * rho, x)
    ex2 = _integrate(x ** 2 * rho, x)
    return ex, np.sqrt(max(ex2 - ex ** 2, 0.0))

def momentum_stats(phi, k, m, hbar):
    """
    Everything that follows from |phi(k)|^2, which doesn't change in time
    (just like |c_n|^2 for the square well):

        integral |phi|^2 dk = 1      (check of Plancherel's theorem)
        <p> = hbar <k>,   Dp = hbar Dk
        <H> = integral |phi|^2 hbar^2 k^2 / (2 m) dk

    Dk_eff is the half-width of the range around the median holding 68% of
    |phi|^2. For a Gaussian this equals Dk, but it stays finite for states
    with jumps (like the box), where Dk itself is infinite.

    phi  : phi(k) on the k grid (FFT order)
    k    : k grid (FFT order)
    m    : particle mass
    hbar : reduced Planck constant

    Returns a dict with norm, ek, Dk, Dk_eff, expH, and the sorted k and
    |phi|^2 for plotting.
    """
    order = np.argsort(k)
    ks = k[order]
    prob = np.abs(phi[order]) ** 2
    dk = ks[1] - ks[0]

    norm = np.sum(prob) * dk
    ek = np.sum(ks * prob) * dk / norm
    ek2 = np.sum(ks ** 2 * prob) * dk / norm
    Dk = np.sqrt(max(ek2 - ek ** 2, 0.0))
    expH = hbar ** 2 * ek2 / (2 * m)

    cdf = np.cumsum(prob) * dk / norm
    k_16, k_84 = np.interp([0.16, 0.84], cdf, ks)
    Dk_eff = (k_84 - k_16) / 2

    return {'norm': norm, 'ek': ek, 'Dk': Dk, 'Dk_eff': Dk_eff, 'expH': expH,
            'k_sorted': ks, 'prob_sorted': prob, 'cdf': cdf}

def k_range(stats, coverage):
    """
    Smallest symmetric-in-probability range of k that holds a fraction
    'coverage' of |phi(k)|^2.

    stats    : dict from momentum_stats
    coverage : fraction, e.g. 0.999

    Returns (k_low, k_high).
    """
    tail = (1 - coverage) / 2
    return tuple(np.interp([tail, 1 - tail], stats['cdf'], stats['k_sorted']))

def spreading_time(Dx0, m, hbar):
    """
    Spreading time tau = 2 m Dx0^2 / hbar. For a Gaussian packet the width
    grows as Dx(t) = Dx0 * sqrt(1 + (t/tau)^2), so it is sqrt(2) times wider
    at t = tau.

    Dx0  : initial width of the packet
    m    : particle mass
    hbar : reduced Planck constant

    Returns tau.
    """
    return 2 * m * Dx0 ** 2 / hbar

def plan_simulation(x0, Dx0, stats, tau, t_max, m, hbar):
    """
    Choose the x range to show and the x grid to simulate on, so that
    the packet stays in view until t_max, the FFT's periodic copies stay
    far outside the view, and the grid is fine enough for both Psi(x,0)
    and phi(k).

    x0, Dx0 : <x> and Dx at t = 0
    stats   : dict from momentum_stats
    tau     : spreading time
    t_max   : run time
    m, hbar : physical constants

    Returns (view, x_min, x_max, N):
      view         : (x_left, x_right) shown in the plots
      x_min, x_max : ends of the simulation grid
      N            : number of grid points (a power of 2)
    """
    v_group = hbar * stats['ek'] / m                          # <p>/m, speed of the packet's centre
    width_end = np.sqrt(Dx0 ** 2 + (hbar * stats['Dk_eff'] * t_max / m) ** 2)
    x_end = x0 + v_group * t_max
    view = (min(x0, x_end) - VIEW_MARGIN * width_end, max(x0, x_end) + VIEW_MARGIN * width_end)

    view_width = view[1] - view[0]
    x_min = view[0] - DOMAIN_PADDING * view_width
    x_max = view[1] + DOMAIN_PADDING * view_width

    k_lo, k_hi = k_range(stats, K_COVERAGE)
    dx_needed = min(Dx0 / POINTS_PER_WIDTH, np.pi / max(abs(k_lo), abs(k_hi)))
    N = int(2 ** np.ceil(np.log2((x_max - x_min) / dx_needed)))
    if N > N_SIM_MAX:
        print(f'  NOTE: grid capped at {N_SIM_MAX} points; the finest details of Psi may be smoothed out.')
        N = N_SIM_MAX
    return view, x_min, x_max, N

def print_momentum(stats, Dx0, tau):
    """
    Print the checks and results that follow from phi(k).

    stats : dict from momentum_stats
    Dx0   : initial width of the packet
    tau   : spreading time
    """
    print("\n--- phi(k) from Plancherel's theorem ---")
    print(f'  integral |phi(k)|^2 dk = {stats["norm"]:.8f}   (should be 1)')
    print(f'  <p> = hbar <k> = {HBAR * stats["ek"]:.6g} kg m/s   (<k> = {stats["ek"]:.6g} m^-1)')
    print(f'  Dp  = hbar Dk  = {HBAR * stats["Dk"]:.6g} kg m/s')
    print(f'  Dx0 * Dp = {Dx0 * stats["Dk"]:.4f} hbar   (uncertainty principle: >= 0.5 hbar)')
    print(f'  <H> = integral |phi|^2 hbar^2 k^2 / 2m dk = {stats["expH"]:.6g} J = {stats["expH"] / EV:.6g} eV')
    print(f'  spreading time tau = 2 m Dx0^2 / hbar = {tau:.4g} s')
    if stats['Dk'] > JUMP_RATIO * stats['Dk_eff']:
        print('  NOTE: Psi(x,0) seems to have a jump, so |phi(k)|^2 falls off slowly and Dp and <H>'
              ' are really infinite. The values above are limited by the grid spacing; tau uses Dx0 only.')

def animate(x, view, psi0, phi, k, stats, m, hbar, tau, t_max, title_expr, A_text, speed, frames=None):
    """
    Build the animation window:

    - top panel    : Re Psi, Im Psi and +-|Psi|
    - middle panel : |Psi|^2 with <x>, and |Psi(x,0)|^2 for reference
    - bottom-left  : |phi(k)|^2 (probability density in k, constant in time)
    - bottom-right : info box (A, integral |phi|^2, <H>, <p>, tau)

    x          : simulation x grid
    view       : (x_left, x_right) shown in the plots
    psi0       : normalised Psi(x,0) on the x grid
    phi        : phi(k) from get_phi
    k          : k grid
    stats      : dict from momentum_stats
    m, hbar    : physical constants
    tau        : spreading time
    t_max      : run time, after which the animation restarts at t = 0
    title_expr : Psi(x,0) as a (LaTeX) string for the figure title
    A_text     : label for the normalisation constant
    speed      : animation speed multiplier
    frames     : number of frames (None = run forever, used when showing live)

    Returns (fig, anim).
    """
    # Only the part of the grid inside the view is drawn, thinned out to MAX_PLOT_POINTS
    in_view = np.where((x >= view[0]) & (x <= view[1]))[0]
    stride = max(1, len(in_view) // MAX_PLOT_POINTS)
    idx = in_view[::stride]
    xp = x[idx]

    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 2.2, 1.4], width_ratios=[3, 1.4],
                          hspace=0.45, wspace=0.25)
    ax_psi = fig.add_subplot(gs[0, :])
    ax_rho = fig.add_subplot(gs[1, :], sharex=ax_psi)
    ax_k = fig.add_subplot(gs[2, 0])
    ax_info = fig.add_subplot(gs[2, 1])
    ax_info.axis('off')

    fig.suptitle(r'Free particle:  $\Psi(x,0) = $ ' + title_expr, fontsize=12)

    # Wave function panel
    ymax = 1.15 * np.max(np.abs(psi0))
    ax_psi.set_xlim(*view)
    ax_psi.set_ylim(-ymax, ymax)
    ax_psi.axhline(0, color='k', lw=0.6)
    line_re, = ax_psi.plot([], [], color=COLOUR_RE, lw=1.6, label=r'Re $\Psi$')
    line_im, = ax_psi.plot([], [], color=COLOUR_IM, lw=1.6, label=r'Im $\Psi$')
    line_abs, = ax_psi.plot([], [], color=COLOUR_ABS, lw=1.0, ls='--', label=r'$\pm|\Psi|$')
    line_nabs, = ax_psi.plot([], [], color=COLOUR_ABS, lw=1.0, ls='--')
    ax_psi.set_ylabel(r'$\Psi(x,t)$ (m$^{-1/2}$)')
    ax_psi.legend(loc='upper right', fontsize=9, ncol=3)
    time_txt = ax_psi.text(0.01, 0.92, '', transform=ax_psi.transAxes, fontsize=11,
                           bbox=dict(fc='white', ec='0.7', alpha=0.85))

    # Probability density panel
    rho0 = np.abs(psi0) ** 2
    ax_rho.set_ylim(0, 1.15 * np.max(rho0))
    ax_rho.plot(xp, rho0[idx], color=COLOUR_RHO_0, lw=1, ls=':', label=r'$|\Psi(x,0)|^2$')
    line_rho, = ax_rho.plot([], [], color=COLOUR_RHO, lw=1.8, label=r'$|\Psi(x,t)|^2$')
    fill = [ax_rho.fill_between(xp, 0, 0, color=COLOUR_RHO, alpha=0.25)]
    exp_x_line = ax_rho.axvline(0, color=COLOUR_EXP_X, lw=1.2, ls='--', label=r'$\langle x\rangle$')
    ax_rho.set_xlabel(r'$x$ (m)')
    ax_rho.set_ylabel(r'$|\Psi(x,t)|^2$ (m$^{-1}$)')
    ax_rho.legend(loc='upper right', fontsize=9, ncol=3)

    # |phi(k)|^2 panel
    # Show most of |phi|^2, but at most PHI_PLOT_WIDTHS widths around <k> (states
    # with jumps have very long tails that would squash the plot)
    k_lo, k_hi = k_range(stats, K_PLOT_COVERAGE)
    k_lo = max(k_lo, stats['ek'] - PHI_PLOT_WIDTHS * stats['Dk_eff'])
    k_hi = min(k_hi, stats['ek'] + PHI_PLOT_WIDTHS * stats['Dk_eff'])
    pad = 0.1 * (k_hi - k_lo)
    shown = (stats['k_sorted'] >= k_lo - pad) & (stats['k_sorted'] <= k_hi + pad)
    ax_k.plot(stats['k_sorted'][shown], stats['prob_sorted'][shown], color=COLOUR_PHI, lw=1.6)
    ax_k.fill_between(stats['k_sorted'][shown], 0, stats['prob_sorted'][shown], color=COLOUR_PHI, alpha=0.3)
    ax_k.axvline(stats['ek'], color=COLOUR_EXP_X, lw=1.2, ls='--', label=r'$\langle k\rangle$')
    ax_k.set_xlim(k_lo - pad, k_hi + pad)
    ax_k.set_ylim(bottom=0)
    ax_k.set_xlabel(r'$k$ (m$^{-1}$)')
    ax_k.set_ylabel(r'$|\phi(k)|^2$ (m)')
    ax_k.set_title(r'probability density in $k$  ($p = \hbar k$)', fontsize=10)
    ax_k.legend(loc='upper right', fontsize=9)

    # Info box
    expH = stats['expH']
    info = (f'{A_text}\n'
            rf'$\int |\phi|^2 dk$ = {stats["norm"]:.6f}' + '\n'
            rf'$\langle H\rangle$ = {expH:.4g} J = {expH / EV:.4g} eV' + '\n'
            rf'$\langle p\rangle$ = {hbar * stats["ek"]:.4g} kg m/s' + '\n'
            rf'spreading time $\tau = 2m\Delta x_0^2/\hbar$ = {tau:.4g} s' + '\n\n'
            'SPACE pause   ↑/↓ speed   R restart')
    ax_info.text(0, 1, info, va='top', fontsize=9.5, family='monospace')

    state = {'t': 0.0, 'running': True, 'dt': speed * tau / FRAMES_PER_TAU}

    def on_key(ev):
        if ev.key == ' ':
            state['running'] = not state['running']
        elif ev.key == 'up':
            state['dt'] *= 1.5
        elif ev.key == 'down':
            state['dt'] /= 1.5
        elif ev.key in ('r', 'R'):
            state['t'] = 0.0

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(_frame):
        if state['running']:
            state['t'] += state['dt']
            if state['t'] > t_max:  # start again from t = 0
                state['t'] = 0.0
        t = state['t']
        psi = psi_at_time(t, phi, x, k, m, hbar)
        ex, Dx = position_stats(psi, x)
        psi_p = psi[idx]
        rho_p = np.abs(psi_p) ** 2
        line_re.set_data(xp, psi_p.real)
        line_im.set_data(xp, psi_p.imag)
        line_abs.set_data(xp, np.abs(psi_p))
        line_nabs.set_data(xp, -np.abs(psi_p))
        line_rho.set_data(xp, rho_p)
        fill[0].remove()
        fill[0] = ax_rho.fill_between(xp, 0, rho_p, color=COLOUR_RHO, alpha=0.25)
        exp_x_line.set_xdata([ex, ex])
        time_txt.set_text(f't = {t:.4e} s   ( t/tau = {t / tau:.3f} )   Dx = {Dx:.4g} m')
        return line_re, line_im, line_abs, line_nabs, line_rho, exp_x_line, time_txt

    # Draw the first frame at t = 0
    update(0)
    state['t'] = 0.0
    anim = FuncAnimation(fig, update, frames=frames, interval=FRAME_INTERVAL_MS, blit=False,
                         cache_frame_data=False)
    return fig, anim

#################################################################################################################################################################
"""Main"""

def main():
    """
    Run the full pipeline:
    read Psi(x,0) -> find A by normalisation -> compute phi(k) (Plancherel)
    -> build Psi(x,t) (integral over k) -> animate (or save the animation).
    """
    p = argparse.ArgumentParser(description='Free particle time-evolution simulator',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=EXAMPLES)
    p.add_argument('--psi0', type=str, default=None, help='initial wave function Psi(x,0)')
    p.add_argument('--a', type=float, default=LENGTH_SCALE_DEFAULT, help='length scale a in m (default 1e-9)')
    p.add_argument('--m', type=float, default=MASS_DEFAULT, help='particle mass in kg (default: electron)')
    p.add_argument('--tmax', type=float, default=T_MAX_DEFAULT, help='run time in units of tau (default 4)')
    p.add_argument('--probe', type=float, default=PROBE_HALF_WIDTH,
                   help='Psi(x,0) must vanish outside [-probe*a, probe*a] (default 50)')
    p.add_argument('--speed', type=float, default=SPEED_DEFAULT, help='animation speed multiplier (default 1)')
    p.add_argument('--save', type=str, default=None,
                   help='save the animation instead of showing it (e.g. free.gif or free.mp4)')
    p.add_argument('--frames', type=int, default=None,
                   help='frames to write when using --save (default: one full run up to --tmax)')
    args = p.parse_args()

    expr_str = args.psi0
    if expr_str is None:
        print(EXAMPLES)
        expr_str = input('\nPsi(x,0) = ').strip() or DEFAULT_PSI0

    a, m, hbar = args.a, args.m, HBAR

    # Step 1: parse Psi(x,0) and look at it on a wide probe grid
    x_probe, k_probe = make_grid(-args.probe * a, args.probe * a, N_PROBE)
    try:
        f_np, sym = parse_psi0(expr_str, a)
        f_probe = f_np(x_probe)
    except Exception as err:
        sys.exit(f'Could not understand Psi(x,0) = {expr_str!r}:\n  {err}')

    # Step 2: normalise (find A)
    check_localised(f_probe, args.probe)
    A_val, A_text = normalise(f_probe, x_probe, sym, a)
    psi0_probe = A_val * f_probe
    x0, Dx0 = position_stats(psi0_probe, x_probe)
    print(f'  <x>(0) = {x0:.6g} m,   Dx0 = {Dx0:.6g} m')

    # Step 3: phi(k) on the probe grid, used to plan the simulation grid
    stats_probe = momentum_stats(get_phi(psi0_probe, x_probe, k_probe), k_probe, m, hbar)
    tau = spreading_time(Dx0, m, hbar)
    t_max = args.tmax * tau
    view, x_min, x_max, N = plan_simulation(x0, Dx0, stats_probe, tau, t_max, m, hbar)

    # Step 3 (again): phi(k) on the final simulation grid
    x, k = make_grid(x_min, x_max, N)
    psi0 = A_val * f_np(x)
    # Tiny correction so the sampled Psi(x,0) is normalised exactly on this grid
    # (matters for functions with jumps, where the grid points shift the edges)
    psi0 = psi0 / np.sqrt(np.sum(np.abs(psi0) ** 2) * (x[1] - x[0]))
    phi = get_phi(psi0, x, k)
    stats = momentum_stats(phi, k, m, hbar)
    print_momentum(stats, Dx0, tau)

    # Check how much of the packet is still inside the view at the end of the run
    psi_end = psi_at_time(t_max, phi, x, k, m, hbar)
    in_view = (x >= view[0]) & (x <= view[1])
    inside = _integrate(np.abs(psi_end[in_view]) ** 2, x[in_view])
    print(f'  grid: {N} points from {x_min:.3g} m to {x_max:.3g} m;'
          f' {100 * inside:.2f}% of the probability is still in view at t = tmax')

    # Step 4: animate Psi(x,t)
    if sp is not None:
        title_expr = '$' + sp.latex(sp.sympify(expr_str, locals={'I': sp.I, 'pi': sp.pi,
                                                                 'Piecewise': sp.Piecewise})) + '$'
    else:
        title_expr = expr_str

    frames = None
    if args.save:
        plt.switch_backend('Agg')  # no window needed when only saving
        frames = args.frames or int(np.ceil(args.tmax * FRAMES_PER_TAU / args.speed))

    fig, anim = animate(x, view, psi0, phi, k, stats, m, hbar, tau, t_max, title_expr, A_text,
                        args.speed, frames=frames)

    if args.save:
        writer = PillowWriter(fps=SAVE_FPS) if args.save.lower().endswith('.gif') else FFMpegWriter(fps=SAVE_FPS)
        print(f'\nSaving {frames} frames to {args.save} ...')
        anim.save(args.save, writer=writer)
        print('done.')
    else:
        plt.show()  # blocks until the figure window is closed

if __name__ == '__main__':
    main()
