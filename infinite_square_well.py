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

# Time evolution of a wave function in the 1D infinite square well (0 < x < a),
# using Griffiths eqs. (2.39) and (2.40):
#
#     Psi(x,t) = sum_n c_n sqrt(2/a) sin(n pi x / a) exp(-i n^2 pi^2 hbar t / (2 m a^2))
#
#     c_n      = sqrt(2/a) * integral_0^a sin(n pi x / a) Psi(x,0) dx
#
# Workflow:
#   1. You type in Psi(x,0), which may contain an unknown constant A.
#   2. A is found from normalisation: integral_0^a |Psi(x,0)|^2 dx = 1
#      (exactly with SymPy when possible, otherwise numerically).
#   3. The coefficients c_n are computed with Fourier's trick (eq. 2.40).
#   4. Psi(x,t) is built from eq. 2.39 and animated.
#
# Usage:
#   python infinite_square_well.py                       # asks for Psi(x,0)
#   python infinite_square_well.py --psi0 "A*x*(a-x)"
#   python infinite_square_well.py --psi0 "A*(sin(pi*x/a) + sin(2*pi*x/a))"
#   python infinite_square_well.py --psi0 "Piecewise((A*x, x < a/2), (A*(a-x), True))"
#   python infinite_square_well.py --psi0 "A*exp(-(x-a/3)**2/(2*(0.05*a)**2))*exp(I*40*x/a)"
#
# Input syntax (SymPy): use x, a, A, pi, I (imaginary unit), sqrt, sin, cos, exp,
# Abs, Heaviside, Piecewise((expr, condition), (expr, True)), ...
#
# Controls in the animation window: SPACE = pause/play, Up/Down = faster/slower,
# R = restart at t = 0.
#
# Units: SI. By default the particle is an electron in a well of width
# a = 1 nm (change with --m and --a). x is in m, t in s, E in J (also shown
# in eV), Psi in m^-1/2 and |Psi|^2 in m^-1.
# Write lengths in Psi(x,0) in terms of a (e.g. a/3, 0.05*a, 40/a), so the
# same expression works for any well width.
# Time is also shown as a fraction of the revival time T = 4 m a^2 / (pi hbar),
# after which Psi returns exactly to Psi(x,0). For the default electron in
# a 1 nm well, T is about 11 fs.
#
# Requires: numpy, matplotlib, sympy   (pip install numpy matplotlib sympy)

#################################################################################################################################################################
"""Plot styling"""

FIGURE_SIZE = (11, 7.5)         # inches
FRAME_INTERVAL_MS = 30          # ms between animation frames
SAVE_FPS = 30                   # frames per second when saving to .gif/.mp4

# Colours per plotted quantity
COLOUR_RE = 'tab:blue'          # Re Psi
COLOUR_IM = 'tab:orange'        # Im Psi
COLOUR_ABS = 'k'                # +-|Psi| envelope
COLOUR_RHO = 'tab:purple'       # |Psi|^2
COLOUR_RHO_0 = '0.6'            # |Psi(x,0)|^2 reference curve
COLOUR_EXP_X = 'tab:red'        # <x> marker
COLOUR_CN = 'tab:green'         # |c_n|^2 bars

#################################################################################################################################################################
"""Constants"""

HBAR = 1.054571817e-34          # J s   (reduced Planck constant)
ELECTRON_MASS = 9.1093837015e-31  # kg  (electron rest mass)
EV = 1.602176634e-19            # J     (1 electronvolt, for printing energies in eV)

WELL_WIDTH_DEFAULT = 1e-9       # m     (width of the well a, 1 nm)
MASS_DEFAULT = ELECTRON_MASS    # kg    (particle mass, an electron by default)

N_STATES_DEFAULT = 200          # number of stationary states kept in the sum (2.39)
N_X_DEFAULT = 2000              # number of grid points in x
N_NORM_POINTS = 20001           # grid points used for the numerical normalisation integral

SPEED_DEFAULT = 1.0             # animation speed multiplier
FRAMES_PER_REVIVAL = 600        # at speed 1, one revival time T takes this many frames
SAVE_FRAMES_DEFAULT = 600       # frames written when using --save

BOUNDARY_TOLERANCE = 1e-3       # |Psi(0)|,|Psi(a)| relative to max|Psi| above which a warning is printed
N_COEFFS_PRINTED = 8            # how many c_n are printed to the terminal

DEFAULT_PSI0 = 'A*x*(a-x)'      # used when the user just presses Enter

EXAMPLES = """Examples of Psi(x,0)  (use x, a, A, pi, I, sqrt, sin, cos, exp, Piecewise, ...):
  A*x*(a-x)                                         (Griffiths Ex. 2.2)
  A*(sin(pi*x/a) + sin(2*pi*x/a))                   (superposition of n=1 and n=2)
  A*sin(pi*x/a)**3
  Piecewise((A*x, x < a/2), (A*(a-x), True))        (triangle)
  A*exp(-(x-a/3)**2/(2*(0.05*a)**2))*exp(I*40*x/a) (moving Gaussian wave packet)"""

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

    expr_str : Psi(x,0) as a string, e.g. 'A*x*(a-x)'
    a_val    : width of the well a

    Returns (f_np, sym):
      f_np : function x -> Psi(x,0) with A = 1, as a complex numpy array
      sym  : (expr, x, a) SymPy objects for the exact normalisation, or None
             when SymPy is not available
    """
    if sp is not None:
        x, a, A = sp.symbols('x a A', positive=True)
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

        1 = |A|^2 * integral_0^a |f(x)|^2 dx

    where f is Psi(x,0) with A = 1.

    sym   : (expr, x, a) SymPy objects from parse_psi0, or None
    a_val : width of the well a (used to check that A evaluates to a number)

    Returns A as a SymPy expression in a, or None if SymPy is missing or
    the integral can't be done in closed form.
    """
    if sym is None:
        return None
    expr, x, a = sym
    try:
        integrand = sp.simplify(expr * sp.conjugate(expr))
        I_sym = sp.integrate(integrand, (x, 0, a))
        if I_sym.has(sp.Integral):
            return None
        A_sym = sp.simplify(1 / sp.sqrt(I_sym))
        # Sanity check that it evaluates to a number
        complex(A_sym.subs(a, a_val).evalf())
        return A_sym
    except Exception:
        return None

def numeric_normalisation(f_np, a_val):
    """
    Find A numerically from

        A = 1 / sqrt( integral_0^a |f(x)|^2 dx )

    f_np  : function x -> Psi(x,0) with A = 1
    a_val : width of the well a

    Returns A (float).
    """
    xs = np.linspace(0, a_val, N_NORM_POINTS)
    norm2 = _integrate(np.abs(f_np(xs)) ** 2, xs)
    if not np.isfinite(norm2) or norm2 <= 0:
        raise ValueError('Psi(x,0) has zero or infinite norm on [0, a] -- cannot normalise.')
    return 1.0 / np.sqrt(norm2)

def normalise(f_np, sym, a_val):
    """
    Find the normalisation constant A (exactly if possible, otherwise
    numerically) and print the result.

    f_np  : function x -> Psi(x,0) with A = 1
    sym   : (expr, x, a) SymPy objects from parse_psi0, or None
    a_val : width of the well a

    Returns (A_val, A_text):
      A_val  : A as a float
      A_text : label for A, used in the info box of the animation
    """
    A_num = numeric_normalisation(f_np, a_val)
    A_sym = exact_normalisation(sym, a_val)

    print('\n--- Normalisation:  integral_0^a |Psi(x,0)|^2 dx = 1 ---')
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

def check_boundaries(psi0):
    """
    Warn if Psi(x,0) doesn't vanish at the walls. Such a state isn't really
    allowed in the infinite well, and the sine series (2.39) will show
    Gibbs ringing there.

    psi0 : normalised Psi(x,0) on the x grid
    """
    edge = max(abs(psi0[0]), abs(psi0[-1])) / max(np.max(np.abs(psi0)), 1e-300)
    if edge > BOUNDARY_TOLERANCE:
        print("  WARNING: Psi(x,0) does not vanish at the walls; the series (2.39) will show"
              " Gibbs ringing there (that's physical: such a state isn't really allowed).")

def stationary_states(x, a, N):
    """
    Stationary states of the infinite square well:

        psi_n(x) = sqrt(2/a) * sin(n pi x / a),   n = 1, ..., N

    x : 1D array of grid points
    a : width of the well
    N : number of states

    Returns an array of shape (N, len(x)), row n-1 holds psi_n(x).
    """
    n = np.arange(1, N + 1)[:, None]
    return np.sqrt(2 / a) * np.sin(n * np.pi * x[None, :] / a)

def energies(N, a, m, hbar):
    """
    Allowed energies E_n = n^2 pi^2 hbar^2 / (2 m a^2),   n = 1, ..., N

    N    : number of states
    a    : width of the well
    m    : particle mass
    hbar : reduced Planck constant

    Returns a 1D array of E_n.
    """
    n = np.arange(1, N + 1)
    return n ** 2 * np.pi ** 2 * hbar ** 2 / (2 * m * a ** 2)

def coefficients(psi0, x, psi_n):
    """
    Expansion coefficients by Fourier's trick (eq. 2.40):

        c_n = integral_0^a psi_n(x)* Psi(x,0) dx

    (the psi_n are real, so no complex conjugate is needed)

    psi0  : normalised Psi(x,0) on the x grid
    x     : 1D array of grid points
    psi_n : stationary states from stationary_states

    Returns a 1D complex array of c_n.
    """
    return _integrate(psi_n * psi0[None, :], x)

def print_coefficients(c, E):
    """
    Print the first few c_n and the checks sum |c_n|^2 = 1 and
    <H> = sum |c_n|^2 E_n.

    c : expansion coefficients c_n
    E : energies E_n
    """
    N = len(c)
    print('\n--- Expansion coefficients c_n (eq. 2.40) ---')
    for k in range(min(N_COEFFS_PRINTED, N)):
        ck = c[k]
        ck_str = f'{ck.real:+.6f}' if abs(ck.imag) < 1e-10 else f'{ck.real:+.6f} {ck.imag:+.6f}i'
        print(f'  c_{k + 1:<3d} = {ck_str:<26s} |c|^2 = {abs(ck) ** 2:.6f}')
    print(f'  sum |c_n|^2 (n <= {N}) = {np.sum(np.abs(c) ** 2):.8f}   (should be 1)')
    expH = np.sum(np.abs(c) ** 2 * E)
    print(f'  <H> = sum |c_n|^2 E_n = {expH:.6g} J = {expH / EV:.6g} eV')

def revival_time(a, m, hbar):
    """
    Revival time T = 4 m a^2 / (pi hbar). After T, every phase factor in
    (2.39) has gone round a whole number of times, so Psi(x,T) = Psi(x,0).

    a    : width of the well
    m    : particle mass
    hbar : reduced Planck constant

    Returns T.
    """
    return 4 * m * a ** 2 / (np.pi * hbar)

def psi_at_time(t, c, psi_n, E, hbar):
    """
    Psi(x,t) from eq. 2.39:

        Psi(x,t) = sum_n c_n psi_n(x) exp(-i E_n t / hbar)

    t     : time
    c     : expansion coefficients c_n
    psi_n : stationary states from stationary_states
    E     : energies E_n
    hbar  : reduced Planck constant

    Returns Psi(x,t) on the x grid (complex array).
    """
    return (c * np.exp(-1j * E * t / hbar)) @ psi_n

def animate(x, psi0, c, psi_n, E, hbar, a, m, title_expr, A_text, speed, frames=None):
    """
    Build the animation window:

    - top panel    : Re Psi, Im Psi and +-|Psi|
    - middle panel : |Psi|^2 with <x>, and |Psi(x,0)|^2 for reference
    - bottom-left  : |c_n|^2 bar chart (probability of measuring E_n)
    - bottom-right : info box (A, sum |c_n|^2, <H>, E_1, revival time)

    x          : 1D array of grid points
    psi0       : normalised Psi(x,0) on the x grid
    c          : expansion coefficients c_n
    psi_n      : stationary states from stationary_states
    E          : energies E_n
    hbar, a, m : physical constants
    title_expr : Psi(x,0) as a (LaTeX) string for the figure title
    A_text     : label for the normalisation constant
    speed      : animation speed multiplier
    frames     : number of frames (None = run forever, used when showing live)

    Returns (fig, anim).
    """
    N = len(c)
    T_rev = revival_time(a, m, hbar)

    prob = np.abs(c) ** 2
    expH = np.sum(prob * E)
    psi_rec = psi_at_time(0.0, c, psi_n, E, hbar)  # Psi(x,0) rebuilt from the series

    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 2.2, 1.4], width_ratios=[3, 1.4],
                          hspace=0.45, wspace=0.25)
    ax_psi = fig.add_subplot(gs[0, :])
    ax_rho = fig.add_subplot(gs[1, :], sharex=ax_psi)
    ax_c = fig.add_subplot(gs[2, 0])
    ax_info = fig.add_subplot(gs[2, 1])
    ax_info.axis('off')

    fig.suptitle(r'Infinite square well:  $\Psi(x,0) = $ ' + title_expr, fontsize=12)

    # Wave function panel
    ymax = 1.15 * max(np.max(np.abs(psi0)), np.max(np.abs(psi_rec)))
    ax_psi.set_xlim(0, a)
    ax_psi.set_ylim(-ymax, ymax)
    ax_psi.axhline(0, color='k', lw=0.6)
    for xw in (0, a):
        ax_psi.axvline(xw, color='k', lw=3)
        ax_rho.axvline(xw, color='k', lw=3)
    line_re, = ax_psi.plot([], [], color=COLOUR_RE, lw=1.6, label=r'Re $\Psi$')
    line_im, = ax_psi.plot([], [], color=COLOUR_IM, lw=1.6, label=r'Im $\Psi$')
    line_abs, = ax_psi.plot([], [], color=COLOUR_ABS, lw=1.0, ls='--', label=r'$\pm|\Psi|$')
    line_nabs, = ax_psi.plot([], [], color=COLOUR_ABS, lw=1.0, ls='--')
    ax_psi.set_ylabel(r'$\Psi(x,t)$ (m$^{-1/2}$)')
    ax_psi.legend(loc='upper right', fontsize=9, ncol=3)
    time_txt = ax_psi.text(0.01, 0.92, '', transform=ax_psi.transAxes, fontsize=11,
                           bbox=dict(fc='white', ec='0.7', alpha=0.85))

    # Probability density panel
    rho_max = 1.15 * max(np.max(np.abs(psi0) ** 2), np.max(np.abs(psi_rec) ** 2))
    ax_rho.set_ylim(0, rho_max)
    ax_rho.plot(x, np.abs(psi0) ** 2, color=COLOUR_RHO_0, lw=1, ls=':', label=r'$|\Psi(x,0)|^2$')
    line_rho, = ax_rho.plot([], [], color=COLOUR_RHO, lw=1.8, label=r'$|\Psi(x,t)|^2$')
    fill = [ax_rho.fill_between(x, 0, 0, color=COLOUR_RHO, alpha=0.25)]
    exp_x_line = ax_rho.axvline(a / 2, color=COLOUR_EXP_X, lw=1.2, ls='--', label=r'$\langle x\rangle$')
    ax_rho.set_xlabel(r'$x$ (m)')
    ax_rho.set_ylabel(r'$|\Psi(x,t)|^2$ (m$^{-1}$)')
    ax_rho.legend(loc='upper right', fontsize=9, ncol=3)

    # |c_n|^2 bar chart: show up to where 99.99% of the probability is covered
    n_show = int(min(N, max(10, np.searchsorted(np.cumsum(prob), 0.9999) + 3)))
    ax_c.bar(np.arange(1, n_show + 1), prob[:n_show], color=COLOUR_CN)
    ax_c.set_xlabel(r'$n$')
    ax_c.set_ylabel(r'$|c_n|^2$')
    ax_c.set_title('probability of measuring $E_n$', fontsize=10)
    ax_c.set_xlim(0.3, n_show + 0.7)

    # Info box
    info = (f'{A_text}\n'
            rf'$\sum |c_n|^2$ = {np.sum(prob):.6f}' + f'  (N = {N})\n'
            rf'$\langle H\rangle$ = {expH:.4g} J = {expH / EV:.4g} eV' + '\n'
            rf'$E_1 = \pi^2\hbar^2/2ma^2$ = {E[0]:.4g} J = {E[0] / EV:.4g} eV' + '\n'
            rf'revival time $T = 4ma^2/\pi\hbar$ = {T_rev:.4g} s' + '\n\n'
            'SPACE pause   ↑/↓ speed   R restart')
    ax_info.text(0, 1, info, va='top', fontsize=9.5, family='monospace')

    state = {'t': 0.0, 'running': True, 'dt': speed * T_rev / FRAMES_PER_REVIVAL}

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
        t = state['t']
        psi = psi_at_time(t, c, psi_n, E, hbar)
        rho = np.abs(psi) ** 2
        line_re.set_data(x, psi.real)
        line_im.set_data(x, psi.imag)
        line_abs.set_data(x, np.abs(psi))
        line_nabs.set_data(x, -np.abs(psi))
        line_rho.set_data(x, rho)
        fill[0].remove()
        fill[0] = ax_rho.fill_between(x, 0, rho, color=COLOUR_RHO, alpha=0.25)
        ex = _integrate(x * rho, x)
        exp_x_line.set_xdata([ex, ex])
        time_txt.set_text(f't = {t:.4e} s   ( t/T = {t / T_rev:.4f} )')
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
    read Psi(x,0) -> find A by normalisation -> compute c_n (eq. 2.40)
    -> build Psi(x,t) (eq. 2.39) -> animate (or save the animation).
    """
    p = argparse.ArgumentParser(description='Infinite square well time-evolution simulator',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=EXAMPLES)
    p.add_argument('--psi0', type=str, default=None, help='initial wave function Psi(x,0)')
    p.add_argument('--a', type=float, default=WELL_WIDTH_DEFAULT, help='width of the well in m (default 1e-9)')
    p.add_argument('--m', type=float, default=MASS_DEFAULT, help='particle mass in kg (default: electron)')
    p.add_argument('--N', type=int, default=N_STATES_DEFAULT, help='number of stationary states in the sum (default 200)')
    p.add_argument('--nx', type=int, default=N_X_DEFAULT, help='number of grid points in x (default 2000)')
    p.add_argument('--speed', type=float, default=SPEED_DEFAULT, help='animation speed multiplier (default 1)')
    p.add_argument('--save', type=str, default=None,
                   help='save the animation instead of showing it (e.g. well.gif or well.mp4)')
    p.add_argument('--frames', type=int, default=SAVE_FRAMES_DEFAULT, help='frames to write when using --save')
    args = p.parse_args()

    expr_str = args.psi0
    if expr_str is None:
        print(EXAMPLES)
        expr_str = input('\nPsi(x,0) = ').strip() or DEFAULT_PSI0

    a, m, hbar, N = args.a, args.m, HBAR, args.N
    x = np.linspace(0, a, args.nx)

    # Step 1: parse Psi(x,0)
    try:
        f_np, sym = parse_psi0(expr_str, a)
        f_vals = f_np(x)
    except Exception as err:
        sys.exit(f'Could not understand Psi(x,0) = {expr_str!r}:\n  {err}')

    # Step 2: normalise (find A)
    A_val, A_text = normalise(f_np, sym, a)
    psi0 = A_val * f_vals
    check_boundaries(psi0)

    # Step 3: coefficients c_n (eq. 2.40)
    psi_n = stationary_states(x, a, N)
    c = coefficients(psi0, x, psi_n)
    E = energies(N, a, m, hbar)
    print_coefficients(c, E)

    # Step 4: animate Psi(x,t) (eq. 2.39)
    if sp is not None:
        title_expr = '$' + sp.latex(sp.sympify(expr_str, locals={'I': sp.I, 'pi': sp.pi,
                                                                 'Piecewise': sp.Piecewise})) + '$'
    else:
        title_expr = expr_str

    if args.save:
        plt.switch_backend('Agg')  # no window needed when only saving

    fig, anim = animate(x, psi0, c, psi_n, E, hbar, a, m, title_expr, A_text, args.speed,
                        frames=args.frames if args.save else None)

    if args.save:
        writer = PillowWriter(fps=SAVE_FPS) if args.save.lower().endswith('.gif') else FFMpegWriter(fps=SAVE_FPS)
        print(f'\nSaving {args.frames} frames to {args.save} ...')
        anim.save(args.save, writer=writer)
        print('done.')
    else:
        plt.show()  # blocks until the figure window is closed

if __name__ == '__main__':
    main()
