import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import streamlit as st
from scipy.special import jn, jnp_zeros

st.set_page_config(page_title="Chladni Patterns Generator", layout="centered")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "serif"


# ---------- Cartesian helpers ----------

def X_basis(x, n):
    return np.mod(n + 1, 2) * np.cos(n * np.pi * x / 2.0) + np.mod(n, 2) * np.sin(n * np.pi * x / 2.0)


def u_square(x, y, pairs, signs):
    eq = np.zeros_like(x)
    for (n, m), s in zip(pairs, signs):
        eq += s * X_basis(x, n) * X_basis(y, m)
    return eq


def find_pairs(k):
    return [(n, m) for n in range(int(np.sqrt(k)) + 1)
            for m in range(int(np.sqrt(k)) + 1) if n * n + m * m == k]


def sign_combos(pairs):
    if not pairs:
        return [[]]
    combos = list(itertools.product([1, -1], repeat=len(pairs) - 1))
    return [[1] + list(c) for c in combos]


def square_variants_for_S(S):
    pairs = find_pairs(S)
    if not pairs:
        return []
    variants = [(pairs, signs) for signs in sign_combos(pairs)]
    non_zero = pairs[1:-1]
    if non_zero:
        variants.extend((non_zero, signs) for signs in sign_combos(non_zero))
    return variants


@st.cache_data(show_spinner=False)
def all_valid_S(max_S=300):
    return [k for k in range(1, max_S + 1)
            if any(n * n + m * m == k
                   for n in range(int(np.sqrt(k)) + 1)
                   for m in range(int(np.sqrt(k)) + 1))]


def square_freq(c, a, S):
    # Plate x,y ∈ [-a, a]; eigenvalue (π/(2a))²·(n²+m²); ω = c·k; f = c·√S/(4a)
    return c * np.sqrt(S) / (4 * a)


# ---------- Polar helpers ----------

def bessel_deriv_zero(n, m):
    if n == 0:
        zeros = list(jnp_zeros(0, m + 1))
        zeros.insert(0, 0)
        return zeros[m]
    return jnp_zeros(n, m + 1)[-1]


@st.cache_data(show_spinner=False)
def polar_mode_table(n_max=15, m_max=10):
    out = []
    for n in range(n_max):
        for m in range(m_max):
            z = float(bessel_deriv_zero(n, m))
            out.append((n, m, z))
    return sorted(out, key=lambda x: x[2])


def polar_freq(c, a, z):
    # ω = c·z_nm/a; f = c·z_nm/(2π·a)
    return c * z / (2 * np.pi * a)


def f_polar(r, theta):
    return r * np.cos(theta)


@st.cache_data(show_spinner=False)
def polar_field(n, m, resolution=120):
    a_norm = 1.0
    z = bessel_deriv_zero(n, m)
    num_a = integrate.dblquad(
        lambda theta, r: jn(n, z * r / a_norm) * np.cos(n * theta) * f_polar(r, theta) * r,
        0, a_norm, lambda r: 0, lambda r: 2 * np.pi,
    )[0]
    den = integrate.quad(lambda r: jn(n, z * r / a_norm) ** 2 * r, 0, a_norm)[0]
    A = num_a / (2 * np.pi * den)

    num_b = integrate.dblquad(
        lambda theta, r: jn(n, z * r / a_norm) * np.sin(n * theta) * f_polar(r, theta) * r,
        0, a_norm, lambda r: 0, lambda r: 2 * np.pi,
    )[0]
    B = num_b / (2 * np.pi * den)

    r_grid = np.linspace(0, a_norm, resolution)
    theta_grid = np.linspace(0, 2 * np.pi, resolution)
    R, T = np.meshgrid(r_grid, theta_grid)
    Z = jn(n, z * R / a_norm) * (A * np.cos(n * T) + B * np.sin(n * T))
    return T, R, Z


# ---------- Renderers (one diagram) ----------

def render_one_square(pairs, signs, *, plate_color, contour_color, thickness, delta=0.005):
    xrange = np.arange(-1.0, 1.0, delta)
    yrange = np.arange(-1.0, 1.0, delta)
    x, y = np.meshgrid(xrange, yrange)
    eq = u_square(x, y, pairs, signs)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_alpha(0)  # transparent fig bg — only the square plate is visible
    ax.set_facecolor(plate_color)
    ax.contour(x, y, eq, levels=[0], colors=contour_color, linewidths=thickness)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def render_one_polar(n, m, *, plate_color, contour_color, thickness):
    T, R, Z = polar_field(n, m)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    fig.patch.set_alpha(0)  # transparent fig bg — only the disk is visible
    ax.set_facecolor(plate_color)
    ax.contour(T, R, Z, levels=[0], colors=contour_color, linewidths=thickness)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


# ---------- UI ----------

COLOR_OPTIONS = {
    "Black":  "#000000",
    "White":  "#FFFFFF",
    "Red":    "#E63946",
    "Blue":   "#3B82F6",
    "Green":  "#10B981",
    "Yellow": "#F59E0B",
}
COLOR_NAMES = list(COLOR_OPTIONS)

st.title("Chladni Patterns Generator")
st.caption("Drive the plate at a real frequency — the closest natural mode lights up.")

# Reserve slots at the top of the sidebar so the Hz controls render above the
# inputs that determine their range.
freq_slot = st.sidebar.empty()
exact_slot = st.sidebar.empty()

shape = st.sidebar.radio("Plate shape", ["Square", "Circular"])
plate_color_name = st.sidebar.radio(
    "Plate color", COLOR_NAMES, index=0, horizontal=True,
)
contour_color_name = st.sidebar.radio(
    "Contour color", COLOR_NAMES, index=1, horizontal=True,
)
thickness = st.sidebar.slider("Line thickness", 0.5, 3.0, 1.5, 0.1)

with st.sidebar.expander("Advanced", expanded=False):
    c = st.number_input("Wave speed c (m/s)", min_value=1.0, max_value=10000.0,
                        value=100.0, step=10.0)
    a_label = "Half-side a (m)" if shape == "Square" else "Radius a (m)"
    a = st.number_input(a_label, min_value=0.01, max_value=2.0,
                        value=0.1, step=0.01, format="%.3f")
    variant_slot = st.empty()

plate_color = COLOR_OPTIONS[plate_color_name]
contour_color = COLOR_OPTIONS[contour_color_name]

if shape == "Square":
    valid_S = all_valid_S(300)
    f_min = square_freq(c, a, valid_S[0])
    f_max = square_freq(c, a, valid_S[-1])
else:
    table = polar_mode_table()
    nontrivial = [row for row in table if row[2] > 0]
    f_min = polar_freq(c, a, nontrivial[0][2])
    f_max = polar_freq(c, a, table[-1][2])

fmin_i = int(np.floor(f_min))
fmax_i = max(int(np.ceil(f_max)), fmin_i + 1)

# All resonant frequencies (integer Hz) for the current shape/c/a — used by the
# number_input's +/- buttons to step to the next or previous mode rather than
# moving by 1 Hz (which usually maps to the same mode anyway).
if shape == "Square":
    mode_hz = sorted({int(round(square_freq(c, a, S))) for S in valid_S})
else:
    mode_hz = sorted({int(round(polar_freq(c, a, z))) for _, _, z in nontrivial})
mode_hz = [m for m in mode_hz if fmin_i <= m <= fmax_i]
st.session_state._mode_hz = mode_hz  # exposed to callbacks

# Linked slider + number_input for Hz: editing one updates the other.
DEFAULT_FREQ = 1223
if "_freq_slider" not in st.session_state:
    st.session_state._freq_slider = DEFAULT_FREQ
    st.session_state._freq_input = DEFAULT_FREQ

st.session_state._freq_slider = max(fmin_i, min(fmax_i, int(st.session_state._freq_slider)))
st.session_state._freq_input = max(fmin_i, min(fmax_i, int(st.session_state._freq_input)))


def _sync_from_slider():
    st.session_state._freq_input = st.session_state._freq_slider


def _sync_from_input():
    new_val = int(st.session_state._freq_input)
    ref = int(st.session_state._freq_slider)
    modes = st.session_state.get("_mode_hz", [])
    if modes and new_val == ref + 1:
        # +-button click — jump to the next resonant mode above ref.
        nxt = [f for f in modes if f > ref]
        target = nxt[0] if nxt else modes[-1]
        st.session_state._freq_input = target
        st.session_state._freq_slider = target
    elif modes and new_val == ref - 1:
        # −-button click — jump to the previous resonant mode below ref.
        prv = [f for f in modes if f < ref]
        target = prv[-1] if prv else modes[0]
        st.session_state._freq_input = target
        st.session_state._freq_slider = target
    else:
        # Typed value — take it as-is; the diagram still snaps to the nearest mode.
        st.session_state._freq_slider = new_val


freq_slot.slider(
    "Frequency (Hz)",
    min_value=fmin_i, max_value=fmax_i, step=1,
    key="_freq_slider", on_change=_sync_from_slider,
)
exact_slot.number_input(
    "Or type exact (Hz)",
    min_value=fmin_i, max_value=fmax_i, step=1,
    key="_freq_input", on_change=_sync_from_input,
)
freq = int(st.session_state._freq_slider)

if shape == "Square":
    target_S = (4 * a * freq / c) ** 2
    S = min(valid_S, key=lambda s: abs(s - target_S))
    actual_freq = square_freq(c, a, S)
    variants = square_variants_for_S(S)

    if len(variants) > 1:
        variant_idx = variant_slot.slider(
            f"Degenerate variant (1–{len(variants)})",
            1, len(variants), 1, key=f"variant_S{S}",
        ) - 1
    else:
        variant_idx = 0
    pairs, signs = variants[variant_idx]

    fig = render_one_square(pairs, signs, plate_color=plate_color,
                            contour_color=contour_color, thickness=thickness)
    st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Resonant freq", f"{actual_freq:.1f} Hz")
    c2.metric("Mode S = n²+m²", str(S))
    c3.metric("Variant", f"{variant_idx + 1} / {len(variants)}")
    st.caption(f"(n,m) pairs: `{pairs}` · signs: `{signs}`")
    st.caption(r"$f = \dfrac{c}{4a}\sqrt{n^2+m^2}$")

else:
    target_z = 2 * np.pi * a * freq / c
    n, m, z = min(nontrivial, key=lambda row: abs(row[2] - target_z))
    actual_freq = polar_freq(c, a, z)

    fig = render_one_polar(n, m, plate_color=plate_color,
                           contour_color=contour_color, thickness=thickness)
    st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Resonant freq", f"{actual_freq:.1f} Hz")
    c2.metric("(n, m)", f"({n}, {m})")
    c3.metric("Bessel zero z_{n,m}", f"{z:.3f}")
    st.caption(r"$f = \dfrac{c \, z_{n,m}}{2\pi a}$")
