import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import streamlit as st
from scipy.special import jn, jnp_zeros

st.set_page_config(page_title="Chladni Patterns Generator", layout="wide")
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "serif"


# ---------- Material / plate physics ----------
# Kirchhoff–Love thin plate, simply-supported edges. Frequencies follow
# f ∝ k² · √(D/(ρh)), where D = E h³ / (12(1 − ν²)).

MATERIALS = {
    "Aluminum": {"E": 69e9,  "nu": 0.33, "rho": 2700.0},
    "Steel":    {"E": 200e9, "nu": 0.30, "rho": 7850.0},
    "Brass":    {"E": 100e9, "nu": 0.34, "rho": 8500.0},
}


def flexural_rigidity(E, nu, h):
    return E * h**3 / (12.0 * (1.0 - nu**2))


def plate_speed(E, nu, rho, h):
    """sqrt(D/(ρh)) — the prefactor that scales every plate frequency."""
    return np.sqrt(flexural_rigidity(E, nu, h) / (rho * h))


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
def all_valid_S(max_S=500):
    return [k for k in range(1, max_S + 1)
            if any(n * n + m * m == k
                   for n in range(int(np.sqrt(k)) + 1)
                   for m in range(int(np.sqrt(k)) + 1))]


def square_freq(E, nu, rho, h, a, S):
    # Biharmonic eigenvalue k² = (π/(2a))²·S; f = (k²/2π)·√(D/ρh) = (πS)/(8a²)·√(D/ρh)
    return (np.pi * S) / (8.0 * a ** 2) * plate_speed(E, nu, rho, h)


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


def polar_freq(E, nu, rho, h, a, z):
    # Biharmonic: f = z²/(2π·a²)·√(D/ρh)
    return (z ** 2) / (2.0 * np.pi * a ** 2) * plate_speed(E, nu, rho, h)


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

def render_one_square(pairs, signs, *, plate_color, contour_color, thickness,
                      delta=0.005, figsize=(6, 6), bg_color=None):
    xrange = np.arange(-1.0, 1.0, delta)
    yrange = np.arange(-1.0, 1.0, delta)
    x, y = np.meshgrid(xrange, yrange)
    eq = u_square(x, y, pairs, signs)

    fig, ax = plt.subplots(figsize=figsize)
    if bg_color is None:
        fig.patch.set_alpha(0)
    else:
        fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(plate_color)
    ax.contour(x, y, eq, levels=[0], colors=contour_color, linewidths=thickness)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def render_one_polar(n, m, *, plate_color, contour_color, thickness, figsize=(6, 6),
                     bg_color=None):
    T, R, Z = polar_field(n, m)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    if bg_color is None:
        fig.patch.set_alpha(0)
    else:
        fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(plate_color)
    ax.contour(T, R, Z, levels=[0], colors=contour_color, linewidths=thickness)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


# ---------- List-view helpers ----------

DEFAULT_FREQ_LIST = (
    "174, 285, 396, 417, 432, 473, 528, 639, 697, 741, 852, 963,\n"
    "1053, 1176, 1500, 1820, 2222, 2473, 3333, 3370, 3975, 4254, 4444"
)


def parse_freq_list(text):
    out, seen = [], set()
    for tok in text.replace("\n", ",").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(round(float(tok)))
        except ValueError:
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def square_snap(freq, E, nu, rho, h, a, valid_S):
    cp = plate_speed(E, nu, rho, h)
    target_S = 8.0 * a ** 2 * freq / (np.pi * cp)
    S = min(valid_S, key=lambda s: abs(s - target_S))
    return S, square_freq(E, nu, rho, h, a, S)


def polar_snap(freq, E, nu, rho, h, a, nontrivial):
    cp = plate_speed(E, nu, rho, h)
    target_z = float(np.sqrt(max(2.0 * np.pi * a ** 2 * freq / cp, 0.0)))
    n, m, z = min(nontrivial, key=lambda r: abs(r[2] - target_z))
    return n, m, z, polar_freq(E, nu, rho, h, a, z)


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

OOB_BG = "#ffd6d6"

st.title("Chladni Patterns Generator")
st.caption("Thin-plate (Kirchhoff–Love) biharmonic model, simply-supported edges.")

view_mode = st.sidebar.radio(
    "View", ["Single frequency", "Frequency list", "Parameter grid"],
)

# Slots reserved so the single-mode Hz controls render above the inputs that
# determine their range. Stay empty in other modes.
freq_slot = st.sidebar.empty()
exact_slot = st.sidebar.empty()

if view_mode == "Frequency list":
    freq_text = st.sidebar.text_area(
        "Frequencies (Hz)", value=DEFAULT_FREQ_LIST, height=180,
        help="Comma- or newline-separated integers.",
    )
    n_cols = st.sidebar.slider("Grid columns", 2, 6, 4)
elif view_mode == "Parameter grid":
    grid_freq = st.sidebar.number_input(
        "Target frequency (Hz)", min_value=1, max_value=20000,
        value=528, step=1,
    )
    h_count = st.sidebar.slider("h cells", 2, 9, 5)
    h_step_mm = st.sidebar.number_input(
        "Δh per cell (mm)", min_value=0.01, max_value=5.0,
        value=0.1, step=0.01, format="%.2f",
    )
    a_count = st.sidebar.slider("a cells", 2, 9, 5)
    a_step = st.sidebar.number_input(
        "Δa per cell (m)", min_value=0.001, max_value=0.5,
        value=0.005, step=0.001, format="%.3f",
    )

shape = st.sidebar.radio("Plate shape", ["Square", "Circular"])
plate_color_name = st.sidebar.radio(
    "Plate color", COLOR_NAMES, index=0, horizontal=True,
)
contour_color_name = st.sidebar.radio(
    "Contour color", COLOR_NAMES, index=1, horizontal=True,
)
thickness = st.sidebar.slider("Line thickness", 0.5, 3.0, 1.5, 0.1)

with st.sidebar.expander("Advanced", expanded=False):
    material_name = st.selectbox(
        "Material", list(MATERIALS.keys()) + ["Custom"], index=0,
    )
    if material_name == "Custom":
        E = st.number_input("Young's modulus E (GPa)",
                            min_value=1.0, max_value=1000.0,
                            value=69.0, step=1.0) * 1e9
        nu = st.number_input("Poisson's ratio ν",
                             min_value=0.0, max_value=0.5,
                             value=0.33, step=0.01)
        rho = st.number_input("Density ρ (kg/m³)",
                              min_value=100.0, max_value=20000.0,
                              value=2700.0, step=100.0)
    else:
        preset = MATERIALS[material_name]
        E, nu, rho = preset["E"], preset["nu"], preset["rho"]
        st.caption(f"E = {E/1e9:.0f} GPa,  ν = {nu},  ρ = {rho:.0f} kg/m³")
    h = st.number_input("Thickness h (mm)", min_value=0.1, max_value=20.0,
                        value=1.0, step=0.1) / 1000.0
    a_label = "Half-side a (m)" if shape == "Square" else "Radius a (m)"
    a = st.number_input(a_label, min_value=0.01, max_value=2.0,
                        value=0.1, step=0.01, format="%.3f")
    variant_slot = st.empty()

plate_color = COLOR_OPTIONS[plate_color_name]
contour_color = COLOR_OPTIONS[contour_color_name]

if shape == "Square":
    valid_S = all_valid_S(500)
    f_min = square_freq(E, nu, rho, h, a, valid_S[0])
    f_max = square_freq(E, nu, rho, h, a, valid_S[-1])
else:
    table = polar_mode_table()
    nontrivial = [row for row in table if row[2] > 0]
    f_min = polar_freq(E, nu, rho, h, a, nontrivial[0][2])
    f_max = polar_freq(E, nu, rho, h, a, table[-1][2])

fmin_i = int(np.floor(f_min))
fmax_i = max(int(np.ceil(f_max)), fmin_i + 1)


if view_mode == "Single frequency":
    # Resonant frequencies (integer Hz) for the current plate — used by the
    # number_input's +/- buttons to step to the next or previous mode.
    if shape == "Square":
        mode_hz = sorted({int(round(square_freq(E, nu, rho, h, a, S))) for S in valid_S})
    else:
        mode_hz = sorted({int(round(polar_freq(E, nu, rho, h, a, z))) for _, _, z in nontrivial})
    mode_hz = [m for m in mode_hz if fmin_i <= m <= fmax_i]
    st.session_state._mode_hz = mode_hz

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
            nxt = [f for f in modes if f > ref]
            target = nxt[0] if nxt else modes[-1]
            st.session_state._freq_input = target
            st.session_state._freq_slider = target
        elif modes and new_val == ref - 1:
            prv = [f for f in modes if f < ref]
            target = prv[-1] if prv else modes[0]
            st.session_state._freq_input = target
            st.session_state._freq_slider = target
        else:
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
        S, actual_freq = square_snap(freq, E, nu, rho, h, a, valid_S)
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
        plt.close(fig)

        c1, c2, c3 = st.columns(3)
        c1.metric("Resonant freq", f"{actual_freq:.1f} Hz")
        c2.metric("Mode S = n²+m²", str(S))
        c3.metric("Variant", f"{variant_idx + 1} / {len(variants)}")
        st.caption(f"(n,m) pairs: `{pairs}` · signs: `{signs}`")
        st.caption(r"$f = \dfrac{\pi (n^2+m^2)}{8 a^2}\,\sqrt{D/(\rho h)},\quad D = \dfrac{E h^3}{12(1-\nu^2)}$")

    else:
        n, m, z, actual_freq = polar_snap(freq, E, nu, rho, h, a, nontrivial)
        fig = render_one_polar(n, m, plate_color=plate_color,
                               contour_color=contour_color, thickness=thickness)
        st.pyplot(fig)
        plt.close(fig)

        c1, c2, c3 = st.columns(3)
        c1.metric("Resonant freq", f"{actual_freq:.1f} Hz")
        c2.metric("(n, m)", f"({n}, {m})")
        c3.metric("Bessel zero z_{n,m}", f"{z:.3f}")
        st.caption(r"$f = \dfrac{z_{n,m}^2}{2\pi a^2}\,\sqrt{D/(\rho h)}$")

elif view_mode == "Frequency list":
    # ---------- Frequency list view ----------
    freqs = parse_freq_list(freq_text)
    if not freqs:
        st.info("Enter at least one frequency in the sidebar.")
    else:
        oob_set = {f for f in freqs if f < fmin_i or f > fmax_i}
        if oob_set:
            st.markdown(
                f"<div style='background:#ffd6d6;border:2px solid #d33;"
                f"border-radius:8px;padding:1rem 1.25rem;margin:0.5rem 0 1rem;"
                f"font-size:1.4rem;font-weight:600;color:#7a0000;'>"
                f"<span style='font-size:2rem;'>🚨</span>&nbsp;"
                f"{len(oob_set)} value(s) outside the resonant range "
                f"[{fmin_i}, {fmax_i}] Hz will snap to the nearest available mode."
                f"&nbsp;<span style='font-size:2rem;'>🚨</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        for row_start in range(0, len(freqs), n_cols):
            row = st.columns(n_cols)
            for col, target in zip(row, freqs[row_start:row_start + n_cols]):
                with col:
                    is_oob = target in oob_set
                    bg = OOB_BG if is_oob else None
                    if shape == "Square":
                        S, actual_freq = square_snap(target, E, nu, rho, h, a, valid_S)
                        variants = square_variants_for_S(S)
                        key = f"vidx_sq_{target}"
                        idx = st.session_state.get(key, 0) % max(1, len(variants))
                        pairs, signs = variants[idx]
                        fig = render_one_square(
                            pairs, signs,
                            plate_color=plate_color, contour_color=contour_color,
                            thickness=thickness, figsize=(3, 3), bg_color=bg,
                        )
                        sub = f"S={S} · {idx + 1}/{len(variants)}"
                        n_variants = len(variants)
                    else:
                        n, m, z, actual_freq = polar_snap(target, E, nu, rho, h, a, nontrivial)
                        key = f"vidx_pol_{n}_{m}"
                        fig = render_one_polar(
                            n, m,
                            plate_color=plate_color, contour_color=contour_color,
                            thickness=thickness, figsize=(3, 3), bg_color=bg,
                        )
                        sub = f"(n,m)=({n},{m})"
                        n_variants = 1

                    label = f"**{target} Hz** → {actual_freq:.0f} Hz · {sub}"
                    if is_oob:
                        label = f"🚨 {label}"
                    st.markdown(label)
                    st.pyplot(fig)
                    plt.close(fig)

                    if n_variants > 1:
                        bcols = st.columns(2)
                        if bcols[0].button("◀", key=f"prev_{key}_{target}",
                                           use_container_width=True):
                            st.session_state[key] = (idx - 1) % n_variants
                            st.rerun()
                        if bcols[1].button("▶", key=f"next_{key}_{target}",
                                           use_container_width=True):
                            st.session_state[key] = (idx + 1) % n_variants
                            st.rerun()

else:
    # ---------- Parameter grid view ----------
    # Sweep (h, a) over a cartesian product centered on the Advanced inputs,
    # rendering the snapped pattern at `grid_freq` for every (h, a) cell.
    h_step = h_step_mm / 1000.0
    h_offsets = (np.arange(h_count) - (h_count - 1) / 2.0) * h_step
    a_offsets = (np.arange(a_count) - (a_count - 1) / 2.0) * a_step
    h_values = [round(float(h + x), 6) for x in h_offsets if 0.0001 <= h + x <= 0.02]
    a_values = [round(float(a + x), 3) for x in a_offsets if 0.01 <= a + x <= 2.0]

    if not h_values or not a_values:
        st.info("Sweep stepped outside the allowed range — reduce Δh or Δa.")
    else:
        st.markdown(
            f"Target **{grid_freq} Hz** · {material_name} · "
            f"center h = {h*1000:.2f} mm, a = {a:.3f} m · "
            f"sweeping {len(h_values)} × {len(a_values)} = "
            f"{len(h_values) * len(a_values)} cells"
        )

        if shape == "Circular":
            z_max = nontrivial[-1][2]
            z_min = nontrivial[0][2]

        for a_val in a_values:
            row = st.columns(len(h_values))
            for col, h_val in zip(row, h_values):
                with col:
                    if shape == "Square":
                        f_min_cell = square_freq(E, nu, rho, h_val, a_val, valid_S[0])
                        f_max_cell = square_freq(E, nu, rho, h_val, a_val, valid_S[-1])
                        is_oob = grid_freq < f_min_cell or grid_freq > f_max_cell
                        bg = OOB_BG if is_oob else None
                        S, actual = square_snap(grid_freq, E, nu, rho, h_val, a_val, valid_S)
                        pairs, signs = square_variants_for_S(S)[0]
                        fig = render_one_square(
                            pairs, signs,
                            plate_color=plate_color, contour_color=contour_color,
                            thickness=thickness, figsize=(2.4, 2.4), bg_color=bg,
                        )
                        sub = f"S={S}"
                    else:
                        f_min_cell = polar_freq(E, nu, rho, h_val, a_val, z_min)
                        f_max_cell = polar_freq(E, nu, rho, h_val, a_val, z_max)
                        is_oob = grid_freq < f_min_cell or grid_freq > f_max_cell
                        bg = OOB_BG if is_oob else None
                        n, m, z, actual = polar_snap(grid_freq, E, nu, rho, h_val, a_val, nontrivial)
                        fig = render_one_polar(
                            n, m,
                            plate_color=plate_color, contour_color=contour_color,
                            thickness=thickness, figsize=(2.4, 2.4), bg_color=bg,
                        )
                        sub = f"({n},{m})"

                    prefix = "🚨 " if is_oob else ""
                    st.markdown(
                        f"{prefix}**h={h_val*1000:.2f} mm, a={a_val:.3f}**  \n"
                        f"{actual:.0f} Hz · {sub}"
                    )
                    st.pyplot(fig)
                    plt.close(fig)
