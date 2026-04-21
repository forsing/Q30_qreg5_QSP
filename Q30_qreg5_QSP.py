#!/usr/bin/env python3
"""
Q30 Kvantna regresija (5/5) — tehnika: Quantum Signal Processing (QSP)
polinomna regresija P(A)·b preko block-encoding + qubitization walk operatora
(čisto kvantno: Chebyshev polinomna spektralna transformacija BEZ QPE, BEZ klasične
regresije, bez hibrida).

Koncept:
  QSP/QSVT (Low & Chuang 2017; Gilyén et al. 2019) omogućava primenu POLINOMNE
  transformacije P(A) na b BEZ Quantum Phase Estimation-a (QPE). Ideja:
    1) Block-encoding: konstruiši unitarno U_A tako da je gornji-levi 2×2-blok
          (⟨0|_a ⊗ I) U_A (|0⟩_a ⊗ I) = A.
       Za Hermitsku A sa |λ_i| ≤ 1:   U_A = [[A, √(I−A²)], [√(I−A²), −A]].
    2) Qubitization walk: W = (Z_a ⊗ I) · U_A. Tada:
          (⟨0|_a ⊗ I) W^d (|0⟩_a ⊗ I) = T_d(A)
       gde je T_d d-ti Chebyshev polinom prve vrste. Dokaz indukcijom:
          W^1 anc=0-blok = A (prvi red), T_1(x) = x. ✓
          W^2 anc=0-blok = 2A² − I = T_2(A). ✓  (videti recurrenciju T_{d+1}=2x·T_d−T_{d−1})
    3) Između primena U_A može da se umetne Rz(φ_k) na ancilla — QSP phase sequence
       {φ_k} parametrizuje PROIZVOLJAN polinom P(x) degree ≤ d. Za φ_k = 0 dobijamo
       čist T_d(A) (Chebyshev regresija).

  Regresija: P(A) · b u eigenbazi A daje Σ_i P(λ_i) · ⟨v_i|b⟩ · |v_i⟩ — polinomna
  emphasis/filtracija eigenvalue spektra. Različiti P daju različite regresione
  modele (linearni, kvadratni, kubni, ...) bez QPE overhead-a.

Razlika u odnosu na Q26 (HHL), Q27 (QPE regresija), Q29 (QSVE):
  Q26 (HHL):    QPE + 1/λ rotacija             → A⁻¹·b.              nq_b + n_phase + 1 qubit.
  Q27 (QPE):    QPE + λ^β rotacija             → A^β·b.             nq_b + n_phase + 1 qubit.
  Q29 (QSVE):   dilacija + QPE + odd rotacija  → A⁺·b (Tikhonov).   2 + nq_b + n_phase qubit.
  Q30 (QSP):    block-encoding + walk, BEZ QPE → T_d(A)·b (Chebyshev). 1 + nq_b qubit.

  QSP je algoritamski najefikasniji (NEMA phase-registra!) — nasleđuje samo 1 qubit
  za block-encoding ancilla. Polinom se bira PRE izvršavanja preko phase-sequence
  (umesto preko spektra-estimacije u QPE-baziranim algoritmima).

Matrica A (ista kao u Q26/Q27 radi konzistentnosti — Hermitska, spektar u [−1, 1]):
  A = (P + diag(f) + α·I) / scale, scale biran tako da |λ_i| ≤ 0.9
  (margin od 0.1 stabilizuje √(I−A²) blok-encoding-a).

Sve deterministički: seed=39; A, b iz CELOG CSV-a (pravilo 10).
Deterministička grid-optimizacija (nq_b, d, alpha, phase_profile) po cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

NQ_B = 6
GRID_D = (2, 3, 4, 5)
GRID_ALPHA = (0.5, 1.0)
PHASE_PROFILES = ("zero", "pi8_alt")
BE_EIG_MARGIN = 0.9


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


# =========================
# Hermitska matrica A (CELI CSV) — ista struktura kao Q26/Q27
# =========================
def pair_matrix(H: np.ndarray) -> np.ndarray:
    P = np.zeros((N_MAX, N_MAX), dtype=np.float64)
    for row in H:
        for a in row:
            for b in row:
                if a != b and 1 <= a <= N_MAX and 1 <= b <= N_MAX:
                    P[a - 1, b - 1] += 1.0
    return P


def build_matrix_A(H: np.ndarray, nq: int, alpha: float) -> Tuple[np.ndarray, float]:
    P = pair_matrix(H)
    f = freq_vector(H)
    f_mean = float(f.mean()) + 1e-18
    P_n = P / f_mean
    D = np.diag(f / f_mean)
    A = P_n + D + float(alpha) * np.eye(N_MAX)

    dim = 2 ** nq
    A_pad = float(alpha) * np.eye(dim, dtype=np.float64)
    A_pad[:N_MAX, :N_MAX] = A

    eigs = np.linalg.eigvalsh(A_pad)
    max_abs = float(max(abs(eigs.max()), abs(eigs.min())))
    scale = max_abs / BE_EIG_MARGIN if max_abs > 1e-18 else 1.0
    A_scaled = A_pad / scale
    return A_scaled, scale


# =========================
# Block-encoding: U_A = [[A, √(I−A²)], [√(I−A²), −A]]   (dim 2·2^nq × 2·2^nq)
# Hermitska A, |λ_i| ≤ BE_EIG_MARGIN < 1 ⇒ I − A² je PSD ⇒ √ dobro definisan.
# =========================
def block_encode_hermitian(A: np.ndarray) -> np.ndarray:
    eigs, V = np.linalg.eigh(A)
    one_minus_sq = 1.0 - eigs ** 2
    one_minus_sq = np.clip(one_minus_sq, 0.0, None)
    sqrt_diag = np.sqrt(one_minus_sq)
    B = V @ np.diag(sqrt_diag) @ V.conj().T
    dim = A.shape[0]
    U = np.zeros((2 * dim, 2 * dim), dtype=np.complex128)
    U[:dim, :dim] = A
    U[:dim, dim:] = B
    U[dim:, :dim] = B
    U[dim:, dim:] = -A
    return U


# =========================
# QSP phase-sequence (parametrizuje polinom P)
# =========================
def phase_sequence(profile: str, d: int) -> List[float]:
    if profile == "zero":
        return [0.0] * max(0, d - 1)
    if profile == "pi8_alt":
        # alternirajuća mala rotacija — stabilna QSP perturbacija oko Chebyshev T_d
        base = float(np.pi / 8.0)
        return [base * ((-1) ** k) for k in range(max(0, d - 1))]
    return [0.0] * max(0, d - 1)


# =========================
# QSP kolo:
#   Registri: anc_be (1), b (nq_b). Ukupno 1 + nq_b qubit-a (BEZ phase-registra!).
#   Walk: W = (Z_a ⊗ I) · U_A.  Sekvenca: U_A (Rz(φ_k) Z_a U_A)^{d−1}.
#   Post-select anc_be=0 → b-registar drži (aproksimativno) P(A)·|b⟩, gde je
#   P = T_d za profile="zero" ili mala perturbacija oko T_d za ostale profile.
# =========================
def build_qsp_circuit(
    A_scaled: np.ndarray, b_amp: np.ndarray, n_b: int, d: int, phases: List[float]
) -> QuantumCircuit:
    anc = QuantumRegister(1, name="a")
    b_reg = QuantumRegister(n_b, name="b")
    qc = QuantumCircuit(anc, b_reg)

    qc.append(StatePreparation(b_amp.tolist()), b_reg)

    U_A = block_encode_hermitian(A_scaled)
    UA_gate = UnitaryGate(U_A, label="U_A")

    for k in range(d):
        qc.append(UA_gate, list(anc) + list(b_reg))
        if k < d - 1:
            phi_k = float(phases[k]) if k < len(phases) else 0.0
            if abs(phi_k) > 1e-14:
                qc.rz(2.0 * phi_k, anc[0])
            qc.z(anc[0])

    return qc


def qsp_state_probs(
    H: np.ndarray, n_b: int, d: int, alpha: float, profile: str
) -> Tuple[np.ndarray, float]:
    A_scaled, _scale = build_matrix_A(H, n_b, alpha)
    b_amp = amp_from_freq(freq_vector(H), n_b)
    phases = phase_sequence(profile, d)
    qc = build_qsp_circuit(A_scaled, b_amp, n_b, d, phases)

    sv = Statevector(qc)
    p = np.abs(sv.data) ** 2

    dim_b = 2 ** n_b
    # Qiskit little-endian: qubit 0 = anc_be (LSB), qubit 1..n_b = b.
    # Reshape (MSB → LSB): (b, anc).
    mat = p.reshape(dim_b, 2)
    p_b = mat[:, 0]
    p_post = float(p_b.sum())
    if p_post < 1e-18:
        return np.zeros(dim_b, dtype=np.float64), 0.0
    return p_b / p_post, p_post


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija (d, alpha, phase_profile)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for d in GRID_D:
        for alpha in GRID_ALPHA:
            for profile in PHASE_PROFILES:
                try:
                    p_b, p_post = qsp_state_probs(
                        H, NQ_B, int(d), float(alpha), str(profile)
                    )
                    bi = bias_39(p_b)
                    score = cosine(bi, f_csv_n)
                except Exception:
                    continue
                key = (score, int(d), -float(alpha), str(profile))
                if best is None or key > best[0]:
                    best = (
                        key,
                        dict(
                            d=int(d),
                            alpha=float(alpha),
                            profile=str(profile),
                            score=float(score),
                            p_post=float(p_post),
                        ),
                    )
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q30 Kvantna regresija (5/5) — QSP Chebyshev polinomna (T_d(A)·b): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED, "| nq_b:", NQ_B, "| BE margin:", BE_EIG_MARGIN)

    A_dbg, _sc = build_matrix_A(H, NQ_B, float(GRID_ALPHA[0]))
    eigs_dbg = np.linalg.eigvalsh(A_dbg)
    print("--- A (skalirana, |λ_i| ≤ BE_EIG_MARGIN) eigenvalue info ---")
    print(
        f"  min(λ)={float(eigs_dbg.min()):.6f}  max(λ)={float(eigs_dbg.max()):.6f}  "
        f"max|λ|={float(np.max(np.abs(eigs_dbg))):.6f}"
    )

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "d (polinom degree)=", best["d"],
        "| α (ridge):", best["alpha"],
        "| profile:", best["profile"],
        "| P(anc=0):", round(float(best["p_post"]), 6),
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX

    print("--- demonstracija efekta degree d (Chebyshev T_d emphasis) za profile=zero, α=", best["alpha"], "---")
    for d_demo in GRID_D:
        p_d, p_post_d = qsp_state_probs(H, NQ_B, int(d_demo), float(best["alpha"]), "zero")
        pred_d = pick_next_combination(p_d)
        cos_d = cosine(bias_39(p_d), f_csv_n)
        print(f"  d={d_demo:d}  P(anc=0)={p_post_d:.6f}  cos={cos_d:.6f}  NEXT={pred_d}")

    p_b, _p_post = qsp_state_probs(
        H, NQ_B, int(best["d"]), float(best["alpha"]), str(best["profile"])
    )
    pred = pick_next_combination(p_b)
    print("--- glavna predikcija (QSP Chebyshev polinomna regresija P(A)·b) ---")
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q30 Kvantna regresija (5/5) — QSP Chebyshev polinomna (T_d(A)·b): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39 | nq_b: 6 | BE margin: 0.9
--- A (skalirana, |λ_i| ≤ BE_EIG_MARGIN) eigenvalue info ---
  min(λ)=0.059905  max(λ)=0.900000  max|λ|=0.900000
BEST hparam: d (polinom degree)= 3 | α (ridge): 1.0 | profile: pi8_alt | P(anc=0): 0.664997 | cos(bias, freq_csv): 0.821654
--- demonstracija efekta degree d (Chebyshev T_d emphasis) za profile=zero, α= 1.0 ---
  d=2  P(anc=0)=0.769158  cos=0.804430  NEXT=(4, 9, 14, 17, 19, 22, 25)
  d=3  P(anc=0)=0.646079  cos=0.814929  NEXT=(15, 20, 21, 23, 24, 28, 32)
  d=4  P(anc=0)=0.654775  cos=0.776004  NEXT=(8, 13, 14, 16, 17, 22, 25)
  d=5  P(anc=0)=0.788463  cos=0.771215  NEXT=(4, 9, 12, 15, 20, 21, 23)
--- glavna predikcija (QSP Chebyshev polinomna regresija P(A)·b) ---
predikcija NEXT: (18, 20, x, y, z, 28, 32)
"""



"""
Q30_qreg5_QSP.py — tehnika: Quantum Signal Processing (QSP) polinomna regresija
P(A)·b preko block-encoding + qubitization walk operatora.

Koncept:
QSP/QSVT parametrizuje POLINOMNU transformaciju P(A) spektra A kroz phase sequence
{φ_k} umesto preko QPE estimacije. Za φ_k = 0 dobija se čist Chebyshev polinom
T_d(A) prve vrste. Regresija: P(A)·b filtrira/pojačava eigenvalue strukturu
deterministički i efikasno (BEZ phase-registra, BEZ QPE).

Kolo (1 + nq_b qubit-a):
  StatePreparation(b_amp) na b; anc_be = |0⟩ (start anc=0-blok = A-blok).
  U_A = block-encoding Hermitske A: [[A, √(I-A²)], [√(I-A²), -A]]   (UnitaryGate).
  Sekvenca d primena: za k ∈ 0..d-1:
    qc.append(U_A, anc ⊕ b).
    Ako k < d-1: Rz(2φ_k) na anc, zatim Z na anc (qubitization walk faktor).
Readout:
  Post-select anc_be = 0 → b-registar drži (aproksimativno) P(A)·|b⟩.
  Za profile="zero": P = T_d (čist Chebyshev, matematički tačno).
  Za profile="pi8_alt": P ≈ T_d sa malim perturbacijama (demonstracija QSP phase-param.).

A matrica (ista kao Q26/Q27):
  A = P + diag(freq_csv) + α·I, padded na 2^nq_b, skalirana da |λ_i| ≤ 0.9
  (margin od 0.1 stabilizuje √(I-A²) u block-encoding-u).

Razlika od Q26 (HHL), Q27 (QPE regresija), Q29 (QSVE):
  Q26/Q27/Q29 KORISTE QPE (phase-registar za estimaciju eigenvalue-a).
  Q30 NE koristi QPE — polinom se bira PRE izvršavanja preko {φ_k} sequence.
  Qubit budget:
    Q26: nq_b + n_phase + 1  (npr 6+5+1=12).
    Q27: nq_b + n_phase + 1  (npr 6+5+1=12).
    Q29: 2 + nq_b + n_phase  (npr 2+4+5=11).
    Q30: 1 + nq_b            (npr 1+6=7)  — ★ najefikasniji.

Tehnike:
Block-encoding Hermitske matrice: U_A = [[A, √(I-A²)], [√(I-A²), -A]] (klasična kompilacija).
Qubitization walk: (Z_a ⊗ I) · U_A između uzastopnih primena.
QSP phase sequence {φ_k} — parametrizuje polinom preko ancilla Rz-rotacija.
Chebyshev T_d(A) regresija za profile="zero" (dokazano indukcijom iz walk recurrencije).
Post-selekcija anc=0 za extrakciju A-bloka nakon d primena.
Deterministička grid-optimizacija (d, α, profile).

Prednosti:
★ Najmanji qubit budget među QPE-baziranim regresijama (nema phase registra).
Polinomno izražajan P(A) preko Chebyshev baze — fundamentalna ekspresivnost QSVT-a.
Čisto kvantno: block-encoding + walk + rotacije + post-selekcija su unitarne/projekcione.
Ceo CSV (pravilo 10): A i b iz CELOG CSV-a.

Nedostaci:
Block-encoding zahteva klasičan √(I - A²) — eigendecomposition u KOMPILACIJI
  (isti princip kao klasični eigendecomposition za e^{iAt} u Q26/Q27/Q29 QPE-u).
|λ_i| ≤ 1 zahtev za block-encoding — margin BE_EIG_MARGIN trade-off preciznost vs stabilnost.
Pun QSP phase-finding (za proizvoljan polinom P) je netrivijalan optimizacioni problem;
  ovde koristimo Chebyshev T_d (zero phases) — glavna klasa polinoma u QSVT-u.
P(anc=0) može biti niska za veliko d (degradacija nakon više walk-ova).
mod-39 readout — inherentno za sve 2^nq_b ≠ 39 implementacije.
"""



"""
QSP polinomna regresija (Quantum Signal Processing) 
Block-encoding operatora A (izgrađenog iz CSV-a) + QSP niz faza da implementira polinom p(A) na kvantnom nivou. p(A)·|b⟩ daje nelinearnu regresiju (npr. A², sinh(A)) za bogatiji model. Najsnažniji od svih, ali najsloženiji.

Quantum Signal Processing polinomna regresija P(A)·b preko block-encoding (U_A = [[A, √(I−A²)], [√(I−A²), −A]]) i qubitization walk operatora. Ključna razlika: NEMA QPE, pa je qubit budget samo 1 + nq_b (najefikasniji od svih 5 kvantnih regresija). Za zero-phase profil daje tačno Chebyshev T_d(A). 

QSP Chebyshev polinomna (T_d(A)·b).
"""
