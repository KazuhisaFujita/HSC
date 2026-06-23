# HierarchicalSparseCoding.py
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image


# =========================================================
# Utility
# =========================================================
def normalize_dictionary_(D: torch.Tensor, eps: float = 1e-12):
    """
    D: [input_dim, n_atoms]
    列ごとにL2正規化
    """
    with torch.no_grad():
        norms = D.norm(dim=0, keepdim=True).clamp_min(eps)
        D.div_(norms)


def soft_threshold(v: torch.Tensor, thr):
    return torch.sign(v) * torch.clamp(v.abs() - thr, min=0.0)


@torch.no_grad()
def estimate_lipschitz(D: torch.Tensor, n_power_iter: int = 10):
    """
    ||D||_2^2 の近似
    固定初期ベクトルで power iteration を回す
    """
    K = D.shape[1]
    v = torch.ones(K, device=D.device, dtype=D.dtype)
    v = v / v.norm().clamp_min(1e-12)

    for _ in range(n_power_iter):
        v = D.t() @ (D @ v) # D^T (D v)
        v = v / v.norm().clamp_min(1e-12) # (D^T D v)/||D^T D v||

    Gv = D.t() @ (D @ v)
    L = torch.dot(v, Gv).item() # v^T (D^T D v) = ||D v||^2 (最大固有値の近似)
    return max(L, 1e-6)


@torch.no_grad()
def sparsity_stats(a: torch.Tensor, tau: float = 1e-3):
    l1 = a.abs().mean().item()
    active = (a.abs() > tau).float().mean().item()
    return {"l1_mean": l1, "active_frac": active}


def make_run_dir(root: str = "data/raw_data", prefix: str = "hsc"):
    """
    実行ごとに新しい保存先を作る
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{prefix}_{timestamp}"

    idx = 0
    while run_dir.exists():
        idx += 1
        run_dir = root / f"{prefix}_{timestamp}_{idx:02d}"

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def expand_param_list(values: Union[float, Sequence[float]], n: int, name: str) -> List[float]:
    """
    float なら n 個に複製
    list/tuple なら長さチェック
    """
    if n == 0:
        return []

    if isinstance(values, (int, float)):
        return [float(values)] * n

    values = list(values)
    if len(values) == n:
        return [float(v) for v in values]
    if len(values) == 1:
        return [float(values[0])] * n

    raise ValueError(f"{name} must have length {n} (or 1 for broadcast), but got {len(values)}")


def compute_effective_dictionaries(Ds: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    D_eff^(1) = D1
    D_eff^(2) = D1 D2
    ...
    D_eff^(L) = D1 D2 ... DL
    """
    eff_dicts = []
    current = Ds[0]
    eff_dicts.append(current)

    for D in Ds[1:]:
        current = current @ D
        eff_dicts.append(current)

    return eff_dicts


def choose_display_levels(n_layers: int) -> List[int]:
    """
    可視化する層番号（1始まり）を選ぶ
    """
    if n_layers <= 3:
        return list(range(1, n_layers + 1))
    return [1, n_layers]


# =========================================================
# N-layer ISTA/MFISTA inference
# =========================================================
def compute_hsc_gradients(
    x: torch.Tensor, # 入力 [B, input_dim]
    Ds: List[torch.Tensor], # 辞書のリスト [input_dim, K1], [K1, K2], ..., [K_{L-1}, KL]
    codes: List[torch.Tensor], # コードのリスト [B, K1], [B, K2], ..., [B, KL]
    betas: List[float], # 層間結合の重み [L-1]
):
    """
    現在の codes における各層の滑らかな部分の勾配を返す
    f(a) = 1/2 ||x - D1 a||^2 
    df/da = D^T (D a - x)
    """
    L = len(Ds)
    grads = []

    for ell in range(L):
        if ell == 0:
            x_rec = codes[0] @ Ds[0].t() # D1 a1
            grad = (x_rec - x) @ Ds[0]   # D1^T (D1 a1 - x)
            if L > 1:
                upper_pred = codes[1] @ Ds[1].t() # D2 a2
                grad = grad + betas[0] * (codes[0] - upper_pred)

        elif ell == L - 1:
            lower_pred = codes[ell] @ Ds[ell].t() # D_L a_L
            grad = betas[ell - 1] * (lower_pred - codes[ell - 1]) @ Ds[ell] # beta_{L-1} D_L^T (D_L a_L - a_{L-1})

        else:
            lower_pred = codes[ell] @ Ds[ell].t() # D_ell a_ell
            upper_pred = codes[ell + 1] @ Ds[ell + 1].t() # D_{ell+1} a_{ell+1}

            grad_lower = betas[ell - 1] * (lower_pred - codes[ell - 1]) @ Ds[ell]
            grad_upper = betas[ell] * (codes[ell] - upper_pred)
            grad = grad_lower + grad_upper

        grads.append(grad)

    return grads # [B, K1], [B, K2], ..., [B, KL]


@torch.no_grad()
def compute_hsc_layer_steps(
    Ds: List[torch.Tensor],
    betas: List[float],
    eta_scale: float = 1.0,
):
    """
    block ISTA/MFISTA 用の各層 step size
    Lipschitz 定数の近似を使う
    """
    L = len(Ds)
    if len(betas) != max(L - 1, 0):
        raise ValueError(f"betas must have length {max(L - 1, 0)}, but got {len(betas)}")

    lips = []

    for ell in range(L):
        if ell == 0:
            lip = estimate_lipschitz(Ds[0])
            if L > 1:
                lip += betas[0]

        elif ell == L - 1:
            lip = betas[ell - 1] * estimate_lipschitz(Ds[ell])

        else:
            lip = betas[ell - 1] * estimate_lipschitz(Ds[ell]) + betas[ell]

        lips.append(max(lip, 1e-6))

    return [eta_scale / lip for lip in lips]


def infer_hsc_ista_nlayer(
    x: torch.Tensor, # 入力 [B, input_dim]
    Ds: List[torch.Tensor], # 辞書のリスト [input_dim, K1], [K1, K2], ..., [K_{L-1}, KL]
    lambdas: List[float], # スパース性の重み [L]
    betas: List[float], # 層間結合の重み [L-1]
    T: int = 30, # 反復回数
    eta_scale: float = 1.0,
    code_inits: List[torch.Tensor] = None, # コードの初期値のリスト [B, K1], [B, K2], ..., [B, KL]
    etas: List[float] = None, # 各層のステップサイズ [L]
):
    """
    N層階層スパースコーディングの逐次 block ISTA
    """
    L = len(Ds)
    B = x.shape[0]
    device = x.device
    dtype = x.dtype

    if len(lambdas) != L:
        raise ValueError(f"lambdas must have length {L}, but got {len(lambdas)}")
    if len(betas) != max(L - 1, 0):
        raise ValueError(f"betas must have length {max(L - 1, 0)}, but got {len(betas)}")
    if etas is not None and len(etas) != L:
        raise ValueError(f"etas must have length {L}, but got {len(etas)}")
    if code_inits is not None and len(code_inits) != L:
        raise ValueError(f"code_inits must have length {L}, but got {len(code_inits)}")

    codes = [] # 各層のコードのリスト [B, K1], [B, K2], ..., [B, KL]
    for ell, D in enumerate(Ds): #ell=0..L-1, D=[input_dim, K_ell]
        K = D.shape[1] # この層のコード次元
        if code_inits is None or code_inits[ell] is None:
            a = torch.zeros(B, K, device=device, dtype=dtype) # コードをゼロ初期化
        else: # 指定された初期値をコピーして使う
            a = code_inits[ell].clone()
        codes.append(a)

    if etas is None: # ステップサイズが指定されていない場合は計算する
        etas = compute_hsc_layer_steps(Ds, betas, eta_scale=eta_scale)

    thrs = [eta * lam for eta, lam in zip(etas, lambdas)] # 各層のソフト閾値の値

    for _ in range(T):
        grads = compute_hsc_gradients(x, Ds, codes, betas) #各層の勾配を計算
        for ell in range(L): # 各層でコードを更新
            codes[ell] = soft_threshold(codes[ell] - etas[ell] * grads[ell], thrs[ell])

    return codes


def infer_hsc_mfista_nlayer(
    x: torch.Tensor,  # 入力 [B, input_dim]
    Ds: List[torch.Tensor],  # 辞書のリスト [input_dim, K1], [K1, K2], ..., [K_{L-1}, KL]
    lambdas: List[float],  # スパース性の重み [L]
    betas: List[float],  # 層間結合の重み [L-1]
    T: int = 30,  # 反復回数
    eta_scale: float = 1.0,
    code_inits: List[torch.Tensor] = None,  # コードの初期値のリスト [B, K1], ..., [B, KL]
    etas: List[float] = None,  # 互換性のため残すが、内部では global eta を使う
):
    """
    N-layer hierarchical sparse coding solved by an MFISTA-style
    proximal gradient method.
    Beck & Teboulle (2009, IEEE TIP) の MFISTA
    
    Notes
    -----
    - The joint latent variable is represented as a list [a1, ..., aL].
    - Gradients are evaluated at the extrapolated point y.
    - A single global fixed step size eta is used across all layers.
    - The update follows the monotone MFISTA selection rule.
    """

    L = len(Ds)
    B = x.shape[0]
    device = x.device
    dtype = x.dtype

    if len(lambdas) != L:
        raise ValueError(f"lambdas must have length {L}, but got {len(lambdas)}")
    if len(betas) != max(L - 1, 0):
        raise ValueError(f"betas must have length {max(L - 1, 0)}, but got {len(betas)}")
    if etas is not None and len(etas) != L:
        raise ValueError(f"etas must have length {L}, but got {len(etas)}")
    if code_inits is not None and len(code_inits) != L:
        raise ValueError(f"code_inits must have length {L}, but got {len(code_inits)}")

    # MFISTA uses a single fixed step size.
    # To minimize changes, reuse the existing layerwise step estimator and
    # take a conservative global step as the minimum across layers.
    if etas is None:
        layer_etas = compute_hsc_layer_steps(Ds, betas, eta_scale=eta_scale)
    else:
        layer_etas = etas

    eta = min(layer_etas)
    if eta <= 0:
        raise ValueError(f"eta must be positive, but got {eta}")

    thrs = [eta * lam for lam in lambdas] # 各層のソフト閾値の値

    def clone_code_list(code_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [a.clone() for a in code_list]

    def energy_of(code_list: List[torch.Tensor]) -> float:
        total, _, _, _ = hierarchical_losses(
            x=x,
            Ds=Ds,
            codes=code_list,
            lambdas=lambdas,
            betas=betas,
        )
        return float(total.detach().item())

    # Step 0: x0 and y1 = x0
    x_prev = []
    for ell, D in enumerate(Ds):
        K = D.shape[1]
        if code_inits is None or code_inits[ell] is None:
            a = torch.zeros(B, K, device=device, dtype=dtype) # ゼロ初期化
        else:
            a = code_inits[ell].clone()
        x_prev.append(a)

    y = clone_code_list(x_prev)
    t = 1.0
    F_x_prev = energy_of(x_prev) # 初期コードのエネルギー

    for _ in range(T):
        # z_k = prox(y_k - eta * grad f(y_k))
        grads = compute_hsc_gradients(x, Ds, y, betas)
        z = [
            soft_threshold(y[ell] - eta * grads[ell], thrs[ell])
            for ell in range(L)
        ]
        F_z = energy_of(z) # z_k のエネルギー

        # Monotone selection:
        # x_k = argmin { F(z_k), F(x_{k-1}) }
        if F_z <= F_x_prev: # z_k の方がエネルギー小さいなら z_k を受け入れる
            x_next = clone_code_list(z)
            F_x_next = F_z
        else:
            x_next = clone_code_list(x_prev)
            F_x_next = F_x_prev

        # t_{k+1}
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))

        # y_{k+1}
        y_next = [
            x_next[ell]
            + (t / t_next) * (z[ell] - x_next[ell])
            + ((t - 1.0) / t_next) * (x_next[ell] - x_prev[ell])
            for ell in range(L)
        ]

        x_prev = x_next
        y = y_next
        t = t_next
        F_x_prev = F_x_next

    return x_prev

# =========================================================
# LISTA blocks
# =========================================================
class SharedLISTABlock(nn.Module):
    """
    1層分の shared-parameter LISTA
    重みを共有するため、反復ステップごとに同じパラメタを使う

    a^{t+1} = S_theta(Wx u + Wa a^t)

    u はこの層への入力
      - 1層目なら x
      - 2層目以降なら 1つ下のコード
    """
    def __init__(self, input_dim: int, code_dim: int, T: int = 5):
        super().__init__()
        self.input_dim = input_dim # 例えば1層目なら画像空間の次元、2層目以降なら1つ下のコードの次元
        self.code_dim = code_dim # コードの次元
        self.T = T # 反復回数

        self.Wx = nn.Parameter(torch.empty(code_dim, input_dim)) # u からコードへの重み行列
        self.Wa = nn.Parameter(torch.empty(code_dim, code_dim)) # 前回のコードからコードへの重み行列
        self.raw_theta = nn.Parameter(torch.zeros(code_dim)) # ソフト閾値 theta の生の値。softplus(raw_theta) で正の値になるようにする

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.Wx)
            self.Wa.zero_()
            self.raw_theta.fill_(math.log(math.exp(0.1) - 1.0 + 1e-8))

    @torch.no_grad()
    def init_from_dictionary(self, D: torch.Tensor, lam: float, eta_scale: float = 1.0):
        """
        ISTA型初期化:
            Wx = eta D^T
            Wa = I - eta D^T D
            theta = eta * lam
        """
        L = estimate_lipschitz(D)
        eta = eta_scale / L

        Dt = D.t()
        G = Dt @ D
        I = torch.eye(G.shape[0], device=D.device, dtype=D.dtype)

        self.Wx.copy_(eta * Dt)
        self.Wa.copy_(I - eta * G)

        theta0 = eta * lam
        raw_theta0 = math.log(math.exp(theta0) - 1.0 + 1e-8)
        self.raw_theta.fill_(raw_theta0)

    def forward(self, u: torch.Tensor):
        batch_size = u.shape[0]
        a = torch.zeros(batch_size, self.code_dim, device=u.device, dtype=u.dtype) # コードをゼロ初期化
        theta = F.softplus(self.raw_theta) # 閾値を正の値に変換

        # Gregor & LeCun の B = We X に対応
        B = F.linear(u, self.Wx)

        if self.T <= 0:
            return a

        # Z^(0) = h_theta(B) を明示
        a = soft_threshold(B, theta)

        # 以後は Z^(t) = h_theta(B + S Z^(t-1))
        for _ in range(self.T - 1):
            a = soft_threshold(B + F.linear(a, self.Wa), theta)

        return a


class UntiedLISTABlock(nn.Module):
    """
    1層分の untied LISTA
    各反復ステップごとに別パラメタを持つ

    a^{t+1} = S_{theta^(t)}( W_x^(t) u + W_a^(t) a^t )
    """
    def __init__(self, input_dim: int, code_dim: int, T: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.T = T

        self.Wx = nn.Parameter(torch.empty(T, code_dim, input_dim))
        self.Wa = nn.Parameter(torch.empty(T, code_dim, code_dim))
        self.raw_theta = nn.Parameter(torch.zeros(T, code_dim))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for t in range(self.T):
                nn.init.xavier_uniform_(self.Wx[t])
                self.Wa[t].zero_()
            self.raw_theta.fill_(math.log(math.exp(0.1) - 1.0 + 1e-8))

    @torch.no_grad()
    def init_from_dictionary(self, D: torch.Tensor, lam: float, eta_scale: float = 1.0):
        """
        ISTA型初期化を全ステップに同じ値で入れる
        """
        L = estimate_lipschitz(D)
        eta = eta_scale / L

        Dt = D.t()
        G = Dt @ D
        I = torch.eye(G.shape[0], device=D.device, dtype=D.dtype)

        Wx0 = eta * Dt
        Wa0 = I - eta * G

        theta0 = eta * lam
        raw_theta0 = math.log(math.exp(theta0) - 1.0 + 1e-8)

        for t in range(self.T):
            self.Wx[t].copy_(Wx0)
            self.Wa[t].copy_(Wa0)
            self.raw_theta[t].fill_(raw_theta0)

    def forward(self, u: torch.Tensor):
        B = u.shape[0]
        K = self.code_dim
        a = torch.zeros(B, K, device=u.device, dtype=u.dtype)

        theta = F.softplus(self.raw_theta)

        for t in range(self.T):
            z = F.linear(u, self.Wx[t]) + F.linear(a, self.Wa[t])
            a = soft_threshold(z, theta[t])

        return a


class NLayerLISTA(nn.Module):
    """
    bottom-up の N層 LISTA
    a1 = f1(x)
    a2 = f2(a1)
    ...
    aL = fL(a_{L-1})

    lista_variant:
      - "shared"
      - "untied"
    """
    def __init__(
        self,
        input_dim: int,
        layer_dims: List[int],
        lista_steps: int = 5,
        lista_variant: str = "shared",
    ):
        super().__init__()
        assert lista_variant in ["shared", "untied"]
        self.lista_variant = lista_variant

        dims = [input_dim] + list(layer_dims)

        if lista_variant == "shared":
            block_cls = SharedLISTABlock
        else:
            block_cls = UntiedLISTABlock

        # 各層の LISTA ブロックを作る
        self.blocks = nn.ModuleList([
            block_cls(dims[ell], dims[ell + 1], T=lista_steps)
            for ell in range(len(layer_dims))
        ])

    @torch.no_grad()
    def init_from_dictionaries(
        self,
        Ds: List[torch.Tensor],
        lambdas: List[float],
        eta_scale: float = 1.0,
    ):
        for block, D, lam in zip(self.blocks, Ds, lambdas):
            block.init_from_dictionary(D, lam=lam, eta_scale=eta_scale)

    def forward(self, x: torch.Tensor):
        codes = []
        u = x
        for block in self.blocks:
            a = block(u)
            codes.append(a)
            u = a
        return codes


# =========================================================
# Unified inference module
# =========================================================
class UnifiedHSCInference(nn.Module):
    """
        mode:
            - "ista"
            - "mfista"
            - "lista"
            - "hybrid"
            - "hybrid_finetune"
            - "hybrid_mfista"

    ista:
        codes = ISTA(x, Ds)

    mfista:
        codes = MFISTA(x, Ds)

    lista:
        codes = NLayerLISTA(x)

    hybrid:
        init_codes = NLayerLISTA(x)
        codes = ISTA(x, Ds, code_inits=init_codes)

    hybrid_finetune:
        inference path is the same as hybrid.
        only the training schedule differs.

    hybrid_mfista:
        init_codes = NLayerLISTA(x)
        codes = MFISTA(x, Ds, code_inits=init_codes)

    lista_variant:
      - "shared"
      - "untied"
    """
    def __init__(
        self,
        mode: str,
        input_dim: int,
        layer_dims: List[int],
        lista_steps: int,
        lista_variant: str = "shared",
    ):
        super().__init__()
        assert mode in ["ista", "mfista", "lista", "hybrid", "hybrid_finetune", "hybrid_mfista"]
        assert lista_variant in ["shared", "untied"]

        self.mode = mode
        self.lista_variant = lista_variant

        if mode in ["lista", "hybrid", "hybrid_finetune", "hybrid_mfista"]:
            self.lista = NLayerLISTA(
                input_dim=input_dim,
                layer_dims=layer_dims,
                lista_steps=lista_steps,
                lista_variant=lista_variant,
            )
        else:
            self.lista = None

    @torch.no_grad()
    def init_from_dictionaries(
        self,
        Ds: List[torch.Tensor],
        lambdas: List[float],
        eta_scale: float = 1.0,
    ):
        if self.lista is not None:
            self.lista.init_from_dictionaries(Ds, lambdas, eta_scale=eta_scale)

    def encoder_parameters(self):
        if self.lista is None:
            return []
        return list(self.lista.parameters())

    def forward(
        self,
        x: torch.Tensor,
        Ds_for_inference: List[torch.Tensor],
        lambdas: List[float],
        betas: List[float],
        infer_steps: int,
        eta_scale: float,
        etas: List[float] = None,
    ):
        if etas is not None and len(etas) != len(Ds_for_inference):
            raise ValueError(
                f"etas must have length {len(Ds_for_inference)}, but got {len(etas)}"
            )

        if self.mode == "ista":
            return infer_hsc_ista_nlayer(
                x=x,
                Ds=Ds_for_inference,
                lambdas=lambdas,
                betas=betas,
                T=infer_steps,
                eta_scale=eta_scale,
                code_inits=None,
                etas=etas,
            )

        if self.mode == "mfista":
            return infer_hsc_mfista_nlayer(
                x=x,
                Ds=Ds_for_inference,
                lambdas=lambdas,
                betas=betas,
                T=infer_steps,
                eta_scale=eta_scale,
                code_inits=None,
                etas=etas,
            )

        if self.mode == "lista":
            return self.lista(x)

        if self.mode in ["hybrid", "hybrid_finetune"]:
            init_codes = self.lista(x) # LISTA で初期コードを生成
            return infer_hsc_ista_nlayer(
                x=x,
                Ds=Ds_for_inference,
                lambdas=lambdas,
                betas=betas,
                T=infer_steps,
                eta_scale=eta_scale,
                code_inits=init_codes,
                etas=etas,
            )

        if self.mode == "hybrid_mfista":
            init_codes = self.lista(x) # LISTA で初期コードを生成
            return infer_hsc_mfista_nlayer(
                x=x,
                Ds=Ds_for_inference,
                lambdas=lambdas,
                betas=betas,
                T=infer_steps,
                eta_scale=eta_scale,
                code_inits=init_codes,
                etas=etas,
            )

        raise ValueError(f"Unknown mode: {self.mode}")


# =========================================================
# Loss / monitoring
# =========================================================
def squared_frobenius_batchmean(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    各サンプルごとに二乗和を取り、その後 batch 平均する
    """
    return ((y_hat - y) ** 2).sum(dim=1).mean()


def l1_batchmean(a: torch.Tensor) -> torch.Tensor:
    """
    各サンプルごとに L1 和を取り、その後 batch 平均する
    """
    return a.abs().sum(dim=1).mean()


def hierarchical_losses(
    x: torch.Tensor,
    Ds: List[torch.Tensor],
    codes: List[torch.Tensor],
    lambdas: List[float],
    betas: List[float],
):
    """
    energy:
      0.5 ||x - D1 a1||^2
      + sum_{l=2..L} 0.5 beta_{l-1} ||a_{l-1} - D_l a_l||^2
      + sum_l lambda_l ||a_l||_1
    ただし最終的には batch 平均
    """
    rec_x = 0.5 * squared_frobenius_batchmean(codes[0] @ Ds[0].t(), x)

    rec_h = x.new_tensor(0.0)
    for ell in range(1, len(Ds)):
        pred_lower = codes[ell] @ Ds[ell].t()
        rec_h = rec_h + 0.5 * betas[ell - 1] * squared_frobenius_batchmean(
            pred_lower, codes[ell - 1]
        )

    sparse = x.new_tensor(0.0)
    for lam, a in zip(lambdas, codes):
        sparse = sparse + lam * l1_batchmean(a)

    total = rec_x + rec_h + sparse
    return total, rec_x, rec_h, sparse


# =========================================================
# Visualization
# =========================================================
def show_dictionary(D: torch.Tensor, H: int, W: int, n_show: int = 64, title: str = "Dictionary"):
    """
    D が画像空間の辞書 [784, K] のとき表示
    """
    D = D.detach().cpu()
    _, K = D.shape
    n = min(n_show, K)
    grid = int(math.ceil(math.sqrt(n)))

    plt.figure(figsize=(grid, grid))
    for i in range(n):
        atom = D[:, i].view(H, W)
        atom = atom - atom.mean()
        m = atom.abs().max().item() + 1e-12
        atom = atom / m

        ax = plt.subplot(grid, grid, i + 1)
        ax.imshow(atom, cmap="gray", vmin=-1.0, vmax=1.0)
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close()


@torch.no_grad()
def save_dictionary_grid(D: torch.Tensor, H: int, W: int, save_path: str, n_show: int = 64):
    K = D.shape[1]
    n = min(n_show, K)

    atoms = D[:, :n].t().contiguous().view(n, 1, H, W)
    atoms = atoms - atoms.mean(dim=(1, 2, 3), keepdim=True)
    atoms = atoms / (atoms.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8)

    nrow = int(math.ceil(math.sqrt(n)))
    grid = make_grid(atoms, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, save_path)


@torch.no_grad()
def save_hsc_recon_examples_nlayer(
    x: torch.Tensor,
    codes: List[torch.Tensor],
    eff_dicts: List[torch.Tensor],
    H: int,
    W: int,
    save_path: str,
    n: int = 8,
    levels_to_show: List[int] = None,
    title: str = "HSC Recon",
):
    x = x.detach().cpu()
    codes = [a.detach().cpu() for a in codes]
    eff_dicts = [D.detach().cpu() for D in eff_dicts]

    L = len(codes)
    if levels_to_show is None:
        levels_to_show = list(range(1, L + 1))

    n = min(n, x.shape[0])
    n_rows = 1 + len(levels_to_show)

    fig = plt.figure(figsize=(2 * n, 2 * n_rows))

    for i in range(n):
        ax = plt.subplot(n_rows, n, i + 1)
        ax.imshow(x[i].view(H, W), cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_title("x")

    for row_idx, level in enumerate(levels_to_show, start=1):
        a = codes[level - 1]
        D_eff = eff_dicts[level - 1]
        x_rec = a @ D_eff.t()

        for i in range(n):
            ax = plt.subplot(n_rows, n, row_idx * n + i + 1)
            ax.imshow(x_rec[i].view(H, W), cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_title(f"D1...D{level} a{level}")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def show_hsc_recon_examples_nlayer(
    x: torch.Tensor,
    codes: List[torch.Tensor],
    eff_dicts: List[torch.Tensor],
    H: int,
    W: int,
    n: int = 8,
    levels_to_show: List[int] = None,
    title: str = "HSC Recon",
):
    """
    1行目: x
    以降: 各層の有効再構成 D_eff^(l) a_l

    levels_to_show は 1始まり
    """
    x = x.detach().cpu()
    codes = [a.detach().cpu() for a in codes]
    eff_dicts = [D.detach().cpu() for D in eff_dicts]

    L = len(codes)
    if levels_to_show is None:
        levels_to_show = list(range(1, L + 1))

    n = min(n, x.shape[0])
    n_rows = 1 + len(levels_to_show)

    plt.figure(figsize=(2 * n, 2 * n_rows))

    for i in range(n):
        ax = plt.subplot(n_rows, n, i + 1)
        ax.imshow(x[i].view(H, W), cmap="gray")
        ax.axis("off")
        if i == 0:
            ax.set_title("x")

    for row_idx, level in enumerate(levels_to_show, start=1):
        a = codes[level - 1]
        D_eff = eff_dicts[level - 1]
        x_rec = a @ D_eff.t()

        for i in range(n):
            ax = plt.subplot(n_rows, n, row_idx * n + i + 1)
            ax.imshow(x_rec[i].view(H, W), cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_title(f"D1...D{level} a{level}")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close()


# =========================================================
# Config
# =========================================================
@dataclass
class Config:
    # "ista" / "mfista" / "lista" / "hybrid" / "hybrid_finetune" / "hybrid_mfista"
    mode: str = "hybrid"

    # LISTAの重み共有方式
    lista_variant: str = "shared"   # "shared" / "untied"

    # デフォルトは2層
    layer_dims: List[int] = field(default_factory=lambda: [256, 64])

    # 各層のL1係数
    lambdas: Union[float, Sequence[float]] = field(default_factory=lambda: [0.05, 0.05])

    # 各層間の coupling 重み
    betas: Union[float, Sequence[float]] = field(default_factory=lambda: [1.0])

    lr_D: float = 1e-3
    lr_E: float = 1e-3

    infer_steps: int = 30
    lista_steps: int = 5
    eta_scale: float = 1.0
    hybrid_pretrain_epochs: int = 0

    batch_size: int = 256
    epochs: int = 10
    dc_center: bool = False

    print_every: int = 200

    save_root: str = "data/raw_data"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
