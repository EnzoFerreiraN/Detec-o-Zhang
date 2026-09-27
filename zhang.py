# -*- coding: utf-8 -*-
"""
zhang.py — Implementação manual do método de calibração de Zhang
(Zhengyou Zhang, "A Flexible New Technique for Camera Calibration", MSR-TR-98-71).

O OpenCV só é usado para detectar os cantos do tabuleiro (cv2.findChessboardCorners /
cornerSubPix) e, em calibrar.py, para a conferência final com cv2.calibrateCamera.
Todo o resto do método (homografia por DLT normalizado, extração dos intrínsecos,
extrínsecos, distorção radial e refinamento por máxima verossimilhança) é implementado
aqui com numpy/scipy, seguindo as seções e equações indicadas nos comentários.

Convenções:
  - Um ponto do modelo (tabuleiro) é (X, Y) em mm, com Z = 0 (Seção 2.2).
  - `Mlist` é uma lista de arrays (m, 2) com os pontos do modelo, um array por imagem
    (normalmente o mesmo array repetido, pois o tabuleiro é fixo).
  - `mlist` é uma lista de arrays (m, 2) com os cantos detectados, em pixels.
  - A matriz intrínseca A segue a eq. (1):
        A = [[alpha, gamma, u0],
             [0,     beta,  v0],
             [0,     0,     1 ]]
"""

import numpy as np
from scipy.optimize import least_squares

try:
    import cv2
except ImportError:  # cv2 só é necessário para detectar_cantos
    cv2 = None


# ---------------------------------------------------------------------------
# Detecção de cantos (única parte que usa o OpenCV como "sensor" de features)
# ---------------------------------------------------------------------------

def detectar_cantos(caminho, nx, ny, escala=0.4, salvar_desenho=None):
    """Detecta os (nx*ny) cantos internos do tabuleiro em `caminho`.

    Estratégia (ver enunciado): `findChessboardCorners` numa cópia reduzida (~escala)
    para ganhar velocidade, e depois `cornerSubPix` na resolução original para refinar.

    Retorna (cantos, metodo, shape_img) em caso de sucesso, com `cantos` um array
    (nx*ny, 2) em pixels na ordem em que o OpenCV varre o tabuleiro (linha a linha,
    u variando mais rápido) — a mesma ordem produzida por `pontos_modelo`.
    Em caso de falha, retorna (None, motivo, shape_img_ou_None).
    """
    if cv2 is None:
        raise RuntimeError("OpenCV não disponível; necessário para detectar_cantos")

    # IMREAD_IGNORE_ORIENTATION: mantém a imagem no referencial nativo do sensor
    # (não aplica a rotação do EXIF). Ver discussão no relatório sobre orientação.
    img = cv2.imread(caminho, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    if img is None:
        return None, "não foi possível ler o arquivo", None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(small, (nx, ny), flags)
    metodo = "padrao"
    if not found:
        # Fallback: detector mais robusto (mais lento), só quando o padrão falha
        found, corners = cv2.findChessboardCornersSB(
            small, (nx, ny), cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        metodo = "SB"

    if not found:
        return None, "tabuleiro não encontrado", img.shape[:2]

    corners = (corners / escala).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    if salvar_desenho is not None:
        vis = img.copy()
        cv2.drawChessboardCorners(vis, (nx, ny), corners, found)
        cv2.imwrite(salvar_desenho, vis)

    return corners.reshape(-1, 2), metodo, img.shape[:2]


def pontos_modelo(nx, ny, s):
    """Grade de pontos (X, Y) do tabuleiro no plano do modelo (Z = 0, Seção 2.2),
    em mm, na mesma ordem de varredura do `findChessboardCorners`
    (linha a linha, com X variando mais rápido que Y).
    """
    xs = np.arange(nx, dtype=float) * s
    ys = np.arange(ny, dtype=float) * s
    X, Y = np.meshgrid(xs, ys)  # X,Y com shape (ny, nx); X varia mais rápido no ravel
    return np.column_stack([X.ravel(), Y.ravel()])


# ---------------------------------------------------------------------------
# Apêndice A — Homografia entre o plano do modelo e sua imagem (DLT normalizado)
# ---------------------------------------------------------------------------

def _normalizar_hartley(pts):
    """Normalização de Hartley [12]: translada o centróide para a origem e
    escala para que a distância média à origem seja sqrt(2).
    Retorna (pts_normalizados, T) tal que pts_normalizados_hom = T @ pts_hom.
    """
    centro = pts.mean(axis=0)
    d = pts - centro
    dist_media = np.sqrt((d ** 2).sum(axis=1)).mean()
    escala = np.sqrt(2.0) / dist_media if dist_media > 1e-12 else 1.0
    T = np.array([
        [escala, 0.0, -escala * centro[0]],
        [0.0, escala, -escala * centro[1]],
        [0.0, 0.0, 1.0],
    ])
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    pts_n = (T @ pts_h.T).T
    return pts_n[:, :2], T


def _refinar_homografia(H0, M, m):
    """Refino não linear da homografia minimizando o erro geométrico (a estimativa
    de máxima verossimilhança descrita no Apêndice A, assumindo Lambda_mi = sigma^2 I,
    o que reduz o problema a mínimos quadrados não lineares comuns)."""
    H0 = H0 / H0[2, 2]
    x0 = H0.flatten()[:8]  # H[2,2] fixado em 1

    Mh = np.column_stack([M, np.ones(len(M))])

    def residuos(x):
        H = np.append(x, 1.0).reshape(3, 3)
        proj = (H @ Mh.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        return (proj - m).flatten()

    res = least_squares(residuos, x0, method="lm")
    return np.append(res.x, 1.0).reshape(3, 3)


def homografia_dlt(M, m, refinar=True):
    """Estima H tal que s*[u,v,1]^T = H*[X,Y,1]^T (eq. 2), por DLT com
    normalização de Hartley (Apêndice A), seguido de refino não linear opcional.
    """
    M = np.asarray(M, dtype=float)
    m = np.asarray(m, dtype=float)
    n = len(M)

    Mn, Tm = _normalizar_hartley(M)
    mn, Ti = _normalizar_hartley(m)

    # Monta L (2n x 9) a partir da forma dada no Apêndice A:
    #   [M~^T  0^T  -u*M~^T]
    #   [0^T  M~^T  -v*M~^T]  x = 0,   x = [h1_linha; h2_linha; h3_linha] (9,)
    L = np.zeros((2 * n, 9))
    for i in range(n):
        X, Y = Mn[i]
        u, v = mn[i]
        Mtil = np.array([X, Y, 1.0])
        L[2 * i, 0:3] = Mtil
        L[2 * i, 6:9] = -u * Mtil
        L[2 * i + 1, 3:6] = Mtil
        L[2 * i + 1, 6:9] = -v * Mtil

    _, _, Vt = np.linalg.svd(L)
    Hn = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(Ti) @ Hn @ Tm
    H = H / H[2, 2]

    if refinar:
        H = _refinar_homografia(H, M, m)

    return H


# ---------------------------------------------------------------------------
# Seção 3.1 — Solução fechada (vetores v_ij, sistema Vb = 0, Apêndice B)
# ---------------------------------------------------------------------------

def _vij(H, i, j):
    """Vetor v_ij (eq. 7-9), a partir das colunas h_i, h_j de H (0-indexado)."""
    hi = H[:, i]
    hj = H[:, j]
    return np.array([
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2],
    ])


def _extrair_A(b):
    """Extrai a matriz intrínseca A a partir do vetor b = [B11,B12,B22,B13,B23,B33]
    (Apêndice B). B = lambda * A^-T A^-1, definida a menos de um fator de escala.
    """
    B11, B12, B22, B13, B23, B33 = b

    denom = B11 * B22 - B12 ** 2
    v0 = (B12 * B13 - B11 * B23) / denom
    lam = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11

    if lam / B11 < 0:
        # b só é definido a menos de sinal (vetor singular); troca o sinal e refaz.
        b = -b
        B11, B12, B22, B13, B23, B33 = b
        denom = B11 * B22 - B12 ** 2
        v0 = (B12 * B13 - B11 * B23) / denom
        lam = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11

    alpha = np.sqrt(lam / B11)
    beta = np.sqrt(lam * B11 / denom)
    gamma = -B12 * alpha ** 2 * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha ** 2 / lam

    return np.array([
        [alpha, gamma, u0],
        [0.0, beta, v0],
        [0.0, 0.0, 1.0],
    ])


def intrinsecos_fechado(Hs, forcar_gamma_zero=None):
    """Monta V (2n x 6) empilhando v_12 e (v_11 - v_22) de cada homografia (eq. 8-9)
    e resolve Vb = 0 por SVD (Seção 3.1). Se n=2, opcionalmente impõe gamma=0
    (linha [0,1,0,0,0,0]) como recomendado no artigo quando faltam imagens.
    Retorna (A, b).

    Atenção de condicionamento numérico: as entradas de b têm ordens de grandeza
    muito diferentes (B11 ~ 1/alpha^2 é da ordem de 1e-6 a 1e-7 para focais em
    pixels, enquanto B33 é O(1)), o que deixa a SVD de V mal condicionada quando
    H está em unidades de pixel "cruas". Por isso `calibrar()` primeiro normaliza
    as homografias (transformação de similaridade N baseada no tamanho da imagem,
    análoga à normalização de Hartley do Apêndice A) antes de chamar esta função,
    e depois desfaz a normalização em A. Ver `calibrar()`.
    """
    n = len(Hs)
    if forcar_gamma_zero is None:
        forcar_gamma_zero = (n == 2)

    linhas = []
    for H in Hs:
        v12 = _vij(H, 0, 1)
        v11 = _vij(H, 0, 0)
        v22 = _vij(H, 1, 1)
        linhas.append(v12)
        linhas.append(v11 - v22)
    if forcar_gamma_zero:
        linhas.append(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))

    V = np.array(linhas)
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]
    A = _extrair_A(b)
    return A, b


def extrinsecos(A, H):
    """Extrai R, t de uma homografia dada A (Seção 3.1):
        r1 = lambda*A^-1*h1, r2 = lambda*A^-1*h2, r3 = r1 x r2, t = lambda*A^-1*h3
    com lambda = 1/||A^-1 h1|| = 1/||A^-1 h2||. Como há ruído, usa-se a média das
    duas normas para lambda, força t_z > 0 e ortogonaliza R via SVD (Apêndice C).
    """
    Ainv = np.linalg.inv(A)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

    lam1 = 1.0 / np.linalg.norm(Ainv @ h1)
    lam2 = 1.0 / np.linalg.norm(Ainv @ h2)
    lam = 0.5 * (lam1 + lam2)

    r1 = lam * (Ainv @ h1)
    r2 = lam * (Ainv @ h2)
    r3 = np.cross(r1, r2)
    t = lam * (Ainv @ h3)

    if t[2] < 0:
        r1, r2, r3, t = -r1, -r2, -r3, -t

    Q = np.column_stack([r1, r2, r3])
    R = _ortogonalizar_rotacao(Q)
    return R, t


def _ortogonalizar_rotacao(Q):
    """Melhor aproximação de Q por uma matriz de rotação (Apêndice C):
    R = U V^T, com Q = U S V^T (SVD); corrige det(R) = -1 se necessário."""
    U, _, Vt = np.linalg.svd(Q)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
    return R


# ---------------------------------------------------------------------------
# Vetor de Rodrigues (implementação própria, usada na MLE — eq. 10 e 14)
# ---------------------------------------------------------------------------

def rodrigues(r):
    """Vetor de Rodrigues r (eixo * ângulo) -> matriz de rotação R."""
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rodrigues_inv(R):
    """Matriz de rotação R -> vetor de Rodrigues r."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-9:
        return np.zeros(3)

    if np.pi - theta < 1e-6:
        # Caso degenerado (theta ~ pi): sin(theta) ~ 0, usa R + I
        A = (R + np.eye(3)) / 2.0
        idx = np.argmax(np.diag(A))
        v = A[:, idx]
        v = v / np.linalg.norm(v)
        return theta * v

    r = theta / (2.0 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    return r


# ---------------------------------------------------------------------------
# Projeção com modelo de distorção radial (Seção 3.3, eqs. 11-12)
# ---------------------------------------------------------------------------

def projetar(A, k, R, t, M3d):
    """Projeta pontos 3D M3d (m,3) na imagem, com pinhole (eq. 2) seguido de
    distorção radial (eqs. 11-12). k = (k1, k2).
    """
    k1, k2 = k
    Mc = (R @ M3d.T).T + t
    x = Mc[:, 0] / Mc[:, 2]
    y = Mc[:, 1] / Mc[:, 2]
    r2 = x ** 2 + y ** 2
    fator = 1.0 + k1 * r2 + k2 * r2 ** 2
    xd = x * fator
    yd = y * fator

    alpha, gamma, u0 = A[0, 0], A[0, 1], A[0, 2]
    beta, v0 = A[1, 1], A[1, 2]
    u = u0 + alpha * xd + gamma * yd
    v = v0 + beta * yd
    return np.column_stack([u, v])


def distorcao_linear(A, Rs, ts, Mlist, mlist):
    """Estima k1, k2 por mínimos quadrados lineares (eq. 13), a partir dos
    intrínsecos/extrínsecos já estimados (assumindo distorção nula neles).
    """
    alpha, gamma, u0 = A[0, 0], A[0, 1], A[0, 2]
    beta, v0 = A[1, 1], A[1, 2]

    D_linhas, d_linhas = [], []
    for R, t, M, mobs in zip(Rs, ts, Mlist, mlist):
        M3 = np.column_stack([M, np.zeros(len(M))])
        Mc = (R @ M3.T).T + t
        x = Mc[:, 0] / Mc[:, 2]
        y = Mc[:, 1] / Mc[:, 2]
        r2 = x ** 2 + y ** 2
        u_id = u0 + alpha * x + gamma * y
        v_id = v0 + beta * y

        for i in range(len(M)):
            D_linhas.append([(u_id[i] - u0) * r2[i], (u_id[i] - u0) * r2[i] ** 2])
            D_linhas.append([(v_id[i] - v0) * r2[i], (v_id[i] - v0) * r2[i] ** 2])
            d_linhas.append(mobs[i, 0] - u_id[i])
            d_linhas.append(mobs[i, 1] - v_id[i])

    D = np.array(D_linhas)
    d = np.array(d_linhas)
    k, *_ = np.linalg.lstsq(D, d, rcond=None)
    return k


# ---------------------------------------------------------------------------
# Seção 3.2/3.3 — Estimação de máxima verossimilhança completa (eq. 14)
# ---------------------------------------------------------------------------

def _empacotar(A, k, Rs, ts, fixar_gamma):
    alpha, gamma, u0 = A[0, 0], A[0, 1], A[0, 2]
    beta, v0 = A[1, 1], A[1, 2]

    intrin = [alpha, beta, u0, v0] if fixar_gamma else [alpha, beta, gamma, u0, v0]
    params = list(intrin) + [k[0], k[1]]
    for R, t in zip(Rs, ts):
        params.extend(rodrigues_inv(R))
        params.extend(t)
    return np.array(params, dtype=float)


def _desempacotar(params, n, fixar_gamma):
    if fixar_gamma:
        alpha, beta, u0, v0 = params[0:4]
        gamma = 0.0
        idx = 4
    else:
        alpha, beta, gamma, u0, v0 = params[0:5]
        idx = 5

    k1, k2 = params[idx], params[idx + 1]
    idx += 2

    A = np.array([[alpha, gamma, u0], [0.0, beta, v0], [0.0, 0.0, 1.0]])

    Rs, ts = [], []
    for _ in range(n):
        r = params[idx:idx + 3]
        t = params[idx + 3:idx + 6]
        idx += 6
        Rs.append(rodrigues(r))
        ts.append(t)

    return A, (k1, k2), Rs, ts


def refinar_mle(A0, k0, Rs0, ts0, Mlist, mlist, fixar_gamma=False, max_nfev=3000):
    """Refina todos os parâmetros minimizando (14) por Levenberg-Marquardt
    (scipy.optimize.least_squares, method='lm'), com rotações parametrizadas
    como vetor de Rodrigues. Calcula RMS de reprojeção e sigma de cada parâmetro
    a partir do jacobiano na solução (aprox. de covariância de mínimos quadrados).
    """
    n = len(Mlist)
    x0 = _empacotar(A0, k0, Rs0, ts0, fixar_gamma)
    n_pontos_total = sum(len(M) for M in Mlist)

    def residuos(x):
        A, k, Rs, ts = _desempacotar(x, n, fixar_gamma)
        partes = []
        for R, t, M, mobs in zip(Rs, ts, Mlist, mlist):
            M3 = np.column_stack([M, np.zeros(len(M))])
            proj = projetar(A, k, R, t, M3)
            partes.append((proj - mobs).flatten())
        return np.concatenate(partes)

    resultado = least_squares(residuos, x0, method="lm", x_scale="jac", max_nfev=max_nfev)

    A, k, Rs, ts = _desempacotar(resultado.x, n, fixar_gamma)
    res_final = resultado.fun
    ssr = float(np.sum(res_final ** 2))
    rms = np.sqrt(ssr / n_pontos_total)

    p = len(x0)
    dof = len(res_final) - p
    J = resultado.jac
    try:
        s2 = ssr / dof if dof > 0 else np.nan
        cov = s2 * np.linalg.pinv(J.T @ J)
        sigmas = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        sigmas = np.full(p, np.nan)

    n_intrin = 4 if fixar_gamma else 5
    sigma_intrinsecos = sigmas[:n_intrin]
    sigma_k = sigmas[n_intrin:n_intrin + 2]

    return {
        "A": A,
        "k": k,
        "Rs": Rs,
        "ts": ts,
        "rms": rms,
        "residuos": res_final.reshape(-1, 2),
        "fixar_gamma": fixar_gamma,
        "sigma_intrinsecos": sigma_intrinsecos,  # ordem: alpha,beta,[gamma,]u0,v0
        "sigma_k": sigma_k,
        "n_params": p,
        "sucesso": resultado.success,
    }


# ---------------------------------------------------------------------------
# Pipeline completo (Seção 3.4)
# ---------------------------------------------------------------------------

def _matriz_normalizacao_imagem(tamanho_imagem, mlist):
    """Similaridade N (eq. análoga à do Apêndice A) para condicionar a SVD de V:
    leva os pixels para uma faixa O(1) em torno da origem, por uma escala
    isotrópica (mesma escala em u e v, preserva a estrutura de A, inclusive a
    condição gamma=0). Usa o tamanho da imagem quando disponível; senão, cai
    para a extensão dos pontos observados.
    """
    if tamanho_imagem is not None:
        w_img, h_img = tamanho_imagem
        cx, cy = w_img / 2.0, h_img / 2.0
    else:
        todos = np.concatenate(mlist, axis=0)
        cx, cy = todos[:, 0].mean(), todos[:, 1].mean()
        w_img = todos[:, 0].max() - todos[:, 0].min()
        h_img = todos[:, 1].max() - todos[:, 1].min()

    w = max(w_img, h_img, 1.0)
    return np.array([
        [1.0 / w, 0.0, -cx / w],
        [0.0, 1.0 / w, -cy / w],
        [0.0, 0.0, 1.0],
    ])


def calibrar(Mlist, mlist, forcar_gamma_zero=None, tamanho_imagem=None):
    """Roda o procedimento recomendado na Seção 3.4, passos 4-6:
      4. Solução fechada dos intrínsecos e extrínsecos (Seção 3.1);
      5. k1, k2 por mínimos quadrados lineares (eq. 13);
      6. Refino de tudo pela MLE (eq. 14).
    Retorna um dicionário com 'fechada', 'mle' e as homografias 'Hs'.

    `tamanho_imagem` = (largura, altura) em pixels, usada para condicionar a
    SVD do sistema Vb=0 (ver nota em `intrinsecos_fechado`). Se None, a escala
    é estimada a partir da dispersão dos pontos detectados.
    """
    n = len(Mlist)
    if forcar_gamma_zero is None:
        forcar_gamma_zero = (n == 2)

    Hs = [homografia_dlt(M, m) for M, m in zip(Mlist, mlist)]

    # Condicionamento numérico: normaliza as homografias antes de montar V,
    # e desfaz a normalização em A0 (A0 = N^-1 A_norm) — ver `_matriz_normalizacao_imagem`.
    N = _matriz_normalizacao_imagem(tamanho_imagem, mlist)
    Hs_norm = [N @ H for H in Hs]
    A_norm, b = intrinsecos_fechado(Hs_norm, forcar_gamma_zero)
    A0 = np.linalg.inv(N) @ A_norm
    A0 = A0 / A0[2, 2]

    Rs0, ts0 = [], []
    for H in Hs:
        R, t = extrinsecos(A0, H)
        Rs0.append(R)
        ts0.append(t)

    k0 = distorcao_linear(A0, Rs0, ts0, Mlist, mlist)

    # RMS da solução fechada (intrínsecos/extrínsecos fechados + k linear)
    n_pontos_total = sum(len(M) for M in Mlist)
    res0 = []
    for R, t, M, mobs in zip(Rs0, ts0, Mlist, mlist):
        M3 = np.column_stack([M, np.zeros(len(M))])
        proj = projetar(A0, k0, R, t, M3)
        res0.append((proj - mobs).flatten())
    res0 = np.concatenate(res0)
    rms0 = np.sqrt(np.sum(res0 ** 2) / n_pontos_total)

    fechada = {"A": A0, "k": k0, "Rs": Rs0, "ts": ts0, "rms": rms0, "b": b}

    mle = refinar_mle(A0, k0, Rs0, ts0, Mlist, mlist, fixar_gamma=forcar_gamma_zero)

    return {"fechada": fechada, "mle": mle, "Hs": Hs, "forcar_gamma_zero": forcar_gamma_zero}


# ---------------------------------------------------------------------------
# Correção de distorção radial de uma imagem (para a figura estilo Fig. 5)
# ---------------------------------------------------------------------------

def corrigir_distorcao(img, A, k):
    """Gera a imagem sem distorção radial: para cada pixel (u,v) da imagem de
    saída (assumida ideal/sem distorção), calcula onde esse ponto realmente caiu
    na imagem capturada (distorcida) e amostra lá (interpolação bilinear).
    """
    from scipy.ndimage import map_coordinates

    h, w = img.shape[:2]
    alpha, gamma, u0 = A[0, 0], A[0, 1], A[0, 2]
    beta, v0 = A[1, 1], A[1, 2]
    Ainv = np.linalg.inv(A)

    us, vs = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    pts = np.stack([us.ravel(), vs.ravel(), np.ones(us.size)], axis=0)
    norm = Ainv @ pts
    x = norm[0] / norm[2]
    y = norm[1] / norm[2]
    r2 = x ** 2 + y ** 2
    fator = 1.0 + k[0] * r2 + k[1] * r2 ** 2
    xd = x * fator
    yd = y * fator
    ud = (u0 + alpha * xd + gamma * yd).reshape(h, w)
    vd = (v0 + beta * yd).reshape(h, w)

    if img.ndim == 2:
        out = map_coordinates(img, [vd, ud], order=1, mode="constant", cval=0)
    else:
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            out[:, :, c] = map_coordinates(img[:, :, c], [vd, ud], order=1, mode="constant", cval=0)
    return out.astype(img.dtype)


# ---------------------------------------------------------------------------
# Auto-teste com câmera sintética (parâmetros da Seção 5.1 do artigo)
# ---------------------------------------------------------------------------

def _teste_sintetico():
    """Valida a implementação com uma câmera sintética, no espírito do experimento
    da Seção 5.1 (câmera com alpha, beta, gamma, u0, v0 conhecidos; ruído gaussiano
    adicionado aos pontos projetados; compara os parâmetros recuperados com a verdade).

    Nota: a configuração de 3 planos com inclinação de só 20° usada literalmente na
    Seção 5.1 do artigo, testada durante o desenvolvimento, cobre uma fração pequena
    do quadro (o tabuleiro projeta em menos de 20% da imagem) e resultou, aqui, num
    sistema V quase degenerado (dois valores singulares pequenos e próximos), muito
    sensível a ruído — o mesmo fenômeno da Seção 4 (planos quase paralelos não dão
    informação nova), só que de forma aproximada em vez de exata. Por isso este
    autoteste usa 8 planos com inclinações maiores (25°-50°, eixos variados) e
    posicionados para cobrir boa parte do quadro, como se recomenda para uma
    calibração de verdade (e como fizemos ao fotografar as 10 imagens de cada
    câmera). Com sigma=0.5 px, o erro fica bem abaixo de 1% em alpha/beta e de
    poucos pixels em u0/v0, compatível com a Fig. 1 do artigo.
    """
    alpha, beta, gamma, u0, v0 = 1000.0, 1000.0, 0.0, 500.0, 500.0
    A_true = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
    k_true = (0.0, 0.0)
    tamanho_imagem = (1000, 1000)

    nx, ny, s = 9, 6, 30.0  # mesmo tabuleiro usado nas câmeras reais (9x6, 30mm)
    M = pontos_modelo(nx, ny, s)
    centro = M.mean(axis=0)

    rng_planos = np.random.default_rng(42)
    n_planos = 8
    rs, ts = [], []
    for _ in range(n_planos):
        eixo = rng_planos.normal(size=3)
        eixo /= np.linalg.norm(eixo)
        ang = np.deg2rad(rng_planos.uniform(25, 50))
        rs.append(eixo * ang)
        z = rng_planos.uniform(500, 900)
        ts.append(np.array([-centro[0], -centro[1], z]))

    rng_ruido = np.random.default_rng(7)
    sigma = 0.5
    Mlist, mlist = [], []
    for r, t in zip(rs, ts):
        R = rodrigues(r)
        pts = projetar(A_true, k_true, R, t, np.column_stack([M, np.zeros(len(M))]))
        pts_ruido = pts + rng_ruido.normal(0, sigma, pts.shape)
        Mlist.append(M)
        mlist.append(pts_ruido)

    resultado = calibrar(Mlist, mlist, forcar_gamma_zero=False, tamanho_imagem=tamanho_imagem)
    A_est = resultado["mle"]["A"]

    erro_alpha = abs(A_est[0, 0] - alpha) / alpha * 100
    erro_beta = abs(A_est[1, 1] - beta) / beta * 100
    erro_u0 = abs(A_est[0, 2] - u0)
    erro_v0 = abs(A_est[1, 2] - v0)

    print("=== Teste sintético (câmera com alpha=beta=1000, u0=v0=500, 8 planos, sigma=0.5px) ===")
    print(f"Verdade:   alpha={alpha:.2f} beta={beta:.2f} gamma={gamma:.5f} u0={u0:.2f} v0={v0:.2f}")
    print(f"Fechada:   alpha={resultado['fechada']['A'][0,0]:.2f} beta={resultado['fechada']['A'][1,1]:.2f} "
          f"gamma={resultado['fechada']['A'][0,1]:.5f} u0={resultado['fechada']['A'][0,2]:.2f} "
          f"v0={resultado['fechada']['A'][1,2]:.2f}  RMS={resultado['fechada']['rms']:.3f}")
    print(f"MLE:       alpha={A_est[0,0]:.2f} beta={A_est[1,1]:.2f} gamma={A_est[0,1]:.5f} "
          f"u0={A_est[0,2]:.2f} v0={A_est[1,2]:.2f}  RMS={resultado['mle']['rms']:.3f}")
    print(f"Erros MLE: alpha={erro_alpha:.3f}%  beta={erro_beta:.3f}%  "
          f"u0={erro_u0:.3f}px  v0={erro_v0:.3f}px")

    assert erro_alpha < 1.0, "erro em alpha maior que o esperado"
    assert erro_beta < 1.0, "erro em beta maior que o esperado"
    assert erro_u0 < 5.0, "erro em u0 maior que o esperado"
    assert erro_v0 < 5.0, "erro em v0 maior que o esperado"
    print("OK: parâmetros recuperados dentro da tolerância esperada.")


if __name__ == "__main__":
    _teste_sintetico()
