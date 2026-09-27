# -*- coding: utf-8 -*-
"""
calibrar.py — Calibra as duas câmeras do Samsung Galaxy S24 Ultra (principal 1x e
ultra-wide 0.6x) pelo método de Zhang, usando as funções implementadas em zhang.py.

Uso:
    python calibrar.py                  # detecção + calibração completa das 2 câmeras
    python calibrar.py --so-deteccao    # roda só a etapa de detecção de cantos
    python calibrar.py --forcar-deteccao  # ignora o cache de cantos e detecta de novo

Gera, para cada câmera, em resultados/<camera>/:
    cantos/*.jpg          fotos com os cantos desenhados
    cantos.npz            cache dos cantos detectados
    calibracao.json       parâmetros finais (A, k, sigma, RMS)
    tabela_solucao.md      Tabela 1 do artigo: fechada x MLE x OpenCV
    tabela_evolucao.md     evolução com n = 2,3,5,7,10 imagens
    tabela_estabilidade.md estabilidade (10 combinações de 9 imagens) — Tabela 2
    tabela_fabricante.md   comparação com a ficha técnica
    fig_residuos.png, fig_rms_por_imagem.png, fig_distorcao.png, fig_poses3d.png
"""

import argparse
import glob
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (necessário para projection='3d')

import zhang

# ---------------------------------------------------------------------------
# Configuração (parâmetros no topo, como pedido no enunciado)
# ---------------------------------------------------------------------------

NX, NY = 9, 6          # cantos internos do tabuleiro (10x7 quadrados)
S = 36.0                # mm, lado do quadrado (medido: 130 px na tela x 0.2767 mm/px)
ESCALA_DETECCAO = 0.4   # fator de redução para a primeira passada de detecção

PASTAS = {
    "principal": "princial_1x",
    "ultrawide": "ultrawide_0.6x",
}

# Dados do fabricante/EXIF usados na comparação (Seção "Comparação com o fabricante").
# Fontes dos tamanhos de sensor: GSMArena / SamMobile (ver relatorio.md para as citações).
EXIF_INFO = {
    "principal": dict(
        f_mm_exif=6.3,
        f35=23,
        titulo="Câmera principal (1x)",
        pixel_pitch_um=2.4,   # ISOCELL HP2, pixel nativo 0.6 um, binning 4x4 -> 2.4 um efetivo em 12MP
        sensor_desc="Samsung ISOCELL HP2, 1/1.3\", 200MP nativo (pixel 0.6 um), "
                    "saida 12MP com binning 4x4 (pixel efetivo 2.4 um)",
    ),
    "ultrawide": dict(
        f_mm_exif=2.2,
        f35=13,
        titulo="Câmera ultra-wide (0.6x)",
        pixel_pitch_um=1.4,   # ficha GSMArena; há fontes que calculam 1.12 um
        sensor_desc="Sensor 12MP, 1/2.55\" (Sony IMX564), pixel 1.4 um conforme ficha "
                    "tecnica do fabricante (analises tecnicas independentes calculam ~1.12 um)",
    ),
}

RESULTADOS_DIR = "resultados"
DIAGONAL_35MM = 43.27  # mm, diagonal do quadro 36x24mm


# ---------------------------------------------------------------------------
# Utilidades gerais
# ---------------------------------------------------------------------------

def listar_fotos(pasta):
    # No Windows o sistema de arquivos é case-insensitive, então glob("*.jpg") já
    # pega "*.JPG"; usar os dois padrões duplicaria os arquivos. Dedup por segurança
    # (caso rode em um SO case-sensitive com nomes mistos).
    candidatos = glob.glob(os.path.join(pasta, "*.jpg")) + glob.glob(os.path.join(pasta, "*.JPG"))
    vistos, unicos = set(), []
    for c in candidatos:
        chave = os.path.normcase(os.path.abspath(c))
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(c)
    return sorted(unicos)


def formatar_tabela_md(cabecalho, linhas):
    md = "| " + " | ".join(cabecalho) + " |\n"
    md += "|" + "|".join(["---"] * len(cabecalho)) + "|\n"
    for linha in linhas:
        md += "| " + " | ".join(str(x) for x in linha) + " |\n"
    return md


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.{nd}f}"


# ---------------------------------------------------------------------------
# Etapa 1: detecção dos cantos (com cache em cantos.npz)
# ---------------------------------------------------------------------------

def detectar_todas(nome_cam, pasta, saida_dir, forcar=False):
    os.makedirs(saida_dir, exist_ok=True)
    cache_path = os.path.join(saida_dir, "cantos.npz")
    caminhos = listar_fotos(pasta)

    if os.path.exists(cache_path) and not forcar:
        d = np.load(cache_path, allow_pickle=True)
        return {
            "nome_cam": nome_cam,
            "Mlist": [zhang.pontos_modelo(NX, NY, S) for _ in d["arquivos"]],
            "mlist": [m for m in d["mlist"]],
            "tamanho_imagem": tuple(int(v) for v in d["tamanho_imagem"]),
            "arquivos": list(d["arquivos"]),
            "aceitas": list(d["aceitas"]),
            "falhas": list(d["falhas"]),
        }

    os.makedirs(os.path.join(saida_dir, "cantos"), exist_ok=True)
    M_modelo = zhang.pontos_modelo(NX, NY, S)

    Mlist, mlist, arquivos_ok = [], [], []
    aceitas, falhas = [], []
    tamanho_imagem = None

    for caminho in caminhos:
        nome = os.path.basename(caminho)
        saida_img = os.path.join(saida_dir, "cantos", nome)
        corners, metodo, shape = zhang.detectar_cantos(
            caminho, NX, NY, ESCALA_DETECCAO, salvar_desenho=saida_img
        )
        if corners is None:
            falhas.append((nome, metodo))
            print(f"  [FALHA] {nome}: {metodo}")
            continue

        h, w = shape
        if tamanho_imagem is None:
            tamanho_imagem = (w, h)
        elif (w, h) != tamanho_imagem:
            print(f"  [AVISO] {nome} tem tamanho {(w, h)}, esperado {tamanho_imagem}")

        aceitas.append((nome, metodo))
        Mlist.append(M_modelo)
        mlist.append(corners)
        arquivos_ok.append(caminho)
        print(f"  [OK]    {nome} ({metodo})")

    np.savez(
        cache_path,
        mlist=np.array(mlist),
        arquivos=np.array(arquivos_ok),
        tamanho_imagem=np.array(tamanho_imagem),
        aceitas=np.array(aceitas, dtype=object),
        falhas=np.array(falhas, dtype=object),
    )

    return {
        "nome_cam": nome_cam,
        "Mlist": Mlist,
        "mlist": mlist,
        "tamanho_imagem": tamanho_imagem,
        "arquivos": arquivos_ok,
        "aceitas": aceitas,
        "falhas": falhas,
    }


# ---------------------------------------------------------------------------
# Etapa 2: comparação com o OpenCV
# ---------------------------------------------------------------------------

def rodar_opencv(Mlist, mlist, tamanho_imagem):
    obj_pts = [np.column_stack([M, np.zeros(len(M))]).astype(np.float32) for M in Mlist]
    img_pts = [m.astype(np.float32) for m in mlist]
    flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, tamanho_imagem, None, None, flags=flags)
    d = dist.flatten()
    return {"A": K, "k": (float(d[0]), float(d[1])), "rms": float(rms)}


# ---------------------------------------------------------------------------
# Etapa 3: tabelas
# ---------------------------------------------------------------------------

def tabela_principal_md(fechada, mle_livre, mle_fix, ocv):
    A0, k0 = fechada["A"], fechada["k"]
    Am, km = mle_livre["A"], mle_livre["k"]
    sig, sigk = mle_livre["sigma_intrinsecos"], mle_livre["sigma_k"]
    Af, kf = mle_fix["A"], mle_fix["k"]
    Ao, ko = ocv["A"], ocv["k"]

    linhas = [
        ("alpha (px)", A0[0, 0], Am[0, 0], sig[0], Af[0, 0], Ao[0, 0]),
        ("beta (px)", A0[1, 1], Am[1, 1], sig[1], Af[1, 1], Ao[1, 1]),
        ("gamma", A0[0, 1], Am[0, 1], sig[2], 0.0, 0.0),
        ("u0 (px)", A0[0, 2], Am[0, 2], sig[3], Af[0, 2], Ao[0, 2]),
        ("v0 (px)", A0[1, 2], Am[1, 2], sig[4], Af[1, 2], Ao[1, 2]),
        ("k1", k0[0], km[0], sigk[0], kf[0], ko[0]),
        ("k2", k0[1], km[1], sigk[1], kf[1], ko[1]),
        ("RMS (px)", fechada["rms"], mle_livre["rms"], None, mle_fix["rms"], ocv["rms"]),
    ]
    cabecalho = ["Parâmetro", "Fechada", "MLE (γ livre)", "σ (MLE)", "MLE (γ=0)", "OpenCV (γ=0)"]
    linhas_fmt = [(n, fmt(v0), fmt(v1), fmt(s, 4) if s is not None else "—", fmt(v2), fmt(v3))
                  for n, v0, v1, s, v2, v3 in linhas]
    return formatar_tabela_md(cabecalho, linhas_fmt)


def tabela_evolucao_md(Mlist, mlist, tamanho_imagem, ns=(2, 3, 5, 7, 10)):
    """Evolução da calibração ao usar as n primeiras imagens (estilo Tabela 1)."""
    linhas_alpha, linhas_beta, linhas_gamma = [], [], []
    linhas_u0, linhas_v0, linhas_k1, linhas_k2, linhas_rms = [], [], [], [], []
    colunas = []

    for n in ns:
        n = min(n, len(Mlist))
        sub_M, sub_m = Mlist[:n], mlist[:n]
        res = zhang.calibrar(sub_M, sub_m, tamanho_imagem=tamanho_imagem)
        A0, Am, sig = res["fechada"]["A"], res["mle"]["A"], res["mle"]["sigma_intrinsecos"]
        k0, km, sigk = res["fechada"]["k"], res["mle"]["k"], res["mle"]["sigma_k"]
        fixado = res["mle"]["fixar_gamma"]

        colunas.append(f"n={n}")
        linhas_alpha.append((fmt(A0[0, 0]), fmt(Am[0, 0]), fmt(sig[0], 3)))
        linhas_beta.append((fmt(A0[1, 1]), fmt(Am[1, 1]), fmt(sig[1], 3)))
        if fixado:
            linhas_gamma.append(("0 (fixo)", "0 (fixo)", "—"))
        else:
            linhas_gamma.append((fmt(A0[0, 1]), fmt(Am[0, 1]), fmt(sig[2], 3)))
        linhas_u0.append((fmt(A0[0, 2]), fmt(Am[0, 2]), fmt(sig[3 if not fixado else 2], 3)))
        linhas_v0.append((fmt(A0[1, 2]), fmt(Am[1, 2]), fmt(sig[4 if not fixado else 3], 3)))
        linhas_k1.append((fmt(k0[0]), fmt(km[0]), fmt(sigk[0], 4)))
        linhas_k2.append((fmt(k0[1]), fmt(km[1]), fmt(sigk[1], 4)))
        linhas_rms.append((fmt(res["fechada"]["rms"], 3), fmt(res["mle"]["rms"], 3), "—"))

    cab_nivel2 = ["Parâmetro"] + sum([[f"{c} inicial", f"{c} final", f"{c} σ"] for c in colunas], [])

    def linha_final(nome, dados):
        r = [nome]
        for tripla in dados:
            r.extend(tripla)
        return r

    linhas = [
        linha_final("alpha", linhas_alpha),
        linha_final("beta", linhas_beta),
        linha_final("gamma", linhas_gamma),
        linha_final("u0", linhas_u0),
        linha_final("v0", linhas_v0),
        linha_final("k1", linhas_k1),
        linha_final("k2", linhas_k2),
        linha_final("RMS", linhas_rms),
    ]
    return formatar_tabela_md(cab_nivel2, linhas)


def tabela_estabilidade_md(Mlist, mlist, tamanho_imagem):
    """Estabilidade: calibração com cada uma das combinações de (n-1) imagens
    (deixando uma de fora por vez), no estilo da Tabela 2 do artigo."""
    n = len(Mlist)
    resultados = []
    for i in range(n):
        idxs = [j for j in range(n) if j != i]
        sub_M = [Mlist[j] for j in idxs]
        sub_m = [mlist[j] for j in idxs]
        res = zhang.calibrar(sub_M, sub_m, forcar_gamma_zero=False, tamanho_imagem=tamanho_imagem)
        resultados.append(res["mle"])

    def coletar(fn):
        return np.array([fn(r) for r in resultados])

    alphas = coletar(lambda r: r["A"][0, 0])
    betas = coletar(lambda r: r["A"][1, 1])
    gammas = coletar(lambda r: r["A"][0, 1])
    u0s = coletar(lambda r: r["A"][0, 2])
    v0s = coletar(lambda r: r["A"][1, 2])
    k1s = coletar(lambda r: r["k"][0])
    k2s = coletar(lambda r: r["k"][1])
    rmss = coletar(lambda r: r["rms"])
    aspecto = alphas / betas

    colunas = [f"sem_{i+1}" for i in range(n)]
    cabecalho = ["Parâmetro"] + colunas + ["média", "desvio"]

    def linha(nome, vals, nd=4):
        return [nome] + [fmt(v, nd) for v in vals] + [fmt(vals.mean(), nd), fmt(vals.std(ddof=1), nd)]

    linhas = [
        linha("alpha", alphas, 2),
        linha("beta", betas, 2),
        linha("gamma", gammas, 4),
        linha("u0", u0s, 2),
        linha("v0", v0s, 2),
        linha("k1", k1s, 5),
        linha("k2", k2s, 5),
        linha("alpha/beta", aspecto, 5),
        linha("RMS", rmss, 3),
    ]
    return formatar_tabela_md(cabecalho, linhas), resultados


# ---------------------------------------------------------------------------
# Etapa 4: comparação com o fabricante
# ---------------------------------------------------------------------------

def tabela_fabricante_md(nome_cam, A_mle, tamanho_imagem):
    info = EXIF_INFO[nome_cam]
    w, h = tamanho_imagem
    diag_px = np.sqrt(w ** 2 + h ** 2)

    f_px_exif = info["f35"] * diag_px / DIAGONAL_35MM
    alpha, beta = A_mle[0, 0], A_mle[1, 1]
    u0, v0 = A_mle[0, 2], A_mle[1, 2]
    cx, cy = w / 2.0, h / 2.0

    f_mm_calibrado = alpha * info["pixel_pitch_um"] / 1000.0

    linhas = [
        ("Distância focal EXIF (mm)", fmt(info["f_mm_exif"], 2)),
        ("Distância focal 35mm equiv. (mm)", fmt(info["f35"], 1)),
        ("f estimado a partir do EXIF (px) = f35*diag_px/43.27", fmt(f_px_exif, 1)),
        ("alpha calibrado (px)", fmt(alpha, 2)),
        ("beta calibrado (px)", fmt(beta, 2)),
        ("Diferença alpha vs f(EXIF) (%)", fmt(100 * (alpha - f_px_exif) / f_px_exif, 2)),
        ("f estimado a partir de alpha (mm) = alpha*pixel_pitch", fmt(f_mm_calibrado, 3)),
        ("Distância focal EXIF (mm), para comparação", fmt(info["f_mm_exif"], 2)),
        ("Ponto principal calibrado (u0,v0)", f"({fmt(u0,1)}, {fmt(v0,1)})"),
        ("Centro geométrico da imagem (w/2,h/2)", f"({fmt(cx,1)}, {fmt(cy,1)})"),
        ("Deslocamento do ponto principal (px)", fmt(np.hypot(u0 - cx, v0 - cy), 2)),
        ("Razão alpha/beta", fmt(alpha / beta, 5)),
        ("Sensor / pixel pitch (fabricante)", info["sensor_desc"]),
    ]
    return formatar_tabela_md(["Grandeza", "Valor"], linhas)


# ---------------------------------------------------------------------------
# Etapa 5: figuras
# ---------------------------------------------------------------------------

def fig_residuos(nome_cam, mle, saida_dir):
    res = mle["residuos"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].scatter(res[:, 0], res[:, 1], s=4, alpha=0.5)
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].axvline(0, color="gray", lw=0.5)
    axes[0].set_xlabel("resíduo em u (px)")
    axes[0].set_ylabel("resíduo em v (px)")
    axes[0].set_title(f"Dispersão dos resíduos — {nome_cam}")
    axes[0].axis("equal")

    dist = np.sqrt((res ** 2).sum(axis=1))
    axes[1].hist(dist, bins=25, color="steelblue", edgecolor="black")
    axes[1].set_xlabel("erro de reprojeção (px)")
    axes[1].set_ylabel("número de pontos")
    axes[1].set_title(f"Histograma do erro — {nome_cam} (RMS={mle['rms']:.3f}px)")

    fig.tight_layout()
    fig.savefig(os.path.join(saida_dir, "fig_residuos.png"), dpi=150)
    plt.close(fig)


def fig_rms_por_imagem(nome_cam, mle, n_pontos_por_imagem, nomes_arquivos, saida_dir):
    res = mle["residuos"]
    n_img = len(nomes_arquivos)
    rms_img = []
    idx = 0
    for i in range(n_img):
        m = n_pontos_por_imagem[i]
        bloco = res[idx: idx + m]
        rms_img.append(np.sqrt((bloco ** 2).sum() / m))
        idx += m

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [os.path.basename(a) for a in nomes_arquivos]
    ax.bar(range(n_img), rms_img, color="darkorange")
    ax.set_xticks(range(n_img))
    ax.set_xticklabels([f"{i+1}" for i in range(n_img)], rotation=0)
    ax.axhline(mle["rms"], color="black", ls="--", lw=1, label=f"RMS geral = {mle['rms']:.3f}px")
    ax.set_xlabel("imagem")
    ax.set_ylabel("RMS (px)")
    ax.set_title(f"RMS de reprojeção por imagem — {nome_cam}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(saida_dir, "fig_rms_por_imagem.png"), dpi=150)
    plt.close(fig)
    return rms_img, labels


def fig_distorcao(nome_cam, caminho_imagem, A, k, saida_dir):
    img = cv2.imread(caminho_imagem, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # reduz para acelerar a correção (mantendo A/pixel_pitch consistentes via reescala)
    escala = 0.5
    pequena = cv2.resize(img_rgb, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    A_pequena = A.copy()
    A_pequena[0, 0] *= escala
    A_pequena[1, 1] *= escala
    A_pequena[0, 1] *= escala
    A_pequena[0, 2] *= escala
    A_pequena[1, 2] *= escala

    corrigida = zhang.corrigir_distorcao(pequena, A_pequena, k)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(pequena)
    axes[0].set_title(f"Antes da correção — {nome_cam}")
    axes[0].axis("off")
    axes[1].imshow(corrigida)
    axes[1].set_title(f"Depois da correção (k1={k[0]:.4f}, k2={k[1]:.4f})")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(saida_dir, "fig_distorcao.png"), dpi=150)
    plt.close(fig)


def fig_poses3d(nome_cam, Rs, ts, nx, ny, s, saida_dir):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    cantos_modelo = np.array([
        [0, 0, 0], [(nx - 1) * s, 0, 0], [(nx - 1) * s, (ny - 1) * s, 0], [0, (ny - 1) * s, 0],
    ])

    cores = plt.cm.viridis(np.linspace(0, 1, len(Rs)))
    for R, t, cor in zip(Rs, ts, cores):
        pts_cam = (R @ cantos_modelo.T).T + t
        poly = np.vstack([pts_cam, pts_cam[0]])
        ax.plot(poly[:, 0], poly[:, 1], poly[:, 2], color=cor)
        ax.scatter(*pts_cam[0], color=cor, s=15)

    # frustum simples da câmera na origem
    esc = np.mean([np.linalg.norm(t) for t in ts]) * 0.15
    cam_pts = np.array([[0, 0, 0], [esc, esc, esc * 1.5], [-esc, esc, esc * 1.5],
                         [-esc, -esc, esc * 1.5], [esc, -esc, esc * 1.5]])
    for i in range(1, 5):
        ax.plot(*zip(cam_pts[0], cam_pts[i]), color="black", lw=1)
    base = np.vstack([cam_pts[1:], cam_pts[1]])
    ax.plot(base[:, 0], base[:, 1], base[:, 2], color="black", lw=1)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Poses do tabuleiro em relação à câmera — {nome_cam}")
    fig.tight_layout()
    fig.savefig(os.path.join(saida_dir, "fig_poses3d.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline por câmera
# ---------------------------------------------------------------------------

def processar_camera(nome_cam, pasta, forcar_deteccao):
    saida_dir = os.path.join(RESULTADOS_DIR, nome_cam)
    print(f"\n=== {nome_cam} ({pasta}) ===")
    dados = detectar_todas(nome_cam, pasta, saida_dir, forcar=forcar_deteccao)

    n_ok = len(dados["arquivos"])
    n_total = n_ok + len(dados["falhas"])
    print(f"Cantos detectados: {n_ok}/{n_total} imagens aceitas. Tamanho: {dados['tamanho_imagem']}")
    if dados["falhas"]:
        print("Falharam:", dados["falhas"])

    return dados


def calibrar_camera(dados):
    nome_cam = dados["nome_cam"]
    saida_dir = os.path.join(RESULTADOS_DIR, nome_cam)
    Mlist, mlist, tamanho_imagem = dados["Mlist"], dados["mlist"], dados["tamanho_imagem"]
    arquivos = dados["arquivos"]

    print(f"Calibrando {nome_cam} com {len(Mlist)} imagens...")

    # --- solução completa (gamma livre) e com gamma fixo, para comparação com OpenCV ---
    res_livre = zhang.calibrar(Mlist, mlist, forcar_gamma_zero=False, tamanho_imagem=tamanho_imagem)
    res_fixo = zhang.calibrar(Mlist, mlist, forcar_gamma_zero=True, tamanho_imagem=tamanho_imagem)
    ocv = rodar_opencv(Mlist, mlist, tamanho_imagem)

    tabela_sol = tabela_principal_md(res_livre["fechada"], res_livre["mle"], res_fixo["mle"], ocv)
    with open(os.path.join(saida_dir, "tabela_solucao.md"), "w", encoding="utf-8") as f:
        f.write(tabela_sol)

    # --- evolução com n imagens ---
    tabela_evo = tabela_evolucao_md(Mlist, mlist, tamanho_imagem)
    with open(os.path.join(saida_dir, "tabela_evolucao.md"), "w", encoding="utf-8") as f:
        f.write(tabela_evo)

    # --- estabilidade (leave-one-out) ---
    tabela_est, _ = tabela_estabilidade_md(Mlist, mlist, tamanho_imagem)
    with open(os.path.join(saida_dir, "tabela_estabilidade.md"), "w", encoding="utf-8") as f:
        f.write(tabela_est)

    # --- comparação com o fabricante ---
    tabela_fab = tabela_fabricante_md(nome_cam, res_livre["mle"]["A"], tamanho_imagem)
    with open(os.path.join(saida_dir, "tabela_fabricante.md"), "w", encoding="utf-8") as f:
        f.write(tabela_fab)

    # --- figuras ---
    mle = res_livre["mle"]
    n_pontos_por_imagem = [len(M) for M in Mlist]
    fig_residuos(nome_cam, mle, saida_dir)
    rms_img, labels = fig_rms_por_imagem(nome_cam, mle, n_pontos_por_imagem, arquivos, saida_dir)
    fig_distorcao(nome_cam, arquivos[0], mle["A"], mle["k"], saida_dir)
    fig_poses3d(nome_cam, mle["Rs"], mle["ts"], NX, NY, S, saida_dir)

    # --- salva calibracao.json ---
    calib = {
        "camera": nome_cam,
        "n_imagens": len(Mlist),
        "tamanho_imagem": tamanho_imagem,
        "fechada": {"A": res_livre["fechada"]["A"].tolist(), "k": list(res_livre["fechada"]["k"]),
                    "rms": res_livre["fechada"]["rms"]},
        "mle_gamma_livre": {"A": mle["A"].tolist(), "k": list(mle["k"]), "rms": mle["rms"],
                             "sigma_intrinsecos": mle["sigma_intrinsecos"].tolist(),
                             "sigma_k": mle["sigma_k"].tolist()},
        "mle_gamma_zero": {"A": res_fixo["mle"]["A"].tolist(), "k": list(res_fixo["mle"]["k"]),
                            "rms": res_fixo["mle"]["rms"]},
        "opencv": {"A": ocv["A"].tolist(), "k": list(ocv["k"]), "rms": ocv["rms"]},
        "rms_por_imagem": rms_img,
        "arquivos": [os.path.basename(a) for a in arquivos],
    }
    with open(os.path.join(saida_dir, "calibracao.json"), "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)

    print(f"  MLE (gamma livre): alpha={mle['A'][0,0]:.2f} beta={mle['A'][1,1]:.2f} "
          f"gamma={mle['A'][0,1]:.4f} u0={mle['A'][0,2]:.2f} v0={mle['A'][1,2]:.2f} "
          f"k1={mle['k'][0]:.4f} k2={mle['k'][1]:.4f} RMS={mle['rms']:.3f}px")
    print(f"  OpenCV (gamma=0):   alpha={ocv['A'][0,0]:.2f} beta={ocv['A'][1,1]:.2f} "
          f"u0={ocv['A'][0,2]:.2f} v0={ocv['A'][1,2]:.2f} "
          f"k1={ocv['k'][0]:.4f} k2={ocv['k'][1]:.4f} RMS={ocv['rms']:.3f}px")

    return calib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so-deteccao", action="store_true", help="roda só a detecção de cantos")
    parser.add_argument("--forcar-deteccao", action="store_true", help="ignora o cache de cantos")
    args = parser.parse_args()

    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    todos_dados = {}
    for nome_cam, pasta in PASTAS.items():
        todos_dados[nome_cam] = processar_camera(nome_cam, pasta, args.forcar_deteccao)

    print("\n=== Resumo da detecção ===")
    for nome_cam, dados in todos_dados.items():
        n_ok = len(dados["arquivos"])
        n_total = n_ok + len(dados["falhas"])
        print(f"{nome_cam}: {n_ok}/{n_total} aceitas")

    if args.so_deteccao:
        return

    for nome_cam, dados in todos_dados.items():
        calibrar_camera(dados)

    print("\nConcluído. Resultados em", RESULTADOS_DIR)


if __name__ == "__main__":
    main()
