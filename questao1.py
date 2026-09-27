# -*- coding: utf-8 -*-
"""
questao1.py — Questão 1 (projeção com uma câmera dada) e o estudo de caso da
Questão 2 (repete a Questão 1 trocando os intrínsecos pelos calibrados de cada
câmera do celular).

Modelo: s*[u,v,1]^T = A*[R_c | t_c]*P_w, com R_c = R^T e t_c = -R^T*T, já que a
transformação dada no enunciado é câmera->mundo (P_w = R*P_c + T), logo
P_c = R^T*(P_w - T). Reaproveita `zhang.projetar` (mesma eq. 2 usada na
calibração) para não duplicar a matemática de projeção/distorção.
"""

import json
import os

import numpy as np

import zhang

RESULTADOS_DIR = "resultados"
SAIDA_DIR = os.path.join(RESULTADOS_DIR, "questao1")

# ---------------------------------------------------------------------------
# Dados do enunciado (Questão 1)
# ---------------------------------------------------------------------------

F_MM = 15.0
SENSOR_W, SENSOR_H = 4.0, 3.0  # mm
IMG_W, IMG_H = 1280, 720       # px
U0, V0 = 630.0, 355.0

R = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
])
T = np.array([0.0, 0.0, 2000.0])

PM = np.array([80.0, 20.0, 1200.0])
PONTOS_B = {
    "P1": np.array([0.0, 0.0, 6500.0]),
    "P2": np.array([200.0, -30.0, 1500.0]),
    "P3": np.array([300.0, -80.0, 5000.0]),
    "P4": np.array([80.0, 20.0, 2000.0]),
    "P5": np.array([90.0, -10.0, 2500.0]),
}


def montar_K(fx, fy, u0, v0):
    return np.array([[fx, 0.0, u0], [0.0, fy, v0], [0.0, 0.0, 1.0]])


def matriz_projecao(A, R_mundo_para_camera_T, T_vetor):
    """P = A [R^T | -R^T T], com R_mundo_para_camera_T a matriz R do enunciado
    (câmera->mundo); R_c = R^T, t_c = -R^T T (Seção 2.1 do artigo de Zhang,
    eq. 1, adaptada para uma transformação extrínseca genérica câmera<->mundo)."""
    Rc = R_mundo_para_camera_T.T
    tc = -Rc @ T_vetor
    return A @ np.column_stack([Rc, tc]), Rc, tc


def analisar_ponto(A, k_dist, Rc, tc, Pw, limites, tol=1e-9):
    """Classifica Pw como: atrás da câmera / no plano focal / fora / dentro,
    verificando Z_c > 0 ANTES de dividir (como pede o enunciado)."""
    Pc = Rc @ Pw + tc
    Zc = Pc[2]
    W, H = limites

    if Zc < -tol:
        return {"Zc": Zc, "classe": "atrás da câmera", "uv_sem": None, "uv_com": None}
    if abs(Zc) <= tol:
        return {"Zc": Zc, "classe": "no plano focal (Zc=0, sem projeção)", "uv_sem": None, "uv_com": None}

    uv_sem = zhang.projetar(A, (0.0, 0.0), Rc, tc, Pw.reshape(1, 3))[0]
    uv_com = zhang.projetar(A, k_dist, Rc, tc, Pw.reshape(1, 3))[0] if k_dist is not None else uv_sem

    dentro = (0 <= uv_sem[0] < W) and (0 <= uv_sem[1] < H)
    classe = "dentro da imagem" if dentro else "fora da imagem"
    return {"Zc": Zc, "classe": classe, "uv_sem": uv_sem, "uv_com": uv_com}


def fmt_uv(uv):
    if uv is None:
        return "—"
    return f"({uv[0]:.2f}, {uv[1]:.2f})"


# ---------------------------------------------------------------------------
# Questão 1a e 1b
# ---------------------------------------------------------------------------

def questao1():
    print("=== Questão 1a ===")
    fx = IMG_W / SENSOR_W * F_MM
    fy = IMG_H / SENSOR_H * F_MM
    print(f"kx = {IMG_W}/{SENSOR_W} = {IMG_W/SENSOR_W:.1f} px/mm ; fx = kx*f = {fx:.1f} px")
    print(f"ky = {IMG_H}/{SENSOR_H} = {IMG_H/SENSOR_H:.1f} px/mm ; fy = ky*f = {fy:.1f} px")

    K = montar_K(fx, fy, U0, V0)
    P, Rc, tc = matriz_projecao(K, R, T)
    print("K =\n", K)
    print("P = K[R^T | -R^T T] =\n", P)

    resultado = analisar_ponto(K, None, Rc, tc, PM, (IMG_W, IMG_H))
    print(f"\nProjeção de Pm = {PM.tolist()} mm -> uv = {fmt_uv(resultado['uv_sem'])}  "
          f"(Zc={resultado['Zc']:.1f} mm)")

    esperado = (750.0, 715.0)
    uv = resultado["uv_sem"]
    assert abs(uv[0] - esperado[0]) < 1e-6 and abs(uv[1] - esperado[1]) < 1e-6, (
        f"Esperado {esperado}, obtido {uv}"
    )
    print(f"OK: bate com o gabarito {esperado}.")

    print("\n=== Questão 1b ===")
    linhas = []
    for nome, Pw in PONTOS_B.items():
        r = analisar_ponto(K, None, Rc, tc, Pw, (IMG_W, IMG_H))
        print(f"{nome} = {Pw.tolist()} mm -> Zc={r['Zc']:.1f}  uv={fmt_uv(r['uv_sem'])}  => {r['classe']}")
        linhas.append((nome, Pw.tolist(), f"{r['Zc']:.1f}", fmt_uv(r["uv_sem"]), r["classe"]))

    os.makedirs(SAIDA_DIR, exist_ok=True)
    with open(os.path.join(SAIDA_DIR, "tabela_q1.md"), "w", encoding="utf-8") as f:
        f.write("## Questão 1a\n\n")
        f.write(f"K =\n\n```\n{K}\n```\n\n")
        f.write(f"P = K[R^T | -R^T T] =\n\n```\n{P}\n```\n\n")
        f.write(f"Projeção de Pm = {PM.tolist()} mm: uv = {fmt_uv(uv)} (esperado (750, 715))\n\n")
        f.write("## Questão 1b\n\n")
        f.write("| Ponto | Coordenadas (mm) | Zc (mm) | (u,v) | Classificação |\n")
        f.write("|---|---|---|---|---|\n")
        for nome, pw, zc, uvst, classe in linhas:
            f.write(f"| {nome} | {pw} | {zc} | {uvst} | {classe} |\n")

    return K, R, T


# ---------------------------------------------------------------------------
# Estudo de caso: mesmos R, T, trocando K pelos A calibrados
# ---------------------------------------------------------------------------

def carregar_calibracao(nome_cam):
    caminho = os.path.join(RESULTADOS_DIR, nome_cam, "calibracao.json")
    with open(caminho, "r", encoding="utf-8") as f:
        d = json.load(f)
    A = np.array(d["mle_gamma_livre"]["A"])
    k = tuple(d["mle_gamma_livre"]["k"])
    tamanho = tuple(d["tamanho_imagem"])
    return A, k, tamanho


def fov_graus(A, tamanho):
    W, H = tamanho
    alpha, beta = A[0, 0], A[1, 1]
    fov_h = 2 * np.degrees(np.arctan(W / (2 * alpha)))
    fov_v = 2 * np.degrees(np.arctan(H / (2 * beta)))
    return fov_h, fov_v


def estudo_de_caso():
    print("\n=== Estudo de caso (Questão 2): mesmos R, T; K trocado pelos A calibrados ===")

    K_enunciado = montar_K(IMG_W / SENSOR_W * F_MM, IMG_H / SENSOR_H * F_MM, U0, V0)
    cameras = {
        "enunciado": (K_enunciado, (0.0, 0.0), (IMG_W, IMG_H)),
        "principal": carregar_calibracao("principal"),
        "ultrawide": carregar_calibracao("ultrawide"),
    }
    # normaliza formato: (A, k, tamanho)
    cameras["enunciado"] = (K_enunciado, (0.0, 0.0), (IMG_W, IMG_H))

    todos_pontos = {"Pm": PM, **PONTOS_B}

    linhas_resumo = []
    detalhes_md = []

    for nome_cam, (A, k, tamanho) in cameras.items():
        Rc = R.T
        tc = -Rc @ T
        fov_h, fov_v = fov_graus(A, tamanho)
        print(f"\n--- Câmera: {nome_cam}  (alpha={A[0,0]:.1f}, beta={A[1,1]:.1f}, "
              f"imagem={tamanho}, FOV=({fov_h:.1f}°,{fov_v:.1f}°), k={k}) ---")

        detalhes_md.append(f"### {nome_cam}\n\nA =\n```\n{A}\n```\nk1,k2 = {k}  "
                            f"Imagem = {tamanho}  FOV ≈ ({fov_h:.1f}°, {fov_v:.1f}°)\n\n"
                            "| Ponto | Zc (mm) | sem distorção (u,v) | com distorção (u,v) | classe |\n"
                            "|---|---|---|---|---|\n")

        for nome_pt, Pw in todos_pontos.items():
            r = analisar_ponto(A, k, Rc, tc, Pw, tamanho)
            print(f"  {nome_pt}: Zc={r['Zc']:.1f}  sem_dist={fmt_uv(r['uv_sem'])}  "
                  f"com_dist={fmt_uv(r['uv_com'])}  => {r['classe']}")
            detalhes_md.append(
                f"| {nome_pt} | {r['Zc']:.1f} | {fmt_uv(r['uv_sem'])} | {fmt_uv(r['uv_com'])} | {r['classe']} |\n"
            )
            linhas_resumo.append((nome_cam, nome_pt, f"{r['Zc']:.1f}",
                                   fmt_uv(r["uv_sem"]), fmt_uv(r["uv_com"]), r["classe"]))
        detalhes_md.append("\n")

    os.makedirs(SAIDA_DIR, exist_ok=True)
    with open(os.path.join(SAIDA_DIR, "tabela_estudo_caso.md"), "w", encoding="utf-8") as f:
        f.write("## Estudo de caso — comparação enunciado x principal x ultra-wide\n\n")
        f.write("| Câmera | Ponto | Zc (mm) | sem distorção (u,v) | com distorção (u,v) | classe |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in linhas_resumo:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n")
        f.writelines(detalhes_md)

    print(f"\nTabelas salvas em {SAIDA_DIR}/")


if __name__ == "__main__":
    questao1()
    estudo_de_caso()
