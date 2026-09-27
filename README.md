# Calibração de câmeras pelo método de Zhang — Atividade 01 (Visão Computacional)

Implementação manual do método de Zhang (MSR-TR-98-71) para calibrar as duas câmeras
de um Samsung Galaxy S24 Ultra (principal 1x e ultra-wide 0.6x), com o estudo de caso
da Questão 1 refeito com os intrínsecos calibrados.

## Estrutura

```
zhang.py          # funções do método de Zhang (homografia, intrínsecos, extrínsecos,
                  # distorção, MLE) — só numpy/scipy; sem OpenCV além da detecção de cantos
calibrar.py       # detecta os cantos e roda a calibração completa das duas câmeras
questao1.py       # Questão 1 (a/b) e o estudo de caso com os A calibrados
princial_1x/      # 10 fotos da câmera principal (nome de pasta como recebido)
ultrawide_0.6x/   # 10 fotos da câmera ultra-wide
resultados/
  principal/      # tabelas, figuras e calibracao.json da câmera principal
  ultrawide/      # idem, ultra-wide
  questao1/       # tabelas da Questão 1 e do estudo de caso
  relatorio.md    # relatório final (procedimento, tabelas, figuras, discussão)
```

## Dependências

Python 3.12 com numpy, opencv-python, scipy, matplotlib e Pillow (já usados no
desenvolvimento: numpy 2.1, opencv 4.10, scipy 1.15, matplotlib 3.9, Pillow 11).

```
pip install numpy opencv-python scipy matplotlib pillow
```

## Como rodar

1. **Só a detecção de cantos** (rápido, ~10s por câmera; útil para conferir se todas
   as fotos foram aceitas antes de calibrar):

   ```
   python calibrar.py --so-deteccao
   ```

   Gera `resultados/<camera>/cantos/*.jpg` (fotos com os cantos desenhados) e
   imprime quantas fotos foram aceitas/rejeitadas em cada câmera.

2. **Calibração completa das duas câmeras** (detecção + tabelas + figuras;
   leva alguns minutos por causa dos ajustes de estabilidade/evolução):

   ```
   python calibrar.py
   ```

   Use `--forcar-deteccao` para ignorar o cache `cantos.npz` e detectar os
   cantos de novo (por exemplo, se as fotos ou o valor de `s` mudarem).

3. **Questão 1 e o estudo de caso** (precisa ter rodado `calibrar.py` antes,
   pois lê `resultados/<camera>/calibracao.json`):

   ```
   python questao1.py
   ```

4. **Auto-teste do módulo `zhang.py`** (valida a implementação com uma câmera
   sintética, sem depender das fotos):

   ```
   python zhang.py
   ```

## Parâmetros do tabuleiro

Definidos no topo de `calibrar.py`: `NX, NY = 9, 6` (cantos internos de um
tabuleiro de 10x7 quadrados) e `S = 36.0` (mm, lado do quadrado). Ajuste esses
valores ali se o padrão mudar.

## Leitura dos resultados

- `resultados/<camera>/tabela_solucao.md` — solução fechada x MLE (γ livre) x
  MLE (γ=0) x OpenCV, no estilo da Tabela 1 do artigo.
- `resultados/<camera>/tabela_evolucao.md` — evolução com n = 2, 3, 5, 7, 10 imagens.
- `resultados/<camera>/tabela_estabilidade.md` — estabilidade (10 combinações
  de 9 imagens), no estilo da Tabela 2.
- `resultados/<camera>/tabela_fabricante.md` — comparação com a ficha técnica.
- `resultados/<camera>/fig_*.png` — resíduos, RMS por imagem, antes/depois da
  correção de distorção, poses 3D do tabuleiro.
- `resultados/<camera>/calibracao.json` — parâmetros finais em formato bruto.
- `resultados/relatorio.md` — o relatório reunindo tudo isso com discussão.
