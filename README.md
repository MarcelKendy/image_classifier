---

# Projeto de Classificação de Imagens

Este projeto utiliza algoritmos de machine learning para classificar imagens em diferentes categorias. Ele foi desenvolvido em Python e requer algumas bibliotecas adicionais para funcionar.

## Pré-requisitos

1. **Python 3.6 ou superior**: Certifique-se de que o Python está instalado na sua máquina. [Baixe aqui](https://www.python.org/downloads/).

2. **Instalação das Bibliotecas Necessárias**: Execute os comandos abaixo para instalar as bibliotecas usadas no projeto:

```bash
pip install numpy scikit-image scikit-learn matplotlib
```

## Estrutura do Projeto

- `mpeg7_mod/`: Pasta contendo subpastas com as imagens organizadas por categoria.
- `classifier.py`: Script principal que executa o projeto de classificação de imagens.

Certifique-se de que as imagens estão organizadas na pasta `mpeg7_mod` em subpastas, com cada subpasta representando uma categoria.

## Como Executar

Para rodar o projeto, utilize o comando:

```bash
python classifier.py
```

Após a execução, os resultados serão exibidos no console e salvos em um arquivo `results.txt` na pasta do projeto. Esse arquivo conterá as métricas de classificação.

---
