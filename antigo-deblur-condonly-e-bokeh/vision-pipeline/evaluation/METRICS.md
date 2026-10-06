# Métricas de Avaliação e Bibliotecas

Este documento descreve as métricas de qualidade de imagem (IQA) e as bibliotecas utilizadas nesta pipeline para avaliar o desempenho do módulo DeblurNet, seguindo as métricas reportadas pelos autores.

## Bibliotecas Principais

A pipeline utiliza a seguinte biblioteca principal para garantir a reprodutibilidade e precisão dos resultados:

1. **[PyIQA](https://github.com/chaofengc/IQA-PyTorch)**: Uma biblioteca abrangente para avaliação de qualidade de imagem baseada em PyTorch. Ela fornece implementações otimizadas para métricas de referência (Full-Reference) e sem referência (No-Reference).

---

## Métricas Utilizadas

### 1. LPIPS (Learned Perceptual Image Patch Similarity)

- **Tipo**: Referência Total (Full-Reference)
- **Biblioteca**: `pyiqa` (modelo `lpips+`)
- **Descrição**: Utiliza uma rede neural profunda para comparar a similaridade perceptual entre a imagem restaurada e o Ground Truth, alinhando-se com o julgamento humano.
- **Interpretação**: **Quanto menor, melhor**.

### 2. DISTS (Deep Image Structure and Texture Similarity)

- **Tipo**: Referência Total (Full-Reference)
- **Biblioteca**: `pyiqa` (modelo `dists`)
- **Descrição**: Uma métrica que avalia a similaridade de estrutura e textura em níveis profundos. É projetada para ser tolerante a pequenas deformações espaciais, focando na qualidade da reconstrução de detalhes.
- **Interpretação**: **Quanto menor, melhor**.

### 3. CLIP-IQA

- **Tipo**: Sem Referência (No-Reference)
- **Biblioteca**: `pyiqa` (modelo `clipiqa+`)
- **Descrição**: Baseada no modelo CLIP (OpenAI), avalia a qualidade da imagem comparando-a com conceitos abstratos de "alta" e "baixa" qualidade.
- **Interpretação**: **Quanto maior, melhor**.

### 4. MANIQA (Multi-dimension Attention Network for IQA)

- **Tipo**: Sem Referência (No-Reference)
- **Biblioteca**: `pyiqa` (modelo `maniqa-kadid`)
- **Descrição**: Utiliza mecanismos de atenção para extrair características globais e locais, sendo sensível a distorções de processamento.
- **Interpretação**: **Quanto maior, melhor**.

### 5. MUSIQ (Multi-scale Image Quality Transformer)

- **Tipo**: Sem Referência (No-Reference)
- **Biblioteca**: `pyiqa` (modelo `musiq`)
- **Descrição**: Emprega arquitetura Transformer para capturar distorções em múltiplas escalas e resoluções variadas.
- **Interpretação**: **Quanto maior, melhor**.

---
