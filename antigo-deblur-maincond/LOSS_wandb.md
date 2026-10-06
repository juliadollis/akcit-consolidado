### 📉 loss vs loss_ema

Em flow matching a loss por step é **muito ruidosa** (sorteia `sigma` + ruído a cada step), então sobe e desce mesmo aprendendo.

* `deblur/loss`: sinal cru, serrilhado.
* `deblur/loss_ema`: média móvel exponencial (`beta=0.98`) que suaviza e mostra a **tendência real**. Convergiu quando achata.
* `deblur/best_loss`: menor loss vista (referência).

**Nosso:** `loss_ema` caiu de ~0.55 para ~0.19 e achatou (convergiu ~step 700). É MSE no latente, então loss baixa não garante nitidez (isso se mede na inferência).
