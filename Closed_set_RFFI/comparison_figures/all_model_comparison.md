| Model | Accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) | Training time (h) | Epochs completed | Parameters (count) | FLOPs/sample (count) | Inference latency (ms/sample) | Confusion errors (samples) | Best validation accuracy (%) | Best epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Residual CNN | 99.5333 | 99.5427 | 99.5333 | 99.5326 | 0.5927 | 75 | 12475422 | 561435478.0000 | 0.2849 | 56 | 85.9667 | 45 |
| Transformer V1 | 84.6167 | 86.6835 | 84.6167 | 83.3800 | 0.6700 | 74 | 192222 | 12975518.0000 | 0.2043 | 1846 | 78.9667 | 44 |
| Transformer V2 | 92.8833 | 93.2841 | 92.8833 | 92.8209 | 2.3168 | 97 | 402014 | 127236004.0000 | 0.6289 | 854 | 77.7333 | 67 |
| Transformer V3 | 96.9667 | 97.0000 | 96.9667 | 96.9658 | 4.1447 | 144 | 453950 | 188197470.0000 | 0.8443 | 364 | 80.9667 | 114 |
| Transformer V4 (revised) | 99.0667 | 99.0763 | 99.0667 | 99.0639 | 3.9958 | 124 | 464254 | 318724830.0000 | 0.9896 | 112 | 84.6000 | 94 |
