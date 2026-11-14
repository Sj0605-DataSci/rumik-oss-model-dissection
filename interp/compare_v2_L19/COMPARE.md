# Steering comparison, layer 19 (R=26.2)

Target = P(label) from external WavLM probe (speaker/tone/laugh) or duration ratio (pace). Held-out = text 1, 4 seeds. 'n/a' = <2/3 clips intact.

| label | base | top-d SAE | screened SAE | DiffMean |
|---|---|---|---|---|
| speaker=Zoya | 0.01 | 0.83 (f12262@0.3) | 0.92 (f29906@4) | 0.79 (dm@0.3) |
| speaker=Aisha | 0.01 | 0.97 (f25763@0.4) | 1.00 (f25763@4) | 0.83 (dm@0.3) |
| speaker=Siya | 0.00 | n/a | 0.66 (f21611@2) | 0.71 (dm@0.2) |
| tone=excited | 0.07 | 0.66 (f15027@0.3) | 0.93 (f28363@4) | 0.97 (dm@0.4) |
| tone=sad | 0.43 | 0.98 (f18319@0.2) | 0.97 (f18319@1) | 0.97 (dm@0.2) |
| tone=happy | 0.05 | 0.18 (f4777@0.4) | 0.36 (f30670@4) | 0.59 (dm@0.4) |
| tag=<laugh> | 0.02 | 0.06 (f1242@0.2) | 0.18 (f32678@4) | 0.58 (dm@0.4) |
| pace=fast | 1.00 | 1.04 (f11163@0.4) | 1.36 (f10117@4) | 1.44 (dm@0.4) |
| pace=slow | 1.00 | 1.75 (f8202@0.3) | 1.43 (f1020@4) | 1.70 (dm@0.2) |

Probe CV accuracy: speaker 99.4%, tone 54.7%, laugh 90.9%
