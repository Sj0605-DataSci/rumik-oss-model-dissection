# SAE layer 19 - features

## Splice eval (next audio-token CE)

| clean | SAE spliced | mean-ablated | ΔCE | loss recovered |
|---|---|---|---|---|
| 2.1473 | 3.0558 | 12.0394 | +0.9085 | **90.8%** |

## Modality (fires ≥3× more on one type)
audio-only 23855 · text-only 7644 · both 1234 · dead 35

## Top features per label (Cohen's d, clip-mean over audio positions)

**speaker=Aisha**: f25763 (d=5.212, both), f745 (d=5.212, audio), f1988 (d=4.006, audio), f30703 (d=3.268, audio), f22570 (d=3.077, audio)

**speaker=Ira**: f5694 (d=4.033, audio), f26979 (d=3.963, audio), f1547 (d=3.372, audio), f17496 (d=3.303, audio), f7004 (d=3.232, text)

**speaker=Siya**: f21611 (d=5.485, audio), f16615 (d=3.378, audio), f25652 (d=3.371, audio), f9452 (d=3.195, audio), f3069 (d=3.125, audio)

**speaker=Zoya**: f12262 (d=8.35, audio), f6026 (d=4.963, audio), f28148 (d=4.364, audio), f7372 (d=3.928, audio), f10585 (d=3.534, both)

**tone=angry**: f16043 (d=2.164, audio), f13022 (d=1.176, audio), f11902 (d=1.165, audio), f25812 (d=1.108, both), f12634 (d=1.042, audio)

**tone=excited**: f15027 (d=0.922, both), f17023 (d=0.911, audio), f7317 (d=0.887, audio), f28363 (d=0.869, audio), f23016 (d=0.868, audio)

**tone=happy**: f4777 (d=0.52, text), f26864 (d=0.507, audio), f24114 (d=0.502, audio), f12607 (d=0.492, audio), f9508 (d=0.45, audio)

**tone=professional**: f1461 (d=0.74, audio), f22820 (d=0.529, audio), f15260 (d=0.487, audio), f20316 (d=0.464, audio), f30651 (d=0.436, audio)

**tone=sad**: f18319 (d=1.232, audio), f29637 (d=1.056, audio), f12829 (d=0.953, audio), f29082 (d=0.851, audio), f6913 (d=0.802, audio)

**accent=Hindi accent**: f22820 (d=0.227, audio), f23491 (d=0.216, audio), f24444 (d=0.215, audio), f20947 (d=0.212, audio), f2161 (d=0.209, audio)

**accent=Indian English**: f26989 (d=0.207, audio), f31590 (d=0.168, audio), f24605 (d=0.165, audio), f1601 (d=0.162, audio), f29833 (d=0.16, audio)

**lang=en**: f6775 (d=1.64, audio), f28543 (d=1.501, audio), f25320 (d=1.493, audio), f26273 (d=1.425, audio), f28867 (d=1.399, audio)

**lang=hi**: f18570 (d=1.869, audio), f11066 (d=1.84, audio), f13003 (d=1.572, audio), f8287 (d=1.559, audio), f1201 (d=1.554, audio)

**pace=fast**: f11163 (d=1.145, audio), f3399 (d=1.121, audio), f24298 (d=1.11, audio), f10117 (d=1.071, audio), f15027 (d=1.024, both)

**pace=slow**: f8202 (d=0.911, audio), f1020 (d=0.736, audio), f14500 (d=0.618, audio), f12607 (d=0.6, audio), f31856 (d=0.59, audio)

**pace=steady**: f1461 (d=0.39, audio), f22820 (d=0.353, audio), f15260 (d=0.336, audio), f9671 (d=0.319, audio), f18319 (d=0.293, audio)

**tag=<laugh>**: f1242 (d=1.264, audio), f28390 (d=1.257, audio), f9138 (d=1.212, audio), f24696 (d=1.188, audio), f16005 (d=1.176, audio)

**tag=<sigh>**: f13557 (d=1.816, audio), f15255 (d=1.275, audio), f23313 (d=1.214, audio), f31362 (d=1.172, audio), f18521 (d=1.057, audio)

**tag=<chuckle>**: f5415 (d=0.937, audio), f15040 (d=0.875, audio), f24696 (d=0.856, audio), f26465 (d=0.842, audio), f1242 (d=0.823, audio)
