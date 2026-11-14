# SAE layer 19 - features

## Splice eval (next audio-token CE)

| clean | SAE spliced | mean-ablated | ΔCE | loss recovered |
|---|---|---|---|---|
| 2.1520 | 3.4444 | 12.0234 | +1.2924 | **86.9%** |

## Modality (fires ≥3× more on one type)
audio-only 25216 · text-only 6599 · both 951 · dead 2

## Top features per label (Cohen's d, clip-mean over audio positions)

**speaker=Aisha**: f17660 (d=5.565, audio), f1531 (d=4.583, audio), f5255 (d=3.451, audio), f25681 (d=2.86, audio), f2802 (d=2.69, audio)

**speaker=Ira**: f14927 (d=3.957, audio), f31758 (d=2.973, audio), f9122 (d=2.717, text), f8728 (d=2.626, audio), f23625 (d=2.463, audio)

**speaker=Siya**: f17191 (d=4.083, audio), f11059 (d=3.935, audio), f5451 (d=3.815, audio), f13191 (d=3.754, both), f29091 (d=3.611, audio)

**speaker=Zoya**: f29304 (d=7.563, audio), f14355 (d=4.876, audio), f32355 (d=4.149, audio), f29833 (d=3.767, audio), f5647 (d=3.587, audio)

**tone=angry**: f12064 (d=1.184, audio), f11568 (d=0.993, both), f15235 (d=0.976, audio), f8589 (d=0.905, audio), f10536 (d=0.896, audio)

**tone=excited**: f1821 (d=0.94, audio), f27418 (d=0.927, audio), f3296 (d=0.887, audio), f31186 (d=0.88, audio), f20447 (d=0.863, audio)

**tone=happy**: f15037 (d=0.454, audio), f5884 (d=0.424, audio), f545 (d=0.41, audio), f10948 (d=0.407, audio), f10199 (d=0.4, audio)

**tone=professional**: f19527 (d=0.468, audio), f31423 (d=0.465, both), f18426 (d=0.441, audio), f8994 (d=0.432, audio), f7217 (d=0.418, audio)

**tone=sad**: f27558 (d=0.873, audio), f8601 (d=0.81, audio), f29312 (d=0.766, audio), f15286 (d=0.736, audio), f30709 (d=0.735, audio)

**accent=Hindi accent**: f3336 (d=2.011, audio), f18418 (d=1.861, audio), f6181 (d=1.793, audio), f27144 (d=1.791, audio), f32518 (d=1.781, audio)

**accent=Indian English**: f1628 (d=2.188, audio), f25575 (d=2.088, audio), f18538 (d=1.896, audio), f2672 (d=1.876, audio), f22509 (d=1.762, audio)

**pace=fast**: f18860 (d=1.285, audio), f12446 (d=1.032, audio), f21384 (d=1.016, audio), f8589 (d=1.015, audio), f24942 (d=0.991, audio)

**pace=slow**: f11724 (d=0.703, audio), f31193 (d=0.641, audio), f3346 (d=0.633, audio), f13850 (d=0.557, audio), f8986 (d=0.553, audio)

**pace=steady**: f7724 (d=0.347, audio), f22229 (d=0.308, audio), f632 (d=0.301, audio), f29004 (d=0.28, audio), f18113 (d=0.276, audio)

**tag=<laugh>**: f26479 (d=1.515, both), f673 (d=1.471, audio), f7237 (d=1.309, audio), f12055 (d=1.276, audio), f31276 (d=1.239, audio)

**tag=<sigh>**: f10633 (d=1.24, audio), f3943 (d=1.084, audio), f10924 (d=0.99, audio), f415 (d=0.971, audio), f10639 (d=0.963, audio)

**tag=<chuckle>**: f19136 (d=0.886, audio), f7237 (d=0.861, audio), f23959 (d=0.812, audio), f23036 (d=0.797, audio), f5030 (d=0.784, audio)
