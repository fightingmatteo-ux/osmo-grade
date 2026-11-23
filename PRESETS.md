# Pocket 4P shooting presets (Thailand, 50Hz)

Full guide with setup steps: the "Osmo Pocket 4P Shooting Presets" doc in Claude.

## On-camera custom modes
| Setting | C1 CINE | C2 ACTION |
| --- | --- | --- |
| Mode | Video, Pro, Manual exposure | same |
| Lens / color | 1x, D-Log 2 10-bit, Color Recovery ON | same |
| Resolution, fps | 4K 25p | 4K 50p |
| Shutter | 1/50 | 1/100 |
| ISO start | 100 | 400 |
| White balance | 5600K manual | 5000K manual |
| Texture / NR | -1 / -1 | -1 / -1 |
| Codec | HEVC, bitrate High | same |
| Global | Anti-flicker 50Hz, Overexposure Alert ON, custom button = Switch C1/C2 | |

C1: real estate, Gypsy Amazon, modeling, outdoor, talking head.
C2: Muay Thai gym, bag/pad work, sparring, walking shots, anything slowed 2x.

## What the grading engine should expect from each file
| File looks like | Came from | Profile | Notes |
| --- | --- | --- | --- |
| 4K 25p, 10-bit, log | C1 | dlog2 | |
| 4K 50p, 10-bit, log | C2 | dlog2 | |
| 10-bit log, zoomed or tele | C1/C2 + D-Log | dlog | ISO 400 minimum |
| 100/120/240/200 fps | Slow Motion mode | normal | log is unavailable in slow motion |
| 4K 30p, contrasty | Low-Light mode or Auto | normal | may flicker under 50Hz lights |

Every new light: 2 seconds of white paper, used with `--wb-ref CLIP@SECONDS`.
