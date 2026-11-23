# osmo_grade: Matteo's Osmo Pocket 4P color pipeline

Grades DJI Osmo Pocket 4P footage automatically: detects the color profile, fixes exposure and
white balance, applies a project look, renders reels or masters, and writes scopes so every grade
can be checked like a colorist would.

## Looks
| project  | for                                   | character |
|----------|---------------------------------------|-----------|
| fights   | @matteofights                         | dark, punchy, desaturated, red kept strong, teal shadows, warm skin |
| gypsy    | Gypsy Amazon                          | warm, soft, human, fabric colors kept true (no hue shifts) |
| samui    | Premium Investment Samui / Crystal Bay View | bright, airy, clean white walls, aqua water pops |
| portrait | modeling / comp card                  | neutral, flattering, unified skin |
| clean    | anything else                         | neutral technical grade |

## Commands
```
python3 osmo_grade.py grade CLIPS... --project fights --out out/            # reel (1080 wide/tall H.264)
python3 osmo_grade.py grade CLIPS... --project gypsy --out out/ --crop 9:16  # horizontal 4K to vertical reel
python3 osmo_grade.py grade CLIPS... --project samui --out out/ --preset master,prores
python3 osmo_grade.py grade CLIPS... --project fights --out out/ --lock       # same light: one shared correction
python3 osmo_grade.py grade CLIPS... --project portrait --out out/ --wb-ref GREYCARD.MP4@1.0
python3 osmo_grade.py grade CLIPS... --project fights --out out/ --input-lut DJI_DLog2_to_Rec709.cube
python3 osmo_grade.py luts --out luts/ [--input-lut DJI_DLog2_to_Rec709.cube]
```
Tweaks on top of auto: `--exposure 0.3` (stops), `--temp 5` (warmer), `--tint -3` (greener),
`--sat 1.1`, `--contrast 0.95`, `--intensity 0.7` (look strength). `--no-render` gives LUT + report only.

## LUTs in /luts (33-point .cube, work in DaVinci, Premiere, Final Cut, LumaFusion and most editors)
- `<project>_from_DLog.cube`: put straight on D-Log clips (3x tele camera, or zoomed main camera).
- `<project>_look_on_Rec709.cube`: the look only. Stack it AFTER DJI's official D-Log 2 to Rec.709 LUT,
  or put it on Normal footage.
- `DLog_to_Rec709_neutral.cube`: neutral D-Log conversion from DJI's published D-Log/D-Gamut math.
- `<project>_from_DLog2.cube` appears once DJI's official D-Log 2 LUT is fed in with `luts --input-lut`.
A LUT alone cannot fix exposure or white balance per clip. The `grade` command does that automatically.

## Shooting settings that make the grade work
1. Main camera at 1x: D-Log 2, 10-bit, ISO 100 (D-Log 2 has no auto ISO). Outdoors in Thai sun use the ND
   filters so shutter stays near double the frame rate (1/50 at 25p, 1/60 at 30p).
2. 3x tele, or any zoom on the main camera: D-Log (D-Log 2 only works at 1x). Its minimum ISO is 400.
3. Lock white balance manually (daylight around 5600K, shade around 6500K, indoor gym LEDs around 4000-5000K).
   Never auto: it drifts mid-clip and breaks shot matching.
4. Expose a little bright: D-Log 2 puts middle grey low (around 31 IRE), so the flat image should look
   slightly bright on the screen, never dark. Underexposed log turns noisy in the shadows when lifted.
5. New location or new light: film 2 seconds of a grey card or plain white paper in that light.
   That clip makes the white balance perfect (`--wb-ref`).
6. Same lighting setup = same settings across clips, then grade with `--lock`.
7. Vertical: portrait mode records 3K natively; horizontal 4K cropped to 9:16 is 1215x2160, still above
   1080x1920 reel output. Both work.

See PRESETS.md for the on-camera C1/C2 settings these looks expect.

Needs ffmpeg, numpy, Pillow. No network access, no keys, no secrets anywhere in the code.
