"""Several pictures on one labelled sheet.
  python sheet.py out.jpg <columns> <scale> label=file label=file ..."""
import sys
from PIL import Image, ImageDraw, ImageFont
out, cols, scale = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
items = [a.split('=', 1) if '=' in a else ['', a] for a in sys.argv[4:]]
ims = [Image.open(f).convert('RGB') for _, f in items]
w, h = int(ims[0].width * scale), int(ims[0].height * scale)
rows = (len(ims) + cols - 1) // cols
W = Image.new('RGB', (w * cols, h * rows), (20, 22, 28))
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', max(13, int(h / 30)))
except Exception: font = ImageFont.load_default()
d = ImageDraw.Draw(W)
for i, (im, (lab, _)) in enumerate(zip(ims, items)):
    x, y = (i % cols) * w, (i // cols) * h
    W.paste(im.resize((w, h), Image.LANCZOS), (x, y))
    if lab:
        d.text((x + 9, y + 7), lab, fill=(0, 0, 0), font=font)
        d.text((x + 8, y + 6), lab, fill=(255, 255, 255), font=font)
W.save(out, quality=90)
