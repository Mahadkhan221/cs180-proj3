"""Produce a self-contained writeup_embedded.html with every image inlined as a
base64 data URI (downscaled), so the page renders anywhere with no local files.
"""

import base64
import io
import os
import re

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def embed(match):
    rel = match.group(1)
    path = os.path.join(HERE, *rel.split("/"))
    im = Image.open(path).convert("RGB")
    if im.width > 1100:
        im = im.resize((1100, round(im.height * 1100 / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'src="data:image/jpeg;base64,{b64}"'


def main():
    html = open(os.path.join(HERE, "writeup.html"), encoding="utf-8").read()
    out = re.sub(r'src="([^"]+\.(?:jpg|png))"', embed, html)
    dst = os.path.join(HERE, "writeup_embedded.html")
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"[ok] wrote {dst} ({len(out)/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
