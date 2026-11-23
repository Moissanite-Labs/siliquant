"""Build the GitHub Pages report and its print edition from one HTML source."""

import argparse
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrome", default="google-chrome", help="Chrome or Chromium executable")
    args = parser.parse_args()
    source = Path(__file__).resolve().with_name("results.html")
    docs = source.parents[1] / "docs"
    with TemporaryDirectory(prefix="siliquant-results-") as work:
        work = Path(work)
        pdf = work / "siliquant-results.pdf"
        subprocess.run(
            [
                args.chrome,
                "--headless",
                f"--user-data-dir={work / 'chrome'}",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}",
                source.as_uri(),
            ],
            check=True,
        )
        shutil.copyfile(source, docs / "index.html")
        shutil.copyfile(pdf, docs / pdf.name)
    print("Built docs/index.html and docs/siliquant-results.pdf")


if __name__ == "__main__":
    main()
