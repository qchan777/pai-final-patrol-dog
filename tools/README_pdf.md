# レポート PDF の作り方
LibreOffice の HTML → PDF 直接変換は 1 ページ目が空白になる (原因不明) ので、HTML → DOCX → PDF の二段で変換する:
```bash
soffice --headless --convert-to docx:"MS Word 2007 XML" docs/report.html && soffice --headless --convert-to pdf --outdir docs report.docx
```
