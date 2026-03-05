set -ex
pdoc3 adseq --html --force --output-dir /tmp
rm -fr docs
mv /tmp/adseq/ docs
tree docs

(
awk '/DOCS/ {exit} 1' README.template.md
python3 build-doc-tree.py
awk 'a {print} /DOCS/ {a=1}' README.template.md
) > README.md
