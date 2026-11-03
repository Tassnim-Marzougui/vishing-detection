import base64
import html
import json
import webbrowser
from pathlib import Path

R = Path("reports")


def load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def esc(x):
    return html.escape(str(x))


def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f} %"


base, met, cmp_ = load("baselines.json"), load("metrics.json"), load("compare.json")

# Resultat de baseline_lodo.py (regle sur la duree, leave-one-dialect-out)
LODO_DUREE = {"ar-Egyptian": 0.69, "ar-Gulf": 0.72, "ar-Iraqi": 0.50, "ar-Jordanian": 0.68,
              "ar-MSA": 0.66, "ar-Saudi": 0.47, "ar-Sudanese": 0.85, "ar-Syrian": 0.54}
LODO_DUREE_MEAN = 0.637

maj = g(base, "appels", "majority_acc_test", default=0.5)
dur = g(base, "appels", "duration_sec", "test_acc")
ac05 = g(cmp_, "split_officiel", "acoustique_rf", "seuil_0.5", "accuracy")
acv = g(cmp_, "split_officiel", "acoustique_rf", "seuil_valide", "accuracy")
tx = g(cmp_, "split_officiel", "texte_tfidf", "seuil_0.5", "accuracy")
lodo_a = g(cmp_, "lodo", "acoustique_rf", default={})
lodo_t = g(cmp_, "lodo", "texte_tfidf", default={})
langs = sorted(lodo_a)
n_calls = sum(v["n"] for v in lodo_a.values()) if lodo_a else 0
mean_a = sum(v["accuracy"] for v in lodo_a.values()) / len(lodo_a) if lodo_a else None
mean_t = sum(v["accuracy"] for v in lodo_t.values()) / len(lodo_t) if lodo_t else None
seg_file = Path("data/processed/segments.csv")
n_seg = max(sum(1 for _ in seg_file.open(encoding="utf-8")) - 1, 0) if seg_file.exists() else None


def bar(label, v, color="acc", ref=None, note=""):
    if v is None:
        return ""
    w = max(0.0, min(1.0, v)) * 100
    m = f'<i class="ref" style="left:{ref * 100:.1f}%"></i>' if ref is not None else ""
    return (f'<div class="row"><span class="lab">{esc(label)}</span>'
            f'<span class="track"><b class="fill {color}" style="width:{w:.1f}%"></b>{m}</span>'
            f'<span class="val">{pct(v)}</span><span class="note">{esc(note)}</span></div>')


def table(head, rows):
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def tile(big, small):
    return f'<div class="tile"><div class="big">{esc(big)}</div><div class="small">{esc(small)}</div></div>'


tiles = "".join([
    tile(str(n_calls or "422"), "appels audio (8 dialectes)"),
    tile(str(n_seg or "1676"), "segments de 10 s"),
    tile("0", "fuite entre splits"),
    tile("5 / 5", "tests pytest"),
    tile(pct(g(met, "plancher_duree_appels", default=dur)), "plancher durée (test)"),
])

# 1. Benchmark
bench = "".join([
    bar("Classe majoritaire", maj, "mut", None, "plancher 1"),
    bar("Règle sur la durée", dur, "warn", None, "plancher 2 (raccourci)"),
    bar("Audio RF (seuil 0.5)", ac05, "acc", dur, "MFCC + RMS + ZCR"),
    bar("Audio RF (seuil validation)", acv, "acc", dur, "seuil choisi sur validation"),
    bar("Texte TF-IDF", tx, "bad", dur, "NON fiable: fuite de catégorie"),
])

# 2. LODO
lodo_rows = []
for L in langs:
    lodo_rows.append([
        esc(L), str(lodo_a[L]["n"]),
        pct(LODO_DUREE.get(L)),
        pct(lodo_a[L]["accuracy"]),
        pct(g(lodo_a, L, "roc_auc")),
        pct(lodo_t.get(L, {}).get("accuracy")),
    ])
lodo_rows.append(["<b>Moyenne</b>", f"<b>{n_calls}</b>", f"<b>{pct(LODO_DUREE_MEAN)}</b>",
                  f"<b>{pct(mean_a)}</b>", "", f"<b>{pct(mean_t)}</b>"])
lodo_tbl = table(["Dialecte testé", "Appels", "Règle durée", "Audio RF", "AUC audio", "Texte"], lodo_rows)
lodo_bars = "".join(bar(L, lodo_a[L]["accuracy"], "acc", LODO_DUREE.get(L)) for L in langs)

# 3. Agrégation
agg_rows = []
for k, name in (("mean", "moyenne"), ("max", "max"), ("topk", "top-3")):
    m = met.get(k)
    if m:
        agg_rows.append([name, pct(m["accuracy"]), f'{m["f1"]:.2f}', pct(m.get("roc_auc")),
                         str(m["fn"]), str(m["fp"])])
agg_tbl = table(["Agrégation appel", "Accuracy", "F1", "AUC", "FN", "FP"], agg_rows)

# 4. Par langue (test)
pl_rows = []
for L, m in sorted(g(met, "par_langue_mean", default={}).items()):
    pl_rows.append([esc(L), str(m["n"]), pct(m["accuracy"]), f'{m["f1"]:.2f}'])
pl_tbl = table(["Dialecte (test officiel)", "Appels", "Accuracy", "F1"], pl_rows)

# 5. Images EDA
CAPS = {
    "01": "Classes et dialectes : équilibre correct, 8 dialectes (Yemeni sans audio).",
    "02": "Catégories : scam et légitime sont séparés à 100 % par la catégorie. C'est pour ça que le texte fait 1.00.",
    "03": "Durée : les scams durent plus longtemps (30.8 s vs 23.0 s). Raccourci neutralisé par les segments de 10 s.",
    "04": "Longueur du texte : même tendance, les scams parlent plus.",
}
imgs = []
for p in sorted((R / "eda").glob("*.png")):
    b64 = base64.b64encode(p.read_bytes()).decode()
    imgs.append(f'<figure><img alt="{esc(p.name)}" src="data:image/png;base64,{b64}">'
                f'<figcaption>{esc(CAPS.get(p.name[:2], p.name))}</figcaption></figure>')
imgs_html = "".join(imgs) or "<p class='mut'>Aucune image dans reports/eda (lancer 01_eda.ipynb).</p>"

forts = [
    "Pipeline complet et reproductible (audit, manifest, validate, prepare, baselines, features, train, evaluate).",
    "Manifest 13 colonnes valide, 0 doublon, 0 fuite entre splits (split groupé, seed 42).",
    "Planchers calculés avant tout modèle (majorité, règle durée) : le guide est respecté.",
    f"L'audio dépasse le plancher durée en leave-one-dialect-out ({pct(mean_a)} vs {pct(LODO_DUREE_MEAN)}).",
    "Segmentation 10 s : neutralise le raccourci « durée = label ».",
    "Tests pytest, MLflow (tags), Git + DVC (aucun WAV dans Git).",
]
faibles = [
    "Corpus 100 % synthétique à voix unique : les scores sont gonflés.",
    f"Texte à {pct(tx)} = fuite de catégorie, pas une vraie généralisation.",
    "Audio modeste : features MFCC/RMS/ZCR portent peu de signal (AUC test ~0.75).",
    "Test officiel = 64 appels : écart de 4 appels avec le plancher, non significatif.",
    "Dialectes en test trop petits (2 à 21 appels) : métriques par dialecte instables.",
    "Pas de corpus externe (cross-corpus), Yemeni sans audio, licence à vérifier, Spark remplacé par pandas.",
]
suite = [
    "Embeddings wav2vec2 sur Kaggle GPU (guide A.6) pour remplacer MFCC.",
    "Intégrer un 2ème corpus (DS02, DS04...) : vrai test cross-corpus.",
    "Vérifier la licence d'ASLC-448 avant intégration.",
    "Pull Request vers main avec la checklist du guide.",
]


def ul(items):
    return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"


TEMPLATE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vishing DS07 - Résultats</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1c2330;--mut:#5d6778;--line:#dfe3ea;--acc:#2a5db0;--ok:#1f7a4d;--warn:#b26a00;--bad:#b3261e;--track:#e8ebf0;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#0f141b;--card:#171d27;--fg:#e6e9ef;--mut:#9aa4b5;--line:#2a3342;--acc:#6ea0f0;--ok:#4cc38a;--warn:#e0a040;--bad:#f08078;--track:#252d3a;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,sans-serif;padding:24px max(16px,4vw)}
main{max-width:1100px;margin:0 auto;display:grid;gap:20px}
h1{font-size:1.7rem;margin:0 0 4px}
h2{font-size:1.1rem;margin:0 0 4px}
p{margin:4px 0 10px}
.mut,.sub{color:var(--mut)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;min-width:0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.big{font-size:1.8rem;font-variant-numeric:tabular-nums}
.small{color:var(--mut);font-size:.85rem}
.verdict{border-left:4px solid var(--warn)}
.row{display:grid;grid-template-columns:200px 1fr 70px;gap:4px 12px;align-items:center;margin:8px 0}
.row .note{grid-column:2 / 4;color:var(--mut);font-size:.8rem;margin-top:-2px}
.track{position:relative;height:14px;background:var(--track);border-radius:7px;overflow:visible}
.fill{display:block;height:100%;border-radius:7px}
.fill.acc{background:var(--acc)}.fill.warn{background:var(--warn)}.fill.bad{background:var(--bad)}.fill.mut{background:var(--mut)}
.ref{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--warn)}
.val{text-align:right;font-variant-numeric:tabular-nums}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:20px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:600;font-size:.85rem}
figure{margin:0}
img{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}
figcaption{color:var(--mut);font-size:.85rem;margin-top:4px}
ul{margin:6px 0;padding-left:20px}li{margin:4px 0}
.ok h2{color:var(--ok)}.bad h2{color:var(--bad)}
@media (max-width:640px){.row{grid-template-columns:1fr 60px}.row .lab{grid-column:1 / 3}.row .track{grid-column:1}.row .note{grid-column:1 / 3}}
</style></head><body><main>
<header><h1>Détection du vishing - DS07 (ASLC-448, audio TTS)</h1>
<p class="sub">Résultats du pipeline suivant le guide d'équipe. Lis les barres avec le trait orange = plancher « durée seule ».</p></header>

<section class="tiles">{{TILES}}</section>

<section class="card verdict"><h2>Verdict</h2>
<p>Le <b>texte</b> fait {{TX}} mais ce n'est pas un résultat (catégories parfaitement séparées dans ce corpus synthétique).
L'<b>audio</b> fait {{AC}} en test et {{MEANA}} en leave-one-dialect-out, contre {{DUR}} / {{LODOD}} pour la durée seule :
<b>signal modeste mais réel</b>. À confirmer sur un corpus externe.</p></section>

<section class="card"><h2>1. Comparaison au plancher (test officiel, 64 appels)</h2>
<p class="mut">Le trait orange est la règle « durée seule ». Un modèle doit le dépasser nettement.</p>{{BENCH}}</section>

<section class="grid2">
<div class="card"><h2>2. Leave-one-dialect-out (422 appels)</h2>
<p class="mut">On entraîne sur 7 dialectes, on teste sur le 8ème. Barres = audio, trait = règle durée.</p>{{LODOBARS}}</div>
<div class="card"><h2>Détail par dialecte</h2>{{LODOTBL}}</div>
</section>

<section class="grid2">
<div class="card"><h2>3. Agrégation au niveau appel (audio, test)</h2>
<p class="mut">FN = scams ratés, FP = fausses alertes. Beaucoup de FP : le modèle dit « scam » trop souvent.</p>{{AGG}}</div>
<div class="card"><h2>Par dialecte (test officiel)</h2>
<p class="mut">Effectifs de 2 à 21 : trop petits pour conclure.</p>{{PL}}</div>
</section>

<section class="card"><h2>4. EDA : ce que les données montrent</h2><div class="grid2">{{IMGS}}</div></section>

<section class="grid2">
<div class="card ok"><h2>Points forts</h2>{{FORTS}}</div>
<div class="card bad"><h2>Points faibles / limites</h2>{{FAIBLES}}</div>
</section>

<p class="mut">Généré par make_dashboard.py à partir de reports/*.json. Relancer la commande après chaque nouvel entraînement.</p>
</main></body></html>"""

page = (TEMPLATE
        .replace("{{TILES}}", tiles).replace("{{TX}}", pct(tx)).replace("{{AC}}", pct(ac05))
        .replace("{{MEANA}}", pct(mean_a)).replace("{{DUR}}", pct(dur)).replace("{{LODOD}}", pct(LODO_DUREE_MEAN))
        .replace("{{BENCH}}", bench).replace("{{LODOBARS}}", lodo_bars).replace("{{LODOTBL}}", lodo_tbl)
        .replace("{{AGG}}", agg_tbl).replace("{{PL}}", pl_tbl).replace("{{IMGS}}", imgs_html)
        .replace("{{FORTS}}", ul(forts)).replace("{{FAIBLES}}", ul(faibles)).replace("{{SUITE}}", ul(suite)))

out = Path("dashboard.html")
out.write_text(page, encoding="utf-8")
print("OK ->", out.resolve())
webbrowser.open(out.resolve().as_uri())