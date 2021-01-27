import argparse
from collections import defaultdict

import pandas as pd
import os
from tqdm import tqdm
from html import escape

HTML_MAIN_TEMPLATE = """
<html>
<head>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/2.4.1/semantic.min.css" integrity="sha512-8bHTC73gkZ7rZ7vpqUQThUDhqcNFyYi2xgDgPDHc+GXVGHXq+xPjynxIopALmOPqzo9JZj0k6OqqewdGO3EsrQ==" crossorigin="anonymous" />
</head>
<body>
<div class="ui container" style="padding-top: 3em;">
<table class="ui celled padded table fixed">
<thead>
<tr>
<th>Number</th>
<th>Sentence</th>
<th>Label</th>
<th>Label 1</th>
<th>Label 2</th>
<th>Label 3</th>
<th>Label 4</th>
<th>Label 5</th>
</tr>
</thead>
<tbody>
{body}
</tbody>
</table>
</div>
</body>
</html>
"""

HTML_MAIN_LINE_TEMPLATE = """<tr>
<td><a href="./{tsv_name}_{number}.html" style="font-size: 28px;">{number}</a></td>
<td>{sentence}</td>
<td>{label}</td>
<td>{label_1}</td>
<td>{label_2}</td>
<td>{label_3}</td>
<td>{label_4}</td>
<td>{label_5}</td>
</tr>
"""

INSTANCE_TEMPLATE = """
<html>
<head>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/semantic-ui/2.4.1/semantic.min.css" integrity="sha512-8bHTC73gkZ7rZ7vpqUQThUDhqcNFyYi2xgDgPDHc+GXVGHXq+xPjynxIopALmOPqzo9JZj0k6OqqewdGO3EsrQ==" crossorigin="anonymous" />
</head>
<body>
<div class="ui container" style="padding-top: 3em;">
<p><strong>Sentence:</strong> {sentence}</p>
<p><strong>Label:</strong> {label}</p>
<p><strong>Rarity:</strong> {rarity}</p>
<p><strong>Results:</strong></p>
<table class="ui celled padded table">
<thead>
<tr>
<th>Number</th>
<th>Label</th>
<th>Lemma</th>
<th>Distance</th>
<th>Sentence</th>
</tr>
</thead>
<tbody>
{body}
</tbody>
</table>
</div>
</body>
</html>
"""


INSTANCE_LINE_TEMPLATE = """<tr>
<td>{number}</td>
<td>{label}</td>
<td>{lemma}</td>
<td>{distance}</td>
<td>{sentence}</td>
</tr>
"""

def enh_sent(escaped_sent):
    sent = escaped_sent.replace("&gt;&gt;", '<strong style="color: #b70000">&gt;&gt;')
    sent = sent.replace("&lt;&lt;", "&lt;&lt;</strong>")
    i = sent.find('</strong>')
    i = sent.find(' ', i)
    j = sent.find(' ', i)
    return sent


def generate_instance_page(number, base_dir, tsv_name, row):
    body = ""
    for i in range(1,51):
        format_result = lambda attr: lambda s: f'<span style="background-color:#fff791; padding:2px; border-radius:3px;">{s}</span>' if s != getattr(row, attr) else s
        body += INSTANCE_LINE_TEMPLATE.format(
            number=i,
            label=format_result('label')(getattr(row, f'label_{i}')),
            lemma=format_result('lemma')(getattr(row, f'lemma_{i}')),
            distance=str(getattr(row, f'distance_{i}'))[:6],
            sentence=enh_sent(escape(str(getattr(row, f'sentence_{i}')))),
        )

    html_str = INSTANCE_TEMPLATE.format(
        sentence=enh_sent(escape(row.sentence)),
        label=row.label,
        body=body,
        rarity=str(row.rarity)
                   + f" ({LABEL_FREQS[row.label]}/{LEMMA_FREQS[row.lemma]})"
    )
    with open(f'{base_dir}/html/{tsv_name}_{number}.html', 'w') as f:
        f.write(html_str)


def write_pages(df, base_dir, tsv_name):
    body_str = ""
    # pandas preserves row index on sorting, so use our own i
    buckets = [0.05, 0.10, 0.20, 0.50, 1]
    bucket2body = {i: "" for i in buckets}
    i = 0
    for _, row in tqdm(df.sort_values('rarity').iterrows()):
        format_result = lambda s: '<span style="background-color:#fff791; padding:2px; border-radius:3px;">' + s + '</span>' if s != row.label else s
        s = HTML_MAIN_LINE_TEMPLATE.format(
            number=i,
            tsv_name=tsv_name,
            sentence=enh_sent(escape(row.sentence)),
            label=escape(row.label),
            label_1=format_result(escape(row.label_1)),
            label_2=format_result(escape(row.label_2)),
            label_3=format_result(escape(row.label_3)),
            label_4=format_result(escape(row.label_4)),
            label_5=format_result(escape(row.label_5)),
        )
        body_str += s
        generate_instance_page(i, base_dir, tsv_name, row)

        if i != 0 and (i % 1000 == 0 or i == df.shape[0] - 1):
            with open(f'{base_dir}/html/{tsv_name}_index_{i // 1000}.html', 'w') as f:
                f.write(HTML_MAIN_TEMPLATE.format(body=body_str))
            body_str = ""

        for bucket in buckets:
            if not (row.rarity > bucket):
                bucket2body[bucket] += s
                break
        i += 1

    for bucket, body in bucket2body.items():
        with open(f'{base_dir}/html/{tsv_name}_bucket_{bucket}.html', 'w') as f:
            f.write(HTML_MAIN_TEMPLATE.format(body=body))


def main(tsv_filepath):
    base_dir = os.path.dirname(tsv_filepath)
    tsv_name = tsv_filepath.split(os.sep)[-1]
    print(dir, tsv_name)

    df = pd.read_csv(tsv_filepath, sep="\t")
    df['rarity'] = df.label.apply(lambda label: LABEL_FREQS[label]) / df.lemma.apply(lambda lemma: LEMMA_FREQS[lemma])
    write_pages(df, base_dir, tsv_name)

    #for rarity_threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    #    with open(f'{base_dir}/html/{tsv_name}_index.{int(rarity_threshold * 100)}.html', 'w') as f:
    #        f.write(HTML_MAIN_TEMPLATE.format(body=rarity2body[int(rarity_threshold * 100)]))


if __name__ == '__main__':
    with open('cache/ontonotes_stats/train_label_freq.tsv', 'r') as f:
        LABEL_FREQS = {k: int(v) for k, v in map(lambda l: l.strip().split('\t'), f)}
    with open('cache/ontonotes_stats/train_lemma_freq.tsv', 'r') as f:
        LEMMA_FREQS = {k: int(v) for k, v in map(lambda l: l.strip().split('\t'), f)}
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "tsv_file",
    )
    args = ap.parse_args()
    main(args.tsv_file)
