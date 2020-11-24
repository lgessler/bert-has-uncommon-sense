import argparse
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
<p><strong>Results:</strong></p>
<table class="ui celled padded table">
<thead>
<tr>
<th>Number</th>
<th>Label</th>
<th>Synset</th>
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
<td>{synset}</td>
<td>{lemma}</td>
<td>{distance}</td>
<td>{sentence}</td>
</tr>
"""

def enh_sent(escaped_sent):
    return escaped_sent.replace("&gt;&gt;", '<strong style="color: #b70000">&gt;&gt;').replace("&lt;&lt;", "&lt;&lt;</strong>")


def generate_instance_page(number, base_dir, tsv_name, row):
    body = ""
    for i in range(1,51):
        format_result = lambda attr: lambda s: f'<span style="background-color:#fff791; padding:2px; border-radius:3px;">{s}</span>' if s != getattr(row, attr) else s
        body += INSTANCE_LINE_TEMPLATE.format(
            number=i,
            label=format_result('label')(getattr(row, f'label_{i}')),
            synset=format_result('synset')(getattr(row, f'synset_{i}')),
            lemma=format_result('lemma')(getattr(row, f'lemma_{i}')),
            distance=str(getattr(row, f'distance_{i}'))[:6],
            sentence=enh_sent(escape(getattr(row, f'sentence_{i}'))),
        )

    html_str = INSTANCE_TEMPLATE.format(
        sentence=enh_sent(escape(row.sentence)),
        label=row.label,
        body=body
    )
    with open(f'{base_dir}/html/{tsv_name}_{number}.html', 'w') as f:
        f.write(html_str)


def main(tsv_filepath):
    base_dir = os.path.dirname(tsv_filepath)
    tsv_name = tsv_filepath.split(os.sep)[-1]
    print(dir, tsv_name)

    body_str = ""
    df = pd.read_csv(tsv_filepath, sep="\t")
    for i, row in tqdm(df.iterrows()):
        format_result = lambda s: '<span style="background-color:#fff791; padding:2px; border-radius:3px;">' + s + '</span>' if s != row.label else s
        body_str += HTML_MAIN_LINE_TEMPLATE.format(
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
        generate_instance_page(i, base_dir, tsv_name, row)

    with open(f'{base_dir}/html/{tsv_name}_index.html', 'w') as f:
        f.write(HTML_MAIN_TEMPLATE.format(body=body_str))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "tsv_file",
    )
    args = ap.parse_args()
    main(args.tsv_file)
