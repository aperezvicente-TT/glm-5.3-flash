"""Gate for the served chat template (program 2026-10-23T15:02:43Z, review round 5 P2-4 stop/sequence): the
thinking-switch template must render byte-identically to the model's own template in every case except a generation
prompt with enable_thinking=false, which must end '<|assistant|><think></think>'. Host only (tokenizer files).
Exit 0 and one JSON line on success; exit 1 otherwise. usage: check_chat_template.py <model snapshot dir> <template>"""

import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
new = open(sys.argv[2]).read()
convs = [
    [{"role": "user", "content": "Count from 1 to 20, separated by commas."}],
    [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"},
     {"role": "assistant", "content": "hello"}, {"role": "user", "content": "again"}],
    [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<think>r</think>a"}, {"role": "user", "content": "q2"}],
]  # fmt: skip
kwsets = [{}, {"enable_thinking": True}, {"enable_thinking": False}, {"reasoning_effort": "low"},
          {"reasoning_effort": "high", "enable_thinking": False}]  # fmt: skip
bad, n = [], 0
for i, msgs in enumerate(convs):
    for gen in (True, False):
        for kw in kwsets:
            a = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, **kw)
            b = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, chat_template=new, **kw)
            n += 1
            if gen and kw.get("enable_thinking") is False:
                ok = a.endswith("<|assistant|><think>") and b == a + "</think>"
            else:
                ok = a == b
            if not ok:
                bad.append({"conv": i, "gen": gen, "kw": kw, "orig_tail": a[-40:], "new_tail": b[-40:]})
print(json.dumps({"renders": n, "failures": bad, "ok": not bad}))
sys.exit(0 if not bad else 1)
