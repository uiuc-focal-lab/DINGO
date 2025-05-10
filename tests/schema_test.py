import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dataset import load_dataset_by_name
from dfa import DFAMatrixStore
from transformers import AutoTokenizer
from jsonschema import validate
import json
from outlines_core.fsm.json_schema import build_regex_from_schema


ds, _ = load_dataset_by_name('json_mode')

ds = ds.select(range(1, 2))

schema = ds['schema'][0]

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct')

dfa_store = DFAMatrixStore('json_mode', True, True).build(schema, 1, tokenizer)


dfa_store.dfa.matches(ds['completion'][0])

validate(instance = json.loads(ds['completion'][0]), schema = json.loads(schema))

rgx = build_regex_from_schema(schema)
import pdb;pdb.set_trace()