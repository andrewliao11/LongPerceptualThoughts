import os
import fire
import json

from tqdm import tqdm
from datetime import datetime
from jinja2.sandbox import SandboxedEnvironment
from pathlib import Path

from stage_1_mcq_gen import generate_mcq_from_captions
from stage_2_simple_cot import generate_simple_cot, collect_simple_cot
from stage_3_expand_cot import generate_extended_cot, collect_extended_cot


if __name__ == '__main__':
    fire.Fire({
        # stage 1: ask
        'generate_mcq_from_captions': generate_mcq_from_captions, 
        # stage 2: think
        'generate_simple_cot': generate_simple_cot,
        'collect_simple_cot': collect_simple_cot, 
        # stage 3: think harder
        'generate_extended_cot': generate_extended_cot, 
        'collect_extended_cot': collect_extended_cot, 
    })