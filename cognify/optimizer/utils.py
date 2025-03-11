import logging
from functools import wraps
from cognify.llm import Model, StructuredModel
import inspect
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from pydantic import BaseModel
from datamodel_code_generator.parser.jsonschema import JsonSchemaParser

from tqdm.auto import tqdm

_cognify_tqdm = tqdm

logger = logging.getLogger(__name__)


def aggregator_factory(lm: Model, code: str):
    agg_func_obj = compile(code, "<string>", "exec")
    local_name_space = {}
    exec(agg_func_obj, {}, local_name_space)
    func_name = agg_func_obj.co_names[0]
    aggregator = local_name_space[func_name]

    if isinstance(lm, StructuredModel):

        @wraps(aggregator)
        def wrapper_kernel(**kwargs):
            result = aggregator(**kwargs)
            mresult = lm.output_format.schema.model_validate(result)
            field = lm.output_format.schema.__name__
            return {field: getattr(mresult, field)}

    else:
        wrapper_kernel = aggregator
    sig = inspect.signature(wrapper_kernel)
    logger.debug(f"Aggregator signature: {sig}")
    return wrapper_kernel


NON_ALPHANUMERIC = re.compile(r"[^a-zA-Z0-9]+")
UPPER_CAMEL_CASE = re.compile(r"[A-Z][a-zA-Z0-9]+")
LOWER_CAMEL_CASE = re.compile(r"[a-z][a-zA-Z0-9]+")


class BadJsonSchema(Exception):
    pass


def _to_camel_case(name: str) -> str:
    if any(NON_ALPHANUMERIC.finditer(name)):
        return "".join(term.lower().title() for term in NON_ALPHANUMERIC.split(name))
    if UPPER_CAMEL_CASE.match(name):
        return name
    if LOWER_CAMEL_CASE.match(name):
        return name[0].upper() + name[1:]
    raise BadJsonSchema(f"Unknown case used for {name}")


def _load_module_from_file(file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name=file_path.stem, location=str(file_path)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[file_path.stem] = module
    spec.loader.exec_module(module)
    return module


def json_schema_to_pydantic_model(json_schema: dict, file_path: str) -> type[BaseModel]:
    json_schema_as_str = json.dumps(json_schema)
    pydantic_models_as_str: str = JsonSchemaParser(json_schema_as_str).parse()

    module_file_path = Path(file_path).resolve()
    with open(module_file_path, "wb+") as f:
        f.write(pydantic_models_as_str.encode())

    module = _load_module_from_file(file_path=module_file_path)

    main_model_name = _to_camel_case(name=json_schema["title"])
    pydantic_model: type[BaseModel] = module.__dict__[main_model_name]
    return pydantic_model

def _report_quality_impv(new, base):
    return (new - base) / base * 100

def _report_cost_reduction(new, base):
    return new / base

def _report_latency_reduction(new, base):
    return new / base