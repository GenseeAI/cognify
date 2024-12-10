# tracing
import os
import socket
import hashlib
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from pathlib import Path
import importlib
import sys
from cognify.optimizer.plugin import translate_workflow
import pkg_resources
from datetime import datetime


def initial_usage_message():
    version = pkg_resources.get_distribution('cognify-ai').version
    # Define the marker file path
    marker_file_name = f".cognify_{version.replace('.', '_')}_initial_run"
    marker_file = Path(__file__).parent / marker_file_name

    if not marker_file.exists():
        # Show the first-run message
        print(f"""
Thank you for using Cognify-{version}! 🚀 To better understand how people use Cognify, we collect the following telemetry data:
    - Whether the optimizer is run in `resume` (-r) mode
    - Whether the original workflow is written in Langchain, DSPy, or Cognify's programming model
    - Information about the search parameters: 
        - light, medium or heavy search (or an application-specific search)
        - number of trials
        - quality constraint
If you would like to opt-out, simply add COGNIFY_TELEMETRY=false to your environment variables.
""")

    # Create the marker file
    with open(marker_file, "w") as f:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"This file marks the first run of Cognify-{version} on {current_time}.\n")


def generate_user_identifier():
    """
    Generate a consistent, anonymized user identifier.
    
    Creates a hash based on:
    - Hostname
    - IP address
    - Current process ID
    """
    # Get basic system information
    hostname = socket.gethostname()
    try:
        # Get primary IP address
        ip_address = socket.gethostbyname(hostname)
    except Exception:
        ip_address = 'unknown'
    
    # Create a consistent hash
    identifier_string = f"{hostname}_{ip_address}"
    user_hash = hashlib.sha256(identifier_string.encode()).hexdigest()
    
    return user_hash

is_telemetry_on = os.getenv("COGNIFY_TELEMETRY", "true").lower() == "true"

resource = Resource.create(attributes={
    "service.name": "cognify",
    "user.id": generate_user_identifier()
})

provider = TracerProvider(resource=resource)
# processor = BatchSpanProcessor(ConsoleSpanExporter())
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://wuklab-01.ucsd.edu:4318/v1/traces"))
provider.add_span_processor(processor)

# Set global default tracer provider
trace.set_tracer_provider(provider)

# Creates a tracer from the global tracer provider
tracer = trace.get_tracer("cognify.tracer")

def trace_cli_args(args):
    if is_telemetry_on:
        with tracer.start_as_current_span("from_cognify_args") as span:
            span.set_attribute("resume", args.resume)

def trace_default_search(search_type, quality_constraint):
    if is_telemetry_on:
        with tracer.start_as_current_span("default_search") as span:
            span.set_attribute("search_type", search_type)
            span.set_attribute("quality_constraint", quality_constraint)

def trace_custom_search(search_mode, n_trials, quality_constraint):
    if is_telemetry_on:
        with tracer.start_as_current_span("custom_search") as span:
            span.set_attribute("search_mode", search_mode)
            span.set_attribute("n_trials", n_trials)
            span.set_attribute("quality_constraint", quality_constraint)

def trace_workflow(module_path: str):
    if is_telemetry_on:
        try:
            path = Path(module_path)
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)

            # reload all cached modules in the same directory
            to_reload = []
            current_directory = os.path.dirname(module.__file__)
            for k,v in sys.modules.items():
                if hasattr(v, '__file__') and v.__file__ and v.__file__.startswith(current_directory):
                    to_reload.append(v)
        
            for mod in to_reload:
                importlib.reload(mod)

            # execute current script as a module
            spec.loader.exec_module(module)
        except Exception:
            raise

        _, translate_data = translate_workflow(module)

        with tracer.start_as_current_span("capture_module_from_fs") as span:
            span.set_attribute("is_manually_translated", translate_data.is_manually_translated)
            span.set_attribute("is_langchain", translate_data.is_langchain)
            span.set_attribute("is_dspy", translate_data.is_dspy)
