"""Technique name normalization: map variant names to canonical forms."""

import logging
from collections.abc import Iterable
from pathlib import Path

import yaml
from rapidfuzz import fuzz, process

from src.config import settings

logger = logging.getLogger(__name__)

# Default canonical mapping
DEFAULT_MAPPING = {
    # SQL injection variants
    "sql_injection": "sql_injection",
    "sqli": "sql_injection",
    "blind_sql_injection": "sql_injection",
    "blind_sqli": "sql_injection",
    "boolean_blind_sqli": "sql_injection",
    "boolean_blind": "sql_injection",
    "time_based_sqli": "sql_injection",
    "time_based": "sql_injection",
    "error_based_sqli": "sql_injection",
    "error_based": "sql_injection",
    "union_sqli": "sql_injection",
    "union_select": "sql_injection",
    "union_based": "sql_injection",
    "stacked_queries": "sql_injection",
    "second_order_sqli": "sql_injection",
    "second_order": "sql_injection",
    "no_sqli": "sql_injection",
    "mysql": "sql_injection",
    "postgresql": "sql_injection",
    "mssql": "sql_injection",
    "sqlite": "sql_injection",
    "sqlite3": "sql_injection",
    "sqlmap": "sql_injection",

    # XSS variants
    "xss": "xss",
    "cross_site_scripting": "xss",
    "reflected_xss": "xss",
    "stored_xss": "xss",
    "dom_xss": "xss",
    "dom_based_xss": "xss",
    "self_xss": "xss",
    "blind_xss": "xss",

    # SSTI variants
    "ssti": "server_side_template_injection",
    "server_side_template_injection": "server_side_template_injection",
    "template_injection": "server_side_template_injection",
    "template_injection_flask": "server_side_template_injection",
    "jinja2": "server_side_template_injection",
    "jinja": "server_side_template_injection",
    "flask": "server_side_template_injection",
    "twig": "server_side_template_injection",
    "smarty": "server_side_template_injection",
    "mako": "server_side_template_injection",

    # Buffer overflow / PWN variants
    "buffer_overflow": "buffer_overflow",
    "stack_overflow": "buffer_overflow",
    "stack_buffer_overflow": "buffer_overflow",
    "ret2libc": "buffer_overflow",
    "ret2libc_attack": "buffer_overflow",
    "rop": "buffer_overflow",
    "return_oriented_programming": "buffer_overflow",
    "rop_chain": "buffer_overflow",
    "ret2csu": "buffer_overflow",
    "ret2dlresolve": "buffer_overflow",
    "ret2plt": "buffer_overflow",
    "ret2syscall": "buffer_overflow",
    "stack_pivot": "buffer_overflow",
    "sigreturn_orsis": "buffer_overflow",
    "sigreturn": "buffer_overflow",
    "srop": "buffer_overflow",

    # Heap exploitation
    "heap_exploitation": "heap_exploitation",
    "heap_overflow": "heap_exploitation",
    "heap_overflow_attack": "heap_exploitation",
    "use_after_free": "heap_exploitation",
    "uaf": "heap_exploitation",
    "tcache": "heap_exploitation",
    "tcache_poisoning": "heap_exploitation",
    "tcache_dup": "heap_exploitation",
    "tcache_perthread_struct": "heap_exploitation",
    "fastbin": "heap_exploitation",
    "fastbin_attack": "heap_exploitation",
    "fastbin_dup": "heap_exploitation",
    "unsafe_unlink": "heap_exploitation",
    "house_of_force": "heap_exploitation",
    "house_of_spirit": "heap_exploitation",
    "house_of_orange": "heap_exploitation",
    "house_of_lore": "heap_exploitation",
    "house_of_storm": "heap_exploitation",
    "fsop": "heap_exploitation",
    "io_file": "heap_exploitation",
    "io_file_attack": "heap_exploitation",
    "malloc_hook": "heap_exploitation",
    "free_hook": "heap_exploitation",
    "double_free": "heap_exploitation",
    "off_by_one": "heap_exploitation",
    "null_byte_off_by_one": "heap_exploitation",

    # Format string
    "format_string": "format_string",
    "format_string_attack": "format_string",
    "printf": "format_string",

    # Shellcode
    "shellcode": "shellcode",
    "shellcode_injection": "shellcode",
    "shellcode_encoding": "shellcode",
    "alphanumeric_shellcode": "shellcode",
    "shellcode_analysis": "shellcode",

    # Reverse engineering
    "reverse_engineering": "reverse_engineering",
    "reverse_engineering_analysis": "reverse_engineering",
    "rev": "reverse_engineering",
    "unpacking": "reverse_engineering",
    "deobfuscation": "reverse_engineering",
    "static_analysis": "reverse_engineering",
    "dynamic_analysis": "reverse_engineering",
    "debugging": "reverse_engineering",

    # Binary exploitation general
    "pwn": "pwn",
    "binary_exploitation": "pwn",
    "exploit": "pwn",
    "exploitation": "pwn",

    # Sandbox escape
    "sandbox_escape": "sandbox_escape",
    "pyjail": "sandbox_escape",
    "python_jail": "sandbox_escape",
    "jail": "sandbox_escape",
    "jail_escape": "sandbox_escape",

    # Deserialization
    "insecure_deserialization": "insecure_deserialization",
    "deserialization": "insecure_deserialization",
    "pickle": "insecure_deserialization",
    "pickle_deserialization": "insecure_deserialization",
    "php_deserialization": "insecure_deserialization",
    "java_deserialization": "insecure_deserialization",
    "ysoserial": "insecure_deserialization",
    "php_unserialize": "insecure_deserialization",
    "python_unpickle": "insecure_deserialization",

    # Command injection
    "command_injection": "command_injection",
    "os_command_injection": "command_injection",
    "rce": "command_injection",
    "remote_code_execution": "command_injection",
    "code_execution": "command_injection",

    # Path traversal / LFI/RFI
    "path_traversal": "path_traversal",
    "lfi": "path_traversal",
    "local_file_inclusion": "path_traversal",
    "file_inclusion": "path_traversal",
    "rfi": "path_traversal",
    "remote_file_inclusion": "path_traversal",
    "zip_slip": "path_traversal",
    "directory_traversal": "path_traversal",
    "path_traversal_attack": "path_traversal",

    # SSRF
    "ssrf": "ssrf",
    "server_side_request_forgery": "ssrf",

    # XXE
    "xxe": "xxe",
    "xml_external_entity": "xxe",
    "xml_external_entity_attack": "xxe",
    "xee": "xxe",

    # Crypto
    "crypto": "crypto",
    "cryptography": "crypto",

    # RSA attacks
    "rsa_attacks": "rsa_attacks",
    "rsa": "rsa_attacks",
    "wiener_attack": "rsa_attacks",
    "coppersmith": "rsa_attacks",
    "common_modulus": "rsa_attacks",
    "common_modulus_attack": "rsa_attacks",
    "hastad_broadcast": "rsa_attacks",
    "franklin_reiter": "rsa_attacks",
    "bleichenbacher": "rsa_attacks",
    "low_exponent": "rsa_attacks",
    "rsa_wiener": "rsa_attacks",
    "rsa_coppersmith": "rsa_attacks",
    "rsa_common_modulus": "rsa_attacks",
    "rsa_hastad": "rsa_attacks",
    "rsa_bleichenbacher": "rsa_attacks",
    "rsa_stereotyped": "rsa_attacks",
    "rsa_related_message": "rsa_attacks",
    "rsa_franklin_reiter": "rsa_attacks",

    # Padding oracle
    "padding_oracle": "padding_oracle",
    "padding_oracle_attack": "padding_oracle",
    "cbc_padding_oracle": "padding_oracle",
    "oracle_attack": "padding_oracle",

    # Timing attacks
    "timing_attack": "timing_attack",
    "timing_side_channel": "timing_attack",
    "side_channel": "timing_attack",
    "side_channel_attack": "timing_attack",

    # Encoding / XOR
    "xor_cipher": "xor_cipher",
    "xor": "xor_cipher",
    "xor_encryption": "xor_cipher",

    # Steganography
    "stegano": "stegano",
    "steganography": "stegano",
    "lsb_steganography": "stegano",
    "lsb": "stegano",

    # Forensics
    "forensics": "forensics",
    "forensic": "forensics",
    "memory_forensics": "forensics",
    "disk_forensics": "forensics",
    "pcap_analysis": "forensics",
    "network_forensics": "forensics",

    # Web general
    "web": "web",
    "web_exploitation": "web",

    # Type juggling
    "type_juggling": "type_juggling",
    "php_type_juggling": "type_juggling",
    "loose_comparison": "type_juggling",

    # JWT
    "jwt": "jwt",
    "jwt_attack": "jwt",
    "json_web_token": "jwt",

    # Prototype pollution
    "prototype_pollution": "prototype_pollution",
    "js_prototype_pollution": "prototype_pollution",

    # Race condition
    "race_condition": "race_condition",
    "race_condition_attack": "race_condition",
    "toctou": "race_condition",

    # HTTP smuggling
    "http_smuggling": "http_smuggling",
    "http_request_smuggling": "http_smuggling",

    # Symbolic execution
    "symbolic_execution": "symbolic_execution",
    "angr": "symbolic_execution",
    "z3": "symbolic_execution",
    "solver": "symbolic_execution",
    "constraint_solving": "symbolic_execution",

    # Fuzzing
    "fuzzing": "fuzzing",
    "fuzz": "fuzzing",
    "fuzz_testing": "fuzzing",

    # Hash length extension
    "hash_length_extension": "hash_length_extension",
    "hash_extension": "hash_length_extension",
    "hash_ext": "hash_length_extension",

    # NoSQL injection
    "nosql_injection": "nosql_injection",
    "no_sql_injection": "nosql_injection",
    "mongodb_injection": "nosql_injection",
    "mongodb": "nosql_injection",

    # LDAP injection
    "ldap_injection": "ldap_injection",
    "ldap": "ldap_injection",

    # File upload
    "file_upload": "file_upload",
    "file_upload_attack": "file_upload",
    "file_upload_bypass": "file_upload",

    # Open redirect
    "open_redirect": "open_redirect",
    "url_redirect": "open_redirect",

    # Clickjacking
    "clickjacking": "clickjacking",
    "click_jacking": "clickjacking",
    "ui_redressing": "clickjacking",

    # OAuth
    "oauth": "oauth",
    "oauth_attack": "oauth",

    # CORS
    "cors": "cors",
    "cross_origin": "cors",

    # Cookie / session
    "cookie": "cookie",
    "cookie_theft": "cookie",
    "session_hijacking": "cookie",
    "session_fixation": "cookie",

    # CSRF
    "csrf": "csrf",
    "cross_site_request_forgery": "csrf",
    "xsrf": "csrf",

    # GraphQL
    "graphql": "graphql",
    "graphql_attack": "graphql",
    "graphql_introspection": "graphql",

    # WebSocket
    "websocket": "websocket",
    "ws": "websocket",
    "web_socket": "websocket",

    # PHP specific
    "php": "php",
    "php_shell": "php",
    "php_wrapper": "php",
    "php_filter": "php",

    # Docker / Container
    "docker": "docker",
    "container": "docker",
    "container_escape": "docker",

    # Kubernetes
    "kubernetes": "kubernetes",
    "k8s": "kubernetes",

    # Cloud
    "cloud": "cloud",
    "aws": "cloud",
    "gcp": "cloud",
    "azure": "cloud",

    # Mobile
    "mobile": "mobile",
    "android": "mobile",
    "ios": "mobile",

    # Browser exploitation
    "browser": "browser",
    "browser_exploitation": "browser",
    "v8": "browser",
    "webkit": "browser",

    # Kernel
    "kernel": "kernel",
    "kernel_exploit": "kernel",
    "kernel_exploitation": "kernel",

    # Windows
    "windows": "windows",
    "windows_exploitation": "windows",
    "powershell": "windows",

    # Linux
    "linux": "linux",
    "linux_exploitation": "linux",

    # Misc / general
    "misc": "misc",
    "miscellaneous": "misc",
    "trivia": "misc",
    "bruteforce": "bruteforce",
    "brute_force": "bruteforce",
    "one_gadget": "one_gadget",
    "seccomp_bypass": "seccomp_bypass",
    "dns_exfiltration": "dns_exfiltration",
    "privilege_escalation": "privilege_escalation",
    "privesc": "privilege_escalation",
    "information_disclosure": "information_disclosure",
    "information_leak": "information_disclosure",
    "osint": "osint",
    "open_source_intelligence": "osint",
    "reconnaissance": "osint",
    "recon": "osint",
    "cve": "cve",
    "cve_exploitation": "cve",
    "patching": "patching",
    "binary_patching": "patching",
    "key_recovery": "key_recovery",
    "key_extraction": "key_recovery",
    "password_cracking": "password_cracking",
    "hash_cracking": "password_cracking",
    "crack": "password_cracking",
    "john": "password_cracking",
    "john_the_ripper": "password_cracking",
    "hashcat": "password_cracking",
    "network": "network",
    "network_analysis": "network",
    "binary_analysis": "binary_analysis",
    "binary_diffing": "binary_analysis",
    "diffing": "binary_analysis",
    "firmware": "firmware",
    "firmware_analysis": "firmware",
    "firmware_extraction": "firmware",
    "embedded": "embedded",
    "embedded_system": "embedded",
    "game": "game",
    "game_hacking": "game",
    "iot": "iot",
    "game_boy": "game",
    "nes": "game",
    "snes": "game",
    "dll": "dll",
    "dll_hijacking": "dll",
    "dll_injection": "dll",
    "process_injection": "process_injection",
    "process_hollowing": "process_injection",
    "api_hooking": "api_hooking",
    "hook": "api_hooking",
    "rootkit": "rootkit",
    "bootkit": "bootkit",
    "driver": "driver",
    "driver_exploitation": "driver",
    "kernel_driver": "driver",
    "wasm": "wasm",
    "webassembly": "wasm",
    "golang": "golang",
    "go": "golang",
    "rust": "rust",
    "dotnet": "dotnet",
    "c#": "dotnet",
    "csharp": "dotnet",
    "arm": "arm",
    "arm_exploitation": "arm",
    "mips": "mips",
    "mips_exploitation": "mips",
    "aarch64": "aarch64",
    "x86": "x86",
    "x64": "x64",
    "x86_64": "x64",
    "amd64": "x64",
    "assembly": "assembly",
    "asm": "assembly",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "node": "javascript",
    "nodejs": "javascript",
    "python": "python",
    "python_exploitation": "python",
    "ruby": "ruby",
    "perl": "perl",
    "lua": "lua",
    "java": "java",
    "webshell": "webshell",
    "web_shell": "webshell",
    "backdoor": "backdoor",
    "trojan": "backdoor",
    "reverse_shell": "reverse_shell",
    "bind_shell": "bind_shell",
    "shell": "shell",
    "bash": "shell",
    "zsh": "shell",
    "sh": "shell",
    "c": "c",
    "c++": "c++",
    "cpp": "c++",
    "injection": "injection",
    "xpath_injection": "xpath_injection",
    "xpath": "xpath_injection",
    "email_injection": "email_injection",
    "smtp_injection": "email_injection",
    "header_injection": "header_injection",
    "http_header_injection": "header_injection",
    "crlf": "header_injection",
    "crlf_injection": "header_injection",
    "response_splitting": "header_injection",
    "idora": "idor",
    "idor": "idor",
    "insecure_direct_object_reference": "idor",
    "mass_assignment": "mass_assignment",
    "parameter_pollution": "parameter_pollution",
    "http_parameter_pollution": "parameter_pollution",
    "hpp": "parameter_pollution",
    "cache_poisoning": "cache_poisoning",
    "web_cache_poisoning": "cache_poisoning",
    "host_header": "host_header",
    "host_header_attack": "host_header",
    "subdomain_takeover": "subdomain_takeover",
    "dns_rebinding": "dns_rebinding",
    "request_smuggling": "request_smuggling",
    "server_side_include": "server_side_include",
    "ssi": "server_side_include",
    "server_side_include_injection": "server_side_include",
    "cgi": "cgi",
    "cgi_attack": "cgi",
    "tomcat": "tomcat",
    "spring": "spring",
    "struts": "struts",
    "shiro": "shiro",
    "weblogic": "weblogic",
    "jboss": "jboss",
    "jenkins": "jenkins",
    "confluence": "confluence",
    "coldfusion": "coldfusion",
    "iis": "iis",
    "apache": "apache",
    "nginx": "nginx",
    "redis": "redis",
    "elasticsearch": "elasticsearch",
    "couchdb": "couchdb",
    "dockerfile": "docker",
    "docker_compose": "docker",
    "docker_registry": "docker",
    "docker_api": "docker",
    "docker_socket": "docker",
    "docker_escape": "docker",
    "kubectl": "kubernetes",
    "istio": "kubernetes",
    "helm": "kubernetes",
    "terraform": "cloud",
    "ansible": "cloud",
    "cloudformation": "cloud",
    "s3": "cloud",
    "s3_bucket": "cloud",
    "iam": "cloud",
    "lambda": "cloud",
    "ec2": "cloud",
    "eks": "cloud",
    "gcs": "cloud",
    "gke": "cloud",
    "azure_blob": "cloud",
    "azure_ad": "cloud",
    "azure_ad_attack": "cloud",
    "azure_key_vault": "cloud",
    "azure_vm": "cloud",
    "azure_container": "cloud",
    "azure_kubernetes": "cloud",
    "aks": "cloud",
    "gcp_storage": "cloud",
    "gcp_compute": "cloud",
    "gcp_function": "cloud",
    "gcp_iam": "cloud",
    "gcp_kms": "cloud",
    "gcp_gke": "cloud",
    "gcp_bigquery": "cloud",
    "gcp_dataflow": "cloud",
    "gcp_pubsub": "cloud",
    "gcp_secret_manager": "cloud",
    "gcp_security_command_center": "cloud",
    "gcp_cloud_shell": "cloud",
    "gcp_cloud_sdk": "cloud",
    "gcp_app_engine": "cloud",
    "gcp_cloud_run": "cloud",
    "gcp_cloud_functions": "cloud",
    "gcp_cloud_build": "cloud",
    "gcp_cloud_source_repositories": "cloud",
    "gcp_cloud_deploy": "cloud",
    "gcp_cloud_scheduler": "cloud",
    "gcp_cloud_tasks": "cloud",
    "gcp_cloud_logging": "cloud",
    "gcp_cloud_monitoring": "cloud",
    "gcp_cloud_trace": "cloud",
    "gcp_cloud_profiler": "cloud",
    "gcp_cloud_error_reporting": "cloud",
    "gcp_cloud_debugger": "cloud",
    "gcp_cloud_test_lab": "cloud",
    "gcp_cloud_iot": "cloud",
    "gcp_cloud_vision": "cloud",
    "gcp_cloud_speech": "cloud",
    "gcp_cloud_translate": "cloud",
    "gcp_cloud_natural_language": "cloud",
    "gcp_cloud_video_intelligence": "cloud",
    "gcp_cloud_recommendations": "cloud",
    "gcp_cloud_talent": "cloud",
    "gcp_cloud_dialogflow": "cloud",
    "gcp_cloud_automl": "cloud",
    "gcp_cloud_inference": "cloud",
    "gcp_cloud_ai_platform": "cloud",
    "gcp_cloud_notebooks": "cloud",
    "gcp_cloud_dataflow": "cloud",
    "gcp_cloud_dataproc": "cloud",
    "gcp_cloud_dataprep": "cloud",
    "gcp_cloud_datafusion": "cloud",
    "gcp_cloud_composer": "cloud",
    "gcp_cloud_life_sciences": "cloud",
    "gcp_cloud_healthcare": "cloud",
    "gcp_cloud_financial_services": "cloud",
    "gcp_cloud_retail": "cloud",
    "gcp_cloud_manufacturing": "cloud",
    "gcp_cloud_gaming": "cloud",
    "gcp_cloud_media": "cloud",
    "gcp_cloud_telecommunications": "cloud",
    "gcp_cloud_energy": "cloud",
    "gcp_cloud_supply_chain": "cloud",
    "gcp_cloud_sustainability": "cloud",
    "gcp_cloud_public_sector": "cloud",
    "gcp_cloud_nonprofit": "cloud",
    "gcp_cloud_education": "cloud",
    "gcp_cloud_healthcare_data_engine": "cloud",
    "gcp_cloud_healthcare_nlp": "cloud",
    "gcp_cloud_healthcare_api": "cloud",
    "gcp_cloud_healthcare_fhir": "cloud",
    "gcp_cloud_healthcare_dicom": "cloud",
    "gcp_cloud_healthcare_hl7v2": "cloud",
    "gcp_cloud_healthcare_consent": "cloud",
    "gcp_cloud_healthcare_deid": "cloud",
    "gcp_cloud_healthcare_imaging": "cloud",
    "gcp_cloud_healthcare_nlp_api": "cloud",
    "gcp_cloud_healthcare_auto_ml": "cloud",
    "gcp_cloud_healthcare_entity_extraction": "cloud",
    "gcp_cloud_healthcare_context_extraction": "cloud",
    "gcp_cloud_healthcare_relationship_extraction": "cloud",
    "gcp_cloud_healthcare_coding": "cloud",
    "gcp_cloud_healthcare_ontology": "cloud",
    "gcp_cloud_healthcare_note_reader": "cloud",
    "gcp_cloud_healthcare_dicom_store": "cloud",
    "gcp_cloud_healthcare_fhir_store": "cloud",
    "gcp_cloud_healthcare_hl7v2_store": "cloud",
    "gcp_cloud_healthcare_consent_store": "cloud",
    "gcp_cloud_healthcare_deid_store": "cloud",
    "gcp_cloud_healthcare_imaging_store": "cloud",
    "gcp_cloud_healthcare_nlp_store": "cloud",
    "gcp_cloud_healthcare_auto_ml_store": "cloud",
    "gcp_cloud_healthcare_entity_extraction_store": "cloud",
    "gcp_cloud_healthcare_context_extraction_store": "cloud",
    "gcp_cloud_healthcare_relationship_extraction_store": "cloud",
    "gcp_cloud_healthcare_coding_store": "cloud",
    "gcp_cloud_healthcare_ontology_store": "cloud",
    "gcp_cloud_healthcare_note_reader_store": "cloud",
    "aws_s3": "cloud",
    "aws_iam": "cloud",
    "aws_lambda": "cloud",
    "aws_ec2": "cloud",
    "aws_eks": "cloud",
    "aws_cloudformation": "cloud",
    "aws_cloudtrail": "cloud",
    "aws_cloudwatch": "cloud",
    "aws_config": "cloud",
    "aws_kms": "cloud",
    "aws_secrets_manager": "cloud",
    "aws_parameter_store": "cloud",
    "aws_sqs": "cloud",
    "aws_sns": "cloud",
    "aws_dynamodb": "cloud",
    "aws_rds": "cloud",
    "aws_aurora": "cloud",
    "aws_redshift": "cloud",
    "aws_elasticache": "cloud",
    "aws_elasticsearch": "cloud",
    "aws_elastic_beanstalk": "cloud",
    "aws_ecs": "cloud",
    "aws_ecr": "cloud",
    "aws_fargate": "cloud",
    "aws_lambda_edge": "cloud",
    "aws_api_gateway": "cloud",
    "aws_cloudfront": "cloud",
    "aws_route53": "cloud",
    "aws_elb": "cloud",
    "aws_alb": "cloud",
    "aws_nlb": "cloud",
    "aws_vpc": "cloud",
    "aws_subnet": "cloud",
    "aws_security_group": "cloud",
    "aws_nacl": "cloud",
    "aws_flow_logs": "cloud",
    "aws_vpc_endpoint": "cloud",
    "aws_vpc_peering": "cloud",
    "aws_vpn": "cloud",
    "aws_direct_connect": "cloud",
    "aws_transit_gateway": "cloud",
    "aws_private_link": "cloud",
    "aws_network_firewall": "cloud",
    "aws_waf": "cloud",
    "aws_shield": "cloud",
    "aws_guardduty": "cloud",
    "aws_macie": "cloud",
    "aws_inspector": "cloud",
    "aws_artifact": "cloud",
    "aws_certificate_manager": "cloud",
    "aws_acm": "cloud",
    "aws_cloudhsm": "cloud",
    "aws_systems_manager": "cloud",
    "aws_opsworks": "cloud",
    "aws_codecommit": "cloud",
    "aws_codebuild": "cloud",
    "aws_codedeploy": "cloud",
    "aws_codepipeline": "cloud",
    "aws_codeartifact": "cloud",
    "aws_codeguru": "cloud",
    "aws_codestar": "cloud",
    "aws_codewhisperer": "cloud",
    "aws_appsync": "cloud",
    "aws_appmesh": "cloud",
    "aws_apprunner": "cloud",
    "aws_appconfig": "cloud",
    "aws_appflow": "cloud",
    "aws_appintegrations": "cloud",
    "aws_appstream": "cloud",
    "aws_amplify": "cloud",
    "aws_device_farm": "cloud",
}


def _clean_name(technique_name: str) -> str:
    """Lowercase and normalize separators/parens from a raw technique name."""
    name = technique_name.lower().strip()
    name = name.replace(" ", "_").replace("-", "_")
    name = name.replace("(", "").replace(")", "")
    return name


def build_deterministic_mapping(
    raw_names: Iterable[str], mapping_path: Path | None = None
) -> dict[str, str]:
    """Build a frozen ``{raw_name: canonical}`` map independent of input order.

    Unlike :meth:`TechniqueNormalizer.normalize`, this never fuzzy-matches
    unknowns against a *growing* set, so the same input always yields the same
    grouping (see [[normalizer-order-dependence]]). Resolution per cleaned name:

      1. exact alias in DEFAULT_MAPPING + custom YAML overrides → canonical
      2. already a canonical value → itself
      3. long chatty name (>40 chars) embedding a known keyword → that canonical
      4. otherwise → self-canonical (auto-discovery, but frozen)

    The ``>40`` substring fold is the only fuzzy-ish step and it is cheap,
    order-independent, and applied against the same sorted keyword list every
    time. Unknown names are NOT fuzzy-matched here — that is deliberately
    conservative: the live normalizer maps e.g. ``json_injection`` →
    ``sql_injection`` via fuzzy (wrong); deterministic mapping keeps them
    distinct and lets :class:`~src.summarization.technique_merger.TechniqueMerger`
    merge near-duplicates only when content or an LLM judge confirms it.
    """
    mapping = dict(DEFAULT_MAPPING)
    if mapping_path is not None and mapping_path.exists():
        try:
            with open(mapping_path, "r") as f:
                custom = yaml.safe_load(f)
            if custom and "technique_mapping" in custom:
                mapping.update(custom["technique_mapping"])
        except Exception:  # noqa: BLE001 (bad mapping file -> use defaults)
            logger.warning("Failed to load custom mapping: %s", mapping_path)

    canonical_values = set(mapping.values())
    # Sorted by descending key length: longest keyword wins the substring match.
    sorted_mapping = sorted(mapping.items(), key=lambda kv: -len(kv[0]))

    result: dict[str, str] = {}
    for raw in sorted(set(raw_names)):  # sorted iteration -> deterministic
        name = _clean_name(raw)
        canonical = mapping.get(name)
        if canonical is None and name not in canonical_values and len(name) > 40:
            for keyword, c in sorted_mapping:
                if keyword in name:
                    canonical = c
                    break
        result[raw] = canonical or name
    return result


class TechniqueNormalizer:
    """Normalize technique names to canonical forms for file organization."""

    def __init__(self, mapping_path: Path | None = None):
        self.mapping_path = mapping_path or settings.technique_mapping_path
        self.mapping = dict(DEFAULT_MAPPING)
        self._load_custom_mapping()

        # Build reverse lookup and canonical set
        self.canonical_names = set(self.mapping.values())
        self.all_known_names = set(self.mapping.keys())
        # Pre-sort mapping items by descending key length for substring matching
        # (DO NOT redo this on every normalize call — 37k × sort = 22M wasted ops).
        self._sorted_mapping = sorted(
            self.mapping.items(), key=lambda kv: -len(kv[0])
        )

    def _load_custom_mapping(self) -> None:
        """Load user-defined mapping overrides from YAML file."""
        if self.mapping_path and self.mapping_path.exists():
            try:
                with open(self.mapping_path, "r") as f:
                    custom = yaml.safe_load(f)
                if custom and "technique_mapping" in custom:
                    self.mapping.update(custom["technique_mapping"])
                    logger.info(
                        "Loaded %d custom mappings from %s",
                        len(custom["technique_mapping"]),
                        self.mapping_path,
                    )
            except Exception as e:  # noqa: BLE001 (bad mapping file -> use defaults)
                logger.warning("Failed to load custom mapping: %s", e)

    def save_custom_mapping(self) -> None:
        """Save the current mapping to YAML for user review."""
        if self.mapping_path:
            self.mapping_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.mapping_path, "w") as f:
                yaml.dump(
                    {"technique_mapping": self.mapping},
                    f,
                    default_flow_style=False,
                    sort_keys=True,
                )
            logger.info("Saved technique mapping to %s", self.mapping_path)

    def normalize(self, technique_name: str) -> str:
        """Map a technique name to its canonical form.

        e.g., "blind_sql_injection" -> "sql_injection"

        Note: this method is order-dependent (it mutates the known-name set as
        it discovers unknowns, and the fuzzy matcher iterates a set). For
        deterministic, order-independent grouping over a whole dataset use
        :meth:`normalize_batch`.
        """
        raw_name = technique_name

        # Clean and normalize
        name = _clean_name(technique_name)

        # Direct match
        if name in self.mapping:
            return self.mapping[name]

        # Check if it's already canonical
        if name in self.canonical_names:
            return name

        # Verbose/chatty technique names (cheap substring check, run BEFORE the
        # expensive fuzzy match): if the name embeds a known keyword as a
        # substring (e.g. "time based blind SQL injection with SLEEP() ..."),
        # map to that canonical instead of creating a new file.
        if len(name) > 40:
            for keyword, canonical in self._sorted_mapping:
                if keyword in name:
                    logger.debug(
                        "Substring matched '%s' -> '%s' -> '%s'",
                        raw_name, keyword, canonical,
                    )
                    return canonical

        # Fuzzy match: find closest known variant
        closest = self._fuzzy_match(name, self.all_known_names)
        if closest:
            result = self.mapping[closest]
            logger.debug("Fuzzy matched '%s' -> '%s' -> '%s'", raw_name, closest, result)
            return result

        # Unknown: add as new canonical
        logger.info("Unknown technique '%s', adding as new canonical", raw_name)
        self.all_known_names.add(name)
        self.canonical_names.add(name)
        self.mapping[name] = name
        return name

    def normalize_batch(self, raw_names: Iterable[str]) -> dict[str, str]:
        """Build a deterministic ``{raw_name: canonical}`` map for a dataset.

        Order-independent and stable (see :func:`build_deterministic_mapping`).
        Use this for grouping/merging decisions where reproducibility matters;
        :meth:`normalize` keeps its live fuzzy auto-discovery for single names.
        """
        return build_deterministic_mapping(raw_names, self.mapping_path)

    def _fuzzy_match(self, name: str, candidates: set[str]) -> str | None:
        """Rapidfuzz-based fuzzy matching with threshold.

        Uses `process.extractOne` (fully C-optimized) instead of the pure-
        Python Levenshtein that was ~100x slower on 15k+ unknown names.
        """
        result = process.extractOne(
            name,
            list(candidates),
            scorer=fuzz.ratio,
            score_cutoff=70,
        )
        return result[0] if result else None