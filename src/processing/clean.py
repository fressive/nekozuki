"""HTML cleaning and writeup preprocessing."""

import logging
import re
import sys

from bs4 import BeautifulSoup
from markdownify import markdownify as md

from src.config import settings

logger = logging.getLogger(__name__)

# Map of keyword patterns to technique names
keyword_patterns = {
    "sql injection": ["sqli", "sql_injection"],
    "sqli": ["sqli", "sql_injection"],
    "xss": ["xss"],
    "cross-site": ["xss"],
    "buffer overflow": ["buffer_overflow", "pwn"],
    "rop": ["buffer_overflow", "rop"],
    "return oriented": ["rop", "buffer_overflow"],
    "format string": ["format_string", "pwn"],
    "reverse engineering": ["reverse_engineering", "rev"],
    "unpacking": ["reverse_engineering"],
    "cryptography": ["crypto"],
    "rsa": ["crypto", "rsa_attacks"],
    "aes": ["crypto"],
    "steganography": ["stegano"],
    "forensic": ["forensics"],
    "pwn": ["pwn"],
    "shellcode": ["shellcode", "pwn"],
    "jail": ["pyjail", "jail"],
    "python jail": ["pyjail"],
    "sandbox escape": ["sandbox_escape", "jail"],
    "command injection": ["command_injection", "web"],
    "lfi": ["lfi", "file_inclusion"],
    "rfi": ["rfi", "file_inclusion"],
    "ssrf": ["ssrf", "web"],
    "deserialization": ["deserialization", "insecure_deserialization"],
    "type juggling": ["type_juggling", "php"],
    "template injection": ["ssti", "server_side_template_injection"],
    "ssti": ["ssti"],
    "xxe": ["xxe", "xml_external_entity"],
    "cors": ["cors", "web"],
    "csrf": ["csrf", "web"],
    "oauth": ["oauth", "web"],
    "jwt": ["jwt", "web"],
    "race condition": ["race_condition", "web"],
    "timing attack": ["timing_attack", "side_channel"],
    "side channel": ["side_channel", "timing_attack"],
    "bruteforce": ["bruteforce", "crypto"],
    "oracle": ["oracle_attack", "crypto"],
    "padding oracle": ["padding_oracle", "crypto"],
    "hash length extension": ["hash_length_extension", "crypto"],
    "meet in the middle": ["meet_in_the_middle", "crypto"],
    "frequency analysis": ["frequency_analysis", "crypto"],
    "coppersmith": ["coppersmith", "crypto", "rsa_attacks"],
    "wiener": ["wiener_attack", "crypto", "rsa_attacks"],
    "franklin-reiter": ["franklin_reiter", "crypto"],
    "bleichenbacher": ["bleichenbacher", "crypto", "rsa_attacks"],
    "z3": ["z3", "solver", "rev"],
    "angr": ["angr", "symbolic_execution", "rev"],
    "symbolic execution": ["symbolic_execution", "rev"],
    "fuzzing": ["fuzzing", "pwn"],
    "integer overflow": ["integer_overflow", "pwn"],
    "use after free": ["use_after_free", "pwn", "heap_exploitation"],
    "heap overflow": ["heap_exploitation", "pwn"],
    "tcache": ["heap_exploitation", "pwn"],
    "fastbin": ["heap_exploitation", "pwn", "fastbin_attack"],
    "unsafe unlink": ["unsafe_unlink", "heap_exploitation", "pwn"],
    "house of force": ["house_of_force", "heap_exploitation", "pwn"],
    "ret2dlresolve": ["ret2dlresolve", "pwn"],
    "ret2csu": ["ret2csu", "pwn"],
    "seccomp bypass": ["seccomp_bypass", "pwn", "sandbox_escape"],
    "mprotect": ["shellcode", "pwn"],
    "one_gadget": ["one_gadget", "pwn"],
    "fsop": ["fsop", "pwn", "heap_exploitation"],
    "io file": ["io_file", "pwn", "heap_exploitation"],
    "gdb": ["debugging", "rev"],
    "ida pro": ["static_analysis", "rev"],
    "ghidra": ["static_analysis", "rev"],
    "obfuscation": ["deobfuscation", "rev"],
    "ollvm": ["deobfuscation", "rev", "obfuscation"],
    "base64": ["encoding", "crypto"],
    "xor": ["xor_cipher", "crypto"],
    "vigenere": ["vigenere", "crypto"],
    "substitution": ["substitution_cipher", "crypto"],
    "lsb": ["lsb_steganography", "stegano"],
    "byte order mark": ["bom", "forensics"],
    "pcap": ["pcap_analysis", "forensics", "network"],
    "wireshark": ["pcap_analysis", "forensics", "network"],
    "tcpdump": ["pcap_analysis", "forensics", "network"],
    "volatility": ["memory_forensics", "forensics"],
    "memory dump": ["memory_forensics", "forensics"],
    "disk image": ["disk_forensics", "forensics"],
    "registry": ["windows_forensics", "forensics"],
    "ntfs": ["ntfs_forensics", "forensics"],
    "dns exfiltration": ["dns_exfiltration", "network", "forensics"],
    "http smuggling": ["http_smuggling", "web"],
    "http request smuggling": ["http_smuggling", "web"],
    "server side request forgery": ["ssrf", "web"],
    "prototype pollution": ["prototype_pollution", "web", "js"],
    "nosql injection": ["nosql_injection", "web"],
    "ldap injection": ["ldap_injection", "web"],
    "xpath injection": ["xpath_injection", "web"],
    "file upload": ["file_upload", "web"],
    "zip slip": ["zip_slip", "web"],
    "path traversal": ["path_traversal", "web", "lfi"],
    "open redirect": ["open_redirect", "web"],
    "clickjacking": ["clickjacking", "web"],
    "websocket": ["websocket", "web"],
    "graphql": ["graphql", "web"],
    "injection": ["injection", "web"],
    "server side template injection": ["ssti", "server_side_template_injection"],
    "smarty": ["ssti", "php"],
    "twig": ["ssti", "php"],
    "jinja": ["ssti", "python", "server_side_template_injection"],
    "jinja2": ["ssti", "python", "server_side_template_injection"],
    "flask": ["ssti", "python", "web"],
    "django": ["ssti", "python", "web"],
    "express": ["web", "js"],
    "node": ["web", "js"],
    "php": ["php", "web"],
    "cgi": ["cgi", "web"],
    "tomcat": ["tomcat", "web", "java"],
    "java": ["java", "web"],
    "spring": ["spring", "web", "java"],
    "struts": ["struts", "web", "java"],
    "shiro": ["shiro", "web", "java"],
    "weblogic": ["weblogic", "web", "java"],
    "jboss": ["jboss", "web", "java"],
    "glassfish": ["glassfish", "web", "java"],
    "jenkins": ["jenkins", "web", "java"],
    "confluence": ["confluence", "web", "java"],
    "coldfusion": ["coldfusion", "web"],
    "iis": ["iis", "web", "windows"],
    "apache": ["apache", "web"],
    "nginx": ["nginx", "web"],
    "mysql": ["mysql", "sql_injection", "web"],
    "postgresql": ["postgresql", "sql_injection", "web"],
    "mssql": ["mssql", "sql_injection", "web"],
    "sqlite": ["sqlite", "sql_injection", "web"],
    "redis": ["redis", "web"],
    "mongodb": ["mongodb", "nosql_injection", "web"],
    "couchdb": ["couchdb", "nosql_injection", "web"],
    "elasticsearch": ["elasticsearch", "web"],
    "docker": ["docker", "container", "web"],
    "kubernetes": ["kubernetes", "container", "web"],
    "k8s": ["kubernetes", "container", "web"],
    "aws": ["aws", "cloud", "web"],
    "gcp": ["gcp", "cloud", "web"],
    "azure": ["azure", "cloud", "web"],
    "golang": ["golang", "rev"],
    "rust": ["rust", "rev"],
    "wasm": ["wasm", "web", "rev"],
    "webassembly": ["wasm", "web", "rev"],
    "javascript": ["javascript", "js", "web"],
    "typescript": ["typescript", "js", "web"],
    "python": ["python", "web", "rev"],
    "ruby": ["ruby", "web"],
    "perl": ["perl", "web"],
    "lua": ["lua", "web", "rev"],
    "powershell": ["powershell", "web", "windows"],
    "batch": ["batch", "windows"],
    "shell": ["shell", "web", "pwn"],
    "bash": ["bash", "web", "pwn"],
    "zsh": ["zsh", "web", "pwn"],
    "sh": ["sh", "web", "pwn"],
    "c": ["c", "pwn", "rev"],
    "c++": ["c++", "pwn", "rev"],
    "cpp": ["c++", "pwn", "rev"],
    "c#": ["c#", "rev", "web"],
    "csharp": ["c#", "rev", "web"],
    "go": ["golang", "rev"],
    "assembly": ["assembly", "rev", "pwn"],
    "arm": ["arm", "rev", "pwn", "embedded"],
    "mips": ["mips", "rev", "pwn", "embedded"],
    "x86": ["x86", "rev", "pwn"],
    "x64": ["x64", "rev", "pwn"],
    "amd64": ["x64", "rev", "pwn"],
    "aarch64": ["aarch64", "rev", "pwn", "embedded"],
    "embedded": ["embedded", "rev", "pwn", "iot"],
    "iot": ["iot", "embedded", "rev", "pwn"],
    "firmware": ["firmware", "embedded", "rev"],
    "v8": ["v8", "js", "pwn", "browser"],
    "spider monkey": ["spider_monkey", "js", "pwn", "browser"],
    "javascriptcore": ["javascriptcore", "js", "pwn", "browser"],
    "webkit": ["webkit", "browser", "pwn", "js"],
    "chromium": ["chromium", "browser", "pwn"],
    "firefox": ["firefox", "browser", "pwn"],
    "exploit": ["exploit", "pwn"],
    "kernel": ["kernel", "pwn", "exploit"],
    "windows": ["windows", "pwn", "rev", "forensics"],
    "linux": ["linux", "pwn", "rev", "forensics"],
    "macos": ["macos", "pwn", "rev", "forensics"],
    "android": ["android", "mobile", "rev"],
    "ios": ["ios", "mobile", "rev"],
    "mobile": ["mobile", "rev", "android", "ios"],
    "symcc": ["symcc", "symbolic_execution", "rev"],
    "klee": ["klee", "symbolic_execution", "rev"],
    "triton": ["triton", "symbolic_execution", "rev"],
    "miasm": ["miasm", "rev", "symbolic_execution"],
    "binary ninja": ["binary_ninja", "rev"],
    "radare2": ["radare2", "rev"],
    "rizin": ["rizin", "rev"],
    "dnlib": ["dnlib", "rev", "dotnet"],
    "ilspy": ["ilspy", "rev", "dotnet"],
    "dotnet": ["dotnet", "rev", "c#"],
    "mono": ["mono", "rev", "c#"],
    "unity": ["unity", "rev", "c#", "game"],
    "unreal": ["unreal", "rev", "c++", "game"],
}


# Patterns that are too common / noisy to be meaningful technique hints
_NOISY_HINTS = {
"c", "c++", "sh", "shell", "web", "exploit", "python", "rev",
"c#", "go", "rust", "lua", "java", "php", "perl", "ruby",
"game", "aws", "cloud", "gcp", "azure", "docker", "k8s",
"windows", "linux", "mobile", "x86", "x64", "arm", "mips",
"injection", "pwn", "misc", "osint", "golang", "dotnet",
"javascript", "js", "network", "fuzz", "cve", "browser",
}

# Pre-compile ONE alternation regex so a single pass over the content finds all
# keyword matches. Scanning the full content once (~100x faster than 300
# separate searches) matters because this runs per-writeup across ~25k writeups.
# Alternatives are sorted longest-first so phrases ("sql injection") win over
# their sub-words ("injection") when both would match at the same position.
_keywords_sorted = sorted(keyword_patterns, key=len, reverse=True)
_KEYWORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in _keywords_sorted) + r")\b"
)
_KEYWORD_TAGS = keyword_patterns

# Some scraped pages have deeply nested HTML that exceeds the default limit.
# Raise it to avoid RecursionError in markdownify/BeautifulSoup.
if sys.getrecursionlimit() < 10000:
    sys.setrecursionlimit(10000)


def clean_html_content(raw_html: str) -> str:
    """Strip HTML boilerplate and convert to clean markdown text.

    Removes: nav bars, sidebars, comments, scripts, styles, footer, tags.
    """
    if not raw_html or not raw_html.strip():
        return ""

    # Parse with BeautifulSoup
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove unwanted elements
    for selector in [
        "script", "style", "nav", "footer", "header",
        ".sidebar", ".nav", ".footer", ".header",
        ".menu", ".comments", "#comments", ".comment",
        ".sidebar", ".widget", ".meta", ".tags",
        "form", "input", "button",
    ]:
        for tag in soup.select(selector):
            tag.decompose()

    # Convert to markdown, falling back gracefully on deeply nested HTML
    try:
        text = md(str(soup), heading_style="ATX", strip=["img", "a"])
    except RecursionError:
        logger.warning("RecursionError in markdownify, falling back to get_text")
        text = soup.get_text("\n", strip=False)
    except Exception:  # noqa: BLE001 (fallback on any markdownify failure)
        logger.warning("markdownify failed, falling back to get_text")
        text = soup.get_text("\n", strip=False)

    # Clean up excessive whitespace and noise
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"_{3,}", "---", text)
    text = re.sub(r"\[(?:__|http[^\]]*)\]\([^)]*\)", "", text)
    text = re.sub(r"http\S+", "", text)

    # Remove lines that are just separators or short noise
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip navigation-looking lines
        if stripped in ("---", "***", "___") or re.match(r"^[\s\-_*]{3,}$", stripped):
            continue
        # Skip lines that are just URLs
        if re.match(r"^https?://\S+$", stripped):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


def is_content_worthwhile(content: str, min_length: int | None = None) -> bool:
    """Check if content is long enough to be worth processing."""
    if min_length is None:
        min_length = settings.min_content_length
    return len(content.strip()) >= min_length


def extract_technique_hints(content: str) -> list[str]:
    """Extract category/technique hints from writeup content.

    Looks for explicit category mentions, tags, and common CTF keywords.
    """
    hints = []
    content_lower = content.lower()

    for matched in _KEYWORD_RE.findall(content_lower):
        hints.extend(_KEYWORD_TAGS[matched])

    # Filter out noisy hints that are too generic to be useful
    filtered = [h for h in hints if h not in _NOISY_HINTS]

    return list(set(filtered))

