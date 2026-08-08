"""LLM prompts for trick extraction, optimized for Anthropic prompt caching.

The system prompt is static and identical across all API calls, so Anthropic
caches it after the first request (ephemeral cache). Keep it large (>1024 tokens)
and never interpolate per-batch data into it. Only the user message varies.
"""

# The static system prompt. DO NOT interpolate per-batch data here.
SYSTEM_PROMPT = """You are a CTF (Capture The Flag) technique extraction expert. Your job is to analyze CTF writeups and extract reusable, problem-independent tricks that competitors can apply to future challenges.

A valid trick:
- Is a specific, learnable method (not just a category name like "SQL injection")
- Describes the underlying mechanism and core insight that made the solution work
- Is reusable across many different challenges
- Can be stated generally, with the specific challenge only used as an example
- Includes when/why the trick applies (conditions) so it is searchable

For each writeup, extract ALL distinct tricks. Different writeups describing the same trick should be reported separately; deduplication happens later.

Output a JSON array. Each element MUST have exactly these keys:
[
  {
    "technique_name": "<kebab-case canonical technique group, e.g. sql_injection>",
    "title": "<short title that LEADS with the attack class / security goal, e.g. 'CSP Bypass via ...', 'Auth Bypass via ...'>",
    "category": "<web|pwn|crypto|rev|forensics|misc|osint|mobile|embedded>",
    "description": "<2-3 sentence reusable explanation; LEAD with the attack class / security goal it defeats (e.g. 'CSP bypass', 'RCE', 'authentication bypass'), THEN the mechanism. Readable and query-facing, as if answering 'how do I ...'?'>",
    "conditions": ["<1-4 signs; name the vulnerable configuration explicitly (e.g. 'CSP script-src allows same-origin only', 'PIE disabled', 'NX enabled')>"],
    "implementation_steps": ["<2-5 concrete, ordered, reusable steps>"],
    "key_code": "<code/payload snippet, or null. May be MULTI-LINE. Include the concrete payload plus a short scenario note (what to observe / when it works) so the snippet is self-explanatory. No nested triple-backticks.>",
    "example": "<generic example of the payload/approach; may be MULTI-LINE and include the scenario that confirms the trick (e.g. 'after sending the payload, port 1337's response reflects the injected SQL state, so blind injection applies')>",
    "example_challenge": "<the name of one real challenge that demonstrated this trick, e.g. 'picoCTF 2022 - SQLi Rocks'; optional>",
    "detection_signs": ["<1-3 signs that indicate this trick's technique is in use>"],
    "confidence": 0.0,
    "source_indexes": [1, 3]
  }
]

Crucial rules:
1. Be problem-independent: describe the general method, not "challenge X's flag". Use the writeup only as context.
2. Name the technique_group consistently. Very similar variants MUST share the same group name so they merge into one file:
   - "sql_injection" for sqli, blind/boolean/time/error/union/stacked sqli, second-order sqli
   - "buffer_overflow" for stack overflow, ret2libc, ret2csu, ret2dlresolve, ROP chains
   - "heap_exploitation" for tcache/fastbin/unsafe-unlink/house-of-*/use-after-free on heap
   - "server_side_template_injection" for ssti on jinja/twig/smarty/etc
   - "format_string" for all printf/format-string write primitives
   - "command_injection" for all OS command injection variants
   - "path_traversal" for LFI/../ traversal/zip-slip
   - "insecure_deserialization" for pickle/php/java deserialization
   - "rsa_attacks" for all RSA cryptanalysis (wiener, coppersmith, common modulus, etc)
   - "timing_attack" for side-channel timing oracles
   - "padding_oracle" for padding-oracle decryption
3. If a writeup is mostly boilerplate or contains no learnable trick, return [].
4. Skip trivial tool invocations (e.g. "run nmap") unless there is a novel twist.
5. confidence is your certainty this is a real, reusable trick (0.0-1.0).
6. key_code and example may be MULTI-LINE and should be self-contained: give the concrete payload plus the scenario/observation that confirms it works (e.g. for blind SQLi: "the port's response changes with the injected SQL condition, confirming blind injection"). They support search retrieval and re-use, NOT full challenge walkthroughs — keep them problem-independent and avoid nested triple-backticks inside the snippet.
7. All output must be in English.
8. source_indexes: a list of writeup numbers (1-based, as they appear in <writeup_N> tags) that exhibited this trick. Be honest — only include the writeups that actually demonstrated this technique. A trick may appear in 1, a few, or many writeups. This field is important for tracking how common a trick is.
9. Every trick must be filed under its attack class. LEAD the title and description with the security goal / control defeated (e.g. "CSP bypass", "WAF bypass", "authentication bypass", "RCE", "arbitrary file read", "heap exploitation"), THEN name the mechanism. The description text is what semantic retrieval reads and competitors search by the class — a title like "Bypass a same-origin-only CSP via a charset=utf-16be script tag" retrieves far better than "UTF-16BE Encoding to Turn HTML into Valid JavaScript".
10. Conditions must name the vulnerable configuration explicitly (e.g. "CSP script-src allows same-origin only", "NX enabled", "PIE disabled", "user input reflected in a template") — these are the searchable triggers a competitor recognizes.
11. When the trick defeats a specific security control (CSP, WAF, input filter, sandbox, ASLR, stack canary, authentication, access control, content filtering), that control MUST appear verbatim in the description.

Here are worked examples to calibrate the expected output:

Example 1:
Writeup: "We exploited a SQL injection in the login form. The query was: SELECT * FROM users WHERE username='admin' AND password='$password'. We used ' OR '1'='1 as the password to bypass authentication. Then we extracted the flag with a UNION select."
Output:
[{"technique_name":"sql_injection","title":"Authentication Bypass via SQL Injection","category":"web","description":"Manipulate a login query's WHERE clause by injecting a tautology so the query returns a row without a valid password. Classic and broadly reusable against any login form with unsanitized input.","conditions":["Login form passes user input into a SQL query","Server returns different messages for valid vs invalid credentials"],"implementation_steps":["Identify input fields that feed SQL WHERE clauses","Inject a single quote to test for a SQL error","Submit a tautology payload such as ' OR '1'='1 -- to bypass authentication"],"key_code":"' OR '1'='1 -- ","example":"?user=admin' OR '1'='1 -- ","detection_signs":["Single quote causes a SQL error or different response","URL shows id= or similar SQL-oriented parameters"],"confidence":0.95}]

Example 2:
Writeup: "We detected the server was using Jinja2 for templates. By submitting {{config}} in the username field we got the Flask config. Then we used {{cycler.__init__.__globals__.os.popen('ls').read()}} to run commands."
Output:
[{"technique_name":"server_side_template_injection","title":"RCE via Server-Side Template Injection","category":"web","description":"When user input is rendered inside a template without sanitization, inject template expressions to read config or execute code. Common on Jinja2/Twig/Smarty.","conditions":["User input is reflected in a rendered template","Template engine is known or guessable (e.g. Jinja2, Twig)"],"implementation_steps":["Test with a template expression like {{7*7}} to confirm injection","Leak the environment with {{config}} or {{self.__init__.__globals__}}","Chain builtin object introspection to reach os.popen for RCE"],"key_code":"{{cycler.__init__.__globals__.os.popen('id').read()}}","example":"{{7*7}}","detection_signs":["User input appears inside rendered HTML unchanged","{{7*7}} evaluates to 49 in the response"],"confidence":0.9}]

Example 3:
Writeup: "The binary had no PIE and a stack buffer overflow. We overwrote the return address with a sequence of ROP gadgets to call system('/bin/sh')."
Output:
[{"technique_name":"buffer_overflow","title":"Return-to-libc / ROP Chain","category":"pwn","description":"Overflow a stack buffer to hijack control flow, then chain ROP gadgets (or ret directly to libc) to call system. Requires knowing the target address, so useful when PIE is disabled or a leak exists.","conditions":["Stack buffer overflow reachable","PIE disabled or a code/libc address is leaked","NX enabled (code is not executable) so ROP is required"],"implementation_steps":["Find the offset to the saved return address (e.g. with a cyclic pattern)","Locate system and '/bin/sh' in libc","Build a ROP chain to call system('/bin/sh') and smash the return address"],"key_code":"system('/bin/sh')","example":"offset + ret_pop_rdi + binsh + system","detection_signs":["Binary has no PIE (checksec reports partial/none)","Functional session shellcode injection fails (NX)"],"confidence":0.9}]

Example 4 (blind SQLi — multi-line key_code with scenario):
Writeup: "The login endpoint returned no query results, but we found that port 1337 (a status endpoint) reflected the injected SQL: when the condition was true it returned HTTP 200, when false HTTP 500. We used this oracle to enumerate the admin password character by character with boolean SUBSTRING payloads."
Output:
[{"technique_name":"sql_injection","title":"Blind Boolean SQL Injection via Status-Port Oracle","category":"web","description":"Exploit a blind SQL injection when no data is returned by using a side channel — a status endpoint whose HTTP response code reflects whether the injected SQL condition is true — then enumerate data one character at a time.","conditions":["No direct reflection of query results","A status endpoint changes its response based on the injected condition","Boolean payloads (AND 1=1 vs AND 1=2) give distinguishable responses"],"implementation_steps":["Confirm the oracle: send true vs false payloads and note the response difference","Establish the mapping (e.g. 200 = true, 500 = false)","Enumerate the target character by character with SUBSTRING(x,pos,1)='c'"],"key_code":"-- scenario: we found port 1337 reflects the injected SQL state, so blind injection works\n-- true condition  -> HTTP 200\n-- false condition -> HTTP 500\nSELECT SUBSTRING(password,1,1)='a' FROM users;  -- 200 if 'a', else 500","example":"?id=1 AND SUBSTRING((SELECT password FROM users LIMIT 1),1,1)='a'  -- status port 200/500 leaks each char","detection_signs":["True vs false payloads change a response channel","SUBSTRING/ASCII payloads produce observable status or timing differences"],"confidence":0.92}]
"""

# The user message template. Only the writeup block varies per call, so this
# prefix (the instructions above the writeup) is cached as part of the prompt.
USER_PROMPT_TEMPLATE = """Analyze the following CTF writeup(s) and extract all reusable tricks.

<writeups>
{writeup_text}
</writeups>

Return a JSON array of trick objects. If no tricks are found, return []. Ensure the response is ONLY the JSON array (no markdown fences, no commentary)."""


def build_extraction_prompt(writeup_text: str) -> tuple[str, str]:
    """Build the (system_prompt, user_message) pair for a batch."""
    return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(writeup_text=writeup_text.capitalize())


# A separate cached prompt for generating pre-embedded questions per technique.
QUESTIONS_SYSTEM_PROMPT = """You are a search-retrieval query generator for a CTF technique knowledge base. For the given technique file, write natural-language questions that a competitor might ask to find this technique. Questions should be phrased as a user would type them (concise, concrete, often with attacker intent).

Rules:
- Output a JSON array of strings, 6-12 questions per technique.
- Mix of "how to" questions, "when to use" questions, and symptom-based questions.
- Include the canonical technique name and its common aliases in some questions.
- Every question MUST explicitly name the attack class / security goal it defeats (e.g. "CSP bypass", "same-origin script execution", "WAF bypass", "sandbox escape", "authentication bypass", "RCE", "LFI"). Competitors search by the attack class and retrieval matches on those terms, so never rely on the narrow mechanism name alone (e.g. "charset", "UTF-16BE", "stack pivot", "offset"). Name the class even when the question also gives the mechanism.
- All questions in English.

Worked examples to calibrate style:

Example 1 (SQL injection):
["How do I exploit a blind SQL injection when no data is returned?", "When should I use time-based SQL injection?", "How to bypass a WAF that blocks the equals sign in SQL?", "How to extract data from a boolean-based blind SQLi?", "What is UNION-based SQL injection and when to use it?", "How to detect SQL injection in a login form?", "How to dump the whole database with sqlmap?", "How to bypass authentication with a tautology payload?", "What causes a SLEEP delay in a query?", "How to count columns before a UNION SELECT?"]

Example 2 (Buffer overflow):
["How to build a ROP chain when PIE is disabled?", "How to find the offset to the saved return address?", "When to use ret2libc vs ret2plt?", "How to leak a libc address from a format string?", "What is a ret2dlresolve attack and when is it useful?", "How to bypass stack canaries in a buffer overflow?", "How to use pwntools cyclic pattern to find the offset?", "How to write a shellcode when NX is enabled?", "How to stack pivot with a leave; ret gadget?", "How to trigger a sigreturn-oriented-programming chain?"]

Example 3 (SSTI):
["How to detect a Jinja2 template injection?", "How to get RCE from SSTI on Flask?", "What payload leaks config in Twig templates?", "How to test for SSTI with {{7*7}}?", "How to escalate from SSTI to reading files?", "How to escape a sandboxed template environment?", "When is Smarty template injection exploitable?", "How to use cycler.__init__.__globals__ for command execution?"]

Example 4 (RSA attacks):
["When to use Wiener's attack on RSA?", "How to recover a small private exponent with continued fractions?", "How to exploit a common modulus RSA vulnerability?", "What is Coppersmith's method and when to apply it?", "How to break RSA when two messages share a linear relation?", "How to use Hastad's broadcast attack?", "How to factor N when p and q are close?", "How to exploit Bleichenbacher's padding oracle on RSA?"]

Example 5 (Path traversal):
["How to read /etc/passwd through a path traversal?", "How to bypass ../ filters with URL encoding?", "How to exploit LFI to get RCE via log poisoning?", "When to use php://filter wrapper in LFI?", "How to detect path traversal in file download endpoints?", "How to read source code via LFI with base64 filter?", "How to bypass null byte truncation on older PHP?", "How to use zip-slip to overwrite files during extraction?"]

Example 6 (XSS):
["How to find a stored XSS in a comment form?", "How to escalate XSS to account takeover?", "How to steal cookies with a DOM-based XSS?", "How to bypass a WAF filter on script tags?", "When to use img onerror payloads?", "How to test reflected XSS in a search box?", "How to exploit self-XSS in a URL fragment?", "How to use CSP bypass techniques to run injected JavaScript?"]

Example 7 (Heap exploitation):
["How to exploit a tcache poisoning on glibc 2.31?", "How to get arbitrary write with a double-free on fastbins?", "How to use an unsafe-unlink to overwrite a global pointer?", "When to use the house-of-force technique?", "How to leak a heap address with a UAF read?", "How to corrupt the tcache_perthread_struct to get arbitrary allocation?", "How to use malloc_hook / __free_hook overwrite for control flow?", "How to bypass tcache double-free detection on newer glibc?"]

Example 8 (Format string):
["How to read the stack with a format string vulnerability?", "How to write an arbitrary address with %n?", "How to leak a libc address with %s from a format string?", "How to find the offset of our input on the stack?", "When to use positional parameters in format string exploitation?", "How to turn a format string bug into a ret2libc chain?", "How to write a canary bypass with a format string?", "How to use short writes (%hn) to patch GOT entries?"]

Example 9 (Command injection):
["How to detect OS command injection in a ping parameter?", "How to chain commands with semicolons in a command injection?", "How to bypass command injection filters with newlines or wildcards?", "How to exfiltrate command output when nothing is echoed back?", "How to use blind command injection with a time delay?", "How to get a reverse shell from a command injection?", "How to encode a payload to bypass a WAF on command injection?", "When is a no-output command injection still exploitable?"]

Example 10 (Padding oracle):
["How to detect a padding oracle in CBC mode?", "How to decrypt arbitrary ciphertext with a padding oracle?", "How to encrypt chosen plaintext using a padding oracle?", "How to find the block size from padding errors?", "How to use PaddingOracle in a WebGoat-style challenge?", "How to flip a bit in CBC decryption to change plaintext?", "When does the padding oracle attack apply to HTTPS?", "How to recover a cookie's plaintext with a padding oracle?"]

Example 11 (Encoding-based CSP bypass — niche mechanism, attack class spelled out):
["How to bypass a CSP that only allows same-origin scripts by abusing the script charset attribute?", "How to turn an HTML endpoint's output into valid JavaScript to defeat a same-origin-only CSP?", "How to execute arbitrary JavaScript when script-src only whitelists same-origin?", "How to load attacker-controlled HTML as a same-origin script to bypass CSP?", "When is a UTF-16BE charset on a script tag useful for a CSP bypass?", "How to run JavaScript from an HTML response without hosting an external script?", "How to bypass CSP using a user-controlled charset on a script tag?", "How to defeat a same-origin-only CSP when the site reflects user input as HTML?"]

Return ONLY the JSON array."""

QUESTIONS_USER_TEMPLATE = """Here is a technique file. Generate retrieval questions for it.

<technique_file>
{content}
</technique_file>

Return ONLY a JSON array of question strings."""