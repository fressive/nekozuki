---
description: |
  Automatically query the nekozuki CTF knowledge base when:
  - The user asks about a CTF technique, exploit, vulnerability, or attack vector you're not fully confident about
  - You need to verify or look up specific CTF-related technical details (e.g. PHP type juggling conditions, SSTI detection signs, SQL injection payload patterns)
  - The user asks a question that could be answered by a distilled CTF writeup knowledge base (39k writeups → technique files)
  - You encounter a security/CTF problem where you suspect there are known techniques, and checking the knowledge base would give a more authoritative answer
  - The user is working on a CTF challenge and you want to suggest relevant techniques
  DO NOT trigger for general programming questions, non-security topics, or when you are already confident in the answer.
version: 1.0.0
allowed-tools: [Bash, Read]
---

# nekozuki RAG Auto-Query

When the user asks a CTF/security question or you encounter a problem where a
known technique might apply, automatically query the nekozuki knowledge base
to get accurate, curated information from 39k CTF writeups.

## How to use

1. Run the query script with the user's question as the search query:

   ```
   bash "$CLAUDE_PLUGIN_ROOT/scripts/query.py" "<your search query>" -n 5
   ```

   If running from the project root, an equivalent alternative is:
   ```
   uv run plugin/scripts/query.py "<your search query>" -n 5
   ```

2. Craft a good query — extract the key technique/vulnerability terms from the
   user's question. Use English queries even if the user asked in Chinese.

3. Present the results inline in your response. The search returns:
   - **technique name** (e.g. `sql_injection`, `php_type_juggling`)
   - **trick title** (short description of the specific trick)
   - **description** (how it works)
   - Source: nekozuki CTF writeup knowledge base

4. If the results are insufficient, try a different query formulation before
   giving up. The knowledge base covers web, pwn, reverse, crypto, forensics,
   misc, and more.

## Example queries

| User asks | Auto-search query |
|---|---|
| "PHP 类型转换怎么绕过" | `"PHP type juggling"` |
| "SSTI 怎么检测" | `"SSTI detection signs"` |
| "这个 SQL 注入怎么搞" | `"SQL injection technique"` |
| "逆向工程里怎么识别加密算法" | `"identify encryption algorithm reverse engineering"` |

## Notes

- The script prefers a running nekozuki API server (fast, ~20-50ms, at
  `http://localhost:8000` or `$NEKOZUKI_URL`). If unavailable, it falls back
  to `uv run nekozuki query` which loads the full index (slower).
- If the index is not built (`nekozuki embed` hasn't been run), the query will
  fail. Inform the user if so.
- The CLI fallback output includes rerank scores; the API output does not.
  Both are useful — the API is just faster.