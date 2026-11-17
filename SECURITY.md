# Security Policy

DenseX is an independent fork of llama.cpp maintained on a best-effort basis. The upstream llama.cpp security program does not apply to this repository, and there is no other private channel (no email address) for this project.

## Reporting a vulnerability

- Not a security issue, or you are not sure: open a public issue at https://github.com/bext1998/DenseX/issues.
- Genuine security vulnerability: report it privately through GitHub Security Advisories for this repository, https://github.com/bext1998/DenseX/security/advisories/new. Do not put exploit details in a public issue.
- If the private advisory form is unavailable, open a public issue that describes the impact and the affected component without exploit details.

Expect a best-effort, non-contracted response; no fix timeline is promised here. Fixes land on the `densex` branch of this repository only.

## Scope

In scope: the DenseX C/C++ sources (`src/`, `ggml/`, `common/`, `tools/`, `include/`), the Python conversion tools, and `vendor/` code as it is used by this repository.

For issues inside a `vendor/` copy itself, report them to the third-party project that owns it. Bugs in third-party prebuilt packages or in upstream llama.cpp do not belong in this repository's tracker.

## Using an inference server and models safely

These are practical notes, not response commitments:

- Run models you did not build yourself in an isolated environment (container/VM); treat model files from unknown sources as untrusted input.
- Treat everything a client can send (HTTP headers, JSON fields, grammar/schema input, uploaded files) as attacker-controlled, and do not expose the server to an untrusted network without authentication in front of it.
- Prompt injection and other model-output-driven actions (tool calling, file access) are a real risk: scope what the server is allowed to do.
- Checksums for model files are published in the install guide (`docs/densex/INSTALL.md`); verify downloads against them.
