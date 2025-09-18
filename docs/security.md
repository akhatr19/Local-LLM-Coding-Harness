# Security model

The orchestration process runs on the host and can read the selected repository, artifact directory,
configured model endpoint, and SearXNG endpoint. Only use repositories and endpoints intended for
local automation.

Implementation commands run as a non-root user in disposable Docker containers with:

- no network;
- no Docker socket;
- a read-only container filesystem;
- dropped Linux capabilities and `no-new-privileges`;
- CPU, memory, process, disk, command, and total-run limits;
- only the exported working copy mounted writable.

API keys are used by the host-side model client and are not passed to implementation containers.
The source repository is exported at a fixed commit, and the final patch is checked against that
same clean revision before it is accepted.

Repository files, issue text, model output, and fetched pages are untrusted input. Path validation
prevents reads and writes outside the selected checkout. Web content is placed inside explicit
untrusted-data delimiters before it reaches a model.

Docker daemon access remains privileged. The harness does not make a generated patch safe to apply;
review the patch and command transcript before changing the source checkout.

The bundled SearXNG service binds to `127.0.0.1`. Change its placeholder secret and add a reverse
proxy with authentication before exposing it beyond the local machine.
