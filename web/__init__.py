"""AIAR watcher GUI — a stdlib http.server web app.

The flow that matters: simulate a prompt -> see the response -> mark it for
evaluation -> evaluate (score 1-10 + reason/correction) -> click Reground to
feed evaluated pairs back into the grounding store -> verify the regrounded
answer improved.

Pure standard library + the AIAR package; no web framework needed.
"""
